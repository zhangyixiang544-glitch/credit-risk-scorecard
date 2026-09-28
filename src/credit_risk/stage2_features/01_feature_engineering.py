"""特征工程脚本：宽表预处理、业务特征衍生与特征初筛。

阶段二第一步。输入阶段一宽表 master_table.csv，完成四件事：
1. 数据预处理：低缺失字段填充（数值中位数/类别众数），中高缺失字段加缺失标识
2. 异常值处理：DAYS_EMPLOYED=365243 加失业标记，疑似录入错误盖帽
3. 业务特征衍生：收入负债比、贷款收入比、年龄、贷款商品价比、外部评分均值
4. 特征初筛：常量列剔除、高相关剔除、VIF 共线性检验，输出候选特征清单

输出：
- processed_features.csv：预处理与衍生后的特征表（供 WOE 分箱使用）
- feature_list.csv：候选特征清单（业务维度、初筛状态、入模理由）
"""

import numpy as np
import pandas as pd
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUT_DIR = os.path.join(PROJECT_ROOT, "output", "stage2")

MASTER_FILE = DATA_DIR + "\\master_table.csv"

# 缺失率分级阈值（与阶段一保持一致）
MISSING_LOW = 0.1
MISSING_HIGH = 0.3

# 高相关阈值与 VIF 阈值
CORR_THRESHOLD = 0.95
VIF_THRESHOLD = 10

# DAYS_EMPLOYED 的特殊取值：365243 表示失业（阶段一结论）
UNEMPLOYED_CODE = 365243


def load_master() -> pd.DataFrame:
    """读取阶段一宽表。

    Returns:
        master_table DataFrame
    """
    df = pd.read_csv(MASTER_FILE)
    df = optimize_dtypes(df)
    return df


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """把数值列类型往小了调，节省内存。

    Args:
        df: 原始 DataFrame

    Returns:
        类型优化后的 DataFrame
    """
    for col in df.columns:
        if df[col].dtype == "int64":
            col_min = df[col].min()
            col_max = df[col].max()
            if col_min >= -128 and col_max <= 127:
                df[col] = df[col].astype("int8")
            elif col_min >= -32768 and col_max <= 32767:
                df[col] = df[col].astype("int16")
            elif col_min >= -2147483648 and col_max <= 2147483647:
                df[col] = df[col].astype("int32")
        elif df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
    return df


def _append_columns(df: pd.DataFrame, new_columns: dict) -> pd.DataFrame:
    """把新列一次性拼接到 DataFrame 上。

    相比逐列插入，一次性拼接不会让 DataFrame 碎片化，性能更好。

    Args:
        df: 原 DataFrame
        new_columns: {新列名: 列值} 字典

    Returns:
        拼接后的 DataFrame
    """
    return pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """缺失处理：低缺失填充，中高缺失加标识特征。

    低缺失（<10%）：数值列填中位数、类别列填众数。
    中高缺失（>=10%）：保留缺失值（WOE 分箱时单独成箱），新增
    {字段名}_MISSING 的 0/1 标识列，让缺失本身成为信息。

    Args:
        df: 宽表 DataFrame

    Returns:
        处理后的 DataFrame（新增缺失标识列）
    """
    new_columns = {}
    for col in df.columns:
        if col in ("SK_ID_CURR", "TARGET"):
            continue
        missing_rate = df[col].isna().mean()
        if missing_rate == 0:
            continue
        if missing_rate < MISSING_LOW:
            if pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mode().iloc[0])
            else:
                df[col] = df[col].fillna(df[col].median())
        else:
            new_columns[col + "_MISSING"] = df[col].isna().astype("int8")

    if new_columns:
        df = _append_columns(df, new_columns)
    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """异常值处理：特殊编码加标记，疑似录入错误盖帽。

    - DAYS_EMPLOYED=365243 表示失业：新增 IS_UNEMPLOYED 标记列，原值保留，
      交给 WOE 分箱时单独成箱（阶段一报告结论）
    - 年金/收入 > 1：盖帽到 1（阶段一标注为疑似录入错误）
    - CNT_CHILDREN > 5：盖帽到 5（阶段一标注为疑似录入错误）
    - AMT_INCOME_TOTAL > 100 万：极端真实样本，保留（阶段一结论）

    Args:
        df: 宽表 DataFrame

    Returns:
        处理后的 DataFrame（新增失业标记列）
    """
    new_columns = {}
    new_columns["IS_UNEMPLOYED"] = (df["DAYS_EMPLOYED"] == UNEMPLOYED_CODE).astype(
        "int8"
    )

    debt_ratio = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    new_columns["DEBT_RATIO_TEMP"] = debt_ratio.replace(
        [np.inf, -np.inf], np.nan
    ).clip(upper=1.0)

    df["CNT_CHILDREN"] = df["CNT_CHILDREN"].clip(upper=5)
    if new_columns:
        df = _append_columns(df, new_columns)
    return df


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """业务特征衍生：基于风控业务常识构造主表衍生特征。

    附表聚合特征（逾期率、申请频次、授信使用率等）阶段一已生成，
    这里补主表层面的业务比值与维度特征。

    Args:
        df: 宽表 DataFrame

    Returns:
        衍生后的 DataFrame
    """
    income = df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    new_columns = {}
    new_columns["DEBT_RATIO"] = df["DEBT_RATIO_TEMP"]
    new_columns["CREDIT_INCOME_RATIO"] = (df["AMT_CREDIT"] / income).replace(
        [np.inf, -np.inf], np.nan
    )
    new_columns["CREDIT_GOODS_RATIO"] = (
        df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
    ).replace([np.inf, -np.inf], np.nan)
    new_columns["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365

    ext_sources = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    new_columns["EXT_SOURCE_MEAN"] = df[ext_sources].mean(axis=1)

    df = _append_columns(df, new_columns)
    df.drop(columns=["DEBT_RATIO_TEMP"], inplace=True)
    return df


def variance_filter(df: pd.DataFrame, numeric_cols: list) -> list:
    """常量列剔除：唯一值只有 1 个的列没有区分度，直接剔除。

    Args:
        df: 特征表 DataFrame
        numeric_cols: 数值特征列名列表

    Returns:
        剔除的列名列表
    """
    dropped = []
    for col in numeric_cols:
        if df[col].nunique() <= 1:
            dropped.append(col)
    return dropped


def correlation_filter(df: pd.DataFrame, numeric_cols: list) -> list:
    """高相关剔除：|相关系数| > 阈值时保留前者、剔除后者。

    Args:
        df: 特征表 DataFrame
        numeric_cols: 数值特征列名列表

    Returns:
        剔除的列名列表
    """
    corr_matrix = df[numeric_cols].corr()
    dropped = []
    cols = list(numeric_cols)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col_i = cols[i]
            col_j = cols[j]
            if col_i in dropped or col_j in dropped:
                continue
            if abs(corr_matrix.loc[col_i, col_j]) > CORR_THRESHOLD:
                dropped.append(col_j)
    return dropped


def calc_vif(df: pd.DataFrame, cols: list) -> dict:
    """计算各特征列的 VIF（方差膨胀因子）。

    用相关矩阵的逆矩阵对角元计算：VIF = diag(pinv(corr))，
    比逐列回归快很多；用伪逆避免特征高度相关时矩阵奇异崩溃。

    Args:
        df: 特征表 DataFrame
        cols: 参与计算的数值列名列表

    Returns:
        {列名: VIF 值} 字典
    """
    x = df[cols].to_numpy(dtype="float64")
    # 只保留没有缺失的行，避免 NaN 混入相关计算
    mask = np.isfinite(x).all(axis=1)
    x = x[mask]
    # 标准化
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-9)
    corr = x.T @ x / x.shape[0]
    corr_inv = np.linalg.pinv(corr)
    vif = np.diag(corr_inv)
    return {cols[i]: float(vif[i]) for i in range(len(cols))}


def vif_filter(df: pd.DataFrame, cols: list, max_rounds: int = 3) -> list:
    """VIF 迭代剔除：每轮剔除 VIF 最大的列，最多迭代 max_rounds 轮。

    Args:
        df: 特征表 DataFrame
        cols: 数值列名列表
        max_rounds: 最大迭代轮数

    Returns:
        被剔除的列名列表
    """
    dropped = []
    current = [c for c in cols if c not in dropped]
    for _ in range(max_rounds):
        vif_dict = calc_vif(df, current)
        if not vif_dict:
            break
        worst_col = max(vif_dict, key=vif_dict.get)
        if vif_dict[worst_col] <= VIF_THRESHOLD:
            break
        dropped.append(worst_col)
        current.remove(worst_col)
    return dropped


def classify_dimension(col: str) -> str:
    """把特征归到四大业务维度（任务书要求输出特征体系框架）。

    规则：按列名前缀与白名单判断，用于特征清单标注。

    Args:
        col: 字段名

    Returns:
        业务维度：基础属性/外部征信/还款行为/衍生指标
    """
    base_fields = [
        "SK_ID_CURR",
        "NAME_",
        "CODE_GENDER",
        "FLAG_OWN_",
        "CNT_CHILDREN",
        "CNT_FAM_MEMBERS",
        "AMT_INCOME_TOTAL",
        "DAYS_BIRTH",
        "DAYS_REGISTRATION",
        "DAYS_ID_PUBLISH",
        "OWN_CAR_AGE",
        "OCCUPATION_TYPE",
        "REGION_",
        "ORGANIZATION_TYPE",
        "WEEKDAY_",
        "HOUR_",
        "FLAG_MOBIL",
        "FLAG_EMP_",
        "FLAG_WORK_",
        "FLAG_CONT_",
        "FLAG_PHONE",
        "FLAG_EMAIL",
        "FLAG_DOCUMENT_",
        "FONDKAPREMONT_",
        "HOUSETYPE_",
        "TOTALAREA_",
        "WALLSMATERIAL_",
        "EMERGENCYSTATE_",
        "LIVE_",
        "REG_",
        "APARTMENTS_",
        "BASEMENTAREA_",
        "YEARS_",
        "COMMONAREA_",
        "ELEVATORS_",
        "ENTRANCES_",
        "FLOORS",
        "LANDAREA_",
        "LIVING",
        "NONLIVING",
    ]
    credit_fields = [
        "EXT_SOURCE_",
        "AMT_REQ_CREDIT_BUREAU_",
        "bureau",
        "bb_",
    ]
    repayment_fields = [
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "DAYS_EMPLOYED",
        "DAYS_LAST_PHONE_CHANGE",
        "OBS_",
        "DEF_",
        "prev_",
        "pos_",
        "cc_",
        "inst_",
    ]
    derived_fields = [
        "IS_UNEMPLOYED",
        "DEBT_RATIO",
        "CREDIT_INCOME_RATIO",
        "CREDIT_GOODS_RATIO",
        "AGE_YEARS",
        "EXT_SOURCE_MEAN",
    ]
    for prefix in credit_fields:
        if col.startswith(prefix):
            return "外部征信"
    for prefix in repayment_fields:
        if col.startswith(prefix):
            return "还款行为"
    if col in derived_fields:
        return "衍生指标"
    for prefix in base_fields:
        if col.startswith(prefix):
            return "基础属性"
    return "其他"


def main() -> None:
    """主流程：预处理、衍生、初筛，输出特征表与特征清单。"""
    print("=" * 60)
    print("特征工程开始（预处理 + 衍生 + 初筛）")
    print("=" * 60)

    df = load_master()
    print(f"宽表维度: {df.shape}")

    # 1. 预处理
    df = fill_missing(df)
    df = handle_outliers(df)
    print(f"处理完成，当前维度: {df.shape}")

    # 2. 业务特征衍生
    df = derive_features(df)
    print(f"衍生完成，当前维度: {df.shape}")

    # 3. 特征初筛（只对数值特征做，类别特征留到 WOE 编码后）
    numeric_cols = [
        col
        for col in df.columns
        if col not in ("SK_ID_CURR", "TARGET")
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    dropped = []
    dropped += variance_filter(df, numeric_cols)
    remaining = [c for c in numeric_cols if c not in dropped]
    dropped += correlation_filter(df, remaining)
    remaining = [c for c in numeric_cols if c not in dropped]
    dropped += vif_filter(df, remaining)

    # 4. 保存处理后的特征表（保留全部特征，初筛结果只在清单中标注）
    out_file = OUT_DIR + "\\processed_features.csv"
    df.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"特征表已保存: {out_file}")

    # 5. 输出特征体系清单
    records = []
    for col in df.columns:
        if col in ("SK_ID_CURR", "TARGET"):
            continue
        missing_rate = df[col].isna().mean()
        if col in dropped:
            status = "初筛剔除"
        else:
            status = "保留"
        records.append(
            {
                "字段名": col,
                "业务维度": classify_dimension(col),
                "数据类型": str(df[col].dtype),
                "缺失率": round(missing_rate, 4),
                "初筛状态": status,
                "入模理由": "",
            }
        )
    feature_list = pd.DataFrame(records)
    list_file = OUT_DIR + "\\feature_list.csv"
    feature_list.to_csv(list_file, index=False, encoding="utf-8-sig")
    print(f"特征清单已保存: {list_file}")


if __name__ == "__main__":
    main()
