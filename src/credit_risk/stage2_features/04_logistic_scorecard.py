"""逻辑回归评分卡脚本：WOE 替换 → 逻辑回归 → 系数显著性 → 评分刻度。

输入阶段一宽表与 02 WOE 分箱结果，流程：
1. 把入模候选特征的原始值按 02 的分箱规则映射为 WOE 值
2. 逻辑回归拟合（statsmodels），输出系数与 p 值，剔除不显著特征
3. 按标准评分卡公式换算评分刻度（基准分 600、每 20 分翻倍）
4. 输出客户评分，供阈值策略脚本测算通过率与坏账率

输出：
- logit_coefficients.csv：最终模型系数、p 值、显著性标记
- model_performance.csv：训练/测试 AUC 与 KS
- scorecard_points.csv：特征分箱分值表（评分卡展示用）
- scores_train.csv / scores_test.csv：客户评分（SK_ID_CURR、TARGET、SCORE）
"""

import math
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


OUT_DIR = os.path.join(PROJECT_ROOT, "output", "stage2")

FEATURE_FILE = OUT_DIR + "\\processed_features.csv"
ENC_FILE = OUT_DIR + "\\woe_encoding_map.csv"
CHECK_FILE = OUT_DIR + "\\loan_availability_check.csv"

# 随机划分与建模参数
RANDOM_STATE = 42
TEST_SIZE = 0.3
P_THRESHOLD = 0.05
MAX_DROP_ROUNDS = 2

# 评分刻度参数（行业常用设定）
BASE_SCORE = 600
BASE_ODDS = 50
PDO = 20

# 分箱参数（与 02 一致）
NUM_BINS = 10
CAT_MIN_SHARE = 0.01
UNEMPLOYED_CODE = 365243
BUSINESS_EDGES = {
    "AGE_YEARS": [18, 25, 30, 35, 40, 45, 50, 60, 120],
    "DEBT_RATIO": [0, 0.2, 0.35, 0.5, 0.7, 1.0],
}


def load_master() -> pd.DataFrame:
    """读取 01 产出的预处理特征表（含缺失标识与衍生特征）。

    Returns:
        特征表 DataFrame
    """
    df = pd.read_csv(FEATURE_FILE)
    return df


def load_candidates() -> list:
    """读取合规清单中的全部入模候选特征。

    Returns:
        候选特征列名列表（2 个经人工复核的计数特征）
    """
    check = pd.read_csv(CHECK_FILE)
    return check["字段名"].tolist()


def build_woe_map() -> dict:
    """把 WOE 映射表转成 {特征名: {箱标签: WOE}} 字典。

    Returns:
        两级嵌套字典，箱标签为字符串（区间或类别名）
    """
    enc = pd.read_csv(ENC_FILE)
    woe_map = {}
    for _, row in enc.iterrows():
        col = row["特征名"]
        if col not in woe_map:
            woe_map[col] = {}
        woe_map[col][str(row["箱"])] = row["WOE"]
    return woe_map


def bin_series(series: pd.Series, col: str) -> pd.Series:
    """单个特征分箱，规则与 02 脚本完全一致。

    Args:
        series: 特征列
        col: 特征名

    Returns:
        箱标签列（字符串，可直接映射 WOE）
    """
    if col in BUSINESS_EDGES:
        bins = pd.cut(series, bins=BUSINESS_EDGES[col], right=True, include_lowest=True)
        labels = bins.astype(str)
        labels[series.isna()] = "缺失"
        return labels
    if col == "DAYS_EMPLOYED":
        labels = pd.Series(dtype="object", index=series.index)
        unemployed = series == UNEMPLOYED_CODE
        valid_mask = ~unemployed & series.notna()
        valid_bins = pd.qcut(
            series.loc[valid_mask], q=NUM_BINS, duplicates="drop"
        ).astype(str)
        labels[valid_mask] = valid_bins.values
        labels[unemployed] = "失业"
        labels[series.isna()] = "缺失"
        return labels
    if pd.api.types.is_string_dtype(series):
        value_share = series.value_counts(normalize=True)
        keep_cats = value_share[value_share >= CAT_MIN_SHARE].index.tolist()
        labels = series.where(series.isin(keep_cats), "其他")
        labels = labels.astype(str)
        labels[series.isna()] = "缺失"
        return labels
    labels = pd.Series(dtype="object", index=series.index)
    valid = series[series.notna()]
    if valid.nunique() < 2:
        labels[series.notna()] = "单一取值"
    else:
        try:
            valid_bins = pd.qcut(valid, q=NUM_BINS, duplicates="drop").astype(str)
            labels[series.notna()] = valid_bins.values
        except ValueError:
            labels[series.notna()] = "单一取值"
    labels[series.isna()] = "缺失"
    return labels


def drop_redundant(X: pd.DataFrame, corr_threshold: float = 0.95) -> tuple:
    """剔除 WOE 矩阵中的常量列与高相关列。

    WOE 转换后部分特征可能趋近常量或强相关，与原始值初筛不完全一致，
    建模前再剔除一轮。

    Args:
        X: WOE 特征矩阵
        corr_threshold: 高相关阈值

    Returns:
        (剔除后的矩阵, 被剔除的列名列表)
    """
    dropped = []
    # 常量列：方差为 0 或唯一值只有 1 个
    for col in X.columns:
        if X[col].nunique() <= 1 or X[col].std() == 0:
            dropped.append(col)
    X = X.drop(columns=dropped)

    # 高相关列：|相关系数| 超过阈值时保留前者、剔除后者
    corr = X.corr()
    cols = list(X.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col_i = cols[i]
            col_j = cols[j]
            if col_i in dropped or col_j in dropped:
                continue
            if abs(corr.loc[col_i, col_j]) > corr_threshold:
                dropped.append(col_j)
    X = X.drop(columns=[c for c in cols if c in dropped])
    return X, dropped


def build_woe_matrix(df: pd.DataFrame, cols: list, woe_map: dict) -> pd.DataFrame:
    """把候选特征原始值替换为 WOE 值，生成模型输入矩阵。

    Args:
        df: 宽表 DataFrame
        cols: 入模特征列名列表
        woe_map: {特征名: {箱标签: WOE}} 字典

    Returns:
        WOE 值矩阵 DataFrame（列名为原特征名）
    """
    woe_cols = {}
    for col in cols:
        labels = bin_series(df[col], col)
        col_woe = labels.map(woe_map.get(col, {}))
        woe_cols[col] = col_woe
    return pd.DataFrame(woe_cols)


def calc_ks(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """计算 KS 统计量：好坏累计占比差的最大值。

    Args:
        y_true: 真实标签数组
        y_score: 预测分数数组

    Returns:
        KS 值
    """
    data = pd.DataFrame({"y": y_true, "score": y_score})
    data = data.sort_values("score", ascending=False)
    data["cum_bad"] = data["y"].cumsum() / data["y"].sum()
    data["cum_good"] = (1 - data["y"]).cumsum() / (1 - data["y"]).sum()
    return float((data["cum_bad"] - data["cum_good"]).abs().max())


def fit_logit(X: pd.DataFrame, y: pd.Series) -> sm.Logit:
    """拟合逻辑回归并返回结果对象。

    Args:
        X: 特征矩阵
        y: 目标列

    Returns:
        statsmodels 拟合结果
    """
    return sm.Logit(y, X).fit(maxiter=100, disp=False)


def significance_filter(
    X: pd.DataFrame, y: pd.Series, max_rounds: int = MAX_DROP_ROUNDS
) -> tuple:
    """迭代剔除 p 值不显著的特征，返回最终列与模型。

    Args:
        X: 特征矩阵
        y: 目标列
        max_rounds: 最大剔除轮数

    Returns:
        最终特征列名列表, 最终模型结果
    """
    cols = list(X.columns)
    for _ in range(max_rounds):
        probe = fit_logit(X[cols], y)
        insignificant = [
            c for c in cols if c != "const" and probe.pvalues[c] >= P_THRESHOLD
        ]
        if not insignificant:
            return cols, probe
        cols = [c for c in cols if c not in insignificant]
    model = fit_logit(X[cols], y)
    return cols, model


def build_scorecard(model: sm.Logit, cols: list, woe_map: dict) -> pd.DataFrame:
    """按评分公式计算每个特征每箱的分值。

    总分 = offset - factor * (常数项 + 各特征系数 * WOE)，
    每个箱的分值 = -factor * 系数 * WOE。

    Args:
        model: 最终逻辑回归模型
        cols: 入模特征列名列表
        woe_map: WOE 映射字典

    Returns:
        分值表 DataFrame
    """
    factor = PDO / math.log(2)
    offset = BASE_SCORE - factor * math.log(BASE_ODDS)
    intercept = model.params.get("const", 0.0)

    records = []
    for col in cols:
        beta = model.params[col]
        for bin_label, woe in woe_map.get(col, {}).items():
            points = round(-factor * beta * woe, 2)
            records.append(
                {
                    "特征名": col,
                    "箱": bin_label,
                    "WOE": round(woe, 4),
                    "系数": round(beta, 5),
                    "分值": points,
                }
            )
    baseline = round(offset - factor * intercept, 2)
    records.append(
        {
            "特征名": "基准分",
            "箱": "常数项",
            "WOE": "",
            "系数": round(intercept, 5),
            "分值": baseline,
        }
    )
    return pd.DataFrame(records)


def calc_score(model: sm.Logit, X: pd.DataFrame) -> pd.Series:
    """按评分公式计算每个客户的最终得分。

    Args:
        model: 最终逻辑回归模型
        X: 特征矩阵（已含常数列）

    Returns:
        每个客户的得分 Series
    """
    factor = PDO / math.log(2)
    offset = BASE_SCORE - factor * math.log(BASE_ODDS)
    logit_value = model.predict(X)
    return offset - factor * np.log(logit_value / (1 - logit_value + 1e-9))


def main() -> None:
    """主流程：WOE 映射、建模、显著性剔除、评分输出。"""
    print("=" * 60)
    title = "逻辑回归评分卡构建开始"
    print(title.center(60, "="))

    df = load_master()
    candidates = load_candidates()
    woe_map = build_woe_map()
    print(f"入模候选特征: {len(candidates)} 个")

    # 1. 生成 WOE 特征矩阵
    X = build_woe_matrix(df, candidates, woe_map)
    X, dropped_cols = drop_redundant(X)
    print(f"WOE 矩阵剔除常量/高相关列 {len(dropped_cols)} 个，剩余特征 {X.shape[1]} 个")
    y = df["TARGET"]
    print(f"WOE 矩阵维度: {X.shape}")

    # 2. 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"训练集: {X_train.shape[0]} 行，测试集: {X_test.shape[0]} 行")

    # 3. 逻辑回归 + 显著性剔除
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)
    final_cols, model = significance_filter(X_train, y_train)
    print(f"最终保留特征: {len(final_cols) - 1} 个（剔除不显著特征）")

    # 4. 模型评估
    train_auc = roc_auc_score(y_train, model.predict(X_train[final_cols]))
    test_auc = roc_auc_score(y_test, model.predict(X_test[final_cols]))
    train_ks = calc_ks(
        y_train.to_numpy(), model.predict(X_train[final_cols]).to_numpy()
    )
    test_ks = calc_ks(y_test.to_numpy(), model.predict(X_test[final_cols]).to_numpy())
    print(f"训练 AUC: {train_auc:.4f}，测试 AUC: {test_auc:.4f}")
    print(f"训练 KS: {train_ks:.4f}，测试 KS: {test_ks:.4f}")

    # 5. 输出结果
    coef_records = []
    for col in final_cols:
        coef_records.append(
            {
                "特征名": col,
                "系数": round(model.params[col], 5),
                "p值": round(model.pvalues[col], 6),
                "显著性": "显著" if model.pvalues[col] < P_THRESHOLD else "不显著",
            }
        )
    coef_df = pd.DataFrame(coef_records)
    perf_df = pd.DataFrame(
        [
            {
                "指标": "AUC",
                "训练集": round(train_auc, 4),
                "测试集": round(test_auc, 4),
            },
            {"指标": "KS", "训练集": round(train_ks, 4), "测试集": round(test_ks, 4)},
        ]
    )
    scorecard = build_scorecard(model, final_cols, woe_map)

    coef_df.to_csv(
        OUT_DIR + "\\logit_coefficients.csv", index=False, encoding="utf-8-sig"
    )
    perf_df.to_csv(
        OUT_DIR + "\\model_performance.csv", index=False, encoding="utf-8-sig"
    )
    scorecard.to_csv(
        OUT_DIR + "\\scorecard_points.csv", index=False, encoding="utf-8-sig"
    )

    train_scores = pd.DataFrame(
        {
            "SK_ID_CURR": df.loc[X_train.index, "SK_ID_CURR"],
            "TARGET": y_train,
            "SCORE": calc_score(model, X_train[final_cols]).round(2),
        }
    )
    test_scores = pd.DataFrame(
        {
            "SK_ID_CURR": df.loc[X_test.index, "SK_ID_CURR"],
            "TARGET": y_test,
            "SCORE": calc_score(model, X_test[final_cols]).round(2),
        }
    )
    train_scores.to_csv(
        OUT_DIR + "\\scores_train.csv", index=False, encoding="utf-8-sig"
    )
    test_scores.to_csv(OUT_DIR + "\\scores_test.csv", index=False, encoding="utf-8-sig")
    print(f"系数表已保存: {OUT_DIR}\\logit_coefficients.csv")
    print(f"性能表已保存: {OUT_DIR}\\model_performance.csv")
    print(f"评分卡分值表已保存: {OUT_DIR}\\scorecard_points.csv")
    print(f"训练评分已保存: {OUT_DIR}\\scores_train.csv")
    print(f"测试评分已保存: {OUT_DIR}\\scores_test.csv")


if __name__ == "__main__":
    main()
