"""核心风控字段单变量分析 + WOE/IV 排序脚本。

对核心风控字段（征信查询、逾期历史、外部评分、负债能力等）做主表单变量分析：
数值字段等频分箱、类别字段按取值分组，计算每箱违约率与 WOE/IV，
按 IV 排序输出（iv_ranking.csv），并画"分箱违约率 + WOE"双轴图。

IV 判断经验值：<0.02 基本无区分度；0.02~0.10 弱；0.10~0.30 中等；>0.30 强。
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


# 指定后端
matplotlib.use("Agg")

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
OUT_DIR = os.path.join(PROJECT_ROOT, "output", "stage1")
FIG_DIR = OUT_DIR + "\\figures"

# 中文字体
sns.set_theme(style="whitegrid", font="Microsoft YaHei")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

os.makedirs(FIG_DIR, exist_ok=True)

# 核心风控字段清单：(字段名, 业务说明, 类型)
# 类型 numeric=数值等频分箱，category=类别按取值分组
CORE_FIELDS = [
    ("EXT_SOURCE_1", "外部信用评分1", "numeric"),
    ("EXT_SOURCE_2", "外部信用评分2", "numeric"),
    ("EXT_SOURCE_3", "外部信用评分3", "numeric"),
    ("AMT_REQ_CREDIT_BUREAU_HOUR", "征信查询次数-申请前1小时", "numeric"),
    ("AMT_REQ_CREDIT_BUREAU_DAY", "征信查询次数-申请前1天", "numeric"),
    ("AMT_REQ_CREDIT_BUREAU_WEEK", "征信查询次数-申请前1周", "numeric"),
    ("AMT_REQ_CREDIT_BUREAU_MON", "征信查询次数-申请前1月", "numeric"),
    ("AMT_REQ_CREDIT_BUREAU_QRT", "征信查询次数-申请前1季度", "numeric"),
    ("AMT_REQ_CREDIT_BUREAU_YEAR", "征信查询次数-申请前1年", "numeric"),
    ("OBS_30_CNT_SOCIAL_CIRCLE", "逾期30天观察-社交圈人数", "numeric"),
    ("DEF_30_CNT_SOCIAL_CIRCLE", "逾期30天违约-社交圈人数", "numeric"),
    ("DEF_60_CNT_SOCIAL_CIRCLE", "逾期60天违约-社交圈人数", "numeric"),
    ("DAYS_BIRTH", "年龄（距申请天数，负数）", "numeric"),
    ("DAYS_EMPLOYED", "工作天数（距申请天数，负数）", "numeric"),
    ("AMT_INCOME_TOTAL", "年收入", "numeric"),
    ("AMT_CREDIT", "贷款金额", "numeric"),
    ("AMT_ANNUITY", "年金（每期还款额）", "numeric"),
    ("CNT_CHILDREN", "子女数量", "numeric"),
    ("REGION_RATING_CLIENT", "地区信用评级", "category"),
    ("NAME_EDUCATION_TYPE", "教育水平", "category"),
]

# 分箱数：数值字段等频分 10 箱
N_BINS = 10

# WOE 平滑系数
SMOOTH = 0.5


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """把 DataFrame 里各列的类型往小了调，可以节省内存。

    Args:
        df: 原始 DataFrame

    Returns:
        类型优化后的 DataFrame
    """
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == "int64":
            col_min = df[col].min()
            col_max = df[col].max()
            if col_min >= -128 and col_max <= 127:
                df[col] = df[col].astype("int8")
            elif col_min >= -32768 and col_max <= 32767:
                df[col] = df[col].astype("int16")
            elif col_min >= -2147483648 and col_max <= 2147483647:
                df[col] = df[col].astype("int32")
        elif col_type == "float64":
            df[col] = df[col].astype("float32")
    return df


def load_train() -> pd.DataFrame:
    """读取主表并做类型优化。

    Returns:
        类型优化后的 application_train DataFrame
    """
    train_df = pd.read_csv(DATA_DIR + "\\application_train.csv")
    train_df = optimize_dtypes(train_df)
    return train_df


def calc_iv_numeric(series: pd.Series, target: pd.Series) -> pd.DataFrame:
    """计算单个数值字段的 WOE/IV（等频分箱 + 缺失单独成箱）。

    Args:
        series: 数值字段列
        target: 目标变量列（0/1）

    Returns:
        分箱明细 DataFrame，含每箱样本数、违约率、WOE、IV 贡献
    """
    data = pd.DataFrame({"x": series, "y": target})
    data["箱"] = data["x"].isna().map({True: "缺失", False: np.nan})

    # 非缺失部分做等频分箱，缺失单独成一箱
    valid = data["x"].notna()
    if valid.sum() > 0:
        try:
            bins = pd.qcut(data.loc[valid, "x"], N_BINS, duplicates="drop")
            data.loc[valid, "箱"] = bins.astype(str)
        except ValueError:
            # 取值太少分不了 10 箱，退化成按取值分组
            data.loc[valid, "箱"] = data.loc[valid, "x"].astype(str)

    return calc_iv_from_bins(data, target)


def calc_iv_category(series: pd.Series, target: pd.Series) -> pd.DataFrame:
    """计算单个类别字段的 WOE/IV（按取值分组，缺失单独成箱）。

    Args:
        series: 类别字段列
        target: 目标变量列（0/1）

    Returns:
        分箱明细 DataFrame，含每箱样本数、违约率、WOE、IV 贡献
    """
    data = pd.DataFrame({"x": series, "y": target})
    data["箱"] = data["x"].fillna("缺失").astype(str)
    return calc_iv_from_bins(data, target)


def calc_iv_from_bins(data: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """根据分箱结果计算 WOE/IV。

    Args:
        data: 含"箱"列的 DataFrame（x 列已分箱）
        target: 目标变量列（0/1）

    Returns:
        分箱明细 DataFrame，含每箱样本数、违约率、WOE、IV 贡献
    """
    good_total = int((target == 0).sum())
    bad_total = int((target == 1).sum())

    group = (
        data.groupby("箱", observed=False)["y"]
        .agg(good=("count"), bad=("sum"))
        .reset_index()
    )
    group["bad"] = group["bad"].astype(int)
    group["good"] = group["good"] - group["bad"]
    group["total"] = group["good"] + group["bad"]
    group["违约率"] = (group["bad"] / group["total"]).mul(100).round(2)

    # 平滑后计算好占比 / 坏占比 / WOE / IV
    good_ratio = (group["good"] + SMOOTH) / (good_total + SMOOTH)
    bad_ratio = (group["bad"] + SMOOTH) / (bad_total + SMOOTH)
    group["woe"] = np.log(bad_ratio / good_ratio).round(4)
    group["iv_contribution"] = ((bad_ratio - good_ratio) * group["woe"]).round(4)

    return group


def plot_woe_chart(field_name: str, field_label: str, iv_table: pd.DataFrame) -> None:
    """画单字段的分箱违约率 + WOE 双轴图。

    Args:
        field_name: 字段名（用于文件名）
        field_label: 字段中文说明（用于标题）
        iv_table: 分箱明细 DataFrame
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # 左轴：每箱违约率柱状图
    ax1.bar(
        range(len(iv_table)),
        iv_table["违约率"],
        color="skyblue",
        alpha=0.8,
        label="违约率(%)",
    )
    ax1.set_ylabel("违约率(%)")
    ax1.set_ylim(0, max(iv_table["违约率"]) * 1.2 + 1)

    # 右轴：每箱 WOE 折线
    ax2 = ax1.twinx()
    ax2.plot(
        range(len(iv_table)),
        iv_table["woe"],
        marker="o",
        color="#E67E22",
        label="WOE",
    )
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_ylabel("WOE")

    # 缺失箱放最右边，标签统一处理
    bin_labels = list(iv_table["箱"])
    ax1.set_xticks(range(len(bin_labels)))
    ax1.set_xticklabels(bin_labels, rotation=45, fontsize=7)

    total_iv = iv_table["iv_contribution"].sum().round(4)
    ax1.set_title(f"{field_label}（{field_name}）  IV={total_iv}")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}\\iv_{field_name}.png", dpi=120)
    plt.close(fig)


def main() -> None:
    """主流程：对核心字段逐一算 IV 并画图，输出 IV 排序表。"""
    print("=" * 60)
    print("核心字段单变量分析 + WOE/IV 排序开始")
    print("=" * 60)

    train_df = load_train()
    print(f"主表样本量: {len(train_df):,}")
    target = train_df["TARGET"]

    results = []
    print(f"\n共 {len(CORE_FIELDS)} 个核心字段")

    for index, (field_name, field_label, field_type) in enumerate(CORE_FIELDS, start=1):
        print(f"  [{index}/{len(CORE_FIELDS)}] {field_name} ...")
        if field_type == "numeric":
            iv_table = calc_iv_numeric(train_df[field_name], target)
        else:
            iv_table = calc_iv_category(train_df[field_name], target)
        total_iv = iv_table["iv_contribution"].sum().round(4)
        results.append(
            {
                "字段名": field_name,
                "业务说明": field_label,
                "类型": field_type,
                "IV": total_iv,
                "分箱数": len(iv_table),
            }
        )
        plot_woe_chart(field_name, field_label, iv_table)

    # IV 排序输出
    iv_ranking = pd.DataFrame(results).sort_values("IV", ascending=False)
    out_file = OUT_DIR + "\\iv_ranking.csv"
    iv_ranking.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"\nIV 排序表已保存: {out_file}")

    # 按 IV 经验值分级
    def iv_level(score: float) -> str:
        if score < 0.02:
            return "基本无区分度"
        if score < 0.10:
            return "弱区分度"
        if score < 0.30:
            return "中等区分度"
        return "强区分度"

    iv_ranking["区分度"] = iv_ranking["IV"].map(iv_level)


if __name__ == "__main__":
    main()
