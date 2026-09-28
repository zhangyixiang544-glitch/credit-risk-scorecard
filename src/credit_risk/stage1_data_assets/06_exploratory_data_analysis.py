"""探索性数据分析（EDA）脚本：围绕主表 application_train 做多维度风险分析。

单变量分布、分组违约率、交叉维度、相关性热力图四类分析，
输出图表到 figures 目录供报告引用。
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


# 指定后端
matplotlib.use("Agg")

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
FIG_DIR = os.path.join(PROJECT_ROOT, "output", "stage1", "figures")

# 创建图表目录
os.makedirs(FIG_DIR, exist_ok=True)

# 定义全局绘图风格
sns.set_theme(style="whitegrid", font="Microsoft YaHei")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """把 DataFrame 里各列的类型往小了调，节省内存。

    与其他脚本重复定义，是为了让本脚本独立可运行。

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


def plot_numeric_distribution(df: pd.DataFrame, column: str, title: str) -> None:
    """画单个数值字段的分布直方图，并叠加分箱违约率曲线。

    左图看分布，右图看字段与违约的关系。

    Args:
        df: 主表 DataFrame
        column: 要分析的数值字段
        title: 图表标题
    """
    # 剔除缺失值，缺失单独看
    data = df[column].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(data, bins=50, color="skyblue", alpha=0.8)
    axes[0].set_title(f"{title} 分布")
    axes[0].set_xlabel(column)
    axes[0].set_ylabel("样本数")

    try:
        df["_bin"] = pd.qcut(df[column], 10, duplicates="drop")
        default_rate = (
            df.groupby("_bin", observed=False)["TARGET"].mean().mul(100).round(2)
        )
        axes[1].plot(
            range(len(default_rate)),
            default_rate.values,
            marker="o",
            color="#E67E22",
        )
        axes[1].set_title(f"{title} 分箱违约率")
        axes[1].set_xlabel("分箱（按十分位）")
        axes[1].set_ylabel("违约率(%)")
        axes[1].set_xticks(range(len(default_rate)))
        axes[1].set_xticklabels(range(1, len(default_rate) + 1))
        df.drop(columns=["_bin"], inplace=True)
    except ValueError:
        # 某些字段取值太少分不了 10 箱，跳过右侧图
        axes[1].text(0.5, 0.5, "取值过少，无法分箱", ha="center")
        axes[1].set_title(f"{title} 分箱违约率")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}\\dist_{column}.png", dpi=120)
    plt.close(fig)


def plot_category_default_rate(df: pd.DataFrame, column: str, title: str) -> None:
    """画类别字段的取值分布和分组违约率对比图。

    左图看各取值人数占比，右图看违约率，快速找出高风险类别。

    Args:
        df: 主表 DataFrame
        column: 要分析的类别字段
        title: 图表标题
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    value_counts = df[column].value_counts(normalize=True).mul(100).round(2)
    value_counts.plot(kind="bar", ax=axes[0], color="skyblue")
    axes[0].set_title(f"{title} 取值占比")
    axes[0].set_xlabel(column)
    axes[0].set_ylabel("占比(%)")
    axes[0].tick_params(axis="x", rotation=45)

    # 右图：分组违约率
    group_stats = (
        df.groupby(column)["TARGET"]
        .agg(default_rate=("mean"), count=("size"))
        .reset_index()
    )
    group_stats["default_rate"] = group_stats["default_rate"].mul(100).round(2)

    bars = axes[1].bar(
        range(len(group_stats)),
        group_stats["default_rate"],
        color="#F5A623",
    )
    axes[1].set_xticks(range(len(group_stats)))
    axes[1].set_xticklabels(group_stats[column], rotation=45)
    axes[1].set_title(f"{title} 分组违约率")
    axes[1].set_ylabel("违约率(%)")

    # 在每个柱子上面标样本数
    for bar, count in zip(bars, group_stats["count"]):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"n={count:,}",
            ha="center",
            fontsize=7,
        )

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}\\cat_{column}.png", dpi=120)
    plt.close(fig)


def plot_cross_default_rate(
    df: pd.DataFrame,
    column_x: str,
    column_y: str,
    title: str,
    max_categories: int = 8,
) -> None:
    """画两个类别字段的交叉违约率热力图。

    Args:
        df: 主表 DataFrame
        column_x: 交叉分析的第一个字段
        column_y: 交叉分析的第二个字段
        title: 图表标题
        max_categories: 每个字段最多保留多少个取值（取值太多图会看不清）
    """

    # 只保留出现最多的几个取值，其余合并为"其他"
    def top_categories(series: pd.Series) -> pd.Series:
        top_values = series.value_counts().head(max_categories).index
        return series.where(series.isin(top_values), "其他")

    df["_x"] = top_categories(df[column_x])
    df["_y"] = top_categories(df[column_y])

    # 交叉违约率透视表
    pivot = (
        df.groupby(["_x", "_y"], observed=False)["TARGET"]
        .mean()
        .mul(100)
        .round(2)
        .unstack()
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="Reds",
        ax=ax,
        cbar_kws={"label": "违约率(%)"},
    )
    ax.set_title(f"{title}（单位：%）")
    ax.set_xlabel(column_y)
    ax.set_ylabel(column_x)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}\\cross_{column_x}_{column_y}.png", dpi=120)
    plt.close(fig)

    df.drop(columns=["_x", "_y"], inplace=True)


def plot_correlation_heatmap(df: pd.DataFrame, columns: list, title: str) -> None:
    """画核心数值字段的相关性热力图（含 TARGET）。

    Args:
        df: 主表 DataFrame
        columns: 参与相关分析的字段列表
        title: 图表标题
    """
    corr = df[columns].corr()
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        ax=ax,
        annot_kws={"fontsize": 8},
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}\\correlation_heatmap.png", dpi=120)
    plt.close(fig)


def main() -> None:
    """主流程：依次执行四类 EDA 分析。"""
    print("=" * 60)
    print("EDA 分析开始")
    print("=" * 60)

    train_df = load_train()
    total = len(train_df)
    default_total = train_df["TARGET"].sum()
    print(
        f"主表样本量: {total:,}，违约样本: {default_total:,}"
        f"（违约率 {default_total / total * 100:.2f}%）"
    )

    # 1. 单变量分布：核心数值字段
    numeric_fields = [
        ("AMT_INCOME_TOTAL", "年收入"),
        ("AMT_CREDIT", "贷款金额"),
        ("AMT_ANNUITY", "每期还款额(年金)"),
        ("AMT_GOODS_PRICE", "商品价格"),
        ("DAYS_BIRTH", "出生天数(距申请日，负数)"),
        ("DAYS_EMPLOYED", "工作天数(距申请日，负数)"),
        ("EXT_SOURCE_2", "外部信用评分2"),
        ("EXT_SOURCE_3", "外部信用评分3"),
        ("CNT_CHILDREN", "子女数量"),
        ("CNT_FAM_MEMBERS", "家庭成员数"),
    ]
    for column, label in numeric_fields:
        plot_numeric_distribution(train_df, column, label)

    # 2. 单变量分布：核心类别字段（分组违约率）
    category_fields = [
        ("CODE_GENDER", "性别"),
        ("NAME_CONTRACT_TYPE", "贷款类型"),
        ("NAME_INCOME_TYPE", "收入类型"),
        ("NAME_EDUCATION_TYPE", "教育水平"),
        ("NAME_FAMILY_STATUS", "婚姻状况"),
        ("NAME_HOUSING_TYPE", "住房类型"),
        ("REGION_RATING_CLIENT", "地区信用评级"),
    ]
    for column, label in category_fields:
        plot_category_default_rate(train_df, column, label)

    # 3. 交叉维度分析
    # 学历 x 收入类型
    plot_cross_default_rate(
        train_df,
        "NAME_EDUCATION_TYPE",
        "NAME_INCOME_TYPE",
        "学历 x 收入类型 交叉违约率",
    )
    # 收入类型 x 住房类型
    plot_cross_default_rate(
        train_df,
        "NAME_INCOME_TYPE",
        "NAME_HOUSING_TYPE",
        "收入类型 x 住房类型 交叉违约率",
    )
    # 地区评级 x 学历
    plot_cross_default_rate(
        train_df,
        "REGION_RATING_CLIENT",
        "NAME_EDUCATION_TYPE",
        "地区信用评级 x 学历 交叉违约率",
    )
    # 逾期社交圈 x 收入类型
    plot_cross_default_rate(
        train_df,
        "NAME_INCOME_TYPE",
        "DEF_30_CNT_SOCIAL_CIRCLE",
        "收入类型 x 30天逾期社交圈人数 交叉违约率",
    )

    # 4. 相关性热力图
    corr_columns = [
        "TARGET",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "DAYS_REGISTRATION",
        "DAYS_ID_PUBLISH",
        "CNT_CHILDREN",
        "CNT_FAM_MEMBERS",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "REGION_POPULATION_RELATIVE",
        "REGION_RATING_CLIENT",
        "OBS_30_CNT_SOCIAL_CIRCLE",
        "DEF_30_CNT_SOCIAL_CIRCLE",
        "DEF_60_CNT_SOCIAL_CIRCLE",
        "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
    ]
    plot_correlation_heatmap(train_df, corr_columns, "核心字段相关性热力图(含TARGET)")

    print(f"\n图表已保存到: {FIG_DIR}")


if __name__ == "__main__":
    main()
