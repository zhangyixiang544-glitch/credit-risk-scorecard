"""补充分析脚本：负债比例异常值 + "逾期社交圈 x 负债比例"交叉违约率。

补齐任务书点名的两项：负债比例（年金/收入）异常值统计，
以及"历史逾期行为  x 负债比例"交叉违约率热力图。
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


matplotlib.use("Agg")

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
OUT_DIR = os.path.join(PROJECT_ROOT, "output", "stage1")
FIG_DIR = OUT_DIR + "\\figures"

# 中文字体
sns.set_theme(style="whitegrid", font="Microsoft YaHei")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

os.makedirs(FIG_DIR, exist_ok=True)


def load_train() -> pd.DataFrame:
    """读取主表（只取需要的列）。

    Returns:
        application_train 主表 DataFrame
    """
    usecols = [
        "SK_ID_CURR",
        "TARGET",
        "AMT_ANNUITY",
        "AMT_INCOME_TOTAL",
        "OBS_30_CNT_SOCIAL_CIRCLE",
    ]
    train_df = pd.read_csv(DATA_DIR + "\\application_train.csv", usecols=usecols)
    return train_df


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
        series = series.astype("object").fillna("缺失")
        top_values = series.value_counts().head(max_categories).index
        return series.where(series.isin(top_values), "其他")

    df["_x"] = top_categories(df[column_x])
    df["_y"] = top_categories(df[column_y])

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


def main() -> None:
    """主流程：负债比例异常值统计 + 交叉违约率热力图。"""
    print("=" * 60)
    print("补充分析：负债比例异常值 + 交叉违约率")
    print("=" * 60)

    train_df = load_train()
    print(f"主表样本量: {len(train_df):,}")

    # 1. 构造负债比例 = 年金 / 年收入（收入为 0 或缺失时置 NaN）
    ratio = train_df["AMT_ANNUITY"] / train_df["AMT_INCOME_TOTAL"]
    train_df["DEBT_RATIO"] = ratio.replace([np.inf, -np.inf], np.nan)

    # 2. 负债比例分箱
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, np.inf]
    labels = ["0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", "0.5-1.0", ">1.0"]
    train_df["DEBT_RATIO_BIN"] = pd.cut(
        train_df["DEBT_RATIO"], bins=bins, labels=labels
    )

    # 3. 交叉违约率：逾期社交圈人数 x 负债比例
    print("\n生成交叉热力图: 逾期社交圈人数 x 负债比例 ...")
    plot_cross_default_rate(
        train_df,
        "OBS_30_CNT_SOCIAL_CIRCLE",
        "DEBT_RATIO_BIN",
        "逾期30天社交圈人数 x 负债比例 交叉违约率",
    )
    print(f"图表已保存: {FIG_DIR}\\cross_OBS_30_CNT_SOCIAL_CIRCLE_DEBT_RATIO_BIN.png")


if __name__ == "__main__":
    main()
