"""WOE 分箱脚本：候选特征分箱与 IV 计算。

阶段二第二步。输入 01 特征工程产出的 processed_features.csv，
对初筛保留的候选特征做 WOE 分箱：
1. 数值特征：等频分箱（最多 10 箱），缺失值单独成箱
2. 类别特征：按取值分箱，低频取值合并为"其他"，缺失单独成箱
3. 业务调整：年龄、负债比例用业务边界分箱，就业天数特殊值单独成箱，
   不纯依赖自动算法
4. 输出 WOE 映射表，供后续逻辑回归评分卡直接替换原始值

输出：
- woe_bins.csv：箱级明细（样本数、坏数、坏率、WOE、IV）
- woe_iv_summary.csv：特征级 IV 汇总，按 IV 降序
- woe_encoding_map.csv：特征 + 箱 -> WOE 值映射
"""

import numpy as np
import pandas as pd
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUT_DIR = os.path.join(PROJECT_ROOT, "output", "stage2")

FEATURE_FILE = OUT_DIR + "\\processed_features.csv"
LIST_FILE = OUT_DIR + "\\feature_list.csv"

# 平滑常数：坏数或好数为 0 时加 0.5
SMOOTH = 0.5

# 等频分箱箱数
NUM_BINS = 10

# 类别特征低频合并阈值
CAT_MIN_SHARE = 0.01

# 年龄按业务常识分段（阶段一：年龄违约率随年龄单调下降）
BUSINESS_EDGES = {
    "AGE_YEARS": [18, 25, 30, 35, 40, 45, 50, 60, 120],
    "DEBT_RATIO": [0, 0.2, 0.35, 0.5, 0.7, 1.0],
}

# DAYS_EMPLOYED 的失业占位符（阶段一结论）
UNEMPLOYED_CODE = 365243


def load_features() -> pd.DataFrame:
    """读取 01 产出的特征表。

    Returns:
        特征表 DataFrame（含 TARGET）
    """
    df = pd.read_csv(FEATURE_FILE)
    return df


def load_candidates() -> list:
    """读取初筛保留的候选特征名。

    Returns:
        候选特征列名列表
    """
    feature_list = pd.read_csv(LIST_FILE)
    kept = feature_list[feature_list["初筛状态"] == "保留"]["字段名"].tolist()
    return kept


def calc_woe_iv(
    bad_count: int, good_count: int, total_bad: int, total_good: int
) -> tuple:
    """计算单个箱的 WOE 与 IV 贡献。

    箱内坏数 / 好数加平滑后除以全局总数，得到坏占比与好占比：
    WOE = ln(坏占比 / 好占比)，IV = (坏占比 - 好占比) * WOE。

    Args:
        bad_count: 箱内违约样本数
        good_count: 箱内非违约样本数
        total_bad: 全局违约样本数
        total_good: 全局非违约样本数

    Returns:
        (woe, iv)：该箱的 WOE 值与 IV 贡献
    """
    bad_pct = (bad_count + SMOOTH) / (total_bad + SMOOTH)
    good_pct = (good_count + SMOOTH) / (total_good + SMOOTH)
    woe = np.log(bad_pct / good_pct)
    iv = (bad_pct - good_pct) * woe
    return float(woe), float(iv)


def build_bin_stats(
    binned: pd.DataFrame, total_bad: int, total_good: int
) -> pd.DataFrame:
    """把带 bin 列的数据汇总成箱级统计表。

    Args:
        binned: 含 bin（箱标签）与 target（0/1）两列的数据
        total_bad: 全局违约样本数
        total_good: 全局非违约样本数

    Returns:
        箱级统计 DataFrame
    """
    stats = (
        binned.groupby("bin", sort=False)
        .agg(样本数=("target", "count"), 坏数=("target", "sum"))
        .reset_index()
        .rename(columns={"bin": "箱"})
    )
    stats["好数"] = stats["样本数"] - stats["坏数"]
    stats["坏率"] = stats["坏数"] / stats["样本数"]
    woe_list = []
    iv_list = []
    for _, row in stats.iterrows():
        woe, iv = calc_woe_iv(int(row["坏数"]), int(row["好数"]), total_bad, total_good)
        woe_list.append(woe)
        iv_list.append(iv)
    stats["WOE"] = woe_list
    stats["IV"] = iv_list
    return stats


def bin_numeric(series: pd.Series, col: str, target: pd.Series) -> pd.DataFrame:
    """数值特征分箱：业务边界或自动等频，缺失单独成箱。

    Args:
        series: 特征列
        col: 特征名
        target: TARGET 列

    Returns:
        箱级统计 DataFrame
    """
    data = pd.DataFrame({"x": series, "target": target})

    if col in BUSINESS_EDGES:
        # 业务分箱：按固定边界切分，缺失行标记为"缺失"
        bins = pd.cut(
            data["x"], bins=BUSINESS_EDGES[col], right=True, include_lowest=True
        )
        data["bin"] = bins.astype(object)
        data.loc[data["x"].isna(), "bin"] = "缺失"
    elif col == "DAYS_EMPLOYED":
        # 失业占位符单独成箱，其余等频分箱
        data["bin"] = pd.Series(dtype="object")
        unemployed = data["x"] == UNEMPLOYED_CODE
        valid_mask = ~unemployed & data["x"].notna()
        valid_bins = pd.qcut(
            data.loc[valid_mask, "x"], q=NUM_BINS, duplicates="drop"
        ).astype(object)
        data.loc[valid_mask, "bin"] = valid_bins.values
        data.loc[unemployed, "bin"] = "失业"
        data.loc[data["x"].isna(), "bin"] = "缺失"
    else:
        # 自动等频分箱：有效行分箱，缺失行单独成箱
        data["bin"] = pd.Series(dtype="object")
        valid = data[data["x"].notna()]
        if valid["x"].nunique() < 2:
            data.loc[data["x"].notna(), "bin"] = "单一取值"
        else:
            try:
                valid_bins = pd.qcut(valid["x"], q=NUM_BINS, duplicates="drop").astype(
                    object
                )
                data.loc[data["x"].notna(), "bin"] = valid_bins.values
            except ValueError:
                data.loc[data["x"].notna(), "bin"] = "单一取值"
        data.loc[data["x"].isna(), "bin"] = "缺失"

    total_bad = int(target.sum())
    total_good = int(len(target) - target.sum())
    return build_bin_stats(data, total_bad, total_good)


def bin_category(series: pd.Series, col: str, target: pd.Series) -> pd.DataFrame:
    """类别特征分箱：按取值分组，低频取值合并为"其他"。

    Args:
        series: 特征列
        col: 特征名
        target: TARGET 列

    Returns:
        箱级统计 DataFrame
    """
    data = pd.DataFrame({"x": series, "target": target})
    value_share = data["x"].value_counts(normalize=True)
    keep_cats = value_share[value_share >= CAT_MIN_SHARE].index.tolist()
    data["bin"] = data["x"].where(data["x"].isin(keep_cats), "其他")
    data.loc[data["x"].isna(), "bin"] = "缺失"
    total_bad = int(target.sum())
    total_good = int(len(target) - target.sum())
    return build_bin_stats(data, total_bad, total_good)


def process_feature(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """单个特征分箱总入口：按类型与业务规则分发。

    Args:
        df: 特征表 DataFrame
        col: 特征名

    Returns:
        箱级统计 DataFrame
    """
    if col in BUSINESS_EDGES or col == "DAYS_EMPLOYED":
        return bin_numeric(df[col], col, df["TARGET"])
    if pd.api.types.is_string_dtype(df[col]):
        return bin_category(df[col], col, df["TARGET"])
    return bin_numeric(df[col], col, df["TARGET"])


def main() -> None:
    """主流程：读取候选特征，逐个分箱，汇总输出三张表。"""
    print("=" * 60)
    title = "WOE 分箱与 IV 计算开始"
    print(title.center(60, "="))

    df = load_features()
    candidates = load_candidates()
    print(f"候选特征: {len(candidates)} 个，样本量: {df.shape[0]}")

    all_bins = []
    iv_records = []
    for i, col in enumerate(candidates, 1):
        print(f"[{i}/{len(candidates)}] {col} 分箱中")
        bins_df = process_feature(df, col)
        bins_df.insert(0, "特征名", col)
        all_bins.append(bins_df)
        iv_total = round(float(bins_df["IV"].sum()), 4)
        iv_records.append(
            {
                "字段名": col,
                "数据类型": "类别" if pd.api.types.is_string_dtype(df[col]) else "数值",
                "箱数": len(bins_df),
                "IV": iv_total,
            }
        )

    print("\n保存结果中")
    woe_bins = pd.concat(all_bins, ignore_index=True)
    iv_summary = pd.DataFrame(iv_records).sort_values("IV", ascending=False)
    encoding_map = woe_bins[["特征名", "箱", "WOE"]].copy()

    woe_bins.to_csv(OUT_DIR + "\\woe_bins.csv", index=False, encoding="utf-8-sig")
    iv_summary.to_csv(
        OUT_DIR + "\\woe_iv_summary.csv", index=False, encoding="utf-8-sig"
    )
    encoding_map.to_csv(
        OUT_DIR + "\\woe_encoding_map.csv", index=False, encoding="utf-8-sig"
    )
    print(f"箱明细已保存: {OUT_DIR}\\woe_bins.csv")
    print(f"IV 汇总已保存: {OUT_DIR}\\woe_iv_summary.csv")
    print(f"WOE 映射已保存: {OUT_DIR}\\woe_encoding_map.csv")


if __name__ == "__main__":
    main()
