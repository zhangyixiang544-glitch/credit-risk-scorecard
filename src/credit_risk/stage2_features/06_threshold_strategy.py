"""双阈值策略脚本：测算不同分数阈值下的通过率与坏账率。

阶段二第六步。基于 04 输出的测试集客户评分，按分数分位
扫描多个阈值，计算每个阈值下的通过率与通过客户的坏账率，
再按"低风险优先""高效率优先"两套业务目标给出建议方案。

输出：
- threshold_grid.csv：全阈值扫描结果
- threshold_strategy.csv：两套方案对比
"""

import os

import pandas as pd

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


OUT_DIR = os.path.join(PROJECT_ROOT, "output", "stage2")

SCORE_FILE = OUT_DIR + "\\scores_test.csv"

# 扫描步长：按分数分位每 5% 取一个阈值
GRID_STEP = 0.05
# 两套方案的目标口径
LOW_RISK_MAX_BAD = 0.02  # 低风险优先：坏账率控制在 2% 以内
HIGH_EFF_MIN_PASS = 0.75  # 高效率优先：通过率不低于 75%


def load_scores() -> pd.DataFrame:
    """读取测试集客户评分。

    Returns:
        评分表 DataFrame（SK_ID_CURR、TARGET、SCORE）
    """
    df = pd.read_csv(SCORE_FILE)
    return df


def scan_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    """按分数分位扫描阈值，计算每个阈值的通过率与坏账率。

    Args:
        df: 评分表 DataFrame

    Returns:
        扫描结果 DataFrame
    """
    quantiles = list(
        range(int(GRID_STEP * 100), int((1 - GRID_STEP) * 100), int(GRID_STEP * 100))
    )
    total = len(df)
    records = []
    for q in quantiles:
        threshold = df["SCORE"].quantile(q / 100)
        passed = df[df["SCORE"] >= threshold]
        pass_rate = len(passed) / total
        bad_rate = passed["TARGET"].mean()
        records.append(
            {
                "阈值": round(threshold, 2),
                "通过样本数": len(passed),
                "通过率": round(pass_rate, 4),
                "通过中违约数": int(passed["TARGET"].sum()),
                "坏账率": round(bad_rate, 4),
            }
        )
    return pd.DataFrame(records)


def pick_low_risk(grid: pd.DataFrame) -> pd.DataFrame:
    """低风险优先：坏账率 3% 以内取通过率最高的一档。

    Args:
        grid: 扫描结果 DataFrame

    Returns:
        选中方案 DataFrame
    """
    candidates = grid[grid["坏账率"] <= LOW_RISK_MAX_BAD]
    if candidates.empty:
        return grid.loc[grid["坏账率"].idxmin()].to_frame().T
    return candidates.loc[candidates["通过率"].idxmax()].to_frame().T


def pick_high_efficiency(grid: pd.DataFrame) -> pd.DataFrame:
    """高效率优先：通过率 60% 以上取坏账率最低的一档。

    Args:
        grid: 扫描结果 DataFrame

    Returns:
        选中方案 DataFrame
    """
    candidates = grid[grid["通过率"] >= HIGH_EFF_MIN_PASS]
    if candidates.empty:
        return grid.loc[grid["坏账率"].idxmin()].to_frame().T
    return candidates.loc[candidates["坏账率"].idxmin()].to_frame().T


def main() -> None:
    """主流程：扫描阈值、挑选两套方案、保存结果。"""
    print("=" * 60)
    title = "双阈值策略测算开始"
    print(title.center(60, "="))

    df = load_scores()
    total_bad_rate = df["TARGET"].mean()
    print(f"测试集样本: {len(df)} 个，整体坏账率: {total_bad_rate:.4f}")

    # 1. 全阈值扫描
    grid = scan_thresholds(df)
    print(f"扫描档位: {len(grid)} 个")

    # 2. 挑选两套方案
    low_risk = pick_low_risk(grid)
    high_eff = pick_high_efficiency(grid)
    low_risk["方案"] = "低风险优先"
    low_risk["口径"] = f"坏账率控制在 {LOW_RISK_MAX_BAD:.0%} 以内"
    high_eff["方案"] = "高效率优先"
    high_eff["口径"] = f"通过率不低于 {HIGH_EFF_MIN_PASS:.0%}"
    strategy = pd.concat([low_risk, high_eff], ignore_index=True)
    print(
        f"低风险优先: 阈值 {low_risk['阈值'].iloc[0]:.2f}，"
        f"通过率 {low_risk['通过率'].iloc[0]:.2%}，坏账率 {low_risk['坏账率'].iloc[0]:.2%}"
    )
    print(
        f"高效率优先: 阈值 {high_eff['阈值'].iloc[0]:.2f}，"
        f"通过率 {high_eff['通过率'].iloc[0]:.2%}，坏账率 {high_eff['坏账率'].iloc[0]:.2%}"
    )

    # 3. 保存结果
    grid.to_csv(OUT_DIR + "\\threshold_grid.csv", index=False, encoding="utf-8-sig")
    strategy.to_csv(
        OUT_DIR + "\\threshold_strategy.csv", index=False, encoding="utf-8-sig"
    )
    print(f"全阈值扫描表已保存: {OUT_DIR}\\threshold_grid.csv")
    print(f"两套方案已保存: {OUT_DIR}\\threshold_strategy.csv")


if __name__ == "__main__":
    main()
