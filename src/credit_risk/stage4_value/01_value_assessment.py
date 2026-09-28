"""方案价值评估脚本：量化风险收益与营运效率提升。

基于阶段三风险分层策略的模拟结果（坏账率、自动审批率、人工审核占比）
，结合行业平均授信额度、资金成本、人工审核成本等业务假设，粗略量化
方案的落地价值。

测算口径（均为业务假设）：
- 申请量基准：10 万笔
- 风险收益：坏账损失减少额 = 减少坏账笔数 × 平均额度 × 坏账损失率
- 营运效率：人工成本节省 = 减少人工审核笔数 × 单笔耗时 × 时薪

输出：
- value_assessment.csv：各方案价值测算表
"""

import os

import pandas as pd

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


OUT_DIR = os.path.join(PROJECT_ROOT, "output", "stage4")

SIM_FILE = os.path.join(PROJECT_ROOT, "output", "stage3", "strategy_simulation.csv")

# 业务假设参数
APPLICATIONS = 100_000  # 测算基准申请量（笔）
AVG_LOAN_AMOUNT = 30_000  # 行业平均单笔授信额度（元）
BAD_LOSS_RATE = 0.5  # 违约后平均本金损失率（假设可回收一半）
MANUAL_MINUTES = 20  # 单笔人工审核耗时（分钟）
HOURLY_COST = 50  # 人工审核时薪（元）


def load_simulation() -> pd.DataFrame:
    """读取阶段三策略模拟表。

    Returns:
        策略模拟 DataFrame
    """
    df = pd.read_csv(SIM_FILE)
    return df


def assess(value: float, pass_rate: float, bad_rate: float, manual_rate: float) -> dict:
    """测算单个方案的价值。

    Args:
        value: 方案名
        pass_rate: 通过率
        bad_rate: 坏账率
        manual_rate: 人工审核占比

    Returns:
        价值测算记录
    """
    passed = int(APPLICATIONS * pass_rate)
    bad_count = int(passed * bad_rate)
    manual_count = int(APPLICATIONS * manual_rate)

    # 风险收益：与纯规则基线（全部通过）相比减少的坏账损失
    base_bad = int(APPLICATIONS * 0.0807)  # 基线坏账率取整体违约率
    reduced_bad = base_bad - bad_count
    loss_saved = reduced_bad * AVG_LOAN_AMOUNT * BAD_LOSS_RATE / 10_000  # 万元

    # 营运效率：与基线（全人工）相比节省的人工成本
    manual_saved = APPLICATIONS - manual_count
    hours_saved = manual_saved * MANUAL_MINUTES / 60
    cost_saved = hours_saved * HOURLY_COST / 10_000  # 万元

    return {
        "方案": value,
        "通过笔数": passed,
        "坏账笔数": bad_count,
        "较基线减少坏账笔数": reduced_bad,
        "坏账损失减少（万元）": round(loss_saved, 1),
        "人工审核笔数": manual_count,
        "节省人工工时（小时）": round(hours_saved, 0),
        "人工成本节省（万元）": round(cost_saved, 1),
        "合计价值（万元）": round(loss_saved + cost_saved, 1),
    }


def main() -> None:
    """主流程：读取模拟结果、逐方案测算、保存价值表。"""
    print("=" * 60)
    title = "方案价值评估开始"
    print(title.center(60, "="))
    print(
        f"测算假设: 申请量 {APPLICATIONS} 笔，平均额度 {AVG_LOAN_AMOUNT} 元，"
        f"坏账损失率 {BAD_LOSS_RATE:.0%}，单笔审核 {MANUAL_MINUTES} 分钟，"
        f"时薪 {HOURLY_COST} 元"
    )

    sim = load_simulation()
    base_row = sim[sim["策略"] == "纯规则基线（全部通过）"]
    if base_row.empty:
        print("未找到纯规则基线行，使用默认整体坏账率 8.07%")

    records = []
    for _, row in sim.iterrows():
        if row["策略"] == "纯规则基线（全部通过）":
            continue
        records.append(
            assess(row["策略"], row["通过率"], row["坏账率"], row["人工审核占比"])
        )
    result_df = pd.DataFrame(records)
    result_df.to_csv(
        OUT_DIR + "\\value_assessment.csv", index=False, encoding="utf-8-sig"
    )
    print(result_df.to_string(index=False))
    print(f"价值测算表已保存: {OUT_DIR}\\value_assessment.csv")


if __name__ == "__main__":
    main()
