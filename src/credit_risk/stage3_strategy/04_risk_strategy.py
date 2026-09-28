"""风险分层与审批策略脚本：三档分层、策略模拟与人工审核规则。

阶段三的最后一步。基于 04 逻辑回归评分卡的测试集评分，
设计三档风险分层体系（低风险自动通过 / 中风险人工审核 /
高风险自动拒绝），输出两套阈值方案，模拟纯规则基线与
模型驱动分层策略下的通过率、坏账率、人工审核占比。

输出：
- risk_strategy.csv：两套方案的三档边界、客群占比与坏账率
- strategy_simulation.csv：纯规则基线 vs 两套模型策略对比
"""

import pandas as pd
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


OUT_DIR = os.path.join(PROJECT_ROOT, "output", "stage3")

SCORE_FILE = os.path.join(PROJECT_ROOT, "output", "stage2", "scores_test.csv")

# 人工审核拦截率假设：中风险档经人工审核后拦截的违约比例，
REVIEW_BLOCK_RATE = 0.5


# 分数从低到高 = 风险从高到低，拒绝线取低分位，通过线取高分位
STRATEGY_EDGES = {
    "稳健方案": {"pass_quantile": 0.60, "reject_quantile": 0.30},
    "进取方案": {"pass_quantile": 0.40, "reject_quantile": 0.20},
}


def load_scores() -> pd.DataFrame:
    """读取测试集客户评分。

    Returns:
        评分表 DataFrame（SK_ID_CURR、TARGET、SCORE）
    """
    df = pd.read_csv(SCORE_FILE)
    return df


def split_three_tiers(
    df: pd.DataFrame, pass_quantile: float, reject_quantile: float
) -> pd.DataFrame:
    """按分位把客户分为低/中/高三档。

    Args:
        df: 评分表 DataFrame
        pass_quantile: 自动通过线分位（分数高于该线自动通过）
        reject_quantile: 自动拒绝线分位（分数低于该线自动拒绝）

    Returns:
        带档位列的 DataFrame
    """
    pass_line = df["SCORE"].quantile(pass_quantile)
    reject_line = df["SCORE"].quantile(reject_quantile)
    result = df.copy()
    result["档位"] = pd.cut(
        result["SCORE"],
        bins=[-float("inf"), reject_line, pass_line, float("inf")],
        labels=["高风险自动拒绝", "中风险人工审核", "低风险自动通过"],
    )
    return result


def tier_summary(df: pd.DataFrame) -> list:
    """统计三档的客群占比与坏账率。

    Args:
        df: 带档位列的 DataFrame

    Returns:
        各档统计记录列表
    """
    total = len(df)
    records = []
    for tier, group in df.groupby("档位", observed=True):
        records.append(
            {
                "档位": tier,
                "样本数": len(group),
                "客群占比": round(len(group) / total, 4),
                "坏账率": round(group["TARGET"].mean(), 4),
            }
        )
    return records


def simulate_strategy(df: pd.DataFrame, block_rate: float) -> dict:
    """模拟模型分层策略：自动通过全放行，中风险人工审核按拦截率放行。

    Args:
        df: 带档位列的 DataFrame
        block_rate: 人工审核拦截违约比例

    Returns:
        策略效果字典
    """
    auto_pass = df[df["档位"] == "低风险自动通过"]
    manual = df[df["档位"] == "中风险人工审核"]
    auto_reject = df[df["档位"] == "高风险自动拒绝"]

    # 自动通过全放行，中风险全部进人工审核，拦截部分违约后放行
    final_pass = pd.concat([auto_pass, manual])
    final_bad = auto_pass["TARGET"].sum() + manual["TARGET"].sum() * (1 - block_rate)

    return {
        "自动审批率": round(len(auto_pass) / len(df), 4),
        "人工审核占比": round(len(manual) / len(df), 4),
        "自动拒绝占比": round(len(auto_reject) / len(df), 4),
        "最终通过率": round(len(final_pass) / len(df), 4),
        "最终坏账率": round(final_bad / len(final_pass), 4),
    }


def main() -> None:
    """主流程：三档分层、两套方案、策略模拟、保存结果。"""
    print("=" * 60)
    title = "风险分层与审批策略设计开始"
    print(title.center(60, "="))

    df = load_scores()
    total_bad = df["TARGET"].mean()
    print(f"测试集样本: {len(df)} 个，整体坏账率: {total_bad:.4f}")

    # 1. 两套方案的三档分层
    strategy_records = []
    for name, edges in STRATEGY_EDGES.items():
        tiered = split_three_tiers(df, edges["pass_quantile"], edges["reject_quantile"])
        tiers = tier_summary(tiered)
        sim = simulate_strategy(tiered, REVIEW_BLOCK_RATE)
        strategy_records.append(
            {
                "方案": name,
                "自动通过线分位": edges["pass_quantile"],
                "自动拒绝线分位": edges["reject_quantile"],
                **{f"{t['档位']}客群占比": t["客群占比"] for t in tiers},
                **{f"{t['档位']}坏账率": t["坏账率"] for t in tiers},
                "自动审批率": sim["自动审批率"],
                "人工审核占比": sim["人工审核占比"],
                "自动拒绝占比": sim["自动拒绝占比"],
                "最终通过率": sim["最终通过率"],
                "最终坏账率": sim["最终坏账率"],
            }
        )
        print(
            f"{name}: 自动通过线分位 {edges['pass_quantile']:.0%}，"
            f"拒绝线分位 {edges['reject_quantile']:.0%}，"
            f"最终通过率 {sim['最终通过率']:.2%}，坏账率 {sim['最终坏账率']:.2%}"
        )

    strategy_df = pd.DataFrame(strategy_records)
    strategy_df.to_csv(
        OUT_DIR + "\\risk_strategy.csv", index=False, encoding="utf-8-sig"
    )
    print(f"两套方案已保存: {OUT_DIR}\\risk_strategy.csv")

    # 2. 纯规则基线对比模型策略模拟
    simulation_records = [
        {
            "策略": "纯规则基线（全部通过）",
            "通过率": 1.0,
            "坏账率": round(total_bad, 4),
            "人工审核占比": 0.0,
            "自动审批率": 1.0,
        }
    ]
    for rec in strategy_records:
        simulation_records.append(
            {
                "策略": rec["方案"],
                "通过率": rec["最终通过率"],
                "坏账率": rec["最终坏账率"],
                "人工审核占比": rec["人工审核占比"],
                "自动审批率": rec["自动审批率"],
            }
        )
    simulation_df = pd.DataFrame(simulation_records)
    simulation_df.to_csv(
        OUT_DIR + "\\strategy_simulation.csv", index=False, encoding="utf-8-sig"
    )
    print(simulation_df.to_string(index=False))
    print(f"策略模拟表已保存: {OUT_DIR}\\strategy_simulation.csv")


if __name__ == "__main__":
    main()
