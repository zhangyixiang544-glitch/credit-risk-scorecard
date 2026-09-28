"""一键复现脚本：按阶段顺序运行全部 19 个脚本，失败即停。

用法（在项目根目录下）：
    python scripts/run_all.py

运行前请确保：
    1. 已安装依赖：pip install -r requirements.txt
    2. 原始数据已放入 data/raw/ 目录
全程约 40~60 分钟，请耐心等待。
"""

import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 按阶段组织的脚本列表（相对项目根的路径）
STAGES = [
    ("阶段一：业务调研与数据资产", [
        "src/credit_risk/stage1_data_assets/01_data_exploration.py",
        "src/credit_risk/stage1_data_assets/02_data_quality_check.py",
        "src/credit_risk/stage1_data_assets/03_field_dictionary_builder.py",
        "src/credit_risk/stage1_data_assets/04_time_window_validation.py",
        "src/credit_risk/stage1_data_assets/05_build_master_table.py",
        "src/credit_risk/stage1_data_assets/06_exploratory_data_analysis.py",
        "src/credit_risk/stage1_data_assets/07_feature_screening.py",
        "src/credit_risk/stage1_data_assets/08_supplementary_cross_analysis.py",
    ]),
    ("阶段二：特征工程与基线模型", [
        "src/credit_risk/stage2_features/01_feature_engineering.py",
        "src/credit_risk/stage2_features/02_woe_binning.py",
        "src/credit_risk/stage2_features/03_loan_availability_check.py",
        "src/credit_risk/stage2_features/04_logistic_scorecard.py",
        "src/credit_risk/stage2_features/05_lgbm_baseline.py",
        "src/credit_risk/stage2_features/06_threshold_strategy.py",
    ]),
    ("阶段三：模型优化与策略", [
        "src/credit_risk/stage3_strategy/01_model_tuning.py",
        "src/credit_risk/stage3_strategy/02_class_imbalance.py",
        "src/credit_risk/stage3_strategy/03_interpretability.py",
        "src/credit_risk/stage3_strategy/04_risk_strategy.py",
    ]),
    ("阶段四：价值评估", [
        "src/credit_risk/stage4_value/01_value_assessment.py",
    ]),
]


def run_script(script_path: str) -> None:
    """运行单个脚本，非零退出码直接终止整个流程。"""
    full_path = os.path.join(PROJECT_ROOT, script_path)
    print(f"\n>>> 运行: {script_path}")
    result = subprocess.run(
        [sys.executable, full_path],
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        print(f"\n[失败] 脚本 {script_path} 退出码 {result.returncode}，流程终止。")
        sys.exit(1)
    print(f"[完成] {script_path}")


def main() -> None:
    """按阶段顺序运行全部脚本。"""
    print("=" * 60)
    print("一键复现开始（全部阶段）")
    print("=" * 60)

    for stage_name, scripts in STAGES:
        print(f"\n{'=' * 60}\n{stage_name}\n{'=' * 60}")
        for script in scripts:
            run_script(script)

    print("\n" + "=" * 60)
    print("全部阶段运行完毕，产物已输出到 output/ 目录。")
    print("=" * 60)


if __name__ == "__main__":
    main()
