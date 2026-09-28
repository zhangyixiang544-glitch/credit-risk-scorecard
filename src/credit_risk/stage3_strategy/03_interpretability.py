"""模型可解释性分析：SHAP 全局/局部解释与合规性校验。

基于 01 的 v1 调优模型（测试 AUC 最优），
输出 SHAP 全局摘要图、特征重要性排序、典型样本力导图，
并校验模型在性别、地域等敏感属性上的差异，符合公平信贷原则。

输出：
- feature_importance.csv：Gain 重要性 + SHAP 均值重要性
- shap_summary.png：SHAP 全局摘要图（前十核心因子）
- shap_force_*.png：高/低风险样本各 3 张力导图
- compliance_check.csv：性别、地域分群预测差异
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


OUT_DIR = os.path.join(PROJECT_ROOT, "output", "stage3")

FEATURE_FILE = os.path.join(PROJECT_ROOT, "output", "stage2", "processed_features.csv")
COEF_FILE = os.path.join(PROJECT_ROOT, "output", "stage2", "logit_coefficients.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.3
# SHAP 摘要图抽样行数
SHAP_SAMPLE_SIZE = 2000
# 局部解释样本数（高/低风险各取）
LOCAL_SAMPLE_N = 3

# 01 网格搜索最优参数（v1）
BEST_PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 8,
    "n_estimators": 300,
    "reg_alpha": 0.5,
}


def load_features() -> pd.DataFrame:
    """读取 01 产出的预处理特征表。

    Returns:
        特征表 DataFrame
    """
    df = pd.read_csv(FEATURE_FILE)
    return df


def load_model_features() -> list:
    """从 04 系数表取最终入模特征名（排除常数项）。

    Returns:
        特征列名列表
    """
    coef = pd.read_csv(COEF_FILE)
    cols = coef["特征名"].tolist()
    return [c for c in cols if c != "const"]


def encode_string_cols(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """对字符串列做标签编码，缺失值先填"缺失"再编码。

    Args:
        df: 特征表 DataFrame
        cols: 入模特征列名列表

    Returns:
        编码后的 DataFrame
    """
    result = df.copy()
    for col in cols:
        if pd.api.types.is_string_dtype(result[col]):
            result[col] = result[col].fillna("缺失")
            encoder = LabelEncoder()
            result[col] = encoder.fit_transform(result[col])
    return result


def set_chinese_font() -> None:
    """设置中文字体，避免图表中文显示为方框。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False


def save_force_plot(
    explainer, shap_value: np.ndarray, sample: pd.Series, path: str
) -> None:
    """保存单个样本的 SHAP 力导图。

    Args:
        explainer: SHAP 解释器
        shap_value: 该样本的 SHAP 值数组
        sample: 样本行
        path: 保存路径
    """
    plt.figure()
    shap.force_plot(
        explainer.expected_value,
        shap_value,
        sample,
        matplotlib=True,
        show=False,
    )
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()


def main() -> None:
    """主流程：训练模型、SHAP 全局/局部解释、合规校验、保存结果。"""
    matplotlib.use("Agg")
    print("=" * 60)
    title = "模型可解释性分析开始"
    print(title.center(60, "="))
    set_chinese_font()

    df = load_features()
    model_cols = load_model_features()
    X = encode_string_cols(df[model_cols], model_cols)
    y = df["TARGET"]
    print(f"入模特征: {len(model_cols)} 个")

    # 1. 训练 v1 模型并预测测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    model = LGBMClassifier(
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, **BEST_PARAMS
    )
    model.fit(X_train, y_train)
    test_prob = model.predict_proba(X_test)[:, 1]
    print("模型训练完成，测试集预测完成")

    # 2. SHAP 全局解释
    sample_idx = np.random.RandomState(RANDOM_STATE).choice(
        len(X_test), size=SHAP_SAMPLE_SIZE, replace=False
    )
    X_sample = X_test.iloc[sample_idx]
    sample_prob = test_prob[sample_idx]
    sample_label = y_test.iloc[sample_idx].to_numpy()

    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_sample)
    shap_values = explanation.values

    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0), index=model_cols, name="SHAP均值"
    ).sort_values(ascending=False)
    importance_df = pd.DataFrame(
        {
            "特征名": model_cols,
            "Gain重要性": model.feature_importances_,
            "SHAP均值重要性": mean_abs_shap.values,
        }
    ).sort_values("SHAP均值重要性", ascending=False)
    importance_df.to_csv(
        OUT_DIR + "\\feature_importance.csv", index=False, encoding="utf-8-sig"
    )
    print(f"SHAP 均值重要性 Top5: {importance_df['特征名'].head(5).tolist()}")

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=10)
    plt.tight_layout()
    plt.savefig(OUT_DIR + "\\shap_summary.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"SHAP 全局摘要图已保存: {OUT_DIR}\\shap_summary.png")

    # 3. 局部解释：在子集中选高风险/低风险样本各 3 个
    prob_df = pd.DataFrame({"prob": sample_prob, "label": sample_label})
    high_risk_pos = (
        prob_df[prob_df["label"] == 1]
        .sort_values("prob", ascending=False)
        .head(LOCAL_SAMPLE_N)
        .index
    )
    low_risk_pos = (
        prob_df[prob_df["label"] == 0].sort_values("prob").head(LOCAL_SAMPLE_N).index
    )
    for pos, i in enumerate(high_risk_pos):
        save_force_plot(
            explainer,
            shap_values[i],
            X_sample.iloc[i],
            OUT_DIR + f"\\shap_force_high_risk_{pos + 1}.png",
        )
    for pos, i in enumerate(low_risk_pos):
        save_force_plot(
            explainer,
            shap_values[i],
            X_sample.iloc[i],
            OUT_DIR + f"\\shap_force_low_risk_{pos + 1}.png",
        )
    print(f"高/低风险力导图各 {LOCAL_SAMPLE_N} 张已保存")

    # 4. 合规性校验：性别、地域分群预测差异
    X_test_full = X_test.copy()
    X_test_full["TARGET"] = y_test.values
    X_test_full["PRED_PROB"] = test_prob
    records = []
    for group_col, group_name in [
        ("CODE_GENDER", "性别"),
        ("REGION_RATING_CLIENT", "地域评级"),
    ]:
        for value, group in X_test_full.groupby(group_col):
            records.append(
                {
                    "分组维度": group_name,
                    "分组取值": value,
                    "样本数": len(group),
                    "实际违约率": round(group["TARGET"].mean(), 4),
                    "平均预测违约概率": round(group["PRED_PROB"].mean(), 4),
                }
            )
    compliance_df = pd.DataFrame(records)
    compliance_df.to_csv(
        OUT_DIR + "\\compliance_check.csv", index=False, encoding="utf-8-sig"
    )
    print(compliance_df.to_string(index=False))
    print(f"合规校验表已保存: {OUT_DIR}\\compliance_check.csv")


if __name__ == "__main__":
    main()
