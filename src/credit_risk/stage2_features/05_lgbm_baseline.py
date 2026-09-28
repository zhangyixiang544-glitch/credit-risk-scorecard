"""LightGBM 基线模型脚本：与逻辑回归评分卡对比，并校验过拟合。

阶段二第五步。使用与 04 逻辑回归相同的训练/测试划分和入模特征，
训练 LightGBM 基线模型，输出训练/测试 AUC 与 KS，对比两个模型，
通过训练与测试的差距判断是否过拟合。

输出：
- lgbm_performance.csv：LightGBM 训练/测试 AUC、KS 与过拟合差距
- model_comparison.csv：LightGBM 与逻辑回归的性能对比
"""

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


OUT_DIR = os.path.join(PROJECT_ROOT, "output", "stage2")

FEATURE_FILE = OUT_DIR + "\\processed_features.csv"
COEF_FILE = OUT_DIR + "\\logit_coefficients.csv"
LR_PERF_FILE = OUT_DIR + "\\model_performance.csv"

# 保持一致划分参数
RANDOM_STATE = 42
TEST_SIZE = 0.3
# 训练与测试 AUC 差距超过该值视为明显过拟合
OVERFIT_GAP = 0.05


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


def calc_ks(y_true: pd.Series, y_score: pd.Series) -> float:
    """计算 KS 统计量：好坏累计占比差的最大值。

    Args:
        y_true: 真实标签
        y_score: 预测概率

    Returns:
        KS 值
    """
    data = pd.DataFrame({"y": y_true, "score": y_score})
    data = data.sort_values("score", ascending=False)
    data["cum_bad"] = data["y"].cumsum() / data["y"].sum()
    data["cum_good"] = (1 - data["y"]).cumsum() / (1 - data["y"]).sum()
    return float((data["cum_bad"] - data["cum_good"]).abs().max())


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


def main() -> None:
    """主流程：取特征、编码、划分、训练 LightGBM、评估对比。"""
    print("=" * 60)
    title = "LightGBM 基线模型构建开始"
    print(title.center(60, "="))

    df = load_features()
    model_cols = load_model_features()
    print(f"对比特征: {len(model_cols)} 个（04 逻辑回归最终入模特征）")

    X = df[model_cols]
    y = df["TARGET"]
    X = encode_string_cols(X, model_cols)
    print(f"特征矩阵维度: {X.shape}")

    # 1. 划分训练/测试集（与 04 相同）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"训练集: {X_train.shape[0]} 行，测试集: {X_test.shape[0]} 行")

    # 2. 训练 LightGBM
    model = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    model.fit(X_train, y_train)

    # 3. 模型评估
    train_pred = model.predict_proba(X_train)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]
    train_auc = roc_auc_score(y_train, train_pred)
    test_auc = roc_auc_score(y_test, test_pred)
    train_ks = calc_ks(y_train, train_pred)
    test_ks = calc_ks(y_test, test_pred)
    gap = train_auc - test_auc
    print(f"训练 AUC: {train_auc:.4f}，测试 AUC: {test_auc:.4f}")
    print(f"训练 KS: {train_ks:.4f}，测试 KS: {test_ks:.4f}")
    print(f"训练-测试 AUC 差距: {gap:.4f}（阈值 {OVERFIT_GAP}）")

    # 4. 输出对比结果
    perf_df = pd.DataFrame(
        [
            {
                "指标": "AUC",
                "训练集": round(train_auc, 4),
                "测试集": round(test_auc, 4),
                "过拟合差距": round(gap, 4),
            },
            {
                "指标": "KS",
                "训练集": round(train_ks, 4),
                "测试集": round(test_ks, 4),
                "过拟合差距": "",
            },
        ]
    )
    perf_df.to_csv(
        OUT_DIR + "\\lgbm_performance.csv", index=False, encoding="utf-8-sig"
    )

    lr_perf = pd.read_csv(LR_PERF_FILE)
    lr_row = {
        "模型": "逻辑回归",
        "训练AUC": lr_perf.loc[lr_perf["指标"] == "AUC", "训练集"].iloc[0],
        "测试AUC": lr_perf.loc[lr_perf["指标"] == "AUC", "测试集"].iloc[0],
        "训练KS": lr_perf.loc[lr_perf["指标"] == "KS", "训练集"].iloc[0],
        "测试KS": lr_perf.loc[lr_perf["指标"] == "KS", "测试集"].iloc[0],
    }
    lgbm_row = {
        "模型": "LightGBM",
        "训练AUC": round(train_auc, 4),
        "测试AUC": round(test_auc, 4),
        "训练KS": round(train_ks, 4),
        "测试KS": round(test_ks, 4),
    }
    compare_df = pd.DataFrame([lr_row, lgbm_row])
    compare_df.to_csv(
        OUT_DIR + "\\model_comparison.csv", index=False, encoding="utf-8-sig"
    )
    print(f"LightGBM 性能表已保存: {OUT_DIR}\\lgbm_performance.csv")
    print(f"模型对比表已保存: {OUT_DIR}\\model_comparison.csv")


if __name__ == "__main__":
    main()
