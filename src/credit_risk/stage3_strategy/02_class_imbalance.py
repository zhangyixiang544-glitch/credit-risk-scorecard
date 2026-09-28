"""样本不均衡处理脚本：类别权重 / SMOTE / 阈值优化三方案对比。

固定模型参数（01 的 v1 最优参数），对比三种不均衡
处理方案，输出各方案的精确率、召回率、F1，选择适配风控场景的方案。

方案定义：
- A 基线：不做处理
- B 类别权重：scale_pos_weight 按正负样本比加权
- C SMOTE：训练集过采样至正负平衡
- D 阈值优化：基线模型预测概率上扫描最优分类阈值

输出：
- imbalance_comparison.csv：四方案指标对比
"""

import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


OUT_DIR = os.path.join(PROJECT_ROOT, "output", "stage3")

FEATURE_FILE = (
    os.path.join(PROJECT_ROOT, "output", "stage2", "processed_features.csv")
)
COEF_FILE = (
    os.path.join(PROJECT_ROOT, "output", "stage2", "logit_coefficients.csv")
)

RANDOM_STATE = 42
TEST_SIZE = 0.3
# 与 01 一致的划分参数
VAL_SIZE = 0.2
# 阈值扫描步长
THRESHOLD_STEP = 0.05

# v1 网格搜索最优参数
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


def eval_with_threshold(y_true: pd.Series, y_prob: pd.Series, threshold: float) -> dict:
    """按给定阈值评估精确率、召回率、F1。

    Args:
        y_true: 真实标签
        y_prob: 预测概率
        threshold: 分类阈值

    Returns:
        指标字典
    """
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "阈值": round(threshold, 2),
        "精确率": round(precision_score(y_true, y_pred), 4),
        "召回率": round(recall_score(y_true, y_pred), 4),
        "F1": round(f1_score(y_true, y_pred), 4),
    }


def main() -> None:
    """主流程：四种方案训练评估、阈值优化、保存对比结果。"""
    print("=" * 60)
    title = "样本不均衡处理对比开始"
    print(title.center(60, "="))

    df = load_features()
    model_cols = load_model_features()
    X = encode_string_cols(df[model_cols], model_cols)
    y = df["TARGET"]
    print(f"入模特征: {len(model_cols)} 个，正样本占比: {y.mean():.4f}")

    # 1. 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    # 从训练集再切验证集，供阈值优化使用
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )
    print(
        f"训练集: {X_tr.shape[0]} 行，验证集: {X_val.shape[0]} 行，测试集: {X_test.shape[0]} 行"
    )

    pos_weight = (len(y_tr) - y_tr.sum()) / y_tr.sum()

    # 2. 方案 A 基线
    base = LGBMClassifier(
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, **BEST_PARAMS
    )
    base.fit(X_tr, y_tr)
    val_prob = base.predict_proba(X_val)[:, 1]

    # 3. 方案 B 类别权重
    weighted = LGBMClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
        **{**BEST_PARAMS, "scale_pos_weight": round(pos_weight, 2)},
    )
    weighted.fit(X_tr, y_tr)

    # 4. 方案 C SMOTE 过采样
    smote = SMOTE(random_state=RANDOM_STATE)
    X_tr_filled = X_tr.fillna(X_tr.median())
    X_rs, y_rs = smote.fit_resample(X_tr_filled, y_tr)
    print(f"SMOTE 后训练集: {X_rs.shape[0]} 行（正样本 {y_rs.sum():.0f} 个）")
    smote_model = LGBMClassifier(
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, **BEST_PARAMS
    )
    smote_model.fit(X_rs, y_rs)

    # 5. 方案 D 阈值优化（基于基线概率，验证集上扫描）
    best_f1 = -1.0
    best_threshold = 0.5
    threshold = THRESHOLD_STEP
    while threshold < 1.0:
        y_val_pred = (val_prob >= threshold).astype(int)
        f1 = f1_score(y_val, y_val_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
        threshold += THRESHOLD_STEP
    print(f"最优分类阈值: {best_threshold:.2f}（验证集 F1 {best_f1:.4f}）")

    test_prob_base = base.predict_proba(X_test)[:, 1]
    test_prob_weighted = weighted.predict_proba(X_test)[:, 1]
    test_prob_smote = smote_model.predict_proba(X_test)[:, 1]

    records = []
    for name, prob in [
        ("A 基线", test_prob_base),
        ("B 类别权重", test_prob_weighted),
        ("C SMOTE", test_prob_smote),
        ("D 阈值优化", test_prob_base),
    ]:
        threshold = 0.5 if name != "D 阈值优化" else best_threshold
        metric = eval_with_threshold(y_test, prob, threshold)
        records.append(
            {
                "方案": name,
                **metric,
                "测试AUC": round(roc_auc_score(y_test, prob), 4),
            }
        )

    result_df = pd.DataFrame(records)
    result_df.to_csv(
        OUT_DIR + "\\imbalance_comparison.csv", index=False, encoding="utf-8-sig"
    )
    print(result_df.to_string(index=False))
    print(f"对比表已保存: {OUT_DIR}\\imbalance_comparison.csv")


if __name__ == "__main__":
    main()
