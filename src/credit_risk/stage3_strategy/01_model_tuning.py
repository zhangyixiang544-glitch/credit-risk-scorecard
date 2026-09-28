"""模型迭代优化脚本：超参数调优与多版本对比。

阶段三第一步。基于阶段二 LightGBM 基线（05），对 LightGBM
做网格搜索调优，再叠加样本不均衡处理，输出至少 3 个版本
的模型性能，形成迭代轨迹。

版本定义：
- v0 基线：默认参数 LightGBM（与 05 一致）
- v1 调优：网格搜索最优参数
- v2 调优+均衡：v1 参数 + 类别权重（scale_pos_weight）

输出：
- lgbm_tuning_results.csv：网格搜索各参数组合的验证集 AUC
- model_versions.csv：3 个版本训练/测试 AUC、KS 与过拟合差距
"""

import os

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


OUT_DIR = os.path.join(PROJECT_ROOT, "output", "stage3")

FEATURE_FILE = os.path.join(PROJECT_ROOT, "output", "stage2", "processed_features.csv")
COEF_FILE = os.path.join(PROJECT_ROOT, "output", "stage2", "logit_coefficients.csv")

# 与阶段二一致的划分参数
RANDOM_STATE = 42
TEST_SIZE = 0.3
# 过拟合差距阈值
OVERFIT_GAP = 0.05
# 网格搜索参数（粗网格，控制运行时长）
PARAM_GRID = {
    "learning_rate": [0.05, 0.1],
    "num_leaves": [31, 63],
    "max_depth": [-1, 8],
    "n_estimators": [200, 300],
    "reg_alpha": [0.0, 0.5],
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


def run_version(
    name: str,
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """训练一个版本并返回性能指标。

    Args:
        name: 版本名
        params: LightGBM 参数
        X_train: 训练特征
        y_train: 训练标签
        X_test: 测试特征
        y_test: 测试标签

    Returns:
        版本性能字典
    """
    model = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, **params)
    model.fit(X_train, y_train)
    train_pred = model.predict_proba(X_train)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]
    train_auc = roc_auc_score(y_train, train_pred)
    test_auc = roc_auc_score(y_test, test_pred)
    train_ks = calc_ks(y_train, train_pred)
    test_ks = calc_ks(y_test, test_pred)
    return {
        "版本": name,
        "参数": str(params),
        "训练AUC": round(train_auc, 4),
        "测试AUC": round(test_auc, 4),
        "训练KS": round(train_ks, 4),
        "测试KS": round(test_ks, 4),
        "过拟合差距": round(train_auc - test_auc, 4),
    }


def main() -> None:
    """主流程：网格搜索、训练 3 个版本、保存对比结果。"""
    print("=" * 60)
    title = "模型迭代优化开始"
    print(title.center(60, "="))

    df = load_features()
    model_cols = load_model_features()
    print(f"入模特征: {len(model_cols)} 个（与 05 一致）")

    X = encode_string_cols(df[model_cols], model_cols)
    y = df["TARGET"]

    # 1. 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"训练集: {X_train.shape[0]} 行，测试集: {X_test.shape[0]} 行")

    # 2. 网格搜索（取三次交叉验证）
    base = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    grid = GridSearchCV(base, PARAM_GRID, cv=3, scoring="roc_auc", n_jobs=1, verbose=0)
    grid.fit(X_train, y_train)
    best_params = grid.best_params_
    print(f"最优参数: {best_params}")
    print(f"最优交叉验证 AUC: {grid.best_score_:.4f}")
    tuning_df = pd.DataFrame(grid.cv_results_)[
        ["params", "mean_test_score", "std_test_score"]
    ].rename(
        columns={
            "params": "参数",
            "mean_test_score": "验证集AUC均值",
            "std_test_score": "验证集AUC标准差",
        }
    )
    tuning_df["参数"] = tuning_df["参数"].astype(str)

    # 3. 训练 3 个版本
    pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    versions = [
        run_version("v0 基线", {}, X_train, y_train, X_test, y_test),
        run_version("v1 调优", best_params, X_train, y_train, X_test, y_test),
        run_version(
            "v2 调优与均衡",
            {**best_params, "scale_pos_weight": round(pos_weight, 2)},
            X_train,
            y_train,
            X_test,
            y_test,
        ),
    ]
    for v in versions:
        print(
            f"{v['版本']}: 测试AUC {v['测试AUC']:.4f}，"
            f"测试KS {v['测试KS']:.4f}，过拟合差距 {v['过拟合差距']:.4f}"
        )

    # 4. 保存结果
    tuning_df.to_csv(
        OUT_DIR + "\\lgbm_tuning_results.csv", index=False, encoding="utf-8-sig"
    )
    versions_df = pd.DataFrame(versions)
    versions_df.to_csv(
        OUT_DIR + "\\model_versions.csv", index=False, encoding="utf-8-sig"
    )
    print(f"网格搜索结果已保存: {OUT_DIR}\\lgbm_tuning_results.csv")
    print(f"版本对比表已保存: {OUT_DIR}\\model_versions.csv")


if __name__ == "__main__":
    main()
