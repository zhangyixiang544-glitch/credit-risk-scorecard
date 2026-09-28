"""时间窗口校验脚本：检查日期类字段相对申请时点的方向，识别贷后数据混入风险。

建模特征必须为"申请时点切片"（贷前信息），本脚本逐字段检查 DAYS_* / MONTHS_*：
负值=贷前（合规）、正值=贷后（混入则模型失效）、0=申请当天。
输出每字段的正值占比、极值、缺失占比与结论，供阶段二聚合时过滤/截断。
"""

import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
OUT_DIR = os.path.join(PROJECT_ROOT, "output", "stage1")

# 主表日期字段（DAYS_* 以申请时点为 0，负数表示申请日之前）
MAIN_TABLE_DATE_FIELDS = [
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE",
]

# 附表日期字段清单：{文件名: [待校验字段]}
# 附表记录的是历史贷款/账户的时点信息，同样以本次申请时点为基准
AUX_TABLE_DATE_FIELDS = {
    "bureau.csv": [
        "DAYS_CREDIT",
        "DAYS_CREDIT_ENDDATE",
        "DAYS_ENDDATE_FACT",
        "DAYS_CREDIT_UPDATE",
    ],
    "previous_application.csv": [
        "DAYS_DECISION",
        "DAYS_FIRST_DRAWING",
        "DAYS_FIRST_DUE",
        "DAYS_LAST_DUE_1ST_VERSION",
        "DAYS_LAST_DUE",
        "DAYS_TERMINATION",
    ],
    "POS_CASH_balance.csv": ["MONTHS_BALANCE"],
    "credit_card_balance.csv": ["MONTHS_BALANCE"],
    "installments_payments.csv": ["DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"],
    "bureau_balance.csv": ["MONTHS_BALANCE"],
}

# 附表行数很大，分块读取
CHUNK_SIZE = 200_000


def check_series(series: pd.Series) -> dict:
    """统计单个日期字段的方向分布。

    Args:
        series: 日期字段列（DAYS_* 或 MONTHS_*）

    Returns:
        统计字典：有效数、正/负/零/缺失数量、最小值、最大值
    """
    total = len(series)
    valid = series.dropna()
    positive = int((valid > 0).sum())
    negative = int((valid < 0).sum())
    zero = int((valid == 0).sum())
    missing = int(series.isna().sum())
    col_min = float(valid.min()) if len(valid) > 0 else np.nan
    col_max = float(valid.max()) if len(valid) > 0 else np.nan
    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "zero": zero,
        "missing": missing,
        "min": col_min,
        "max": col_max,
    }


def check_table_columns(file_path: str, columns: list) -> list:
    """分块读取附表指定列并统计方向分布。

    Args:
        file_path: 附表完整路径
        columns: 待校验的日期字段列表

    Returns:
        每个字段的统计结果列表
    """
    # 分块累加：每块只统计增量，不保留数据，避免占内存
    results = []
    for column in columns:
        total = 0
        positive = 0
        negative = 0
        zero = 0
        missing = 0
        col_min = np.inf
        col_max = -np.inf
        for chunk in pd.read_csv(file_path, usecols=[column], chunksize=CHUNK_SIZE):
            series = chunk[column]
            valid = series.dropna()
            total += len(series)
            positive += int((valid > 0).sum())
            negative += int((valid < 0).sum())
            zero += int((valid == 0).sum())
            missing += int(series.isna().sum())
            if len(valid) > 0:
                col_min = min(col_min, float(valid.min()))
                col_max = max(col_max, float(valid.max()))
        results.append(
            {
                "total": total,
                "positive": positive,
                "negative": negative,
                "zero": zero,
                "missing": missing,
                "min": col_min if total > 0 else np.nan,
                "max": col_max if total > 0 else np.nan,
            }
        )
    return results


def build_conclusion(positive_ratio: float) -> str:
    """根据贷后（正值）占比给出合规结论。

    Args:
        positive_ratio: 正值占比（0~1）

    Returns:
        结论文本
    """
    if positive_ratio > 0.05:
        return "贷后占比高，聚合时必须过滤或截断"
    if positive_ratio > 0:
        return "存在少量贷后数据，聚合时需校验"
    return "全部为贷前数据，合规"


def main() -> None:
    """主流程：校验主表和所有附表的日期字段，输出汇总 CSV。"""
    print("=" * 60)
    print("时间窗口校验开始（申请时点切片检查）")
    print("=" * 60)

    rows = []

    # 1. 主表：读全表，统计日期字段
    main_file = DATA_DIR + "\\application_train.csv"
    main_df = pd.read_csv(main_file)
    for column in MAIN_TABLE_DATE_FIELDS:
        stats = check_series(main_df[column])
        positive_ratio = stats["positive"] / stats["total"] if stats["total"] > 0 else 0
        rows.append(
            {
                "表名": "application_train",
                "字段名": column,
                "记录数": stats["total"],
                "正值数(贷后)": stats["positive"],
                "正值占比": round(positive_ratio, 4),
                "负值占比": (
                    round(stats["negative"] / stats["total"], 4)
                    if stats["total"] > 0
                    else 0
                ),
                "零值占比": (
                    round(stats["zero"] / stats["total"], 4)
                    if stats["total"] > 0
                    else 0
                ),
                "缺失占比": (
                    round(stats["missing"] / stats["total"], 4)
                    if stats["total"] > 0
                    else 0
                ),
                "最小值": stats["min"],
                "最大值": stats["max"],
                "结论": build_conclusion(positive_ratio),
            }
        )
        print(f"  {column}: 正值占比 {positive_ratio:.2%}")

    # 2. 附表：分块读取并统计
    for file_name, columns in AUX_TABLE_DATE_FIELDS.items():
        file_path = DATA_DIR + "\\" + file_name
        table_name = file_name.replace(".csv", "")
        print(f"  {table_name}: {columns}")
        stats_list = check_table_columns(file_path, columns)
        for column, stats in zip(columns, stats_list):
            positive_ratio = (
                stats["positive"] / stats["total"] if stats["total"] > 0 else 0
            )
            rows.append(
                {
                    "表名": table_name,
                    "字段名": column,
                    "记录数": stats["total"],
                    "正值数(贷后)": stats["positive"],
                    "正值占比": round(positive_ratio, 4),
                    "负值占比": (
                        round(stats["negative"] / stats["total"], 4)
                        if stats["total"] > 0
                        else 0
                    ),
                    "零值占比": (
                        round(stats["zero"] / stats["total"], 4)
                        if stats["total"] > 0
                        else 0
                    ),
                    "缺失占比": (
                        round(stats["missing"] / stats["total"], 4)
                        if stats["total"] > 0
                        else 0
                    ),
                    "最小值": stats["min"],
                    "最大值": stats["max"],
                    "结论": build_conclusion(positive_ratio),
                }
            )

    # 3. 输出 CSV 和风险提示
    output = pd.DataFrame(rows)
    out_file = OUT_DIR + "\\time_window_validation.csv"
    output.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"\n校验结果已保存: {out_file}")


if __name__ == "__main__":
    main()
