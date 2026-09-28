"""数据质量巡检脚本：对每一张表统计缺失率、唯一值占比等质量指标。

输出字段级质量统计（缺失率、唯一值数、数据类型），并按任务书口径
做缺失率三级分级（<10% 低、10%-30% 中、>30% 高），结果保存为 CSV。
"""

import gc
import os

import pandas as pd

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
OUT_DIR = os.path.join(PROJECT_ROOT, "output", "stage1")

# 如果输出目录不存在就先创建
os.makedirs(OUT_DIR, exist_ok=True)

# 需要做质量巡检的表
MAIN_TABLE = "application_train.csv"
# 附表：数据量大，用分块方式统计缺失率
SIDE_TABLES = [
    "bureau.csv",
    "bureau_balance.csv",
    "previous_application.csv",
    "POS_CASH_balance.csv",
    "credit_card_balance.csv",
    "installments_payments.csv",
]


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """把 DataFrame 里各列的类型往小了调，节省内存。

    默认整型 int64、浮点 float64，很多列用不到这么宽，转小类型能省一大半内存。

    Args:
        df: 原始 DataFrame

    Returns:
        类型优化后的 DataFrame
    """
    for col in df.columns:
        col_type = df[col].dtype

        # 整型列：按取值范围缩小
        if col_type == "int64":
            col_min = df[col].min()
            col_max = df[col].max()
            if col_min >= -128 and col_max <= 127:
                df[col] = df[col].astype("int8")
            elif col_min >= -32768 and col_max <= 32767:
                df[col] = df[col].astype("int16")
            elif col_min >= -2147483648 and col_max <= 2147483647:
                df[col] = df[col].astype("int32")

        # 浮点列：统一降成 float32
        elif col_type == "float64":
            df[col] = df[col].astype("float32")

    return df


def quality_report(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """对一张已读入内存的表做字段级质量统计。

    逐列统计缺失数量/缺失率、唯一值数量/占比、数据类型，并按任务书口径分三档。

    Args:
        df: 目标表 DataFrame
        table_name: 表名，用于标注

    Returns:
        质量统计 DataFrame，一行一个字段
    """
    total_rows = len(df)
    records = []

    for col in df.columns:
        # 缺失统计：pandas 的 NaN/None 都算缺失
        missing_count = df[col].isna().sum()
        missing_rate = missing_count / total_rows

        unique_count = df[col].nunique()
        unique_rate = unique_count / total_rows

        if missing_rate < 0.1:
            missing_level = "低(<10%)"
        elif missing_rate <= 0.3:
            missing_level = "中(10%-30%)"
        else:
            missing_level = "高(>30%)"

        records.append(
            {
                "表名": table_name,
                "字段名": col,
                "数据类型": str(df[col].dtype),
                "缺失数量": missing_count,
                "缺失率": round(missing_rate, 4),
                "缺失等级": missing_level,
                "唯一值数量": unique_count,
                "唯一值占比": round(unique_rate, 4),
            }
        )

    return pd.DataFrame(records)


def side_table_missing_rate(table_name: str) -> pd.DataFrame:
    """对超大附表分块统计缺失率。

    附表全量读入内存不够，分块读入、每块只累加各列缺失数量，最后汇总。

    Args:
        table_name: 附表文件名

    Returns:
        缺失率统计 DataFrame
    """
    file_path = DATA_DIR + "\\" + table_name
    # 调整分块大小
    chunksize = 200000

    first_chunk = pd.read_csv(file_path, nrows=100)
    columns = first_chunk.columns.tolist()

    total_missing = {col: 0 for col in columns}
    total_rows = 0

    # 数值列按数值读，比全部读成字符串快一点
    for chunk in pd.read_csv(file_path, chunksize=chunksize, low_memory=False):
        total_rows += len(chunk)
        for col in columns:
            total_missing[col] += chunk[col].isna().sum()
        del chunk

    # 主动回收一次内存
    gc.collect()

    records = []
    for col in columns:
        missing_rate = total_missing[col] / total_rows
        if missing_rate < 0.1:
            missing_level = "低(<10%)"
        elif missing_rate <= 0.3:
            missing_level = "中(10%-30%)"
        else:
            missing_level = "高(>30%)"

        records.append(
            {
                "表名": table_name,
                "字段名": col,
                "数据类型": "object(分块)",
                "缺失数量": total_missing[col],
                "缺失率": round(missing_rate, 4),
                "缺失等级": missing_level,
                "唯一值数量": None,
                "唯一值占比": None,
            }
        )

    return pd.DataFrame(records)


def main() -> None:
    """主流程：对主表做完整巡检，对附表做分块缺失率统计。"""
    print("=" * 60)
    print("数据质量巡检开始")
    print("=" * 60)

    all_reports = []

    # 1. 主表：全量读入做完整巡检（含唯一值）
    print(f"\n[1/2] 主表 {MAIN_TABLE} 质量巡检（全量读入）...")
    train_df = pd.read_csv(DATA_DIR + "\\" + MAIN_TABLE)
    train_df = optimize_dtypes(train_df)
    print(
        f"读入完成，优化后内存: {train_df.memory_usage(deep=True).sum() / 1024 / 1024:.0f} MB"
    )
    main_report = quality_report(train_df, MAIN_TABLE)
    all_reports.append(main_report)

    del train_df
    gc.collect()

    # 2. 附表：分块统计缺失率
    for table_name in SIDE_TABLES:
        print(f"  处理 {table_name} ...")
        report = side_table_missing_rate(table_name)
        all_reports.append(report)

    # 3. 合并所有结果并保存
    full_report = pd.concat(all_reports, ignore_index=True)

    # 按表 + 缺失率排序
    full_report = full_report.sort_values(["表名", "缺失率"], ascending=[True, False])

    out_file = OUT_DIR + "\\data_quality_summary.csv"
    full_report.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"\n质量巡检结果已保存: {out_file}")


if __name__ == "__main__":
    main()
