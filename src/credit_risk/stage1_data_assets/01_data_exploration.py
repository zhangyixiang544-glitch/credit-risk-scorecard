"""数据探查脚本：对数据集的每一张表做基础体检。

统计每张表的行数、列数、列类型、全量读入内存估算，
并读入字段说明文件，为后续字段字典做准备。
"""

import pandas as pd
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


# 数据目录：所有 CSV 文件都放在这个文件夹里
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

# 需要探查的 8 张表（含字段说明文件）
TABLE_NAMES = [
    "application_train.csv",
    "application_test.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "previous_application.csv",
    "POS_CASH_balance.csv",
    "credit_card_balance.csv",
    "installments_payments.csv",
]

# 字段说明文件
COLUMNS_DESC_FILE = "credit_columns_description.csv"


def count_rows(file_path: str, chunksize: int = 200000) -> int:
    """分块读取 CSV，累加统计总行数（不包含表头）。

    数据量大时不能用 len(pd.read_csv()) 直接读，会爆内存，所以分块累加。

    Args:
        file_path: CSV 文件的完整路径
        chunksize: 每次读取的块大小，默认 20 万行

    Returns:
        文件的总行数
    """
    total = 0
    # dtype 用 str 可以减少内存，我们只需要行数，不关心具体值
    for chunk in pd.read_csv(file_path, chunksize=chunksize, dtype=str, usecols=[0]):
        total += len(chunk)
    return total


def read_first_rows(file_path: str, nrows: int = 5) -> pd.DataFrame:
    """读取 CSV 的前几行，用来看看长什么样。

    Args:
        file_path: CSV 文件的完整路径
        nrows: 要读的行数，默认 5 行

    Returns:
        前 nrows 行的 DataFrame
    """
    return pd.read_csv(file_path, nrows=nrows)


def estimate_memory(file_path: str, sample_size: int = 100000) -> float:
    """估算全量读入一张表大概需要多少内存（单位 MB）。

    用 sample_size 行的实际内存占用按比例推算全表，估算偏小但够判断能否读入。

    Args:
        file_path: CSV 文件的完整路径
        sample_size: 用于估算的样本行数

    Returns:
        估算的全表内存占用（MB）
    """
    sample = pd.read_csv(file_path, nrows=sample_size)
    per_row_bytes = sample.memory_usage(deep=True).sum() / len(sample)
    total_rows = count_rows(file_path)
    return per_row_bytes * total_rows / 1024 / 1024


def main() -> None:
    """主流程：逐张表打印基本信息。"""
    print("=" * 60)
    print("数据探查开始")
    print("=" * 60)

    # 先看字段说明文件
    desc_path = DATA_DIR + "\\" + COLUMNS_DESC_FILE
    desc_df = pd.read_csv(desc_path, encoding="latin-1")
    print(f"\n[字段说明文件] {COLUMNS_DESC_FILE}")
    print(f"共 {len(desc_df)} 条字段说明，涉及表：{desc_df['Table'].unique().tolist()}")
    print("前 3 条示例：")
    print(desc_df.head(3).to_string(index=False))

    # 逐张表探查
    summary_rows = []
    for table_name in TABLE_NAMES:
        file_path = DATA_DIR + "\\" + table_name
        print(f"\n--- 正在探查: {table_name} ---")

        # 读前几行看结构
        head_df = read_first_rows(file_path)
        print(f"列数: {head_df.shape[1]}, 列名: {head_df.columns.tolist()}")

        # 统计行数
        row_count = count_rows(file_path)
        print(f"行数: {row_count:,}")

        # 估算内存
        mem_mb = estimate_memory(file_path)
        print(f"估算全量读入内存: {mem_mb:.0f} MB")

        summary_rows.append(
            {
                "表名": table_name,
                "行数": row_count,
                "列数": head_df.shape[1],
                "估算内存(MB)": round(mem_mb),
            }
        )

    # 打印汇总表
    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 60)
    print("数据表总览")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print(f"\n全部表估算内存合计: {summary_df['估算内存(MB)'].sum():.0f} MB")


if __name__ == "__main__":
    main()
