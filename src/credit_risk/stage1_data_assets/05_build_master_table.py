"""宽表构建脚本：以申请主表为核心，把 6 张附表聚合回申请粒度，形成分析宽表。

每张表按 SK_ID_CURR聚合（bureau_balance 先经 SK_ID_BUREAU 映射回申请）
，聚合口径包括：征信账户数与状态、逾期月占比、历史申请次数与通过率、POS/信用卡逾期、
额度使用率、还款流水逾期次数与延迟天数等。附表分块读取、聚合完即释放，避免大表常驻
内存。

输出：master_table.csv（主表 122 列 + 附表聚合特征，共 174 列）
"""

import gc

import numpy as np
import pandas as pd
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
OUT_DIR = os.path.join(PROJECT_ROOT, "output", "stage1")

# 附表行数大，分块读取
CHUNK_SIZE = 200_000


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """把 DataFrame 里各列的类型往小了调，节省内存。

    Args:
        df: 原始 DataFrame

    Returns:
        类型优化后的 DataFrame
    """
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == "int64":
            col_min = df[col].min()
            col_max = df[col].max()
            if col_min >= -128 and col_max <= 127:
                df[col] = df[col].astype("int8")
            elif col_min >= -32768 and col_max <= 32767:
                df[col] = df[col].astype("int16")
            elif col_min >= -2147483648 and col_max <= 2147483647:
                df[col] = df[col].astype("int32")
        elif col_type == "float64":
            df[col] = df[col].astype("float32")
    return df


def load_main_table() -> pd.DataFrame:
    """读取主表并做类型优化。

    Returns:
        类型优化后的 application_train DataFrame
    """
    train_df = pd.read_csv(DATA_DIR + "\\application_train.csv")
    train_df = optimize_dtypes(train_df)
    return train_df


def aggregate_bureau() -> pd.DataFrame:
    """聚合外部征信表 bureau 到申请粒度。

    Returns:
        以 SK_ID_CURR 为索引的聚合特征 DataFrame
    """
    print("  读取 bureau ...")
    usecols = [
        "SK_ID_CURR",
        "CREDIT_ACTIVE",
        "DAYS_CREDIT",
        "DAYS_CREDIT_ENDDATE",
        "AMT_CREDIT_SUM",
        "AMT_CREDIT_SUM_DEBT",
        "AMT_CREDIT_SUM_OVERDUE",
        "CREDIT_DAY_OVERDUE",
    ]
    bureau = pd.read_csv(DATA_DIR + "\\bureau.csv", usecols=usecols)
    bureau = optimize_dtypes(bureau)

    # 各状态账户计数（活跃/结清/出售/坏账）
    status_counts = (
        pd.get_dummies(bureau["CREDIT_ACTIVE"], prefix="bureau")
        .groupby(bureau["SK_ID_CURR"])
        .sum()
    )
    # 存续期账户（结束日在申请日后，正值）
    bureau["enddate_positive"] = (bureau["DAYS_CREDIT_ENDDATE"] > 0).astype("int8")

    agg = (
        bureau.groupby("SK_ID_CURR")
        .agg(
            bureau_count=("CREDIT_ACTIVE", "size"),
            bureau_days_credit_mean=("DAYS_CREDIT", "mean"),
            bureau_days_credit_min=("DAYS_CREDIT", "min"),
            bureau_credit_sum=("AMT_CREDIT_SUM", "sum"),
            bureau_credit_mean=("AMT_CREDIT_SUM", "mean"),
            bureau_debt_sum=("AMT_CREDIT_SUM_DEBT", "sum"),
            bureau_debt_mean=("AMT_CREDIT_SUM_DEBT", "mean"),
            bureau_overdue_sum=("AMT_CREDIT_SUM_OVERDUE", "sum"),
            bureau_overdue_max=("AMT_CREDIT_SUM_OVERDUE", "max"),
            bureau_credit_day_overdue_max=("CREDIT_DAY_OVERDUE", "max"),
            bureau_enddate_positive_sum=("enddate_positive", "sum"),
        )
        .join(status_counts)
        .reset_index()
    )
    del bureau, status_counts
    gc.collect()
    return agg


def aggregate_bureau_balance() -> pd.DataFrame:
    """聚合征信月度结余表 bureau_balance 到申请粒度。

    该表没有 SK_ID_CURR，需要先经 bureau 的 SK_ID_BUREAU 映射回申请。

    Returns:
        以 SK_ID_CURR 为索引的聚合特征 DataFrame
    """
    print("  读取 bureau 映射（SK_ID_BUREAU -> SK_ID_CURR）...")
    bureau_map = pd.read_csv(
        DATA_DIR + "\\bureau.csv", usecols=["SK_ID_BUREAU", "SK_ID_CURR"]
    )
    bureau_map = optimize_dtypes(bureau_map)
    id_map = bureau_map.set_index("SK_ID_BUREAU")["SK_ID_CURR"].to_dict()

    print("  分块读取 bureau_balance ...")
    usecols = ["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"]
    agg_parts = []
    for chunk in pd.read_csv(
        DATA_DIR + "\\bureau_balance.csv",
        usecols=usecols,
        chunksize=CHUNK_SIZE,
    ):
        chunk["SK_ID_CURR"] = chunk["SK_ID_BUREAU"].map(id_map)
        chunk = chunk.dropna(subset=["SK_ID_CURR"])
        chunk["SK_ID_CURR"] = chunk["SK_ID_CURR"].astype("int32")
        # 逾期月标记：STATUS 为 1~5 视为逾期，X 为无欠款、C 为结清
        chunk["overdue_flag"] = (
            chunk["STATUS"].isin(["1", "2", "3", "4", "5"]).astype("int8")
        )
        chunk["no_debt_flag"] = (chunk["STATUS"] == "X").astype("int8")
        part = (
            chunk.groupby("SK_ID_CURR")
            .agg(
                bb_months_count=("STATUS", "size"),
                bb_overdue_months=("overdue_flag", "sum"),
                bb_no_debt_months=("no_debt_flag", "sum"),
                bb_balance_min=("MONTHS_BALANCE", "min"),
                bb_balance_mean=("MONTHS_BALANCE", "mean"),
            )
            .reset_index()
        )
        agg_parts.append(part)
        del chunk, part
        gc.collect()

    agg = pd.concat(agg_parts).groupby("SK_ID_CURR").sum(numeric_only=True)
    agg = agg.reset_index()
    # 逾期月占比 = 逾期月数 / 总月数
    agg["bb_overdue_ratio"] = agg["bb_overdue_months"] / agg["bb_months_count"]
    del bureau_map, id_map, agg_parts
    gc.collect()
    return agg


def aggregate_previous() -> pd.DataFrame:
    """聚合历史申请表 previous_application 到申请粒度。

    Returns:
        以 SK_ID_CURR 为索引的聚合特征 DataFrame
    """
    print("  读取 previous_application ...")
    usecols = [
        "SK_ID_CURR",
        "NAME_CONTRACT_STATUS",
        "AMT_APPLICATION",
        "DAYS_DECISION",
        "DAYS_FIRST_DRAWING",
    ]
    prev = pd.read_csv(DATA_DIR + "\\previous_application.csv", usecols=usecols)
    prev = optimize_dtypes(prev)

    # 审批状态计数（通过 / 拒绝 / 取消等）
    status_counts = (
        pd.get_dummies(prev["NAME_CONTRACT_STATUS"], prefix="prev")
        .groupby(prev["SK_ID_CURR"])
        .sum()
    )
    # 放款日在申请日之后（申请后仍有放款动作）计数
    prev["first_drawing_positive"] = (prev["DAYS_FIRST_DRAWING"] > 0).astype("int8")

    agg = (
        prev.groupby("SK_ID_CURR")
        .agg(
            prev_count=("NAME_CONTRACT_STATUS", "size"),
            prev_amt_application_sum=("AMT_APPLICATION", "sum"),
            prev_amt_application_mean=("AMT_APPLICATION", "mean"),
            prev_days_decision_mean=("DAYS_DECISION", "mean"),
            prev_first_drawing_positive_sum=("first_drawing_positive", "sum"),
        )
        .join(status_counts)
        .reset_index()
    )
    # 通过率 = 已通过数 / 申请数（列名 prev_Approved 由 get_dummies 生成）
    approved_cols = [c for c in agg.columns if c.startswith("prev_Approved")]
    if approved_cols:
        agg["prev_approved_ratio"] = agg[approved_cols[0]] / agg["prev_count"]
    del prev, status_counts
    gc.collect()
    return agg


def aggregate_pos() -> pd.DataFrame:
    """聚合消费现金贷月度表 POS_CASH_balance 到申请粒度。

    Returns:
        以 SK_ID_CURR 为索引的聚合特征 DataFrame
    """
    print("  分块读取 POS_CASH_balance ...")
    usecols = ["SK_ID_CURR", "SK_DPD", "SK_DPD_DEF"]
    agg_parts = []
    for chunk in pd.read_csv(
        DATA_DIR + "\\POS_CASH_balance.csv", usecols=usecols, chunksize=CHUNK_SIZE
    ):
        chunk = optimize_dtypes(chunk)
        chunk["overdue_flag"] = (chunk["SK_DPD"] > 0).astype("int8")
        part = (
            chunk.groupby("SK_ID_CURR")
            .agg(
                pos_months_count=("SK_DPD", "size"),
                pos_dpd_max=("SK_DPD", "max"),
                pos_dpd_mean=("SK_DPD", "mean"),
                pos_dpd_def_max=("SK_DPD_DEF", "max"),
                pos_overdue_months=("overdue_flag", "sum"),
            )
            .reset_index()
        )
        agg_parts.append(part)
        del chunk, part
        gc.collect()

    agg = (
        pd.concat(agg_parts)
        .groupby("SK_ID_CURR")
        .agg(
            pos_months_count=("pos_months_count", "sum"),
            pos_dpd_max=("pos_dpd_max", "max"),
            pos_dpd_mean=("pos_dpd_mean", "mean"),
            pos_dpd_def_max=("pos_dpd_def_max", "max"),
            pos_overdue_months=("pos_overdue_months", "sum"),
        )
    )
    agg = agg.reset_index()
    agg["pos_overdue_ratio"] = agg["pos_overdue_months"] / agg["pos_months_count"]
    del agg_parts
    gc.collect()
    return agg


def aggregate_credit_card() -> pd.DataFrame:
    """聚合信用卡月度表 credit_card_balance 到申请粒度。

    Returns:
        以 SK_ID_CURR 为索引的聚合特征 DataFrame
    """
    print("  分块读取 credit_card_balance ...")
    usecols = [
        "SK_ID_CURR",
        "AMT_BALANCE",
        "AMT_CREDIT_LIMIT_ACTUAL",
        "AMT_DRAWINGS_CURRENT",
        "SK_DPD",
    ]
    agg_parts = []
    for chunk in pd.read_csv(
        DATA_DIR + "\\credit_card_balance.csv",
        usecols=usecols,
        chunksize=CHUNK_SIZE,
    ):
        chunk = optimize_dtypes(chunk)
        # 额度使用率 = 当前余额 / 授信额度（额度为 0 时置 NaN 避免除零）
        chunk["utilization"] = np.where(
            chunk["AMT_CREDIT_LIMIT_ACTUAL"] > 0,
            chunk["AMT_BALANCE"] / chunk["AMT_CREDIT_LIMIT_ACTUAL"],
            np.nan,
        )
        chunk["overdue_flag"] = (chunk["SK_DPD"] > 0).astype("int8")
        part = (
            chunk.groupby("SK_ID_CURR")
            .agg(
                cc_months_count=("SK_DPD", "size"),
                cc_balance_mean=("AMT_BALANCE", "mean"),
                cc_balance_max=("AMT_BALANCE", "max"),
                cc_limit_mean=("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
                cc_drawings_mean=("AMT_DRAWINGS_CURRENT", "mean"),
                cc_utilization_mean=("utilization", "mean"),
                cc_dpd_max=("SK_DPD", "max"),
                cc_overdue_months=("overdue_flag", "sum"),
            )
            .reset_index()
        )
        agg_parts.append(part)
        del chunk, part
        gc.collect()

    agg = (
        pd.concat(agg_parts)
        .groupby("SK_ID_CURR")
        .agg(
            cc_months_count=("cc_months_count", "sum"),
            cc_balance_mean=("cc_balance_mean", "mean"),
            cc_balance_max=("cc_balance_max", "max"),
            cc_limit_mean=("cc_limit_mean", "mean"),
            cc_drawings_mean=("cc_drawings_mean", "mean"),
            cc_utilization_mean=("cc_utilization_mean", "mean"),
            cc_dpd_max=("cc_dpd_max", "max"),
            cc_overdue_months=("cc_overdue_months", "sum"),
        )
    )
    agg = agg.reset_index()
    agg["cc_overdue_ratio"] = agg["cc_overdue_months"] / agg["cc_months_count"]
    del agg_parts
    gc.collect()
    return agg


def aggregate_installments() -> pd.DataFrame:
    """聚合分期还款流水表 installments_payments 到申请粒度。

    Returns:
        以 SK_ID_CURR 为索引的聚合特征 DataFrame
    """
    print("  分块读取 installments_payments ...")
    usecols = [
        "SK_ID_CURR",
        "DAYS_INSTALMENT",
        "DAYS_ENTRY_PAYMENT",
        "AMT_INSTALMENT",
        "AMT_PAYMENT",
    ]
    agg_parts = []
    for chunk in pd.read_csv(
        DATA_DIR + "\\installments_payments.csv",
        usecols=usecols,
        chunksize=CHUNK_SIZE,
    ):
        chunk = optimize_dtypes(chunk)
        # 延迟天数：实还日 - 应还日
        chunk["delay_days"] = chunk["DAYS_ENTRY_PAYMENT"] - chunk["DAYS_INSTALMENT"]
        chunk["overdue_flag"] = (chunk["delay_days"] > 0).astype("int8")
        part = (
            chunk.groupby("SK_ID_CURR")
            .agg(
                inst_count=("DAYS_INSTALMENT", "size"),
                inst_overdue_count=("overdue_flag", "sum"),
                inst_delay_days_mean=("delay_days", "mean"),
                inst_payment_sum=("AMT_PAYMENT", "sum"),
                inst_amt_mean=("AMT_INSTALMENT", "mean"),
            )
            .reset_index()
        )
        agg_parts.append(part)
        del chunk, part
        gc.collect()

    agg = (
        pd.concat(agg_parts)
        .groupby("SK_ID_CURR")
        .agg(
            inst_count=("inst_count", "sum"),
            inst_overdue_count=("inst_overdue_count", "sum"),
            inst_delay_days_mean=("inst_delay_days_mean", "mean"),
            inst_payment_sum=("inst_payment_sum", "sum"),
            inst_amt_mean=("inst_amt_mean", "mean"),
        )
    )
    agg = agg.reset_index()
    agg["inst_overdue_ratio"] = agg["inst_overdue_count"] / agg["inst_count"]
    del agg_parts
    gc.collect()
    return agg


def main() -> None:
    """主流程：逐表聚合并合并进宽表，输出 master_table.csv。"""
    print("=" * 60)
    print("宽表构建开始（全表关联整合）")
    print("=" * 60)

    master = load_main_table()
    print(f"主表: {master.shape}")

    # 逐张附表聚合，结果合并进宽表，释放中间对象
    aggregators = [
        ("bureau", aggregate_bureau),
        ("bureau_balance", aggregate_bureau_balance),
        ("previous_application", aggregate_previous),
        ("POS_CASH_balance", aggregate_pos),
        ("credit_card_balance", aggregate_credit_card),
        ("installments_payments", aggregate_installments),
    ]
    for table_name, agg_func in aggregators:
        print(f"\n[{table_name}] 聚合中 ...")
        agg_df = agg_func()
        master = master.merge(agg_df, on="SK_ID_CURR", how="left")
        del agg_df
        gc.collect()
        print(f"  宽表当前维度: {master.shape}")

    # 宽表列名去重
    assert master.columns.is_unique, "宽表存在重复列名，需要检查"

    out_file = OUT_DIR + "\\master_table.csv"
    print(f"\n保存宽表: {out_file} ...")
    master.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"宽表已保存: {master.shape[0]} 行 x {master.shape[1]} 列")

    # 统计附表特征列数
    feature_cols = [
        c
        for c in master.columns
        if c.startswith(("bureau", "bb_", "prev", "pos_", "cc_", "inst_"))
    ]
    print(f"附表聚合特征列数: {len(feature_cols)}")


if __name__ == "__main__":
    main()
