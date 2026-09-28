"""字段字典生成脚本：把所有表的核心字段整理成一份标准业务字典。

合并字段释义文件（credit_columns_description.csv）与数据质量巡检结果，
输出每字段的所属表、业务释义、数据类型、缺失率、四维业务分类（人口属性/
信用历史/还款行为/交易行为），并统计各表字段数量与待分类字段。
"""

import pandas as pd
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
# 质量巡检结果由 data_quality_check.py 生成，路径拆成两段避免单行超长
QUALITY_FILE = (
    os.path.join(PROJECT_ROOT, "output", "stage1")
    + "\\data_quality_summary.csv"
)
OUT_DIR = os.path.join(PROJECT_ROOT, "output", "stage1")

# 四维业务分类：把字段按业务含义分到四个维度。
# 因为字段比较多，写成分组字典的形式，方便检查和维护。
# 人口属性：客户是谁、年龄性别婚姻职业收入、居住和资产情况
DEMOGRAPHIC_FIELDS = [
    # 主表
    "SK_ID_CURR",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "CNT_CHILDREN",
    "NAME_TYPE_SUITE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "REGION_POPULATION_RELATIVE",
    "DAYS_BIRTH",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "OWN_CAR_AGE",
    "OCCUPATION_TYPE",
    "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",
    "REG_REGION_NOT_LIVE_REGION",
    "REG_REGION_NOT_WORK_REGION",
    "LIVE_REGION_NOT_WORK_REGION",
    "REG_CITY_NOT_LIVE_CITY",
    "REG_CITY_NOT_WORK_CITY",
    "LIVE_CITY_NOT_WORK_CITY",
    "ORGANIZATION_TYPE",
    # 房产与居住类字段（_AVG/_MODE/_MEDI 系列）
    "APARTMENTS_AVG",
    "BASEMENTAREA_AVG",
    "YEARS_BEGINEXPLUATATION_AVG",
    "YEARS_BUILD_AVG",
    "COMMONAREA_AVG",
    "ELEVATORS_AVG",
    "ENTRANCES_AVG",
    "FLOORSMAX_AVG",
    "FLOORSMIN_AVG",
    "LANDAREA_AVG",
    "LIVINGAPARTMENTS_AVG",
    "LIVINGAREA_AVG",
    "NONLIVINGAPARTMENTS_AVG",
    "NONLIVINGAREA_AVG",
    "APARTMENTS_MODE",
    "BASEMENTAREA_MODE",
    "YEARS_BEGINEXPLUATATION_MODE",
    "YEARS_BUILD_MODE",
    "COMMONAREA_MODE",
    "ELEVATORS_MODE",
    "ENTRANCES_MODE",
    "FLOORSMAX_MODE",
    "FLOORSMIN_MODE",
    "LANDAREA_MODE",
    "LIVINGAPARTMENTS_MODE",
    "LIVINGAREA_MODE",
    "NONLIVINGAPARTMENTS_MODE",
    "NONLIVINGAREA_MODE",
    "APARTMENTS_MEDI",
    "BASEMENTAREA_MEDI",
    "YEARS_BEGINEXPLUATATION_MEDI",
    "YEARS_BUILD_MEDI",
    "COMMONAREA_MEDI",
    "ELEVATORS_MEDI",
    "ENTRANCES_MEDI",
    "FLOORSMAX_MEDI",
    "FLOORSMIN_MEDI",
    "LANDAREA_MEDI",
    "LIVINGAPARTMENTS_MEDI",
    "LIVINGAREA_MEDI",
    "NONLIVINGAPARTMENTS_MEDI",
    "NONLIVINGAREA_MEDI",
    "FONDKAPREMONT_MODE",
    "HOUSETYPE_MODE",
    "TOTALAREA_MODE",
    "WALLSMATERIAL_MODE",
    "EMERGENCYSTATE_MODE",
    "AMT_INCOME_TOTAL",
]

# 信用历史：外部征信评分、征信查询次数
CREDIT_HISTORY_FIELDS = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AMT_REQ_CREDIT_BUREAU_HOUR",
    "AMT_REQ_CREDIT_BUREAU_DAY",
    "AMT_REQ_CREDIT_BUREAU_WEEK",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
    # bureau 表：外部机构征信
    "SK_ID_BUREAU",
    "CREDIT_ACTIVE",
    "CREDIT_CURRENCY",
    "DAYS_CREDIT",
    "CREDIT_DAY_OVERDUE",
    "DAYS_CREDIT_ENDDATE",
    "DAYS_ENDDATE_FACT",
    "AMT_CREDIT_MAX_OVERDUE",
    "CNT_CREDIT_PROLONG",
    "AMT_CREDIT_SUM",
    "AMT_CREDIT_SUM_DEBT",
    "AMT_CREDIT_SUM_LIMIT",
    "AMT_CREDIT_SUM_OVERDUE",
    "CREDIT_TYPE",
    "DAYS_CREDIT_UPDATE",
    "AMT_ANNUITY",
    # bureau_balance 表：征信账户月度结余
    "MONTHS_BALANCE",
    "STATUS",
]

# 还款行为：还款能力、逾期情况、贷款负担
REPAYMENT_FIELDS = [
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_EMPLOYED",
    "DAYS_LAST_PHONE_CHANGE",
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "OBS_60_CNT_SOCIAL_CIRCLE",
    "DEF_60_CNT_SOCIAL_CIRCLE",
    # POS_CASH_balance 表：消费现金贷月度账户
    "SK_ID_PREV",
    "CNT_INSTALMENT",
    "CNT_INSTALMENT_FUTURE",
    "NAME_CONTRACT_STATUS",
    "SK_DPD",
    "SK_DPD_DEF",
    # credit_card_balance 表：信用卡月度账户
    "AMT_BALANCE",
    "AMT_CREDIT_LIMIT_ACTUAL",
    "AMT_DRAWINGS_ATM_CURRENT",
    "AMT_DRAWINGS_CURRENT",
    "AMT_DRAWINGS_OTHER_CURRENT",
    "AMT_DRAWINGS_POS_CURRENT",
    "AMT_INST_MIN_REGULARITY",
    "AMT_PAYMENT_CURRENT",
    "AMT_PAYMENT_TOTAL_CURRENT",
    "AMT_RECEIVABLE_PRINCIPAL",
    "AMT_RECIVABLE",
    "AMT_TOTAL_RECEIVABLE",
    "CNT_DRAWINGS_ATM_CURRENT",
    "CNT_DRAWINGS_CURRENT",
    "CNT_DRAWINGS_OTHER_CURRENT",
    "CNT_DRAWINGS_POS_CURRENT",
    "CNT_INSTALMENT_MATURE_CUM",
    # installments_payments 表：分期还款流水
    "NUM_INSTALMENT_VERSION",
    "NUM_INSTALMENT_NUMBER",
    "DAYS_INSTALMENT",
    "DAYS_ENTRY_PAYMENT",
    "AMT_INSTALMENT",
    "AMT_PAYMENT",
]

# 交易行为：申请渠道、申请时间、联系方式、材料提交等申请行为
TRANSACTION_FIELDS = [
    "NAME_CONTRACT_TYPE",
    "WEEKDAY_APPR_PROCESS_START",
    "HOUR_APPR_PROCESS_START",
    "FLAG_MOBIL",
    "FLAG_EMP_PHONE",
    "FLAG_WORK_PHONE",
    "FLAG_CONT_MOBILE",
    "FLAG_PHONE",
    "FLAG_EMAIL",
    "FLAG_DOCUMENT_2",
    "FLAG_DOCUMENT_3",
    "FLAG_DOCUMENT_4",
    "FLAG_DOCUMENT_5",
    "FLAG_DOCUMENT_6",
    "FLAG_DOCUMENT_7",
    "FLAG_DOCUMENT_8",
    "FLAG_DOCUMENT_9",
    "FLAG_DOCUMENT_10",
    "FLAG_DOCUMENT_11",
    "FLAG_DOCUMENT_12",
    "FLAG_DOCUMENT_13",
    "FLAG_DOCUMENT_14",
    "FLAG_DOCUMENT_15",
    "FLAG_DOCUMENT_16",
    "FLAG_DOCUMENT_17",
    "FLAG_DOCUMENT_18",
    "FLAG_DOCUMENT_19",
    "FLAG_DOCUMENT_20",
    "FLAG_DOCUMENT_21",
    # previous_application 表：历史申请行为
    "AMT_APPLICATION",
    "AMT_DOWN_PAYMENT",
    "NFLAG_LAST_APPL_PER_CONTRACT",
    "FLAG_LAST_APPL_PER_CONTRACT",
    "NFLAG_LAST_APPL_IN_DAY",
    "RATE_DOWN_PAYMENT",
    "RATE_INTEREST_PRIMARY",
    "RATE_INTEREST_PRIVILEGED",
    "NAME_CASH_LOAN_PURPOSE",
    "NAME_CONTRACT_STATUS",
    "DAYS_DECISION",
    "NAME_PAYMENT_TYPE",
    "CODE_REJECT_REASON",
    "NAME_TYPE_SUITE",
    "NAME_CLIENT_TYPE",
    "NAME_GOODS_CATEGORY",
    "NAME_PORTFOLIO",
    "NAME_PRODUCT_TYPE",
    "CHANNEL_TYPE",
    "SELLERPLACE_AREA",
    "NAME_SELLER_INDUSTRY",
    "CNT_PAYMENT",
    "NAME_YIELD_GROUP",
    "PRODUCT_COMBINATION",
    "DAYS_FIRST_DRAWING",
    "DAYS_FIRST_DUE",
    "DAYS_LAST_DUE_1ST_VERSION",
    "DAYS_LAST_DUE",
    "DAYS_TERMINATION",
    "NFLAG_INSURED_ON_APPROVAL",
]

# 特殊标记字段：说明文件里 Special 列有内容的字段（比如 DAYS_EMPLOYED=365243 表示失业）
SPECIAL_FIELDS = [
    "DAYS_EMPLOYED",
    "DAYS_BIRTH",
    "DAYS_ID_PUBLISH",
    "DAYS_REGISTRATION",
    "DAYS_LAST_PHONE_CHANGE",
    "DAYS_CREDIT",
    "DAYS_CREDIT_ENDDATE",
    "DAYS_ENDDATE_FACT",
    "DAYS_CREDIT_UPDATE",
    "DAYS_INSTALMENT",
    "DAYS_ENTRY_PAYMENT",
    "DAYS_DECISION",
    "DAYS_FIRST_DRAWING",
    "DAYS_FIRST_DUE",
    "DAYS_LAST_DUE",
    "DAYS_LAST_DUE_1ST_VERSION",
    "DAYS_TERMINATION",
    "MONTHS_BALANCE",
    "STATUS",
]


def build_field_group_map() -> dict:
    """把四个分组的字段列表合并成 {字段名: 分类} 的映射字典。

    Returns:
        字段名到四维分类的映射
    """
    field_group_map = {}
    for field in DEMOGRAPHIC_FIELDS:
        field_group_map[field] = "人口属性"
    for field in CREDIT_HISTORY_FIELDS:
        field_group_map[field] = "信用历史"
    for field in REPAYMENT_FIELDS:
        field_group_map[field] = "还款行为"
    for field in TRANSACTION_FIELDS:
        field_group_map[field] = "交易行为"
    return field_group_map


def main() -> None:
    """主流程：合并字段说明和质量结果，输出字段字典 CSV。"""
    print("=" * 60)
    print("字段字典生成开始")
    print("=" * 60)

    # 1. 读取字段说明文件
    desc_path = DATA_DIR + "\\credit_columns_description.csv"
    desc_df = pd.read_csv(desc_path, encoding="latin-1")
    # 表名规范化
    desc_df["规范化表名"] = desc_df["Table"].str.replace(
        "{train|test}", "train", regex=False
    )
    desc_df = desc_df.drop_duplicates(subset=["规范化表名", "Row"], keep="first")

    # 2. 读取数据质量巡检结果
    quality_df = pd.read_csv(QUALITY_FILE)

    # 3. 构建字段分组映射
    field_group_map = build_field_group_map()

    # 4. 合并时以质量结果为基准，再加上业务释义和四维分类
    merged = quality_df.merge(
        desc_df[["规范化表名", "Row", "Description", "Special"]],
        left_on=["表名", "字段名"],
        right_on=["规范化表名", "Row"],
        how="left",
    )

    # 5. 做好四维分类
    field_group_map["TARGET"] = "目标变量"
    merged["业务分类"] = merged["字段名"].map(field_group_map).fillna("待分类")

    # 6. 特殊取值说明
    merged["特殊取值提示"] = merged["Special"].fillna("")

    # 7. 整理输出列顺序
    output = merged[
        [
            "表名",
            "字段名",
            "业务分类",
            "数据类型",
            "缺失数量",
            "缺失率",
            "缺失等级",
            "唯一值数量",
            "唯一值占比",
            "Description",
            "特殊取值提示",
        ]
    ].rename(columns={"Description": "业务释义"})

    # 8. 保存
    out_file = OUT_DIR + "\\field_dictionary.csv"
    output.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"字段字典已保存: {out_file}")

    # 9. 打印统计信息


if __name__ == "__main__":
    main()
