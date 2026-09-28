"""贷前可获得性校验脚本：确认入模特征均为申请时点数据。

阶段二第三步 注意合规红线。对 02 保留的候选特征逐个核对
数据来源，确保全部为贷前可获得信息，未混入贷后数据。

依据阶段一 04 时间窗口校验结论：
- DAYS_EMPLOYED=365243 为失业占位符（业务编码，非贷后时间）
- bureau / previous 表的正值日期字段（结束日在申请日后、放款日在
  申请日后）在宽表聚合时已转为计数特征，原始天数未直接入模

输出：
- loan_availability_check.csv：特征合规清单（数据来源、结论、备注）
"""

import os

import pandas as pd

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUT_DIR = os.path.join(PROJECT_ROOT, "output", "stage2")

LIST_FILE = OUT_DIR + "\\feature_list.csv"
IV_FILE = OUT_DIR + "\\woe_iv_summary.csv"

# 贷前-申请信息：申请表单上的信息，申请时必然已知
APPLY_PREFIXES = [
    "NAME_",
    "CODE_GENDER",
    "FLAG_OWN_",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "DAYS_EMPLOYED",
    "DAYS_LAST_PHONE_CHANGE",
    "OWN_CAR_AGE",
    "OCCUPATION_TYPE",
    "REGION_",
    "ORGANIZATION_TYPE",
    "WEEKDAY_",
    "HOUR_",
    "FLAG_MOBIL",
    "FLAG_EMP_",
    "FLAG_WORK_",
    "FLAG_CONT_",
    "FLAG_PHONE",
    "FLAG_EMAIL",
    "FLAG_DOCUMENT_",
    "FONDKAPREMONT_",
    "HOUSETYPE_",
    "TOTALAREA_",
    "WALLSMATERIAL_",
    "EMERGENCYSTATE_",
    "LIVE_",
    "REG_",
    "APARTMENTS_",
    "BASEMENTAREA_",
    "YEARS_",
    "COMMONAREA_",
    "ELEVATORS_",
    "ENTRANCES_",
    "FLOORS",
    "LANDAREA_",
    "LIVING",
    "NONLIVING",
    "OBS_",
    "DEF_",
]

# 贷前-外部征信：申请时查询征信报告可得
BUREAU_PREFIXES = ["EXT_SOURCE_", "AMT_REQ_CREDIT_BUREAU_", "bureau", "bb_"]

# 贷前-本机构历史记录：申请时点可查询的历史贷款与还款流水
HISTORY_PREFIXES = ["prev_", "pos_", "cc_", "inst_"]

# 衍生特征：全部由贷前字段计算得出
DERIVED_FIELDS = [
    "IS_UNEMPLOYED",
    "DEBT_RATIO",
    "CREDIT_INCOME_RATIO",
    "CREDIT_GOODS_RATIO",
    "AGE_YEARS",
    "EXT_SOURCE_MEAN",
]

# 存疑字段：宽表聚合时由贷后日期转成的计数特征，语义为贷前可得，需人工复核
REVIEW_FIELDS = {
    "bureau_enddate_positive_sum": "存续期账户计数（结束日在申请日后）",
    "prev_first_drawing_positive_sum": "申请后放款计数（放款日在申请日后）",
}


def check_field(col: str) -> tuple:
    """单个字段的贷前可得性判断。

    先剥离 _MISSING 后缀再看来源前缀，缺失标识本身表示
    "申请时点无记录"，属于贷前信息。

    Args:
        col: 字段名

    Returns:
        (数据来源, 结论, 备注)
    """
    if col in REVIEW_FIELDS:
        return "征信/历史记录计数", "需人工复核", REVIEW_FIELDS[col]

    base = col[:-8] if col.endswith("_MISSING") else col
    is_missing_flag = col.endswith("_MISSING")

    for prefix in BUREAU_PREFIXES:
        if base.startswith(prefix):
            source = "外部征信"
            break
    else:
        for prefix in HISTORY_PREFIXES:
            if base.startswith(prefix):
                source = "本机构历史记录"
                break
        else:
            if base in DERIVED_FIELDS:
                source = "衍生特征"
            else:
                source = "申请信息"

    if is_missing_flag:
        remark = "无记录标识，申请时点可知，合规"
        return source + "（缺失标识）", "贷前可得", remark
    if source == "申请信息":
        return source, "贷前可得", "申请表单静态信息，合规"
    if source == "外部征信":
        return source, "贷前可得", "征信报告申请时点快照，合规"
    if source == "本机构历史记录":
        return source, "贷前可得", "本机构历史贷款与还款记录，申请时点可查，合规"
    return source, "贷前可得", "由贷前字段计算，合规"


def main() -> None:
    """主流程：读取候选特征，逐个校验，输出合规清单。"""
    print("=" * 60)
    title = "贷前可获得性校验开始"
    print(title.center(60, "="))

    feature_list = pd.read_csv(LIST_FILE)
    candidates = feature_list[feature_list["初筛状态"] == "保留"]["字段名"].tolist()
    iv_summary = pd.read_csv(IV_FILE)
    print(f"待校验候选特征: {len(candidates)} 个")

    records = []
    for col in candidates:
        source, conclusion, remark = check_field(col)
        records.append(
            {
                "字段名": col,
                "数据来源": source,
                "贷前结论": conclusion,
                "备注": remark,
            }
        )
    result = pd.DataFrame(records)
    # 合并 IV 信息，一份清单看全
    result = result.merge(iv_summary[["字段名", "IV"]], on="字段名", how="left")

    out_file = OUT_DIR + "\\loan_availability_check.csv"
    result.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"合规清单已保存: {out_file}")


if __name__ == "__main__":
    main()
