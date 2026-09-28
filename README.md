# credit-dataset：个人消费信贷智能风险评分模型项目

本项目基于公开去标识化信贷数据集，完成零售信贷风控全链路建模：
数据治理 → 特征工程 → 基线模型（逻辑回归 + LightGBM）→ 风险分层策略。

当前进度：阶段一（业务调研与数据资产）、阶段二（特征工程与基线模型）、
阶段三（模型优化与风险分层策略）均已完成。

## 目录结构

按任务书四阶段组织，代码、文档、数据产物均与阶段一一对应。

```
credit-dataset/
├── data/
│   ├── raw/                        # 原始数据（7 张关联表 + 字段释义）
│   └── processed/                  # 清洗 / 特征工程中间产物（阶段二开始使用）
├── src/
│   └── credit_risk/                # 主包
│       ├── stage1_data_assets/     # 阶段一：业务调研与数据资产
│       │   ├── 01_data_exploration.py             # 数据探查：行数、列数、内存估算
│       │   ├── 02_data_quality_check.py           # 数据质量巡检：缺失率三级分级、异常值
│       │   ├── 03_field_dictionary_builder.py     # 字段字典：四维业务分类
│       │   ├── 04_time_window_validation.py       # 时间窗口校验：申请时点切片合规检查
│       │   ├── 05_build_master_table.py           # 宽表构建：6 张附表聚合回申请粒度
│       │   ├── 06_exploratory_data_analysis.py    # EDA：分布、分组违约率、交叉、相关性
│       │   ├── 07_feature_screening.py            # 核心字段单变量 + WOE/IV 排序
│       │   └── 08_supplementary_cross_analysis.py # 补充交叉分析：负债比例异常值
│       ├── stage2_features/        # 阶段二：特征工程 + 基线模型（脚本唯一来源）
│       │   ├── 01_feature_engineering.py       # 特征工程：缺失/异常/衍生/初筛
│       │   ├── 02_woe_binning.py               # WOE 分箱：自动 + 业务人工调整
│       │   ├── 03_loan_availability_check.py   # 贷前可得性校验（合规红线）
│       │   ├── 04_logistic_scorecard.py        # 逻辑回归评分卡：WOE→显著性→刻度
│       │   ├── 05_lgbm_baseline.py             # LightGBM 基线：对比 + 过拟合校验
│       │   └── 06_threshold_strategy.py        # 双阈值策略：通过率/坏账率测算
│       ├── stage3_strategy/        # 阶段三：模型优化与策略
│       │   ├── 01_model_tuning.py             # 超参调优：网格搜索 + 3 版本对比
│       │   ├── 02_class_imbalance.py          # 样本不均衡：权重/SMOTE/阈值优化
│       │   ├── 03_interpretability.py         # 可解释性：SHAP 全局/局部 + 合规校验
│       │   └── 04_risk_strategy.py            # 风险分层：三档策略 + 效果模拟
│       ├── stage4_value/           # 阶段四：价值评估（预留）
│       └── common/                 # 跨阶段公共工具（预留）
├── tests/                          # 测试目录（unit / integration）
├── scripts/                        # 运维脚本
├── configs/                        # 配置文件
├── output/
│   ├── stage1/                      # 阶段一输出物（CSV / 43 张图）
│   ├── stage2/                      # 阶段二输出物（CSV / 图表）
│   └── stage3/                      # 阶段三输出物（CSV / SHAP 图）
├── docs/
│   ├── stage1_调研与数据资产/       # 阶段一报告（4 份）
│   ├── stage2_特征工程与基线/       # 阶段二报告（1 份）
│   └── stage3_模型优化与策略/       # 阶段三报告（3 份）
├── pyproject.toml                  # 格式化 / lint 配置（Black 88 列等）
├── requirements.txt                # Python 依赖
└── README.md
```

## 运行方式

### 一键复现

首次运行需要先装依赖、放数据：

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 将原始数据表放入 data/raw/（application_train.csv 等 7 张关联表 + 字段释义）

# 3. 一键运行全部阶段（约 40~60 分钟）
python scripts/run_all.py
```

数据不随仓库分发：`data/` 已在 .gitignore 中排除，需自行从数据源下载放入。

脚本输出（CSV / 图表）默认写到项目内 `output\stage1`（相对项目根，脚本自动定位，不依赖绝对路径）。

```bash
# 1. 数据探查
python src/credit_risk/stage1_data_assets/01_data_exploration.py

# 2. 数据质量巡检（输出 data_quality_summary.csv）
python src/credit_risk/stage1_data_assets/02_data_quality_check.py

# 3. 字段字典（输出 field_dictionary.csv）
python src/credit_risk/stage1_data_assets/03_field_dictionary_builder.py

# 4. 时间窗口校验（输出 time_window_validation.csv）
python src/credit_risk/stage1_data_assets/04_time_window_validation.py

# 5. 宽表构建（输出 master_table.csv）
python src/credit_risk/stage1_data_assets/05_build_master_table.py

# 6. EDA 可视化（输出 22 张图）
python src/credit_risk/stage1_data_assets/06_exploratory_data_analysis.py

# 7. 核心字段单变量 + WOE/IV 排序（输出 iv_ranking.csv + 20 张图）
python src/credit_risk/stage1_data_assets/07_feature_screening.py

# 8. 补充交叉分析
python src/credit_risk/stage1_data_assets/08_supplementary_cross_analysis.py
```

### 阶段二（特征工程与基线模型，输出到 `output\stage2`）

```bash
# 1. 特征工程（输出 processed_features.csv）
python src/credit_risk/stage2_features/01_feature_engineering.py

# 2. WOE 分箱（输出 woe_bins.csv、woe_iv_summary.csv、woe_encoding_map.csv）
python src/credit_risk/stage2_features/02_woe_binning.py

# 3. 贷前可得性校验（输出 loan_availability_check.csv）
python src/credit_risk/stage2_features/03_loan_availability_check.py

# 4. 逻辑回归评分卡（输出 5 份 CSV）
python src/credit_risk/stage2_features/04_logistic_scorecard.py

# 5. LightGBM 基线对比 + 过拟合校验
python src/credit_risk/stage2_features/05_lgbm_baseline.py

# 6. 双阈值策略测算
python src/credit_risk/stage2_features/06_threshold_strategy.py
```

> 注意：01–06 需按顺序运行，04 依赖 02 的 WOE 映射，05/06 依赖 04 的输出。

### 阶段三（模型优化与风险分层策略，输出到 `output\stage3`）

```bash
# 1. 超参调优 + 3 版本对比
python src/credit_risk/stage3_strategy/01_model_tuning.py

# 2. 样本不均衡处理对比
python src/credit_risk/stage3_strategy/02_class_imbalance.py

# 3. 可解释性分析 SHAP + 合规校验（输出图与 CSV）
python src/credit_risk/stage3_strategy/03_interpretability.py

# 4. 风险分层与审批策略模拟
python src/credit_risk/stage3_strategy/04_risk_strategy.py
```

> 注意：03 需要 `shap` 库、02 需要 `imbalanced-learn` 库（已加入 requirements.txt）。

## 代码规范

遵循《编码规范》：PEP 8 + Black（行宽 88 列）、Google 风格 docstring、
类型注解、导入分组（标准库/第三方/本地）、src/ 项目结构。
格式化与检查：

```bash
python -m black --line-length 88 src/credit_risk
python -m isort --profile black --line-length 88 src/credit_risk
python -m flake8 --max-line-length=88 --extend-ignore=E203,W503 src/credit_risk
```
