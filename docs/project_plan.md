# 项目计划：贷前准入策略导向

## Phase 0: 业务定位

- 明确业务问题：在贷款申请时，识别未来还款困难概率较高的客户。
- 明确不是“简单预测 TARGET”，而是支持贷前准入策略：
  - 低风险客户自动通过。
  - 中风险客户进入人工复核/补充材料/降额。
  - 高风险客户拒绝或强风控处理。
- 评估指标不仅包括 AUC/KS，还包括：
  - 自动通过率。
  - 人工复核占比。
  - 高风险拒绝组坏样本捕获率。
  - 误拒好客户比例。
  - 分群稳定性与策略可解释性。

## Phase 1: 数据盘点与数据资产理解

- 检查每张表的行数、列数、主键、外键、缺失率、重复记录。
- 梳理 `SK_ID_CURR`、`SK_ID_BUREAU`、`SK_ID_PREV` 的关系。
- 识别每张表的业务含义和可用性：
  - `application_train/test`: 本次贷款申请主表。
  - `bureau`: 外部征信历史贷款。
  - `bureau_balance`: 外部征信月度状态。
  - `previous_application`: Home Credit 历史申请。
  - `installments_payments`: 历史分期还款表现。
  - `credit_card_balance`: 历史信用卡月度余额。
  - `POS_CASH_balance`: 历史 POS / cash loan 月度余额。
- 输出 `outputs/tables/table_profile.csv`。

## Phase 2: 防泄露与口径

- 定义申请时点、历史观察窗口、表现窗口。
- 标记审批前可见字段、审批后字段、可能泄露字段。
- 特别检查：
  - `DAYS_*` 字段是否均相对申请日。
  - 还款表现字段是否只来自历史贷款，而不是当前申请未来结果。
  - test/train 的字段差异。
  - `TARGET` 仅用于训练标签，不能进入特征。
- 输出 `docs/leakage_audit.md` 和 `data/processed/data_dictionary.csv`。

## Phase 3: 特征集市

- 从 bureau、previous application、installments、credit card、POS cash 聚合申请级历史特征。
- 系统生成候选特征，而不是只挑少量看起来业务合理的变量。
- 保留每个特征的来源表、时间窗口、业务解释和编码方式。
- 增强 6/12/24 个月近期窗口、active/closed、approved/refused、cash/revolving 等分组聚合特征。
- 特征组按业务含义拆分：
  - 申请人基本信息与当前申请条件。
  - 外部征信历史负债与逾期。
  - Home Credit 历史申请结果。
  - 历史还款纪律与提前/逾期还款。
  - 额度使用与债务压力。
  - 缺失模式与信息完整度。
- 输出 `data/processed/application_feature_store.parquet`。

## Phase 3.5: 候选特征筛选

- 类别变量生成频率编码和 top 类别 one-hot。
- Target-blind 筛选：
  - 缺失率。
  - 唯一值/常数变量。
  - `application_train` 内部 development / validation PSI 稳定性。
  - `final_holdout` 和官方 test 不参与筛选，避免 test peeking。
- Target-aware 筛选：
  - AUC power。
  - KS。
  - IV。
- 对近期窗口、分状态、分产品变量采用更宽松的单变量筛选口径，避免过早删除有交互价值的业务变量。
- 对通过筛选的变量做相关性去重，避免堆叠同义变量。
- 输出 `data/processed/candidate_feature_matrix.parquet`、`outputs/tables/feature_screening_report.csv`、`outputs/tables/feature_correlation_pairs_ge_0p80.csv`、`outputs/tables/feature_correlation_prune_decisions.csv` 和 `docs/feature_screening_initial.md`。

## Phase 4: 建模与策略

- 训练 Logit baseline 与 LightGBM 主模型。
- 输出 AUC、KS、Lift、TopK 捕获、坏样本率分层。
- 模型冻结后，在 `final_holdout` 上检测 PSI，并比较移除高漂移变量前后的模型表现。
- Logit baseline 先建立可解释、可复现的指标口径；LightGBM 用于后续非线性提升。
- LightGBM 只用 `validation` 做 early stopping，`final_holdout` 保持只读。
- 设计 approve / manual review / decline 三段式策略。
- 将 PD 转换成 A 卡分数：设置 base score、PDO、odds，并输出分数等级。
- 策略输出必须回答：
  - 如果拒绝风险最高的 5% / 10% 申请，能捕获多少坏样本？
  - 人工复核资源有限时，复核 Top 5% / 10% 的命中率是多少？
  - 自动通过组的坏样本率是否显著低于整体？
  - 哪些特征作为 reason code 可以解释拒绝/复核？

## Phase 5: 解释与监控

- 输出 SHAP 全局与个体理由码。
- 做训练集/验证集/时间外样本 PSI 或分布稳定性检查。
- 形成项目报告和简历 bullet。

## Phase 6: 简历与面试表达

- 简历表达重点：
  - 多表原始数据治理。
  - 申请时点可用性与防泄露。
  - 贷前准入三段式策略。
  - 模型解释与策略监控。
- 面试表达重点：
  - “模型不是难点，难点是标签、时点、特征口径、审核产能和业务代价。”
