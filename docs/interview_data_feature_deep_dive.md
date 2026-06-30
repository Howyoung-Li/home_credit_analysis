# 面试深挖：数据选取、特征构建与筛选

这份文档用于回答信用风险 / 金融科技风控面试中关于 Home Credit 项目的深挖问题。重点不是背代码，而是把每一步讲成业务上合理、技术上可复现、风控上防泄露的流程。

## 30 秒项目概述

这个项目使用 Kaggle Home Credit Default Risk 数据，从多张原始信贷表出发，构建贷前 A 卡准入模型。我的目标不是单纯刷分，而是模拟真实风控项目：先盘点申请时点可用的数据，构建申请级特征集市，再通过缺失率、稳定性、AUC/KS/IV、相关性去重筛选特征，训练 Logit baseline 和 LightGBM 主模型，最后把模型分数转成 approve / manual review / decline 策略，并用金额加权成本收益和 SHAP 原因码解释阈值。

核心数字：

| 项目 | 数值 |
|---|---:|
| 有标签 train 样本 | 307,511 |
| 官方无标签 test 样本 | 48,744 |
| 训练集坏账率 | 8.07% |
| 申请级特征集市字段 | 542 |
| 建模候选特征 | 674 |
| 最终入模特征 | 368 |
| LightGBM final_holdout AUC | 0.7960 |
| LightGBM final_holdout KS | 0.4494 |
| final_holdout Top 10% 坏账捕获 | 38.11% |

## 一、数据选取：为什么用这些表

### 1. 业务目标决定数据范围

项目定位是贷前准入 A 卡，因此只应该使用申请时点或申请前已经可获得的信息。我的数据选取逻辑是：

- 当前申请信息：解释本次贷款金额、还款压力、收入、客户画像和资料完整度。
- 外部征信历史：解释申请人在外部机构的历史授信、负债、逾期、多头借贷。
- Home Credit 历史申请：解释客户过去在本机构申请是否被拒、是否通过、产品类型偏好。
- 历史还款和余额表现：解释客户过往还款纪律、额度使用、DPD 和短缺还款。
- 官方 test：只输出预测，不参与筛选、调参或模型选择。

### 2. 原始表与业务含义

| 数据表 | 粒度 | 为什么使用 | 主要业务含义 |
|---|---|---|---|
| `application_train/test` | 当前申请，一行一个 `SK_ID_CURR` | 贷前申请主表 | 申请金额、收入、年龄、就业、家庭、资料完整度、外部评分 |
| `bureau` | 外部征信账户，一人多条 | 外部机构授信历史 | 活跃负债、历史授信、逾期、信用卡/消费贷产品 |
| `bureau_balance` | 征信账户月度状态 | 外部征信月度表现 | 最近月份状态、坏状态比例、严重逾期信号 |
| `previous_application` | 历史申请，一人多条 | 本机构历史申请记录 | 历史通过/拒绝、申请金额、授信金额、产品类型 |
| `installments_payments` | 历史分期还款计划和实还 | 还款纪律 | 是否晚还、晚还天数、短缺还款、还款比例 |
| `credit_card_balance` | 历史信用卡月度余额 | 额度使用行为 | 额度使用率、余额、取现/消费、最低还款压力 |
| `POS_CASH_balance` | 历史 POS/cash loan 月度状态 | 历史现金贷/POS 状态 | DPD、合同状态、剩余期数 |

面试回答：

> 我没有只用 `application_train` 宽表建模，而是先根据贷前准入业务把数据分成当前申请、外部征信、本机构历史申请、历史还款纪律、信用卡额度使用和 POS/cash loan 表现。所有历史表都先聚合到当前申请 `SK_ID_CURR` 粒度，确保最后每个申请只有一行模型输入。

## 二、防泄露原则：哪些数据不能直接用

### 1. 不能用的内容

- `TARGET` 只能作为标签，不能参与任何特征衍生。
- `SK_ID_CURR`、`SK_ID_BUREAU`、`SK_ID_PREV` 只用于连接和聚合，不作为模型业务特征。
- `final_holdout` 不参与特征筛选、调参和 early stopping。
- 官方 Kaggle test 没有标签，只能作为 `external_unlabeled` 输出预测，不能用来筛选变量。
- 类别编码只能在 development 上拟合，再应用到 validation / final_holdout / official test。

### 2. 关于 PSI 和 test 的处理

项目里我没有用官方 test 的 PSI 来提前筛变量。原因是：

- 官方 test 是最终外部样本，如果用它决定哪些变量进入模型，会把测试集分布信息泄露进建模流程。
- 更稳妥的做法是先在 train 内部分 development / validation / final_holdout，在 development 和 validation 上做开发期稳定性筛查。
- 模型冻结后，再在 final_holdout 或官方 test 上做漂移诊断和 ablation，而不是反过来用漂移结果指导前期筛选。

面试回答：

> 我把防泄露分成三层：第一，标签和主键不进入特征；第二，所有类别编码和筛选只用 development 或 development/validation，不看 final_holdout；第三，官方 test 只做最终预测和部署分布观察，不参与任何特征选择。这样避免把未来样本或最终测试样本的信息带入模型开发。

## 三、数据清洗：做了什么

### 1. 特殊值处理

Home Credit 数据里很多 `DAYS_*` 字段用 `365243` 表示异常或无效日期。处理方式：

- 将 `365243` 视为缺失。
- 保留对应哨兵标记字段，记录“原字段是否出现特殊值”。

这样做的原因是：特殊值本身可能代表某种业务状态，但不能当作真实天数参与年龄/时长计算。

### 2. 类别异常值处理

- 将 `XNA`、`Unknown` 统一视为未知/缺失类别。
- 避免把无效类别误当成稳定业务分组。

### 3. 原始数据不修改

- `data/raw/` 中的原始 CSV 不做覆盖修改。
- 清洗后申请底座输出到 `data/interim/application_base.parquet`。
- 后续申请级特征集市输出到 `data/processed/application_feature_store.parquet`。

面试回答：

> 清洗阶段我没有做复杂的模型假设，主要做申请时点字段的标准化：处理特殊日期哨兵值、异常类别、缺失标记，并衍生基础业务比率。原始数据始终不覆盖，所有中间表可追溯可复跑。

## 四、特征构建：怎么从多表变成申请级特征

### 1. 当前申请特征

来自 `application_train/test`。主要构建：

- 年龄、就业年限、证件发布时间、注册时间等时间型特征。
- `credit_to_income_ratio`：授信金额 / 收入。
- `annuity_to_income_ratio`：月供 / 收入。
- `credit_to_annuity_ratio`：授信金额 / 月供，近似期限或还款压力。
- `goods_to_credit_ratio`：商品价格 / 授信金额。
- `income_per_family_member`：家庭人均收入。
- `ext_source_mean/min/max/std`：外部评分的综合风险信号。
- `application_missing_field_count/ratio`：资料完整度。

业务解释：

> 当前申请特征主要回答“这笔贷款本身是否过重、客户基本偿债能力如何、申请资料是否完整、外部评分是否支持”。

### 2. 外部征信 `bureau` 特征

构建思路：

- 按 `SK_ID_CURR` 聚合外部征信账户。
- 区分 active / closed 账户。
- 区分 consumer credit / credit card 等产品。
- 计算授信金额、当前债务、逾期金额、最大逾期金额。
- 构建债务压力比率，例如 `debt_to_credit`。
- 加入最近 6 / 12 / 24 个月窗口，强调近期征信状态。

代表变量：

- `bureau_active_loan_ratio`
- `bureau_total_debt_to_credit_ratio`
- `bureau_credit_card_debt_to_credit_max`
- `bureau_recent_24m_debt_to_credit_max`
- `bureau_max_credit_max_overdue`

业务解释：

> 外部征信特征主要回答“客户在外部机构有没有活跃负债、负债率是否高、是否有历史逾期或多头借贷”。近期窗口用于区分很久以前的历史和当前仍然有效的风险状态。

### 3. 征信月度状态 `bureau_balance` 特征

构建思路：

- 先把每个 `SK_ID_BUREAU` 的月度状态聚合成账户级状态。
- 再通过 `bureau` 映射到 `SK_ID_CURR`。
- 计算最近 6 / 12 / 24 个月的 bad status、severe status 比例。

业务解释：

> 这类变量的业务含义很强，因为月度状态能反映客户近期外部征信是否恶化。但月度状态变量也容易有缺失和窗口差异，因此只在 train 内部做稳定性筛查，最终还要看模型贡献和 reason code 是否合理。

### 4. 历史申请 `previous_application` 特征

构建思路：

- 统计历史申请次数、通过率、拒绝率、取消率。
- 区分 approved / refused / canceled。
- 区分 cash loan / revolving loan 等产品。
- 计算历史申请金额、授信金额、申请金额与授信金额差异。
- 加入最近 6 / 12 / 24 个月历史申请窗口。

代表变量：

- `previous_refusal_rate`
- `previous_recent_24m_refusal_rate`
- `previous_cash_refusal_rate`
- `previous_credit_to_application_mean`

业务解释：

> 历史申请特征反映客户和本机构的交互历史。多次被拒、近期被拒、申请金额和最终授信金额差异较大，都可能表示客户风险或审批策略边界。

### 5. 历史分期还款 `installments_payments` 特征

构建思路：

- 对每期应还金额和实还金额计算差异。
- 构建晚还天数、是否晚还、短缺还款、还款比例。
- 聚合成申请级还款纪律特征。
- 加入最近 6 / 12 / 24 个月窗口。

代表变量：

- `installment_recent_24m_late_days_mean`
- `installment_recent_24m_late_ratio`
- `installment_payment_sum`

业务解释：

> 还款纪律是 A 卡里非常好解释的一类变量。近期晚还、短缺还款、长期还款不稳定，都能转成清晰的人工复核或拒绝原因。

### 6. 信用卡 `credit_card_balance` 特征

构建思路：

- 计算额度使用率：余额 / 信用额度。
- 聚合余额、取现、消费、最低还款、DPD。
- 加入最近 6 / 12 / 24 个月窗口。

代表变量：

- `credit_card_recent_6m_utilization_max`
- `credit_card_recent_12m_utilization_mean`
- `credit_card_recent_24m_balance_mean`

业务解释：

> 信用卡额度使用率能反映客户短期资金压力。近期额度使用率高、余额持续高、最低还款压力高，都是贷前准入中很直观的风险信号。

### 7. POS / cash loan `POS_CASH_balance` 特征

构建思路：

- 统计历史 POS/cash loan 的状态。
- 聚合 DPD、剩余期数、active/completed 状态比例。
- 加入最近 6 / 12 / 24 个月窗口。

业务解释：

> POS/cash loan 月度状态可以补充客户在本机构小额分期或现金贷产品上的履约表现，与外部征信和分期还款形成互补。

## 五、为什么要加时间窗口和分组聚合

面试官可能会问：为什么不只做简单均值、最大值、计数？

回答：

> 简单全历史聚合会把很久以前的表现和近期表现混在一起。真实风控更关心“最近是否恶化”。所以我加了 6 / 12 / 24 个月窗口，比如最近 6 个月信用卡额度使用率、最近 24 个月外部征信负债率、最近 12 个月晚还比例。分组聚合则是为了让变量更接近业务规则，比如 active vs closed、approved vs refused、cash loan vs revolving loan，不同状态和产品的风险含义不同。

这不是为了单纯追分，而是为了让后续 SHAP 和 reason code 有更清晰的业务含义。

## 六、特征筛选：怎么从 674 到 368

### 1. 样本切分

| split | 行数 | 用途 |
|---|---:|---|
| development | 184,507 | 训练、类别编码拟合、单变量筛选 |
| validation | 61,502 | early stopping、稳定性验证、阈值选择 |
| final_holdout | 61,502 | 最终只读评估，不参与筛选 |
| external_unlabeled | 48,744 | 官方 test，仅输出预测 |

### 2. 筛选漏斗

| 阶段 | 规则 | 结果 |
|---|---|---:|
| 候选特征 | 多表聚合 + 类别编码 | 674 |
| 质量筛选 | 缺失、唯一值、development/validation PSI | 672 |
| 单变量筛选 | AUC power / KS / IV | 468 |
| 相关性去重 | 高相关冗余变量去重 | 368 |

### 3. Target-blind 质量筛选

不看标签，只看变量是否可用：

- 缺失率是否过高。
- 是否近似常数。
- development 和 validation 分布是否稳定。
- 类别编码是否只使用 development 拟合。

注意：这里没有用 final_holdout 或官方 test 进行筛选。

### 4. Target-aware 单变量筛选

只在 development 上看标签：

- AUC power >= 0.515
- KS >= 0.020
- IV >= 0.005

这里我故意用了较宽松门槛。原因是：

- 真实业务中有些变量单变量能力不强，但和其他变量有交互。
- 时间窗口变量和分组变量常常需要树模型捕捉非线性。
- 过早剔除会损失业务解释空间。

面试回答：

> 我没有用很强的单变量门槛一刀切，因为贷前风控中很多变量单独看不强，但和收入、额度、历史逾期、产品类型组合后有价值。因此我把单变量筛选定位成“候选压缩”，不是最终定论。最终是否保留，还要看相关性、模型贡献和 SHAP/reason code。

### 5. 相关性去重

对通过初筛的变量做高相关去重：

- 高相关变量容易重复表达同一风险。
- 会放大模型对某类信号的依赖。
- 也会让 SHAP 解释变得分散。

处理方式：

- 对相关性高的变量对保存证据表。
- 相关性 >= 0.95 时，优先保留区分度更强、缺失更低、业务解释更清晰的变量。
- 输出 `feature_correlation_prune_decisions.csv` 记录剔除原因。

### 6. 分组筛选结果

| 特征组 | 候选数 | 最终保留 |
|---|---:|---:|
| 当前申请 | 276 | 91 |
| 历史申请 | 83 | 65 |
| 外部征信 | 86 | 64 |
| 历史还款 | 60 | 52 |
| POS/cash loan | 55 | 39 |
| 信用卡历史 | 73 | 36 |
| 征信月度状态 | 33 | 19 |
| 覆盖率/质量 | 8 | 2 |

## 七、遇到的问题与解决方法

### 问题 1：多表粒度不一致

问题：

- `application` 是一行一个当前申请。
- `bureau`、`previous_application`、`installments`、`credit_card_balance` 都是一对多或多对多。
- 如果直接 join，会造成样本重复和标签污染。

解决：

- 每张历史表先聚合到 `SK_ID_CURR`。
- 只保留申请级宽表作为模型输入。
- `SK_ID_*` 只作为连接键，不进入模型。

面试回答：

> 我没有直接把历史表 join 到主表，而是先按业务粒度聚合。所有模型变量最终都落在当前申请 `SK_ID_CURR` 粒度，这样避免一对多 join 导致样本膨胀。

### 问题 2：申请时点可用性和潜在泄露

问题：

- 历史还款、余额、月度状态表里有时间字段。
- 如果不检查时间窗口，可能会用到当前申请后的信息。

解决：

- 所有历史窗口都基于相对申请日的 `DAYS_*` 或 `MONTHS_BALANCE`。
- 只构造历史窗口聚合。
- `TARGET` 不参与特征衍生。
- `final_holdout` 和官方 test 不参与筛选。

面试回答：

> 对于风控项目，我最关心的是申请时点可用性。我的处理原则是：标签不进特征，主键不进模型，历史表只用申请前的相对时间窗口，最终 holdout 不参与特征筛选。

### 问题 3：缺失率高，但业务含义强

问题：

- 信用卡、征信月度状态等表缺失率较高。
- 缺失可能表示客户没有该类历史产品，而不是纯随机缺失。

解决：

- 不简单按高缺失一刀切删除。
- 保留 history coverage 和缺失信号。
- 对缺失高但单变量/模型贡献较好的变量保留。
- 在解释时说明“该变量只对有历史记录客户有效”。

例子：

- `credit_card_recent_6m_utilization_max` 缺失率高，但在有信用卡历史的人群中业务含义强，且进入了最终短名单。

面试回答：

> 在信贷数据里，缺失本身经常有业务含义，例如没有信用卡历史、没有外部征信记录、没有本机构历史贷款。所以我没有简单用缺失率删除所有变量，而是结合覆盖率、区分度、稳定性和业务解释判断。

### 问题 4：业务含义强的变量被稳定性筛掉怎么办

问题：

- 一些月度状态变量业务含义强，但 development/validation 分布可能不稳定。
- 如果直接拿官方 test 做 PSI 筛选，会有测试集泄露风险。

解决：

- 筛选阶段只用 development/validation 做稳定性检查。
- final_holdout 不参与变量选择。
- 官方 test 只做最终部署分布观察。
- 对高 PSI 变量可以在模型冻结后做 ablation：移除前后比较 AUC、KS、策略收益和 reason code。

面试回答：

> 如果一个变量业务含义强但分布不稳定，我不会直接因为官方 test PSI 高就删除，因为那相当于用了最终样本信息。更合理的是在 train 内部先做稳定性筛查，模型冻结后再在 holdout 做漂移诊断和移除前后对比。

### 问题 5：类别变量怎么编码避免泄露

问题：

- 类别字段很多，直接 one-hot 容易稀疏。
- 如果用全量数据计算类别频率，会泄露 validation/test 分布。

解决：

- 频率编码只用 development 拟合。
- top 类别 one-hot 只根据 development 选择。
- validation/final_holdout/test 只应用已经拟合好的映射。

面试回答：

> 类别变量编码也可能泄露，所以我没有用全量样本算频率或选 top 类别，而是在 development 上拟合编码规则，再应用到后续 split。

### 问题 6：怎么避免“先射箭再画靶”

问题：

- 如果先挑几个看起来合理的变量，再用结果证明它们有用，会有主观筛选偏差。

解决：

- 先系统生成候选变量工厂。
- 再用固定筛选漏斗压缩变量。
- 每一步保留证据表，包括缺失率、PSI、AUC、KS、IV、相关性剔除原因。
- 最后才做 SHAP 和 reason code 业务解释。

面试回答：

> 我把特征工程拆成候选变量工厂和筛选漏斗。业务逻辑指导候选生成，但是否进入模型要经过预设的统计筛选和模型验证。这样避免先有故事再倒推变量。

### 问题 7：Kaggle 分数和业务项目目标的冲突

问题：

- Kaggle 第一名会使用更复杂的 stacking、更多特征和更强调 leaderboard。
- 但真实风控更关心可解释、稳定、可监控和策略落地。

解决：

- 主模型使用 LightGBM，保留 Logit baseline。
- 不做过度 stacking。
- 输出 A 卡分数、三段准入策略、金额加权成本曲线和 SHAP 原因码。
- 更强调业务策略和面试可解释性。

面试回答：

> 这个项目参考了高分方案的多表聚合和 LightGBM 思路，但没有把刷榜作为唯一目标。我更关注贷前准入场景下变量是否申请时点可用、是否稳定、是否能形成 reason code，以及模型分数能否转成 approve/manual review/decline 策略。

## 八、模型结果怎么和特征工程连接起来

模型对比：

| 模型 | final_holdout AUC | KS | Top 10% 坏账捕获 |
|---|---:|---:|---:|
| Logit baseline | 0.7857 | 0.4345 | 36.25% |
| LightGBM | 0.7960 | 0.4494 | 38.11% |

解释：

- Logit baseline 证明特征本身有线性区分能力。
- LightGBM 进一步利用非线性和交互，提升 AUC、KS 和 TopK 捕获。
- final_holdout 没参与筛选和调参，因此结果更接近真实泛化评估。

面试回答：

> 我先用 Logit 做 baseline，确认特征集本身有稳定区分能力；再用 LightGBM 捕捉非线性和交互。LightGBM 在 final_holdout 上 AUC 0.796、KS 0.449，相比 baseline 有提升，说明多表历史特征和时间窗口特征确实提供了增量信息。

## 九、面试高频追问与回答

### Q1：为什么不用所有变量直接扔进模型？

答：

> 直接扔进模型会有三个问题：第一，容易混入泄露或申请时点不可用字段；第二，高缺失、高相关变量会影响稳定性和解释；第三，模型可能依赖不可监控或不可解释信号。所以我先做质量、稳定性、单变量区分度和相关性筛选，再进入模型训练。

### Q2：为什么保留单变量不强的变量？

答：

> 因为风控变量经常存在交互。例如近期信用卡额度使用率可能要和收入、授信金额、历史还款行为一起看才有价值。单变量筛选只是压缩候选空间，不是最终结论，所以我采用宽松阈值，把有业务含义的变量交给 LightGBM 判断交互贡献。

### Q3：为什么时间窗口是 6/12/24 个月？

答：

> 这是兼顾近期风险敏感性和样本覆盖率的折中。6 个月反映短期恶化，12 个月比较接近年度信用周期，24 个月保留较长历史行为。真实业务中这个窗口可以根据产品期限、审批政策和数据覆盖再调。

### Q4：如何解释高缺失特征？

答：

> 高缺失不一定是坏事。例如信用卡历史缺失可能表示客户没有信用卡历史。我的做法是保留覆盖率/缺失信号，并且对高缺失变量看它在有记录客群中的稳定性和模型贡献。上线时这类变量也要单独监控覆盖率变化。

### Q5：如果面试官说你的特征筛选还是有主观性怎么办？

答：

> 主观性无法完全消除，因为业务理解本身会影响候选特征设计。但我把主观业务假设限制在“候选生成”阶段，而筛选阶段用固定、可复现的统计规则，并输出证据表。这样可以解释每个变量为什么保留或剔除。

### Q6：为什么 final_holdout 不参与 PSI 筛选？

答：

> 因为 final_holdout 是最终只读评估集。如果用它决定变量取舍，就会让评估结果乐观。我的做法是 development/validation 用于开发和稳定性筛查，final_holdout 只在模型冻结后做验证和漂移诊断。

### Q7：为什么官方 test 不能用来筛特征？

答：

> 官方 test 没有标签，但它的分布信息仍然是最终样本信息。如果用 test PSI 来决定删除哪些变量，本质上是把最终样本分布泄露进建模流程。可以在模型冻结后报告 test 分布和稳定性，但不应该用它反向影响特征选择。

### Q8：项目里最能体现业务理解的特征是什么？

答：

> 我会举四类：第一，近期信用卡额度使用率，反映短期资金压力；第二，外部征信负债/授信比，反映多头负债和杠杆；第三，历史分期晚还比例，反映还款纪律；第四，历史申请拒绝率，反映客户在本机构审批边界上的风险。

## 十、可以背的总结话术

长版：

> 我这个项目的重点是把 Kaggle 原始多表数据改造成真实风控项目的流程。首先从贷前 A 卡业务出发，只选择申请时点或申请前可用的信息，包括当前申请、外部征信、本机构历史申请、历史还款、信用卡余额和 POS/cash loan 状态。然后把所有一对多历史表聚合到当前申请粒度，构建 542 个申请级字段，再经过类别编码生成 674 个建模候选特征。筛选时我没有直接用最终 test，而是在 train 内部分 development、validation 和 final_holdout，先做缺失率、唯一值、PSI 等 target-blind 质量筛选，再用 development 标签做 AUC/KS/IV 单变量筛选，最后做相关性去重，得到 368 个入模特征。这个流程的关键是防止数据泄露、保留业务可解释性，并为后续 SHAP reason code 和准入策略服务。

短版：

> 我不是直接拿宽表建模，而是先按贷前可用性做数据选取，再把多表历史行为聚合成申请级特征集市，最后用质量、稳定性、区分度和相关性去重形成入模变量。整个过程保留证据表，final_holdout 和官方 test 不参与筛选，确保评估和解释更可信。

## 十一、对应文件

- 数据清洗：[docs/data_cleaning_rules.md](data_cleaning_rules.md)
- 特征集市：[docs/feature_store_initial.md](feature_store_initial.md)
- 特征筛选：[docs/feature_screening_initial.md](feature_screening_initial.md)
- 筛选方法论：[docs/feature_engineering_selection_protocol.md](feature_engineering_selection_protocol.md)
- 防泄露审计：[docs/leakage_audit_initial.md](leakage_audit_initial.md)
- 模型结果：[docs/lgbm_model_initial.md](lgbm_model_initial.md)
- 策略与 SHAP：[docs/strategy_profit_shap_initial.md](strategy_profit_shap_initial.md)

