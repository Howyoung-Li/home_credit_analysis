# 大模型辅助智能风控组件说明

本文件由 `src/09_visualize_strategy.py` 自动生成。

## 定位

项目不把LLM作为直接打分模型，而是把LLM/Agent放在分析编排层：模型负责风险排序，策略阈值负责准入动作，SHAP和变量证据负责解释，Agent负责把这些证据组织成可复核的审核摘要。

## 与度小满风控策略工程师岗位的对应

- 智能化风控组件：A卡模型、准入策略、原因码、审核摘要串成组件化流程。
- 流水/账单解析：Home Credit无真实流水，本项目用收入、年金、授信、征信债务、信用卡利用率和分期还款表现构造偿债能力代理指标。
- KYC智能画像：用职业稳定性、家庭负担、资产居住、外部征信、历史申请与履约表现构建申请人画像。
- 风险排序能力：保留AUC、KS、PR-AUC/AP、TopK坏账率、坏账捕获、Lift作为模型验证指标。
- 稳定性：保留PSI、score drift和final_holdout只读验证口径。
- 业务协同：输出approve/manual_review/decline三段策略、人审清单、误杀/漏放成本与金额加权阈值。

## 输出文件

- `reports/intelligent_risk_agent_demo.html`：智能风控Agent审核摘要页面。
- `outputs/figures/08_repayment_capacity_segments.svg`：偿债能力与KYC画像分层图。
- `outputs/tables/risk_agent_capacity_segment_summary.csv`：三段策略客群画像指标汇总。
- `outputs/tables/risk_agent_case_studies.csv`：三类样例的结构化审核摘要。

## 推荐面试话术

我没有让大模型直接判断客户好坏，因为信贷风控需要可验证、可监控、可复盘。我的做法是传统模型负责风险排序，策略模块负责准入阈值，SHAP和变量筛选证据负责解释，大模型Agent负责把客户画像、模型原因码、策略规则和复核清单组织成结构化审核摘要。这样既能利用大模型处理复杂信息和提升审核效率，又不破坏风控模型的可审计性。
