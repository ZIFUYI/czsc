# 因子研究费用统一与双引擎重放

日期：2026 年 10 月 17 日。承接 [真实数据重跑报告](/Users/chenqiang/Desktop/czsc/docs/research/factor_system_real_data_rerun_20261017.md) 中的费用不一致问题。已有本地改动和原研究产物保留，未提交或推送。

## 1. 统一的执行合同

正式收益、成本和净值均使用固定初始资金 `N`。每个实际有报价的执行时点，按目标权重恢复数量：

```text
target_quantity[t] = weight[t] * N / price[t]
gross_return[t] = held_quantity[t-1] * (price[t] - price[t-1]) / N
turnover[t] = abs(target_quantity[t] - held_quantity[t-1]) * price[t] / N
cost_return[t] = turnover[t] * single_side_fee
net_return[t] = gross_return[t] - cost_return[t]
NAV[t] = 1 + cumulative_sum(net_return)[t]
```

等价的收益率输入账本在调仓前先标记 `pretrade_weight = previous_weight * (1 + price_return)`，再计算 `abs(target_weight - pretrade_weight)`。因此，同权重也可能发生再平衡成交；降低目标权重也可能恰好不需要成交。反手计入原仓平仓和新仓开仓两个数量，退出费用包含价格漂移。

缺报价行冻结数量、保持目标、零价格收益与零成交；恢复报价的收益跨越缺口。该合同不等于“有新价格但涨跌停无法成交”的模型。收盘信号仍在下一根开盘执行，单根输入没有可执行的尾部决策。只有显式给出最后收盘价或零目标时才清仓；未完成交易不补造成准入样本。

## 2. 实现与回归范围

共享算法位于 [research_execution.rs](/Users/chenqiang/Desktop/czsc/crates/czsc-core/src/research_execution.rs)，Rust crate 与 Python 顶层共同导出 `ExecutionLedger`、价格/收益率执行账本、下一开盘决策账本、闭区间归因和标准费用扣减。上述 API 的 Python 门面纯导出，研究调用层保留必要的 DataFrame、时间索引和数组边界处理。

- 正式准入事件、净 Sharpe、换手、Gate 2 救援收益、快速重配权和完整单资产报告使用同一执行账本；诊断三重屏障也按实际进出价格计费，质量权重保持独立。
- 主组合与全部 10 种分配方案接入共享账本，包含缺报价冻结和恢复。风险预算内部的零费用参考也保持冻结数量。
- 滚动权重历史使用没有虚构末尾清仓的账本；完整区间报告显式计入尾部清仓，交易数与活跃率只统计已执行目标。
- 固定资金的加法净值与另行运行的 Backtrader 按当前权益配置账户分别报告，不互相替代。

`czsc.WeightBacktest` 仍是外部 `wbt.WeightBacktest` 的直接导出。本轮使用实际外部引擎以零费用独立计算毛收益，保存为 `wbt_raw`；将该毛收益与共享 Rust 实际成交费用交给 Rust 扣减，保存为 `wbt_standardized`。**标准化 WBT 的费用来自共享账本，独立费用证据来自 Backtrader 的真实订单数量和佣金；没有宣称已修改外部 WBT 原生费用模型。**

核对门槛为毛收益、费用、净收益的逐 bar 绝对误差不超过 `1e-10`，累计绝对误差不超过 `1e-9`，单位为初始资本收益。相对净收益差只作诊断，不能用接近零的分母或彼此抵消的错误通过验收。

## 3. 冻结输入与真实数据结果

本轮复用原运行的真实价格、最终仓位与预先固定的 AG/CU/RB/M 非零控制，未重新拟合因子、重选资产或重新分配权重。旧输出目录保持原样，新输出为 [research_cost_fix_20261017](/Users/chenqiang/Desktop/czsc/examples/results/research_cost_fix_20261017)。[protocol.json](/Users/chenqiang/Desktop/czsc/examples/results/research_cost_fix_20261017/protocol.json) 记录 225 个输入文件的 SHA-256、实现与原生扩展身份、资金及费用合同。

日线正式主组合覆盖 50 商品，日内覆盖 47 商品；各跑零费、基础费、基础加 2/5/10 bp 五档。控制路径的基础费用都是单边 1.5 bp；因此五档总费率分别为 0、1.5、3.5、6.5、11.5 bp。正式组合的 AU 基础费为 1 bp，其余商品为 1.5 bp。附加摩擦统一作为成交名义额上的线性单边费用，不声称已经模拟真实成交价滑点。

日线控制包含 1,550 个执行时点，覆盖 2020-01-02 至 2026-05-28；日内控制包含 15,167 个执行时点、666 个标签日期，覆盖 2024-01-02 至 2026-09-30。表中收益、成本均为这些区间的累计初始资本百分比，不是年化收益。末尾 1 微秒为最后已知收盘的清仓顺序标签，不是新增行情。

基础费率下，原权重差计费的日线成本为 0.561600%，修正为 0.634444%，对应净收益从 3.182811% 修正为 3.109967%；日内成本从 5.457600% 修正为 5.569264%，净收益从 4.684888% 修正为 4.573224%。新结果与原独立 Backtrader 实际数量结果一致，毛收益未变。费用差不是靠给净收益加一个校准常数消除，而是每笔成交重新计算。

**费用口径修复通过本轮冻结数据验收：485 次正式资产重放、10 次非零控制重放全部通过。** 正式 Rust 账本、标准化 WBT 与独立 Backtrader 的毛收益、费用、净收益逐时点及累计一致。最大逐时点绝对误差为 `5.44e-16`，累计最大为 `1.21e-14`，均远小于预设阈值。225 个输入文件与全部已记录实现哈希在结束时再次验证，均未变化。

控制路径的日线毛收益为 **3.744411%**，日内为 **10.142488%**，五档均保持相同：

| 单边费用档 | 日线成本 | 日线净收益 | 日内成本 | 日内净收益 | 核对 |
|---|---:|---:|---:|---:|---|
| 零费用 | 0.000000% | 3.744411% | 0.000000% | 10.142488% | 通过 |
| 基础 1.5 bp | 0.634444% | 3.109967% | 5.569264% | 4.573224% | 通过 |
| 基础 + 2 bp | 1.480369% | 2.264041% | 12.994949% | -2.852461% | 通过 |
| 基础 + 5 bp | 2.749257% | 0.995153% | 24.133477% | -13.990989% | 通过 |
| 基础 + 10 bp | 4.864071% | -1.119660% | 42.697691% | -32.555202% | 通过 |

日线在基础加 10 bp 时为负，日内从基础加 2 bp 起为负。此处“通过”仅表示费用与收益核对通过，不表示高摩擦盈利门槛通过。

全部 20 套冻结正式组合（两阶段各 10 套）另经 Rust 账本完成五档费用压力测试，共 **100 行**；均为空仓，毛收益、成本、净收益和换手为零。只有预先固定主组合逐资产运行实际双引擎，其他方案的压力结果是共享 Rust 账本重计，不冒称也跑了独立 broker。

原始证据：[汇总与逐项误差](/Users/chenqiang/Desktop/czsc/examples/results/research_cost_fix_20261017/summary.json)、[双引擎五档费用表](/Users/chenqiang/Desktop/czsc/examples/results/research_cost_fix_20261017/friction_stress.csv)、[全部方案压力表](/Users/chenqiang/Desktop/czsc/examples/results/research_cost_fix_20261017/scheme_friction_stress.csv)。每个阶段/费用档保留逐资产和控制路径的 bar/day 账本、原生数量、实际 Backtrader 订单，以及 WBT 原始与标准化结果。

另行 [独立复核](/Users/chenqiang/Desktop/czsc/examples/results/research_cost_fix_20261017/validation/independent_artifact_accounting.json) 未重跑引擎，而是直接用保存的价格、目标和订单数量重算：10 个情景的主组合完整面板与分资产汇总逐时点差异均为 0；控制路径独立数量公式与三方账本最大累计差异约 `1.24e-14`；实际 Backtrader 逐笔佣金币值误差不超过 `5.69e-14`。全部方案及控制输入逐表等于旧输入，压力表无重复行。


## 4. 本轮工程验证

在 Rust 修改后先构建 release 扩展，随后执行 Python 测试；类型声明由生成器重新生成。Rust `czsc-core` 全量 **165 项通过，1 项文档测试忽略**。Python 四组互不重复，共 **180 项通过**：共享执行、正式入口及公共 API 74 项，屏障及准入 37 项，组合与滚动配置 48 项，双引擎 21 项。

覆盖多空、杠杆、资金规模不变性、同权重价格漂移、零成交缩仓、反手双腿、清仓、缺报价恢复、无未来补平仓、单根边界及非法时序。独立预期使用成交数量和现金流推导；没有以 Mock 收益证明策略盈利。

正式入口回归实际执行加载、准入和报告链，检查不合格种子不能进入库、全拒绝时正常产出报告，以及只验证模式保留既有研究库。新费用重放脚本复用正式组合账本。公共 API 快照验证顶层纯导出和外部 WBT 身份不变。

`test` extra 新增固定版本 `backtrader==1.9.78.123` 并同步锁文件，双引擎测试取消缺依赖时的整文件跳过。现有 CI 安装 `all` extra，因而会安装该测试依赖；本轮报告的是本地实际执行结果，没有宣称远程 CI 已运行。

20 个本轮修改的 Python 文件通过 Ruff 和格式检查；`cargo fmt --all -- --check`、全 workspace/all-targets Clippy（警告视为错误）及差异空白检查通过。[engineering_checks.json](/Users/chenqiang/Desktop/czsc/examples/results/research_cost_fix_20261017/validation/engineering_checks.json) 保存计数与各组日志哈希，同目录保存完整构建、类型声明生成和测试日志。

## 5. 剩余问题与结论边界

1. **外部 WBT 高精度限制仍存在。** WBT 0.2.1 在一个权重 `0.3942116370214049` 的独立工程反例中，即使 `digits=12` 且 `dfw` 保留高精度，输出毛收益仍与四位小数权重计算结果一致，逐 bar 差异约 `4.7464e-9`；内部舍入位置未确定。Rust 与 Backtrader 差异约 `4.5e-17`。严格校验会拒绝此路径，不舍入冻结输入，也不放宽阈值。六档 digits 探针及全部数字见 [wbt_precision_limit.json](/Users/chenqiang/Desktop/czsc/examples/results/research_cost_fix_20261017/validation/wbt_precision_limit.json)。本轮真实控制只含零、±0.12、0.06，不触发该限制；任意未来高精度组合仍须逐路径核对。
2. **本轮没有重做因子准入。** 原报告的 17,180 项检查和最终零获准是旧成本口径下的历史结果；本轮只重放冻结仓位，不能将旧 CI、胜率、盈亏比或通过状态冒称为新费用下已经重估。下一次重新筛选须使用新原生扩展与费用实现重新计算，不能复用旧统计结论。
3. **正式组合仍为空仓。** 冻结输入的零收益、零成本只证明空仓记账一致；控制路径是预先固定的机械仓位，不是获准 Alpha。高费用下出现亏损属于经济结果，不能用工程一致性通过来解释为盈利稳健。
4. **实际交易边界未补齐。** 仍使用分数数量、乘数 1 与模拟保证金；真实合约、最小手数、换月、盘口、涨跌停、共享现金/保证金约束和交易日历尚待验证。日内商品缺训练历史、微结构元数据不足、5m 候选未运行、历史资产池资格和外层 Walk-Forward 等原报告问题仍保留。

## 6. 重现

保留当前未提交源码和协议中的 native 身份；仅检出旧 Git HEAD 不能重现本轮实现。当前原生扩展 SHA-256 为 `886106eb65804af0cbe36bdfebd75415169df0c85dc12a1cae1fd2b40beba07d`。环境为 Python 3.12.12、CZSC 1.0.1、WBT 0.2.1、Backtrader 1.9.78.123。重放完成后把当前已有的 Backtrader 版本纳入测试依赖并同步锁文件，没有升级其他包版本；同步保留已有项目安装及原生二进制。

```bash
uv run --no-sync python -m examples.multi_factor_cta.replay_execution_costs \
  --source examples/results/research_rerun_20261017 \
  --output examples/results/research_cost_fix_new_run
```

使用新的输出目录；脚本拒绝覆盖已有重放协议或阶段目录，并在结束时重新校验所有输入与实现哈希。新环境先执行 `uv sync --extra test` 安装测试依赖。若修改 Rust 源码，先按仓库指引重新构建扩展，再运行两端回归和重放。
