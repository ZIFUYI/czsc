# 多因子 CTA 四阶段架构与实现审核

审核日期：2026 年 10 月 17 日。需求依据：[Issue 3](https://github.com/ZIFUYI/czsc/issues/3)、[Issue 5](https://github.com/ZIFUYI/czsc/issues/5)、[Issue 6](https://github.com/ZIFUYI/czsc/issues/6)、[Issue 7](https://github.com/ZIFUYI/czsc/issues/7)。

> 以下审核正文保留首次审核的代码基线与反例，行号指向当时版本。用户随后要求的修复已落入工作区；当前状态见文末「第一批修复记录」。不要把历史反例当作修复后仍存在的问题，也不要把工程回归通过视为经济验收通过。

**结论：四阶段拆分有合理基础，但当前实现不能按四个 Issue 的验收标准判定完成，也不足以支持现有因子准入和组合绩效结论。** 主要障碍是未来信息进入历史计算、统计准入样本失真、正式入口未接通新增模块，以及最终仓位与成本没有形成统一账本。应先修复这些基础问题，再评价策略收益和复杂方法是否有增益。

## 审核范围与判断口径

代码基线为 `0e3da7e...6ecbb45`，重点覆盖 Phase 1 提交 `901b96e`、Phase 2 提交 `026f054`、Phase 3/4 提交 `9606931` 及其合并结果。同时检查主研究入口、已有示例和测试是否实际使用新增能力。

规范依据为 [CLAUDE.md](/Users/chenqiang/Desktop/czsc/CLAUDE.md)、[策略开发标准](/Users/chenqiang/Desktop/czsc/docs/strategy_development_guide.md) 与仓库研究 SOP。规范问题和需求实现问题分别列出；既有缺陷但本次要求修复而未修复的，不冒称为新回归。

本地已有的 `tas.rs`、顶层导出、未跟踪 `cta_allocator.py`、脚本及策略目录修改均保留。本次未修改业务源码。运行验证使用重建后的当前工作区扩展，包含用户已有本地信号修改；源码审核针对上述远程提交。尤其，未跟踪的本地 allocator 不能算作远程 Issue 7 已交付内容。

“通过”仅指该项代码契约有依据；“部分”表示有工具或公式但缺完整调用、边界或验收；“无有效验收证据”表示当前源码和相关测试不能支持该项经济结论，不表示证明策略必然亏损。P1 为会污染历史结果、准入判断或实际损益的高优先级问题；P2 为需要修正的契约、边界或集成问题。

## 整体架构和方法的合理性

| 设计 | 评估 | 应保留或调整的部分 |
|---|---|---|
| 原子信号 → 因子评价 → 截面条件 → 组合风险预算 | 合理的职责划分 | 截面条件必须在评价前确定，且研究、回测和实盘走同一个入口；所有拟合和筛选都放进外层 Walk-Forward。 |
| 方向 × 结构质量 Q × 生命周期 D | 有解释性 | Q 是人工评分，尚不是胜率或置信概率；分别保存原始 Q、衰减信号、门控后暴露和最终仓位，避免一列 weights 承担全部含义。 |
| 六簇 × 五周期 | 适合作为候选覆盖框架 | 六簇不是天然统计正交。E 是波动状态，F 是跨周期/跨资产组合维度，均可与 A/B/C/D 重叠；适合多标签分类。 |
| 分簇标尺、ATR 归一、多资产前置准入 | 方向正确 | 统一真实交易记录、收益单位、成本、样本有效期和事件终止时刻；不同标尺的结果不能被当成相同实盘收益。 |
| 状态均衡与 IPW | 可做状态标准化比较 | 当前主要是对已有状态的条件均值取平均。应明确目标状态分布、缺失状态和有效样本量；不能据此宣称已识别因果 Alpha。 |
| Sigmoid 软门控 | 可减少规则阈值突变 | 相对排名不等于绝对趋势；全市场下跌时最抗跌者仍是第一名。排名离散也会造成仓位跳变，应在相同费用下检验实际换手与增量收益。 |
| 70% 池先验 + 30% 局部权重 | 可作为保守收缩基线 | 固定比例和 tanh 映射不等于已估计经验贝叶斯模型；先验必须带训练截止时间、同质池定义和版本，不能把全 IS 统计回填到 IS 起点。 |
| Gram-Schmidt 去重复 | 可控制线性重复暴露 | 信号不相关不推出扣费收益不相关；残差会在原因子静默时产生对冲仓位。应将结构 Alpha 与对冲调整分开，并比较净收益风险贡献。 |
| 动态活跃池与目标波动率 | 有风险管理价值 | 资金利用率升高也会提高集中度和总风险；须在相同风险与杠杆约束下比较，而不是将减少现金本身视为收益来源。 |
| 板块分层分配 | 值得保留 | 当前为板块约束逆波动率分配，不是完整 HRP；Issue 自身把 1/√K 简写成 HRP 也需澄清。 |
| 多指标、多门禁 | 可改善诊断 | 大量指标并非独立证据。持续搜索许多因子、周期、阈值后，应记录所有尝试、控制选择偏差，并保留不反复查看的最终测试集。 |

原始 HRP 方法利用协方差信息与层级结构；仅对板块乘 1/√K 不能等同于该算法。[López de Prado 2016](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678)。波动率管理有相关研究支持，但不能代替当前 CTA 在自身市场、费用和交易约束下的验证。[Moreira 与 Muir 原始研究](https://www.nber.org/system/files/working_papers/w22208/revisions/w22208.rev0.pdf)。重复尝试带来的 Sharpe 选择偏差应独立处理，可将 DSR 等方法作为辅助诊断。[Bailey 与 López de Prado 2014](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)。

## Spec 轴的确定问题

### S01 P1 高周期未来状态改写历史信号

[atomic_factors.py:770](/Users/chenqiang/Desktop/czsc/examples/multi_factor_cta/atomic_factors.py:770) 用全数组 `np.any(ctx.htf_dir != 0)` 决定 FC01 整段使用高周期方向还是本周期均线代理。未来第一次出现有效高周期笔，便改变此前的算法。

实际反例：使用 czsc.mock 行情和受控结构输入，高周期第 200 根才有效；只跑前 150 根得到 110 根非零输出，全量运行的前 150 根却全部为零。现有较晚截断点测试绕过了预热状态切换。

**改进**：高周期是否启用由配置决定；可用性逐时点判断。增加首次有效状态前后、仅改未来数据的前缀不变性验证。

### S02 P1 完成笔退出方向取反

[atomic_factors.py:433](/Users/chenqiang/Desktop/czsc/examples/multi_factor_cta/atomic_factors.py:433) 将 cxt_bi_base 方向解释成完成笔，并据此在 [动态衰减:624](/Users/chenqiang/Desktop/czsc/examples/multi_factor_cta/atomic_factors.py:624) 平仓。但 [cxt.rs:67](/Users/chenqiang/Desktop/czsc/crates/czsc-signals/src/cxt.rs:67) 输出的是与最新完成笔相反的未完成笔方向。

真实增量结构反例：最新完成笔向上，cxt_bi_base 输出向下，被映射成 -1，长仓从 1 降到 0。既会错误终止同向结构，也可能遗漏真正的反向笔退出。

**改进**：Rust 输出带确认时刻、方向和结构 ID 的完成笔事件，消费者使用该事件。不要从未完成笔字符串反推完成事件。

### S03 P1 默认正交化对整个 IS 和 OOS 拟合

[默认分支:147](/Users/chenqiang/Desktop/czsc/examples/multi_factor_cta/synthesis_engine.py:147)、[合成默认参数:599](/Users/chenqiang/Desktop/czsc/examples/multi_factor_cta/synthesis_engine.py:599) 均允许并默认使用全样本。实际 demo 没传因果窗口；[IS 坐标优化:801](/Users/chenqiang/Desktop/czsc/examples/multi_factor_cta/synthesis_engine.py:801) 先合成全历史，再切 IS 评分，OOS 信息已经进入 IS。

反例：400 根行情不变，只修改后 200 根 FC01 信号，前 200 根有 198 根仓位改变，最大差 0.1010854。直接正交算子的前缀比较最大差 0.548424；显式窗口 90 时为 0。

**改进**：交易入口强制历史窗口或训练后冻结；全样本模式独立命名为离线诊断。必须对正式默认入口做未来扰动测试。

### S04 P1 当日收益决定当日资金分配

[活跃池回退:1000](/Users/chenqiang/Desktop/czsc/examples/multi_factor_cta/synthesis_engine.py:1000) 按当日非零收益推断持仓，[收益合成:1105](/Users/chenqiang/Desktop/czsc/examples/multi_factor_cta/synthesis_engine.py:1105) 又将当日预算乘当日收益。缺持仓的 demo 和监控入口会走此路径。

默认参数、8 个同板块资产中，仅将 A 第 60 天收益从 0 改为 0.001，A 同一天预算从 0 变为 0.13630556。零收益本来也不能区分空仓、平价持仓和费用抵消。

**改进**：必须输入决策时已知的有符号仓位，明确何时决策、从哪个收益区间生效；缺失上下文应拒绝分配，不能用实现收益猜持仓。

### S05 P1 风险预热的后向填充泄漏未来

[风险估计:1012](/Users/chenqiang/Desktop/czsc/examples/multi_factor_cta/synthesis_engine.py:1012)、[ATR:1014](/Users/chenqiang/Desktop/czsc/examples/multi_factor_cta/synthesis_engine.py:1014)、[板块:1036](/Users/chenqiang/Desktop/czsc/examples/multi_factor_cta/synthesis_engine.py:1036)、[组合目标波动率:1099](/Users/chenqiang/Desktop/czsc/examples/multi_factor_cta/synthesis_engine.py:1099) 在 shift 后又 bfill。单品种波动率上限也有类似逻辑。这包含既有问题及新增路径扩展。

默认 8 标的、持仓固定，只放大未来第 5～14 天 A 的收益，第 0 天 A 预算由 0.14081 改为 0.09140。

**改进**：预热期不交易或使用预先固定的保守风险先验，只前向填充已知结果。逐层输出风险估计的可用时间。

### S06 P1 IS 不足时自动纳入 OOS 且事件可跨边界

[Gate 1:337](/Users/chenqiang/Desktop/czsc/examples/alpha_mining_agent/forensic_pipeline.py:337)、[IC:162](/Users/chenqiang/Desktop/czsc/examples/alpha_mining_agent/forensic_pipeline.py:162)、[净 Sharpe:252](/Users/chenqiang/Desktop/czsc/examples/alpha_mining_agent/forensic_pipeline.py:252)、[归一化审计:805](/Users/chenqiang/Desktop/czsc/examples/multi_factor_cta/forensic_validator.py:805) 在 IS 少于 20/25 根时改用全部数据。只含 2024 年的行情仍输出“IS”统计；此前反例实际 IS=0，却报告 8 笔 IS 事件。

即使 IS 足够，[事件评价:352](/Users/chenqiang/Desktop/czsc/examples/alpha_mining_agent/forensic_pipeline.py:352) 也将完整行情传给屏障函数，只按触发日期筛选。2023-12-29 触发，仅改变 2024 年 high，IS 事件收益即可从 -0.02% 变成 +2.18%。跨边界是既有问题，本轮扩展仍未解决。

**改进**：不足则返回不可准入；按事件结束时刻划分训练/测试并做 purge/embargo，禁止跨界标签参与拟合。

### S07 P1 池化 t 检验使用构造样本并混用收益单位

[正式入口:245](/Users/chenqiang/Desktop/czsc/examples/alpha_mining_agent/run_evolution_loop.py:245) 向 Gate 1B 传普通 Gate 1 摘要，没有调用要求的归一化审计。[validator:566](/Users/chenqiang/Desktop/czsc/examples/multi_factor_cta/forensic_validator.py:566) 将百分比均值作为 ATR 单位均值；缺逐事件收益时，[583](/Users/chenqiang/Desktop/czsc/examples/multi_factor_cta/forensic_validator.py:583) 构造 linspace 样本计算 t。

反例：5 份 N=100、均值 0.08、标准差 1 的摘要，代码算出 t=6.129 并通过，按输入摘要矩汇总应约为 1.789。正式主链正好会进入该回退，而不是仅测试专用路径。

**改进**：传真实、同单位、带时间戳的逐事件净收益；缺数据拒绝准入。摘要只用于展示，不能生成交易样本。

### S08 P1 同质池要求可被回退逻辑绕过

[同质池选择:531](/Users/chenqiang/Desktop/czsc/examples/multi_factor_cta/forensic_validator.py:531) 在匹配不足两个资产时改用全部资产；低样本资产从分母中删除。[主循环:957](/Users/chenqiang/Desktop/czsc/examples/alpha_mining_agent/run_evolution_loop.py:957) 默认允许单资产降级；没有严格执行 Issue 的资产数下限和商品板块/交易制度分组。

反例：目标权益池只有一个亏损 ETF，另外四个盈利商品也被纳入，按 80% 普适率通过。包装入口不传目标池类别，商品候选也可能按默认权益池评判。

**改进**：池定义先于结果固定，最低规模、所属板块、交易时段和流动性规则都必须满足。数据不足返回 insufficient_data；研究诊断降级不能产生正式 admitted 状态。

### S09 P1 Q 加权统计量和随机基准使用不同检验

[统计量:373](/Users/chenqiang/Desktop/czsc/examples/alpha_mining_agent/forensic_pipeline.py:373) 采用 Q 加权均值，但 [bootstrap:417](/Users/chenqiang/Desktop/czsc/examples/alpha_mining_agent/forensic_pipeline.py:417) 使用 N 个等权样本的均值。极度集中的 Q 会把约一个有效样本误当成几十个独立样本。

隔离反例：30 事件，Q 为 1 和 29 个 0.0001；高 Q 事件 +1%，其余 -1%，对称 ±1% 基准。实现 p=0；采用相同权重的随机基准 p 约 0.5，有效样本数约 1.006。这里证明 p 值错误，不声称该例一定通过全部其他门禁。

**改进**：随机基准使用相同事件权重和抽样结构；记录有效样本量 (ΣQ)²/ΣQ²，按时间块重采样。

### S10 P1 跨资产共振被当成独立样本增加显著性

[pooled samples:581](/Users/chenqiang/Desktop/czsc/examples/multi_factor_cta/forensic_validator.py:581) 直接拼接收益，[t 值:595](/Users/chenqiang/Desktop/czsc/examples/multi_factor_cta/forensic_validator.py:595) 按独立样本标准误计算，没有事件时间戳来识别同日共振。

同一条 20 事件序列复制到 5 个资产，t 从 1.436 变为 3.279；信息完全重复，却跨过门槛。固定 cooldown 还不等于真实区间 purge：cooldown=3 会保留 [0,3]，而屏障实际可能各持有 4 根、仍有重叠。

**改进**：保留 entry/exit 时间；按真实标签区间清除重叠，用日期/资产簇稳健标准误或同步时间块 bootstrap。普适资产数与独立信息量分别报告。

### S11 P1 截面引擎没有接入正式挖掘入口

[主循环加载与预计算:616](/Users/chenqiang/Desktop/czsc/examples/alpha_mining_agent/run_evolution_loop.py:616) 和 [池内评价:244](/Users/chenqiang/Desktop/czsc/examples/alpha_mining_agent/run_evolution_loop.py:244) 都逐品种直接算信号。examples 中截面 API 只有定义和导出，没有正式研究脚本调用。

[缺失截面回退:503](/Users/chenqiang/Desktop/czsc/examples/multi_factor_cta/atomic_factors.py:503) 使用本品种均线偏离/斜率冒充板块龙头、残差和截面强弱。120 根单品种 mock、has_external_cs=False，FC02/03/04 仍产生 8/10/5 次触发。

**改进**：先生成统一时钟的历史有效资产池和截面，再计算、门控和审计；缺截面时 F 因子返回 missing_context，单品种代理使用不同因子身份。

### S12 P1 软门控删除评价事件却保留实际仓位

[atomic_factors.py:341](/Users/chenqiang/Desktop/czsc/examples/multi_factor_cta/atomic_factors.py:341) 用 g≥0.25 过滤触发，但权重只乘 g。B 空头排名从 0.79 到 0.80，仓位由 -0.253506 平滑降为 -0.231475，法医事件却从保留变成删除。

**改进**：保留所有原始触发并用实际执行权重评价；若最小可交易量确实导致不交易，必须同时归零仓位。不能只从收益检验中删除事件。

### S13 P2 门控和截面倾斜可重复应用

门控散布在 [原子层:930](/Users/chenqiang/Desktop/czsc/examples/multi_factor_cta/atomic_factors.py:930)、[注册表:167](/Users/chenqiang/Desktop/czsc/examples/multi_factor_cta/factor_engine.py:167)、[法医层:190](/Users/chenqiang/Desktop/czsc/examples/multi_factor_cta/forensic_validator.py:190)、[合成层:398](/Users/chenqiang/Desktop/czsc/examples/multi_factor_cta/synthesis_engine.py:398)。注册表启用门控后再调用默认法医，R=0.70 的 B 空头可从 -1 变 -0.5，再变 -0.25。

单品种与跨资产预算也均可做同一截面倾斜。当前类型和元数据不能判断输入经过哪些处理。

**改进**：一个阶段拥有门控和一次倾斜；原始信号、门控信号和组合仓位分别定义，保存 transform/version 信息。

### S14 P2 过期行情持续参与排名和龙头选择

[data_engine.py:505](/Users/chenqiang/Desktop/czsc/examples/multi_factor_cta/data_engine.py:505) 无限期 backward asof，只检查数值有限，没有有效龄和可交易性掩码。

反例：STALE 1 月 3 日停止，LIVE 延续到 1 月 8 日，旧行情仍令 LIVE 排名为 0，板块方向 +0.16895；移除 STALE 后排名 0.5、方向 -0.299。

**改进**：区分历史成员资格、报价有效时间、市场时段、停牌和退出；跨市场闭市是否沿用必须显式定义，不能将 asof 自动等同于正确资产池。

### S15 P1 组合调仓未统一扣费且清仓费用可能消失

[组合收益:1098](/Users/chenqiang/Desktop/czsc/examples/multi_factor_cta/synthesis_engine.py:1098) 只是动态预算乘各单品种已扣费收益，没有根据最终标的仓位计算新增交易成本。

平价市场、4 个恒定策略仓位、40 天仅预算轮动，最终预算累计换手 64.35，报告收益仍为 0；10BP 单边成本对应还应计入 6.435% 的累计费用率（不含首次建仓，并非复利净值回撤）。另一反例：A 平仓日净收益 -10BP、日末 position=0，预算被设为零，平仓费用随之从组合收益消失。

**改进**：先形成最终有符号标的仓位，再由统一成交/收益账本计算毛收益与实际换手费用。不能通过动态乘已扣费策略收益代替成交回测。

### S16 P2 参数平原检查未成为准入条件且指标不符

[run_parameter_smoothness_audit:574](/Users/chenqiang/Desktop/czsc/examples/alpha_mining_agent/forensic_pipeline.py:574) 存在但主循环未调用；[入库:1067](/Users/chenqiang/Desktop/czsc/examples/alpha_mining_agent/run_evolution_loop.py:1067) 闯过原三关后直接保存。

内置路径读 is_mean_pct 而不是净 Sharpe，CV 默认 0.40 而非 Issue 0.35；原子因子的 scale 被丢弃，仅扰动一个衰减参数不足以覆盖核心窗口、阈值、止损。均值不变而邻域 Sharpe 全面坍塌的因子可得到“平原通过”。

**改进**：为每族注册有语义的核心参数，统一实际执行净收益指标；把检查结果和参数网格写入准入版本，失败不得入库。

### S17 P2 结构退出和收益口径没有统一

[hypothesis_bank.py:63](/Users/chenqiang/Desktop/czsc/examples/alpha_mining_agent/hypothesis_bank.py:63) 的 stop_plugin 只是字符串，没有实际调用；三重屏障仍是固定 ATR/时间。原子持仓在 [atomic_factors.py:922](/Users/chenqiang/Desktop/czsc/examples/multi_factor_cta/atomic_factors.py:922) 一律用确认时最近三根极值代替 Setup 对应的 FX/BI/ZS/箱体证伪点。

事件按 next-open 入场；持仓 Sharpe 用 close-to-next-close。屏障文档称双边费率，却 [只减一次 fee_rate:136](/Users/chenqiang/Desktop/czsc/examples/alpha_mining_agent/forensic_pipeline.py:136)；持仓收益按每次权重变化扣费，而 AssetSpec 明确它是单边费率。MFE/MAE 又先计入整个退出 bar 的高低点，再判断止损止盈，可能包含成交后的价格。

**改进**：区分“固定标尺诊断”和“真实执行验证”，不得混用结论。Setup 保留结构对象和证伪价格；统一成交时钟、进出成本与区间标签，对 OHLC 内先后顺序不明给出保守边界或使用更细数据。

### S18 P2 因子上下文缓存未覆盖真实输入

[atomic_factors.py:1015](/Users/chenqiang/Desktop/czsc/examples/multi_factor_cta/atomic_factors.py:1015) 缓存主要依赖长度、close、HTF 对象 ID 与部分截面指纹，遗漏其他 OHLCV、原生 score/笔状态，以及 HTF 原地内容变化。

同一行情先计算 DC01 得到 0 活跃，仅更新资金流信号列再算仍为 0；清除 attrs 后得到 249 根活跃。

**改进**：缓存绑定不可变数据版本与完整配置，或显式失效；覆盖同 close、不同高低价/成交量/score/HTF 内容的测试。

### S19 P2 新配权模块未形成统一正式调用链

[demo:338](/Users/chenqiang/Desktop/czsc/examples/multi_factor_cta/run_framework_demo.py:338) 仍用静态 EB 和 IS 优化；RollingEmpiricalBayesWeighter 只有定义/导出，生产示例没有调用。状态机默认 None。demo、监控及 50 商品脚本仍按名称或簇硬删除因子。

远程 HEAD 没有接入 ContinuousCtaAllocator.allocate_active_weights。这里不是导入错误，而是根本未接线；本地未跟踪文件也不能补足远程交付。种子初始化 [run_evolution_loop.py:664](/Users/chenqiang/Desktop/czsc/examples/alpha_mining_agent/run_evolution_loop.py:664) 记录 Gate 1 结果却不判断 passed，也不跑 Gate 1B；旧库没有统一重新准入。

**改进**：建立一个版本化研究入口，统一候选、种子和旧库的准入；让研究、回测、监控复用同一条因果计算链。

### S20 P2 最终死区失效且端到端测试丢失空头方向

[deadband:1087](/Users/chenqiang/Desktop/czsc/examples/multi_factor_cta/synthesis_engine.py:1087) 作用在组合杠杆缩放之前，[最终权重:1104](/Users/chenqiang/Desktop/czsc/examples/multi_factor_cta/synthesis_engine.py:1104) 又每天变化。deadzone=0.05 的反例中，20 日后基础预算换手为 0，最终仍有 144 笔小于 0.05 的微调，总换手 0.18206。

[端到端测试:644](/Users/chenqiang/Desktop/czsc/tests/unit/test_phase4_portfolio_synthesis_and_allocation.py:644) 将日均绝对仓位作为 positions 传入，倾斜模块因而把空头当多头。弱势 A/强势 B 的空头预算应为 0.3625/0.1375，传绝对值后变成 0.1375/0.3625。

**改进**：死区作用于风险约束后的最终可交易仓位，并明确风险减仓豁免；activation 和 signed exposure 分开传递。

## 逐条需求核对

### Issue 3 连续结构因子和六簇体系

| 要求 | 判定 | 实现与差距 |
|---|---|---|
| 六簇 × 五周期 | 标签层通过 | [hypothesis_bank.py:572](/Users/chenqiang/Desktop/czsc/examples/alpha_mining_agent/hypothesis_bank.py:572) 和 atomic 目录具备六簇及五周期；不能据此证明统计正交。 |
| 旧 D 日内制度迁移 | 通过 | 历史类别映射及账本迁移存在，执行模式独立。 |
| Rust 分类全面分发 | 通过 | [registry.rs:117](/Users/chenqiang/Desktop/czsc/crates/czsc-signals/src/registry.rs:117) 与 PyO3 共用分类映射，17 类含 strength。 |
| 高价值 K 线算子挂载 | 基本通过 | [hypothesis_bank.py:813](/Users/chenqiang/Desktop/czsc/examples/alpha_mining_agent/hypothesis_bank.py:813) 配置扩展，新增六个强度算子。 |
| pos/trader 状态研究入口 | 未完成 | [factor_engine.py:179](/Users/chenqiang/Desktop/czsc/examples/multi_factor_cta/factor_engine.py:179) 的 Position 路径仍返回全零，属旧 stub 未补全。 |
| 去除 Python 重复指标 | 部分 | SMA 已原生化，RSI/MACD/OBV 仍手算且 RSI 口径不同，见 Standards。 |
| FX 连续 Q | 实现存在 | Rust 使用真实分型、ATR 和多项混合强度；权重为经验设定，经济单调性未验证。 |
| BI 背驰 Q | 部分 | 有斜率、实体面积和极值检查，但质量下限 0.1 与需求 [0,1] 不同。min 比率意味着任一项衰减即可，不是双重衰减。 |
| ZS 回归及三买卖 Q | 实现存在 | 从真实 BI 构建 ZS，使用相对距离和回踩安全边际；有效性仍待真实数据验证。 |
| ATR 自适应 D 和结构断路 | 部分且有错 | ATR 衰减存在；完成笔方向、结构锚点问题见 S02、S17。 |
| Signal.score 0～100 | 基本通过 | Rust 写分、Python 解码和范围测试具备；代表量化评分，不代表校准概率。 |
| D 量价、E 波动压缩补齐 | 算子层通过 | strength_flow、strength_squeeze 及 D/E 因子存在。 |
| Setup 先于 Confirmation | 基本通过 | [two_stage_trigger:562](/Users/chenqiang/Desktop/czsc/examples/multi_factor_cta/atomic_factors.py:562) 要求先前 Setup；结构身份与失效语义仍不足。 |
| 零前视 | 不通过 | FC01 确定反例见 S01。 |
| 多空对称 | 部分工程证据 | 有镜像测试，但部分测试允许 0.5% 不一致，不能声称已证明全路径 100% 对称。 |
| Task 1～3 整体完成 | 部分 | 核心算子与目录已形成，状态链路、结构证伪、指标统一和因果边界仍需修复。 |

### Issue 5 因子评价和前置准入

| 要求 | 判定 | 实现与差距 |
|---|---|---|
| Tier 1 A/E/F，15～30 跨资产、60%、t≥2.5 | 部分且统计无效 | 阈值配置存在；池规模、真实归一化收益、独立性未满足，见 S07～S10。 |
| Tier 2 B/C，权益 75～80%、商品板块≥65% | 部分 | 仅 equity/commodity 粗分类，没有逐商品板块门禁，匹配不足还混池。 |
| Tier 3 D 及 5m/15m 同制度/流动性池≥70% | 部分 | [tier:77](/Users/chenqiang/Desktop/czsc/examples/multi_factor_cta/forensic_validator.py:77) 排除了 A/B/C/E/F 高频规则；没有交易制度和流动性分组。 |
| Tier 3 净 Sharpe 显著为正 | 未充分实现 | 平均池 Sharpe 阈值不等于显著性检验，也不能替代各池有效性。 |
| 所有入池标的先 ATR 与状态归一 | 正式入口不通过 | runner 调普通 Gate 1，摘要缺真实 ATR/IPW 事件数据。 |
| A 趋势标尺 + pos_fx_stop | 参数存在，结构退出缺失 | 字符串插件未执行，见 S17。 |
| B 反转/背驰的结构目标 | 部分 | 固定 ATR/时间标尺存在，没有完整结构目标和退出链。 |
| C 中枢回归到中轴 | 部分 | 以固定 ATR 代替真正中轴退出。 |
| D 量价短持有期 | 参数基本具备 | 仍缺交易制度和成本口径一致性。 |
| E 箱体止损与动态扩张目标 | 部分 | 固定 ATR 标尺不能替代特定箱体证伪点与动态止盈。 |
| F 高周期和截面条件 | 部分 | 参数存在，但真实截面未在入口接入，见 S11。 |
| Q 加权评价 | 实现有误 | 加权统计量与 bootstrap 不一致，有效样本量缺失，见 S09。 |
| Rank IC 1/3/5/10/20、ICIR、自相关和半衰期 | 输出存在 | 是单标的时序秩相关诊断；半衰期来自权重 AR(1) 持久性，不是 IC 预测能力衰减。重叠期限和 IS 回退也影响解释。 |
| MFE/MAE | 输出存在 | 退出 bar 路径先后、成本和实际退出一致性仍需修正。 |
| 参数 ±20%、保持率≥70%、CV<0.35 | 不通过 | 不在主准入链，内置指标和 CV 阈值不符，见 S16。 |
| Active-union 相关 | 部分 | 实现混合 Pearson 与余弦规则；可作去重启发式，但不是标准 Pearson，缺交叠时返回 0 不等于独立性已证。 |
| 强制前置 Gate 1B | 部分 | 新挑战者有入口，但默认降级，种子/旧库绕过，且上游数据错误。 |
| 旧因子库全量重审及 Graveyard | 部分 | 拒绝新候选的记录能力存在；没有对旧库全部重新准入的完整证据。 |
| Task 1～3 整体完成 | 不通过准入可信度要求 | 问题不是缺指标数量，而是统计对象、样本与主流程不一致。 |

### Issue 6 截面特征和软门控

| 要求 | 判定 | 实现与差距 |
|---|---|---|
| 风险标准化 cs_slope_rank | 部分 | 有 ATR 标准化动量与平均秩，缺历史活跃池/有效龄。 |
| 板块动量龙头的大级别连续趋势 | 部分 | 实际取排除自身后同级动量绝对值最大者，不是动量第一名的大级别趋势。 |
| 板块相对价格残差 Z-score | 近似替代 | 实际是累计 ATR 标准化步长差，须另行定义并验证均值回复。 |
| B Sigmoid 多空公式 | 公式通过、流程不通过 | 又硬删事件，并可重复门控，见 S12、S13。 |
| C 居中度 + 低 ER 增强 | 部分且偏离 | 居中度只用于 direction=0；实际多/空分别走单调公式。ER=0.9 时 C 多头 R=.5 门控 .777，R=.95 为 .996。 |
| A 趋势增强、震荡阻尼 | 工具层部分 | 有平滑函数，正式入口及真实增益未验证。 |
| B/C 在 Phase 2 前置绑定 | 正式入口缺失 | validator 单独支持，主挖掘没有真实截面。 |
| F01 龙头补涨/补跌，对应 FC02 | 部分且定义放宽 | 近 5 日变默认 5 根 bar，中枢下沿变近 10 根低点且允许破 0.5ATR，确认还可由普通突破替代。 |
| F02 极端残差加背驰，对应 FC03 | 部分且定义放宽 | Z 的绝对值门限从 2 改为 1.5；背驰可由距 MA20 超过 .45ATR 代理。 |
| F03 Top25% × 日线/60m 突破，对应 FC04 | 部分且定义放宽 | rank≥.70，即前 30%；注册覆盖 5m～1d。 |
| Task 1 因果/asof 测试 | 部分 | 已有截断测试遗漏首次 HTF 有效、过期资产和实际默认调用链。 |
| Task 2 50 商品 B/C OOS 重验 | 无有效验收证据 | 新测试只构造 8 标的并改写赢家/输家路径，可验证机制，不能证明真实经济效果。 |
| Task 3 F01～F03 通过 Phase 2 入库 | 无有效验收证据 | [测试:603](/Users/chenqiang/Desktop/czsc/tests/unit/test_phase3_cross_sectional_engine.py:603) 只验证输出字段，未要求 passed=True。 |

### Issue 7 合成和组合风险预算

| 要求 | 判定 | 实现与差距 |
|---|---|---|
| Rolling/Expanding EB，无未来样本 | 工具层部分 | 滚动函数取 [start,t)，但正式 demo 不调用；全链仍有 S03～S05。 |
| 70% 同质池先验 + 30% 局部 | 部分 | 固定收缩存在，缺滚动同质池先验生成和训练截止时间。 |
| 全面启用并去除人工删因子 | 未完成 | demo、监控、50 商品脚本仍按名称/簇硬删。 |
| 六簇合成 A/F/E/D/B/C 顺序 | 结构层通过 | 类与六簇入口存在；新类是旧类的显式别名子类。 |
| 滚动 active-union 残差回归 | 部分且默认不通过 | 有 causal 参数但默认全样本；因果分支并未严格限定 active-union 回归样本。 |
| 正交后滚动风险平价/最大 IR | 未完成 | [synthesis_engine.py:312](/Users/chenqiang/Desktop/czsc/examples/multi_factor_cta/synthesis_engine.py:312) 仍固定簇预算与经验混合。 |
| u2p 迟滞状态机 | 工具层部分 | 实现存在，默认不启用，正式调用也未明确启用。 |
| 截面强弱倾斜 | 工具层基本具备 | 方向调整有合理性；需防重复应用和绝对仓位误传。 |
| ContinuousCtaAllocator 接入 | 未完成 | 未见正式调用，远程 HEAD 未交付该连接。 |
| 动态活跃池 | 部分且默认有错 | 显式 positions 路径可用，缺省按当日收益推断，见 S04。 |
| ATR 逆波动率 | 可选能力 | demo 不传 ATR，风险预热又有 bfill。 |
| 板块 HRP | 简化分配 | 逆波动率 × 1/√K，没有相关矩阵、聚类和递归分配。 |
| 事前目标波动率 | 部分 | 用历史变动组合的实现波动率，不是当前目标仓位的 sqrt(w'Σw)；有 bfill，最低 .70 缩放限制去风险。 |
| 最终调仓死区 | 不通过最终仓位契约 | 缩放后重新产生微调，见 S20。 |
| Task 1 簇净值相关平均绝对值<.25 | 无对应验收证据 | 测试检查全样本拟合后的信号相关，未检查 OOS 扣费收益相关。需求本身也宜改为收益而非累计净值相关。 |
| Task 2 统一闭环 | 未完成 | 新模块存在，但生产调用与最终成交账本不统一。 |
| Task 3 权益 + 50 商品、IS/OOS、WF、双引擎、≥65%、10BP | 无有效验收证据 | 旧 50 商品脚本不是完整新体系验收；新增 mock 测试直接加正收益，且没有完整双引擎/WF结果。 |

## Standards 轴

### N01 P2 Rust 与 Python 指标统一要求尚未完成

[CLAUDE.md](/Users/chenqiang/Desktop/czsc/CLAUDE.md) 第一条与 Issue 3 Task 1 要求统一底层算法。[hypothesis_bank.py:721](/Users/chenqiang/Desktop/czsc/examples/alpha_mining_agent/hypothesis_bank.py:721) 仍用简单均值 RSI，而 [Rust calc_rsi:1029](/Users/chenqiang/Desktop/czsc/crates/czsc-signals/src/utils/ta.rs:1029) 为 Wilder 平滑；[Python MACD/OBV:754](/Users/chenqiang/Desktop/czsc/examples/alpha_mining_agent/hypothesis_bank.py:754) 仍有重复实现。

这是本次明确清理任务没有完成，不应将全部 Python 研究编排一概判为违规。应统一到共享 Rust 算子和明确参数，再增加两端等价输入验证。

### N02 P2 新信号命名违反仓库约定

[CLAUDE.md:193](/Users/chenqiang/Desktop/czsc/CLAUDE.md:193) 明确不再新增 V<日期> 后缀，六个 [strength 信号:121](/Users/chenqiang/Desktop/czsc/crates/czsc-signals/src/strength.rs:121) 仍采用该命名。修复时应考虑已有名称的兼容与迁移，优先级低于收益和因果缺陷。

### N03 P1 工程模拟测试不能承担真实收益验收

仓库要求真实全量数据、无正向漂移的经济验证和双引擎交叉检验。[Phase 4 测试:637](/Users/chenqiang/Desktop/czsc/tests/unit/test_phase4_portfolio_synthesis_and_allocation.py:637) 直接向收益加入 `0.0018 × |w| × (0.6 + 0.4 × |trend|)`，满仓每根额外 10.8～18BP，再断言 10BP 成本后的 Sharpe 达标。该测试还走默认全样本 GS。Phase 3 的人为赢家/输家路径同样只能验证机制。

Mock 用于工程测试符合规范；问题是把注入优势后的收益当作经济验收。应保留确定性的工程性质测试，将真实行情的成本、WF、双引擎报告作为独立验收。

交易边界也应补齐：[ExecutionMode:693](/Users/chenqiang/Desktop/czsc/examples/multi_factor_cta/synthesis_engine.py:693) 仅裁剪多空权重或 14:55 归零，不管理当日可卖库存、借券和结算。国内股票 ETF 的二级市场买入实施 T+1；因此将 510500 等行情直接套双向/日内权重，或将“截为纯多头”称为现货可执行，都缺必要执行条件。若 ETF 仅是信号参考、实际交易期货/期权，应明确映射并用实际工具价格和费用验证。[上交所 ETF 交易说明](https://www.sse.com.cn/assortment/fund/etf/question/)。

### N04 可维护性判断 重复目录和隐式处理链增加漂移风险

[hypothesis_bank.py:569](/Users/chenqiang/Desktop/czsc/examples/alpha_mining_agent/hypothesis_bank.py:569) 与 [atomic_factors.py:866](/Users/chenqiang/Desktop/czsc/examples/multi_factor_cta/atomic_factors.py:866) 双份维护因子元数据，以测试维持一致；缓存塞在 DataFrame.attrs，门控/先验/权重缺明确类型。这是重复数据与职责分散的设计判断，不单独作为收益错误。

可提取中立 factor_catalog，定义不可变 SignalFrame、EventTable、FoldConfig、RiskBudget 和 ExecutionLedger；共享算法下沉 Rust，Python 保留 IO 与编排。

### N05 P2 正式准入没有落实仓库研究 SOP 的基础门槛

[CLAUDE.md:312](/Users/chenqiang/Desktop/czsc/CLAUDE.md:312) 要求信号阶段 N≥30、胜率>50%、95% 置信区间下界为正；持仓阶段要求交易数≥30、R>1.5、安全边际>10%。[Gate 1:342](/Users/chenqiang/Desktop/czsc/examples/alpha_mining_agent/forensic_pipeline.py:342) 却允许日线 8、60m 12、其他周期 15/20 事件；[标尺配置:74](/Users/chenqiang/Desktop/czsc/examples/alpha_mining_agent/hypothesis_bank.py:74) 安全边际仅 3%～4.5%，bootstrap p 上限约 0.20～0.26，且没有等价的置信下界验收。

不同诊断可以使用不同阈值，但当前主循环用这些结果直接晋升因子，没有后续统一 SOP 验收来补足。应明确探索诊断与正式准入状态，并按仓库硬约束执行。分层借力可以是另行研究的方法，不能默认为放宽现有标准的授权。

## 改进顺序与验收标准

| 顺序 | 工作 | 完成标准 |
|---|---|---|
| 1 修复因果性 | S01/S03/S04/S05/S06；先修默认入口 | 任意未来数据扰动不改变已决策的信号、准入参数、仓位和预算；覆盖预热、缺失和 IS/OOS 边界。 |
| 2 建立真实事件与成交账本 | S02/S07/S09/S10/S12/S15/S17 | 保存事件结构、起止时间、实际仓位、成本；平价市场调仓只产生费用，清仓费用不消失；复制资产不虚增独立信息。 |
| 3 接通唯一研究链 | S08/S11/S13/S16/S19 | 实际入口先建截面、门控一次、全部候选/种子/旧库统一准入，再滚动合成和分配；缺必需输入直接不可准入。 |
| 4 修正边界和方法命名 | S14/S18/S20、N01/N02/N04 | 输入版本使缓存失效；历史有效资产池正确；最终死区有效；Rust/Python 口径一致，HRP/EB/残差名称与实现匹配。 |
| 5 重新做经济验收 | N03、四个 Issue 的真实数据任务 | 权益同质池与 50 商品池，2020～2023 训练、2024～可用的 2026 数据验证，滚动前推；全部筛选只在训练窗完成。 |

经济验收应按同一资产池、费用、杠杆和目标风险做逐层消融：简单固定预算基线 → 连续 Q/D → 截面门控 → 收缩 → 残差 → 动态风险预算。报告每层的扣费日收益、Sharpe/Calmar、换手、有效样本、板块风险、尾部共同亏损及配对时间块置信区间。70% 收缩、排名门限、簇预算与基底顺序作为训练参数冻结，不能依据 OOS 再筛选。

双引擎应核对逐笔/逐日收益与费用差异，并明确数据调整、换月、交易时段、乘数、保证金、最小下单单位以及可卖/可空约束。因子评价与执行尽量复用仓库 Position/Event 路径；纯组合调仓可用权重回测，但必须与实际执行时钟一致。

## 验证记录

本次已成功重建原生扩展，确认新增 strength 分类可用，避免沿用旧扩展造成误判。Rust 信号 crate 测试 **54 项通过**，文档测试 0 项。

Python 验证覆盖以下五个文件，**87 项全部通过，耗时 77.42 秒**：

- test_signal_dispatcher_categories.py
- test_six_cluster_atomic_factors.py
- test_phase2_forensic_pooled_validator.py
- test_phase3_cross_sectional_engine.py
- test_phase4_portfolio_synthesis_and_allocation.py

上述 S01～S10、S12～S15、S18、S20 中的数值反例来自本次及本轮前序审核的直接调用；行情类反例以 czsc.mock 生成数据，必要时控制结构信号/标签以隔离待验证行为。它们证明具体契约失效，不用来推断真实市场收益。

本次审核没有重跑真实权益池和全部 50 商品的多年回测，因此不报告真实投资收益是否达标。即使现有工程测试全部通过，也不能消除已复现的默认路径、统计样本和费用问题。

Spec 轴记录 20 项问题，最严重的一组直接破坏因果性和准入可信度；Standards 轴记录 5 项，其中 4 项为规范要求、1 项为维护性判断，最高风险是经济验收使用注入收益且缺真实执行证据。

## 第一批修复记录（2026-10-17）

本批优先修复因果性、准入统计、实际仓位成本，并验证正式研究入口。未提交或推送。用户原有的 Rust 信号、顶层导出、`cta_allocator.py`、研究脚本和策略库改动均保留。本批没有修改 Rust 源码或 Python 公共 native API。

| 审核项 | 本批结果 |
|---|---|
| S01 / S03 / S04 / S05 | 正式研究链改为逐时点 HTF 可用性、默认历史窗口正交化、显式有符号仓位和历史风险估计；删除主链的风险后向填充。跨周期权重统一按实际时间向后对齐，移除长度整除推断。高周期选择、换月识别、未来上市资产也加入前缀不变性验证。 |
| S02 | 修正完成笔方向的解释和退出行为；独立、带结构 ID 的 Rust 完成笔事件 API 尚未新增。 |
| S06 | IS 不足拒绝准入；屏障事件必须在训练边界内完成；Gate2、参数检查、重配权不使用越界收益。 |
| S07 / S08 | 只接受包含起止时间、实际 ATR 收益、净收益、Q、状态的事件表；取消摘要造样本、跨同质池替代和单资产自动通过。15/5 个资产的分层最低池规模不可下调，微观结构池须有明确元数据。 |
| S09 / S10 | Q 加权观测与随机基准一致，检验有效样本量；按同步时间块计算加权比率估计量的 HAC 标准误，保留状态均衡目标。复制同一时间的资产不增加独立证据；有效正向事件池仍能通过。 |
| S11 / S12 / S13 | 正式入口先批量构建截面再算信号；F 类缺真实截面明确拒绝。门控保留触发事件，由 Registry 应用一次，下游显式关闭；参数检查也避免读取 attrs 再叠加一次。 |
| S14 / S18 | 截面使用报价有效龄与历史成员掩码；缓存覆盖 OHLCV、品种、配置、实现、原生扩展和外部上下文内容。 |
| S15 / S20 | 根据最终有符号仓位扣开仓、调仓、反手及清仓费用；死区置于最终仓位层，减风险优先。主分配器及 50 商品全部 10 个比较方案都让缺报价仓位冻结并占用预算，无法交易的预算超额单独输出。 |
| S16 | 正式准入必须通过参数平原；使用实际生效的衰减参数和扣费 Sharpe，CV 上限为 0.35。 |
| S17 | 已统一主研究、单资产报告、组合输入和重配权的下一根开盘成交口径，补末次清仓及多空跳空穿止损费用/价格处理。真正结构止损锚点和 stop_plugin 完整闭环仍待完成。 |
| S19 | 种子、旧库、新候选共同准入；demo/monitor/50 商品及主要历史调用复用共同准备与审计入口；滚动权重、因果合成与状态机接入多资产入口，IS 选择不回填成可交易 IS 业绩。完整 Rust Position 路径仍未实现，现明确抛错。 |
| N03 / N05 | 删除人工注入优势收益的盈利验收；恢复 N≥30、有效 N、胜率、95% 置信下界、R、超过 10% 安全边际等基础门槛。Mock 仅用于工程反例，真实多年收益验收未宣称完成。 |

额外复核修复了状态 IPW 在时间块求均值时被抵消、冻结仓位重复占用预算，以及 Gate2 救援收益、滚动配权和快速重配权沿用旧成交时钟的问题。均增加了独立数值回归；滚动历史不虚构样本末尾清仓，完整回测则计入终点清仓。

### 入口与执行契约

- `run_evolution_loop.py --validate-only --audit-unit FC02L@30m` 运行真实加载、截面、信号及准入，写入 `entrypoint_validation.json`，不修改因子库。`--pool-metadata` 接受每个品种的 `asset_group`、`sector`、`microstructure_group`、`fee_rate`。
- 完整单资产主循环在无合格因子时正常输出空库和报告；不再强制写入基石种子。完整多资产 demo 同样允许全部拒绝，报告实际准入数及原因。
- `build_asset_execution_inputs` 默认把收盘信号移到下一根开盘执行，末尾使用最后已知收盘价清仓。组合要求原始价格收益、有符号执行仓位、单边费率和显式时钟；`asset_weights_df` 已是最终仓位，不可再次乘信号。`budget_weights_df` 是独立预算。
- 实时监控使用当前已知收盘信息计算待执行目标，不将这个目标快照当作已经按下一根开盘成交的业绩。
- 9/14 资产和主要商品比较入口已迁移；Top24 比较不再使用事后赢家名单或固定绩效数字，改用 IS 已知历史覆盖选择子集，并保存选择截止时间。

### 验证与尚未完成的部分

最终验证：17 个相关测试文件 **204 项全部通过，耗时 90.73 秒**。其中保留原有五文件的 87 项覆盖，新增 12 个文件、117 项回归；修正旧测试的输入契约，没有通过放宽因果性或费用断言取得通过。修改过的 33 个 Python 文件全部通过 Ruff 检查及格式检查，`git diff --check` 通过。

测试覆盖实际入口的加载至报告路径，以及正向准入、错误准入拒绝、未来扰动、时间边界、费用与冻结预算；下一根开盘执行账本还与现有 WeightBacktest 对照，按其收益舍入精度检查总损益。完整主循环无合格种子、多资产 demo 全拒绝、只验证模式保留现有因子库均通过。控制收益/结构的 Mock 场景只验证工程契约。本轮未改 Rust 源码，因此未重复构建扩展或运行 Rust crate 测试；前述首次审核的 Rust 验证记录不冒称为本批新执行结果。

仍需后续独立工作：N01 的共享指标 Rust 统一、N02 信号命名迁移、N04 目录与数据契约整理；真正结构止损/Position 执行链；ETF 实际可卖可空与交易制度约束；完整 HRP/经验贝叶斯方法校准、搜索选择偏差和外层 Walk-Forward。旧的 20/32 商品及 Plan A/B 等一次性实验脚本尚未整体迁移，仍可能含后向填充与净收益再配权，不能沿用其历史经济结论。它们不是本次通过验证的正式入口。

本批未运行真实权益池、全部 50 商品的多年重新筛选或 Backtrader 双引擎经济验收。修复后的数据门禁可能让原冠军库全部拒绝，这应作为证据不足的结果保存，不能通过降低阈值恢复旧曲线。


## 第二批修复：结构退出、跨语言一致性和旧入口迁移

本批继续保留已有工作区改动，未提交或推送。Rust 核心负责新增算法，Python 研究层只做行情与数组边界转换、参数编排和结果表关联。

| 审核项 | 本批实现 |
| --- | --- |
| S02 / S17 | 原生逐时点结构快照提供 FX、完成 BI、ZS 的价格/ID与确认事件，快照绑定输入时刻；反向退出不再从未完成笔文本推断。Setup 冻结最先有效结构，Confirmation 严格在后；所有因子通过统一 Rust 状态机执行结构破位、反向完成笔、1.3% 收盘跟踪、衰减和超时退出。风险退出不能被同一根旧 Setup 再次开回原方向抵消。 |
| S17 插件闭环 | A 使用已确认 FX，B 使用完成 BI 外沿，C 使用原中枢外沿（旧 `pos_zs_mid` 名称保持兼容，中轴是回归目标而非初始止损），D 使用 Setup 资金流 K 线极值，E 使用 Setup 时已完成箱体，F 使用当时已收盘 HTF 均线；缺结构不武装。 |
| S17 收益口径 | 新增原生 `execution_episodes` / `execution_net_returns`，真实连续仓位的缩放、反转双腿、退出全部扣费；只统计已完成交易，不把同向刷新当成新独立样本，也不在训练边界补造清仓。固定 ATR 保留为明确的诊断项，正式准入改用实际执行、实际事件池化与参数平原。OHLC 止损/止盈无法确定先后时使用保守路径边界。 |
| S19 Position 路径 | 新增共享 `Position.evaluate_signals`，研究注册器调用原生 Event/Position；从空仓配置副本回放，保留 T+1、时间与标的校验，避免研究污染模板运行状态。共享研究行情转换支持 OHLCV、保留已有成交额；批量回放使用连续编号，避免旧输入编号全为 0 导致超时不执行。 |
| N01 | RSI / MACD 共享 Rust TA 核心，研究侧不再独立计算；标准 OBV 也使用 Rust。MACD 完整预热、柱线不倍增，RSI 为 Wilder。历史实体方向资金流信号保留原语义并与标准 OBV 区分。 |
| N02 | 六个 strength 新默认调用名无日期后缀，宏支持无日期命名，历史名称为相同输出的兼容入口；反解析稳定选新名称，已有信号键和 Event 配置不被改写。 |
| 旧实验入口 | 20/32 商品、加权比较、Plan A/B 与旧图表、沪银、三个权益参考价格迁移实验及 TD walkthrough 复用共同数据准备、强制准入和最终成交账本。移除重复净收益再缩放和风险后向填充；品种子集只由 IS 历史覆盖选择，导入不执行回测。 |

执行语义：结构破位和跟踪止盈采用**收盘确认、下一开盘执行**。这不是盘中挂单止损，也不承诺止损价成交。快照的结构 ID 是结构终点，确认事件所在输入行才是可知时间。退出原因和冻结锚点保留用于复核。F 簇没有可用 HTF 均线时不产生持仓，最高周期也不回退到本周期极值。

来源一致性也纳入回归：结构快照与信号使用同一环境配置，中枢回归的 Setup 与止损共同使用最近 20 根完成笔。配置变化会使研究缓存失效。输入 OHLCV 无成交额时由共享 Rust 边界显式记为 0（未知）；已有成交额保持原值，空表返回空结果，不修改原始 DataFrame。

镜像验收修正：仿射价格镜像保持结构价差，却不保持百分比回撤。因而分别检查 Setup / Confirmation、质量和结构止损的镜像关系，以及多空两侧相同的 1.3% 跟踪阈值；不再以最终持仓逐值反号作为百分比风控的验收条件。正式风控阈值未因测试而放宽。

严格镜像回归进一步复现并修复了共享核心的两处边界错误：初始两根互相包含时，仅按最高价判断方向会使原序列和镜像序列同时被判为下行；十字星被固定按阳线合成，会让同一分型的镜像评分从 37 变为 16。现在仅在双极值明确同向时合并，十字星的开收价保持相等并限制在合并区间内。两端都有内包、外包、并列极值和十字星回归，没有放宽镜像容差。

历史核心快照另行核实后作有限维护：原导入实现固定 `min_bi_len=6`、默认 `max_bi_num=50`，旧测试却受 `tests/__init__.py` 的环境配置影响。现显式锁定参数，并增加大小写环境变量隔离回归。初始化包含修复使 FX 从 175 变为 173、首笔长度从 8 变为 6；43 笔方向以及后 42 笔长度逐项不变。快照保存了原来源、配置和修订证据，未盲目重生成预期结果。

迁移影响：旧库候选 ID 可以读入，但须重新准入；旧指标、旧仓位、旧曲线不能作为新实现的验收结果。权益 ETF/指数参考价格实验明确为因子诊断，尚不能当作对应实盘工具的可卖、可空与库存约束证明。Mock 回归验证工程合同，不构成真实策略收益证据。

### 本批最终验证

- 在全部 Rust 源码修复后重新构建并安装 release 原生扩展，再运行 Python 回归；同时重新生成类型声明。
- Rust：`czsc-core` 156 项通过；`czsc-ta / czsc-signals / czsc-trader / czsc-signal-macros` 合计 108 项通过，总计 **264 项通过**，另有 1 项既有文档测试忽略。
- Python：**31 个文件、365 项全部通过，耗时 107.93 秒**。覆盖第一批因果性、准入统计和组合成本回归，以及结构快照/止损/退出、包含镜像、指标共享、原生 Position、OHLCV 边界、真实执行准入、旧实验迁移、公共 API、核心基线和用户已有的 CTA 分配器/斜率信号测试。
- 质量检查：59 个修改过的 Python 文件通过 Ruff 检查与格式检查；`cargo fmt --all -- --check`、`cargo clippy --workspace --all-targets -- -D warnings`、`git diff --check` 全部通过。生成的 stub 未手工编辑。

正式入口验收覆盖真实调用链：`--validate-only` 从 OHLCV 加载到准入报告并保留已有研究库；完整研究循环拒绝不合格种子；多资产入口允许全部拒绝并生成 HTML 与逐资产准入记录。12 个旧实验入口统一调用这些流程，导入模块不会自动运行研究。没有沿用历史冠军系数或固定收益曲线来证明迁移成功。

### 仍需独立完成的审核工作

N04 的统一因子目录与显式研究数据类型整理尚未完成。完整 HRP/经验贝叶斯方法校准、搜索选择偏差控制和外层 Walk-Forward 仍须独立研究；当前简化方法不能据此升级命名或宣称统计有效。ETF 实际交易工具的可卖、可空、库存与结算约束仍须用真实工具验证。

本批未执行真实权益池、全部 50 商品的多年重新筛选及 Backtrader 双引擎经济验收。所有工程回归通过只说明实现契约受控，不说明修订后的策略收益达标；旧研究结果须重新计算和准入。已有本地改动和未跟踪文件继续保留，本批未提交或推送。

## 第三批验证：真实数据重跑与双引擎核对

后续已使用修复后的正式入口重新执行真实数据研究，详见 [真实数据重跑报告](/Users/chenqiang/Desktop/czsc/docs/research/factor_system_real_data_rerun_20261017.md)。以上“未执行真实数据重跑”是前两批的历史状态，不代表本轮最终状态。

- 日线 61 个候选 × 60 个资产，共 3,660 项检查；日内 260 个候选 × 52 个资产，共 13,520 项检查。合计 17,180 项，最终获准为 0。日内权益有 11 项通过单资产执行门禁，但均未通过池化；日内商品缺少足够 IS 历史。不能将全部拒绝解释为所有因子都已被证明无效。
- 两阶段各 10 个正式商品组合均为空仓，收益、成本和换手为 0；50 个日线商品与 47 个日内商品的实际 Backtrader / CZSC WeightBacktest 逐资产重放及总账核对完成。空仓一致不构成经济验收通过。
- AG/CU/RB/M 固定非零控制的毛收益在双引擎逐 bar 对齐，但新增确认了费用模型差异：权重差计费未计入固定初始名义资本下的价格漂移再平衡。额外增加 2 bp 费用时，日线、日内净收益相对差分别为 6.98% 和 10.05%，超过 5% 门槛；成本一致性验收未通过。完整收益、成本、订单与五档压力结果均已归档。
- 新增及相关 Python 回归共 100 项通过；本轮未修改 Rust 源码，不把前批 264 项 Rust 测试冒称为重新运行。数据来源、时间修正、源码及 native 哈希、窗口与费用假设已记录。

剩余优先事项为统一实际数量费用、补齐日内训练历史和真实合约/交易日历/微结构元数据、继续迁移旧 Python 规则；外层 Walk-Forward、历史资产池成员资格、分配方法校准及 N04 仍需独立完成。已有本地改动和未跟踪文件保留，本轮没有提交或推送。

## 第四批修复：固定资金与实际成交费用统一

详见 [费用统一与双引擎重放报告](/Users/chenqiang/Desktop/czsc/docs/research/factor_system_cost_alignment_20261017.md)。本批修复第三批发现的费用差异，保留原研究产物与已有本地改动。

- 共享 Rust 账本统一固定初始资金、每个有报价时点的数量再平衡，以及按实际成交数量计费。正式准入执行、单资产报告、滚动配置、Gate 2、组合及全部 10 种分配方案接入；缺报价冻结、恢复、反手和清仓均纳入两端回归。
- 实际外部 WeightBacktest 以零费独立核对毛收益，标准化结果使用共享 Rust 费用；Backtrader 独立按订单数量计费。保持明确来源区分，没有声称修改了外部 WBT 费用模型。
- 原冻结真实数据完成 **485 次正式资产 + 10 次非零控制双引擎重放**，五档费用均通过逐时点与累计绝对误差校验；全部 20 套组合另完成 **100 项 Rust 费用压力测试**。新目录与原目录独立，225 个输入文件及记录的实现哈希在重放后复核未变。
- 基础单边 1.5 bp 下，非零控制共同净收益为日线 **3.109967%**、日内 **4.573224%**，成本分别为 **0.634444%**、**5.569264%**。日线基础加 10 bp、日内基础加 2 bp 起出现净亏损，工程对齐不等于经济验收。
- Rust 核心 **165 项通过**，Python 四组 **180 项通过**；release 扩展重建、类型声明生成、格式及静态检查完成。独立产物复核确认主组合完整面板与分资产账本逐项相等。

本轮没有重新拟合或准入。第三批的 17,180 项检查与零获准是旧费用口径的历史证据，不能当作新口径重估后的结论。外部 WBT 0.2.1 的高精度毛收益限制另已复现并纳入严格拒绝回归；当前简单冻结权重不受影响。日内训练数据、真实合约与日历、非零正式组合保证金验证及其他方法论问题仍未完成。本批未提交或推送。
