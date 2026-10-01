# CTA 策略开发标准流程

本文档面向 CZSC 1.0 之后的项目结构。缠论核心、信号注册、交易器和策略回测主路径由 Rust 实现；用户通过 `czsc.*` 公共 API 调用。策略代码负责声明规则、数据接入、研究编排和结果分析；新增共享算法必须遵循 [CLAUDE.md](CLAUDE.md) 的 Rust/Python 行为一致约束。

标准流程是：

```text
策略规格 -> 数据准备 -> 信号验证 -> 事件/持仓配置 -> 回测/绩效分析
         -> 参数冻结与样本外验证 -> 执行压力测试 -> 归档验收
```

公共交易接口的主路径是：

```text
Signal -> Event -> Position -> CzscTrader
```

能用 `Signal -> Event -> Position` 表达的策略优先使用该体系回测，随后可用 `WeightBacktest` 分析持仓权重。纯权重时序或组合调仓研究可直接走权重回测，但必须声明收益、成交与成本口径。

下文代码片段按顺序构成一个 30 分钟 mock 数据示例，用于验证 API 与产物链路；第 1 节的 RBreaker 是规格示例，单独的聚宽研究脚本需要外部数据权限。

## 1. 明确策略规格

开发策略前，先把规则写清楚，不要直接进入编码。

需要明确：

- 交易标的：指数、ETF、股票、期货、数字货币等
- 基础周期：`1分钟`、`5分钟`、`30分钟`、`日线` 等
- 是否多周期联立：例如日线定方向，5分钟找入场
- 开仓逻辑：什么信号组合触发开多 / 开空
- 平仓逻辑：反向信号、固定时间、止损、止盈、超时等
- 风控规则：是否 T0、是否隔夜、单日最多交易次数、冷却间隔
- 回测区间：`bar_sdt`、`sdt`、`edt`
- 数据源：本地缓存、聚宽、Tushare、天勤、CCXT 等
- 执行口径：信号何时可知、使用哪个价格成交、是否允许做空、手续费和滑点如何扣除
- 验证方案：训练期、验证期、最终测试期，以及样本外与压力测试的验收标准；在调参前固定

RBreaker 规格示例：

```text
标的：000852.XSHG
基础周期：5分钟
开仓：RBreaker 突破价位触发趋势开仓
平仓：14:55 强制平仓
隔夜：不允许
单日重复开仓：不允许
```

规格中的次数、隔夜与定时平仓约束必须落实到信号/事件或执行规则，并通过交易记录核验。`interval` 是同类型开仓间隔，不等同于“每天只开一次”；`t0` 也不等同于隔夜控制。

## 2. 准备数据

信号生成接口使用 `list[RawBar]`；策略 `backtest` 支持 RawBar 列表、DataFrame 或 Arrow IPC 字节，`run_research` 接受 DataFrame 或 Arrow IPC 字节。使用 DataFrame 时，标准字段是 `symbol / dt / open / close / high / low / vol / amount`；通过顶层 API 转换成 RawBar 列表。

完整 mock 示例先准备早于正式回测起点的数据：

```python
from czsc import CZSC, Freq, format_standard_kline
from czsc.mock import generate_symbol_kines

BAR_SDT = "20230101"
BACKTEST_SDT = "2024-01-01"
EDT = "20240601"
INIT_N = 500

df_bars = generate_symbol_kines("000001", "30分钟", BAR_SDT, EDT, seed=42)
bars = format_standard_kline(df_bars, freq=Freq.F30)
assert len(bars) > INIT_N
c = CZSC(bars)
```

时间与预热约定：

| 配置 | 含义 |
|---|---|
| `bar_sdt` / `BAR_SDT` | 数据获取起点，是研究脚本变量；应早于正式运行起点，供指标和结构预热 |
| `sdt` / `BACKTEST_SDT` | 信号或策略正式运行的目标起点；各入口的边界与预热规则需分别核对 |
| `edt` / `EDT` | 数据获取终点；由数据源或脚本裁剪，不是 `backtest` 的通用关键字参数 |
| `init_n` | `generate_czsc_signals` 的预热根数，默认 500，按输入基础周期计数 |
| `include_sdt_bar` | 策略研究的起始边界选项；默认 False，日期预热分界按 `dt <= sdt` / `dt > sdt` 划分 |

`generate_czsc_signals` 按日期和 `init_n` 切分数据；数据不足或起点超出范围时存在边界处理，不能把传入 `sdt` 当成实际输出起点保证。策略研究也有预热回退规则，且 Trader 状态信号的预热路径有所区别。两类入口不能仅凭相同 `sdt` 假定完全对齐，必须检查实际输出时间范围、行数和非 `其他` 信号分布。500 根只是默认值，应为所需的大周期和指标窗口预留足够历史。

真实数据需检查时间递增、同品种同周期无重复、OHLCV 有限值、交易日和夜盘归属、复权或连续合约规则。大周期正在形成的 K 线与已收盘 K 线需区分，不得把最终大周期结果回填到更早的小周期时点。

如果使用聚宽 JQData，先安装可选依赖：

```bash
uv sync --extra jq
```

或临时运行：

```bash
uv run --extra jq python examples/signals_dev/backtest_rbreaker_000852_5m.py
```

上述研究脚本使用 JQData SDK，账号密码通过环境变量读取；`czsc.connectors.jq_connector` 的 HTTP token 路径是另一套配置入口，使用时以对应代码为准：

```text
JQDATA_USERNAME=...
JQDATA_PASSWORD=...
```

不要把真实账号密码提交到 Git。

## 3. 选择或开发信号

CZSC 1.0 后，信号函数主体位于 Rust：

```text
crates/czsc-signals/src/
```

常见模块：

- `bar.rs`：原始 K 线特征、涨跌停、时间段、RBreaker 等
- `cxt.rs`：分型、笔、中枢、一二三类买卖点等缠论结构
- `tas.rs`：均线、MACD、KDJ、BOLL、RSI、ATR 等技术指标
- `vol.rs`：成交量特征
- `zdy.rs`：自定义指标
- `pos.rs` / `cat.rs` / `cxt_trader.rs` / `zdy_trader.rs`：依赖 Trader 或持仓状态的信号

新增信号应在 Rust 侧实现，并用 `#[signal(...)]` 宏注册。Python 侧不要再新增 `czsc/signals/*.py` 形式的信号函数。

优先通过公共 CLI 查询已编译扩展中的信号目录与参数模板：

```bash
uv run --no-sync czsc signals list --json
uv run --no-sync czsc signals doc cxt_bi_status_V230101 --json
```

下文带 `V...` 后缀的名称和事件字符串仍是当前注册表中的有效历史契约，不要直接去掉后缀。新增名称遵循仓库命名规范，并以注册表和参数模板核对。

查看当前信号模块速查：

```bash
uv run --no-sync python scripts/dump_signal_catalog.py
```

查看详细信号元信息：

```bash
uv run --no-sync python scripts/dump_signal_details.py --format json
```

## 4. 验证单个 K 线信号

用户代码可通过公共 `generate_czsc_signals` 指定一个信号并查看输出：

```python
from czsc import generate_czsc_signals

single_config = [{"name": "bar_r_breaker_V230326", "freq": "30分钟", "params": {}}]
df_single = generate_czsc_signals(
    bars, signals_config=single_config, sdt=BACKTEST_SDT, init_n=INIT_N, df=True
)
print(df_single.tail())
```

带参数的例子：

```python
macd_config = [
    {"name": "zdy_macd_bc_V230422", "freq": "30分钟", "params": {"di": 1, "th": 50}}
]
df_macd = generate_czsc_signals(
    bars, signals_config=macd_config, sdt=BACKTEST_SDT, init_n=INIT_N, df=True
)
print(df_macd.tail())
```

内部 K 线分发器位于 `czsc._native.signals`，依赖 Trader 或持仓状态的信号不能走 K 线单次分发路径。跨周期信号通过相应运行时配置调度；依赖真实持仓的信号需放入包含 Position 的交易器或策略研究路径验证。

还需核对信号算法与策略规格是否一致：例如内置 `bar_r_breaker_V230326` 使用当前周期的前一根 K 线 HLC，第 1 节所引用的日内 RBreaker 研究脚本使用前一交易日日线 HLC。不能因名称相同就把两者视为等价规则。

## 5. 批量生成信号

研究和特征工程中，使用 `generate_czsc_signals`：

```python
from czsc import generate_czsc_signals

signals_config = [
    {"name": "cxt_bi_status_V230101", "freq": "30分钟", "params": {}},
    {"name": "bar_zdt_V230331", "freq": "30分钟", "params": {"di": 1}},
]

df = generate_czsc_signals(
    bars,
    signals_config=signals_config,
    sdt=BACKTEST_SDT,
    init_n=INIT_N,
    df=True,
)
assert not df.empty
print(df["dt"].iloc[0], df["dt"].iloc[-1], len(df))
```

流式场景中，使用同一份 30 分钟数据，并先预热 BarGenerator：

```python
from czsc import BarGenerator, CzscSignals

bg = BarGenerator(base_freq="30分钟", freqs=["30分钟"], max_count=5000)
for bar in bars[:INIT_N]:
    bg.update(bar)
cs = CzscSignals(bg, signals_config=signals_config)

for bar in bars[INIT_N:]:
    cs.update_signals(bar)

print(cs.s)
```

这里的流式示例只演示初始化与更新，不按 `BACKTEST_SDT` 对齐批量输出。若要做批量/流式一致性比较，需使用相同的预热切分、配置和输出时间范围。使用 1 分钟基础周期时应另取 1 分钟数据；BarGenerator 只能合成相应的大周期，不能把 30 分钟数据拆成 1 分钟数据。

## 6. 验证信号

信号验证重点：

- 对约定了默认状态的信号，非目标状态是否稳定返回 `其他`；完全分类信号按各自模板检查
- 目标状态是否不漏触发
- 触发频率是否合理
- 信号 key / value 是否符合事件匹配规则
- 多周期信号是否对齐正确
- 同一历史前缀在线计算与分段回放的结果是否一致；关键状态改变后已有历史决策是否被未来数据改写
- 不同参数是否造成大量 `其他`、错误字段或静默未调度

推荐验证方式：

```python
sig_cols = [c for c in df.columns if len(c.split("_")) == 3]
assert sig_cols

for col in sig_cols:
    print(col, df[col].value_counts().head(10))
```

需要图形检查时，优先使用 lightweight HTML：

```python
from czsc.utils.plotting.lightweight import plot_czsc_signals
from pathlib import Path

OUT_DIR = Path("docs/examples/_output/cta_workflow")
OUT_DIR.mkdir(parents=True, exist_ok=True)

plot_czsc_signals(
    bars,
    signals_config=signals_config,
    path=OUT_DIR / "signals_check.html",
    sdt=BACKTEST_SDT,
    init_n=INIT_N,
)
```

## 7. 组合事件和持仓

事件由信号组合形成。信号完整字符串格式通常是：

```text
{k1}_{k2}_{k3}_{v1}_{v2}_{v3}_{score}
```

以下事件示例用于 30 分钟笔表里关系策略，和第 1 节的 RBreaker 规格分别说明：

```python
from czsc import Event, Position

open_long = Event.load(
    {
        "name": "30分钟表里向上开多",
        "operate": "开多",
        "signals_all": ["30分钟_D1_表里关系V230101_向上_任意_任意_0"],
        "signals_not": ["30分钟_D1_涨跌停V230331_涨停_任意_任意_0"],
    }
)

open_short = Event.load(
    {
        "name": "30分钟表里向下开空",
        "operate": "开空",
        "signals_all": ["30分钟_D1_表里关系V230101_向下_任意_任意_0"],
        "signals_not": ["30分钟_D1_涨跌停V230331_跌停_任意_任意_0"],
    }
)

def build_position(symbol: str) -> Position:
    return Position(
        name="30分钟笔非多即空",
        symbol=symbol,
        opens=[open_long, open_short],
        exits=[],
        interval=3600 * 4,
        timeout=16 * 30,
        stop_loss=500,
        t0=False,
    )
```

事件通过 `signals_all / signals_any / signals_not` 声明组合关系。先用 `get_signals_config` 验证字符串能被解析，再检查产出的 key/value 与事件条件匹配。上述“涨跌停”信号是其模板定义的技术状态，不能替代实际成交约束校验。

Position 参数必须写明单位和策略含义：

| 参数 | 当前含义 | 示例解释 |
|---|---|---|
| `interval` | 同类型开仓事件的间隔，单位秒 | `3600 * 4` 为 4 小时，不是 4 根 K 线，也不是每日次数上限 |
| `timeout` | 最近一次开仓事件之后的基础周期 K 线数量限制 | `16 * 30` 为 480 根 30 分钟 K 线，不是 480 分钟 |
| `stop_loss` | BP 阈值，以最近一次开仓事件触发价格为基准 | `500` 对应 5%；不是 500 元或 500% |
| `t0` | 是否允许日内平仓，Python 参数名为小写 | 示例 False；日内策略需显式确认设为 True |

`exits=[]` 不代表永不平仓：反向开仓事件、止损、超时仍可能改变持仓，且受 `interval / t0` 等限制。检查真实交易记录；止损是按更新价格检查的规则，不保证跳空或无法成交时恰好在阈值成交。

## 8. 组织策略类

推荐继承 `CzscStrategyBase`，只实现 `positions`：

```python
from czsc import CzscStrategyBase, Position


class MyStrategy(CzscStrategyBase):
    @property
    def positions(self) -> list[Position]:
        return [build_position(self.symbol)]
```

框架会自动派生：

- `base_freq`
- `freqs`
- `signals_config`
- `unique_signals`

用实例的 `self.symbol` 构造持仓，避免多个标的共享可变的全局 Position。此示例的周期固定为 30 分钟；若周期也可配置，事件字符串和输入数据周期必须一起修改。

完整示例见：

```text
docs/examples/07_strategy_backtest.py
examples/intraday_0935_direction_strategy.py
```

## 9. 回测和回放

内存回测：

```python
tactic = MyStrategy(symbol="000001")
res = tactic.backtest(bars, sdt=BACKTEST_SDT, include_sdt_bar=False, emit_signals=True)
signals = res.signals_df()
pairs = res.pairs_df()
holds = res.holds_df()
assert not holds.empty
print(res.meta)
print(signals.shape, pairs.shape, holds.shape)
print(holds["dt"].min(), holds["dt"].max())
```

回放落盘：

```python
replay_res = tactic.replay(
    bars,
    res_path=OUT_DIR / "replay",
    sdt=BACKTEST_SDT,
    include_sdt_bar=False,
    emit_signals=True,
    exist_ok=True,
)
```

`exist_ok=True` 允许在已有目录重新执行；每次正式实验应使用独立运行目录。`refresh=True` 会先删除输出目录，只有明确要清理旧结果时才使用。结果返回 Arrow IPC 字节流及可选文件路径，优先通过 `signals_df() / pairs_df() / holds_df()` 读取。不要把“回测运行成功”当作策略通过验收。

1.0 之后不再使用旧的 `CTAResearch` 入口。研究入口统一看：

```python
from czsc import run_research, run_replay, run_optimize_batch
```

策略对象自身的入口是上述 `tactic.backtest` 和 `tactic.replay`。

## 10. 保存和复用策略

`CzscStrategyBase` 支持保存 / 加载持仓配置：

```python
position_dir = OUT_DIR / "positions"
tactic.save_positions(position_dir)
```

重新加载：

```python
from czsc import CzscJsonStrategy

files_position = sorted(str(path) for path in position_dir.glob("*.json"))
assert files_position, "没有可加载的 Position JSON"
loaded_tactic = CzscJsonStrategy(
    symbol="000001",
    files_position=files_position,
    check_position=True,
)
assert len(loaded_tactic.positions) == len(tactic.positions)
assert loaded_tactic.signals_config == tactic.signals_config
```

`files_position` 是文件路径列表，不接受目录字符串。Position 的序列化和 checksum 校验在 Rust 中实现，保存新配置使用 `sha256(canonical JSON)`。JSON 只保存持仓规则；数据源、执行成本、训练窗口等研究设置必须另行归档。

## 11. 绩效分析

策略回测产物通常包括：

- `signals`：逐根 K 线信号
- `pairs`：完整开平仓交易对
- `holds`：持仓权重序列

用 `res.holds_df()` 转成 `WeightBacktest` 需要的 `dt / symbol / weight / price` 表。下例对同标的多个 Position 等权平均；正式研究需预先声明组合比例及缺失持仓的处理规则。

```python
from czsc import WeightBacktest

dfw = (
    holds[["dt", "symbol", "pos", "price"]]
    .rename(columns={"pos": "weight"})
    .groupby(["dt", "symbol"], as_index=False)
    .agg(weight=("weight", "mean"), price=("price", "first"))
    .sort_values(["symbol", "dt"])
    .reset_index(drop=True)
)
assert not dfw.empty
assert not dfw.isna().any().any()
assert not dfw.duplicated(["dt", "symbol"]).any()

FEE_RATE = 0.0002  # 演示单边成本 2 BP，正式研究按执行模型设置
YEARLY_DAYS = 252  # 此处的年化假设；其他市场按统计口径设置
wb = WeightBacktest(
    data=dfw, digits=2, fee_rate=FEE_RATE, weight_type="ts", yearly_days=YEARLY_DAYS
)
report = wb.stats
print(report)
```

绩效报告至少注明实际统计区间、年化口径、净收益与最大回撤、逐年表现、交易数、换手、仓位暴露及成本假设。交易数等指标需注明来自 Position 的 pairs 还是权重回测统计，不能假定两者相同。交易信号形成、持仓生效和收益区间必须一致：先用少量交易核对更新时间、成交价与下一段收益，再确认是否需要额外延迟，避免偷用同根收益或重复移位。

`pairs` 用于交易级胜率、盈亏、持仓时间和开平仓规则分析；只把完整交易收益连乘，会遗漏未平仓浮动损益和交易内回撤，不能替代账户逐期净值。权重回测也不自动等于券商成交仿真，下一开盘成交、滑点和不可成交约束需另行验证。

完整 Event → holds → 权重回测示例见 `docs/examples/13_event_weight_backtest.py`。以下 RBreaker 脚本包含自定义交易模拟，可用于对照执行假设；能用 Position/Event 表达的新增策略仍优先使用标准路径：

```text
examples/signals_dev/backtest_rbreaker_000852_5m.py
examples/signals_dev/backtest_rbreaker_000852_5m_s03_2020.py
```

## 12. 测试规范

Python 测试放在 `tests/`，使用 pytest；Rust 逻辑测试放在对应 crate。新增共享算法必须在 Rust 和 Python 公共调用路径分别覆盖等价输入、边界与错误行为。

推荐命令：

```bash
# Rust 源码或 Cargo 配置改动后，先重建 Python 扩展
uv run --no-sync maturin develop --release

# 按改动选择聚焦用例；下面是已有策略序列化回归测试
uv run --no-sync pytest tests/unit/test_strategy_save_load_parity.py -v

# 修改信号或交易逻辑时，分别跑对应 Rust crate
export PYO3_PYTHON="$(uv run --no-sync python -c 'import sys; print(sys.executable)')"
cargo test -p czsc-signals --no-fail-fast
cargo test -p czsc-trader --no-fail-fast

# 修改 PyO3 公共 API 后生成 stub，两项 feature 参数必须同时使用
cargo run --bin stub_gen -p czsc-python --no-default-features --features stub-gen
git diff -- czsc/_native/__init__.pyi
```

测试数据优先使用 `czsc.mock.generate_symbol_kines`：

```python
from czsc import Freq, format_standard_kline
from czsc.mock import generate_symbol_kines

df_test = generate_symbol_kines("000001", "30分钟", "20240101", "20240601", seed=42)
bars_test = format_standard_kline(df_test, freq=Freq.F30)
```

涉及外部数据源的测试不要默认进主测试集，避免依赖账号、网络和行情服务状态。

聚焦验证应覆盖：批量/流式同区间结果、预热与起点、事件匹配、止损/超时/日内规则、换标的绑定、保存加载一致性，以及成本和收益时点。依赖 `time.sleep` 或子进程冷启动的测试标记 `slow`。完整开发命令、逐 crate 链接限制和 CI 范围见 [CLAUDE.md](CLAUDE.md)；发布前跑包含慢用例的全套检查。

## 13. 旧文档迁移对照

| 旧路径 / 旧入口 | 1.0 后替代方式 |
|---|---|
| `from czsc.core import CZSC, RawBar, Freq` | `from czsc import CZSC, RawBar, Freq` |
| `from czsc.traders.base import generate_czsc_signals` | `from czsc.traders import generate_czsc_signals` |
| `czsc.signals.*` Python 信号函数 | Rust `#[signal]` 注册 + `czsc.generate_czsc_signals` / 交易器公共 API |
| `CTAResearch` | `CzscStrategyBase.backtest/replay` 或 `czsc.run_research/run_replay` |
| `czsc.sensors.*` | 按需求改用 `czsc.traders`、`czsc.research` 或本地研究脚本 |
| `test/` | `tests/` |

## 14. 参数冻结与样本外验证

在开始调参前，把候选范围、选择目标、数据窗口和验收标准写入实验配置。按时间顺序切分训练、验证和最终测试区间：

1. 训练区间用于产生候选和选择参数；验证区间用于评估稳定性。任何使用验证结果改规则的行为都应计入研究尝试记录。
2. 冻结候选信号、事件、持仓和组合规则后，再运行最终测试区间。测试结果参与新一轮调参后，该区间不再是未见过的最终测试集。
3. 做滚动或扩展窗口 walk-forward：每一折只用此前数据选规则，下一折参数冻结。预热可以使用此前历史，但正式持仓、区间边界和统计输出必须按已声明规则处理。
4. 检查分年份、不同标的、弱窗口、参数邻域及成本扰动表现，并与事先选定的基线对照。记录尝试次数及被拒绝候选，不能只保留最优结果。

发现未来数据泄漏、同根收益错配或窗口边界错误时，受影响结果作废并重跑。未满足预设标准的候选应标记不通过；可以回到研究阶段，但需要新的验证记录，不得以样本内最优结果替代样本外验收。

仓库参考记录：

- [纯日线 CTA 滚动样本外验证记录](docs/research/纯日线CTA滚动样本外验证记录.md)
- [商品期货 CTA 反过拟合验证报告](docs/research/商品期货CTA反过拟合验证报告.md)

mock 数据仅验证流程与确定性，策略有效性结论必须基于所声明的真实数据和独立验证区间。

## 15. 执行压力测试

先声明信号资产与交易工具之间的映射，再检查研究规则在执行约束下是否仍满足预设标准。对指数风险敞口、连续期货或 ETF 代理数据，要单独记录工具映射、跟踪误差或换月处理，不能把研究价格直接当成可成交报价。

至少覆盖策略实际涉及的情景：

- 信号完成后延迟成交或下一根开盘成交，以及时间戳和价格对齐。
- 手续费、滑点与换手成本上调；空头、融资或展期成本按适用场景另计。
- 价格跳空、涨跌停、停牌、缺失报价与未成交订单；记录无法执行时的持仓规则。
- 仓位上限、杠杆、保证金、最小下单单位、调仓频率和换手限制。
- 不允许做空、不允许日内交易或限制隔夜时的对照结果。

报告中分开列出信号/权重基准结果与执行约束结果；说明模型覆盖和未覆盖的约束。无法合理映射交易工具或未达到执行验收标准的候选，只保留为研究结果。

执行口径与工具映射参考 [日线 CTA 策略最终研究报告](docs/research/日线CTA策略最终研究报告.md)。

## 16. 归档与验收

每次实验使用独立运行目录，并保存足以复现结论的材料：

| 材料 | 最低内容 |
|---|---|
| 实验配置 | 标的、周期、起止时间、预热规则、数据源、执行成本、组合比例、随机种子 |
| 代码环境 | Git commit、工作区是否有未提交改动、CZSC/Python 版本、依赖锁文件或其哈希 |
| 数据清单 | 数据获取时间、实际范围、行数、复权/换月规则、文件哈希与存储位置；不含账号密钥 |
| 策略规则 | Position JSON、signals_config、参数选择规则、各折训练与测试范围 |
| 产物 | signals/pairs/holds、权重表、逐期净值、指标摘要、HTML 检查图及日志 |
| 验收结论 | 样本外和执行测试是否通过、失败项、尝试记录、适用边界与后续动作 |

运行目录不要混入上一次产物。大型行情和报告使用约定的本地目录或外部存储，Git 保存规则、脚本、数据索引与结论；不提交账号密钥。mock 示例默认输出到 `docs/examples/_output/cta_workflow/`，正式研究应另取独立运行目录。

## 17. 开发检查清单

- [ ] 策略规格已经写清楚
- [ ] 输入格式与所用 API 匹配，数据标的、周期、时间顺序和质量已核验
- [ ] 数据预热充分，实际输出起点和时间范围已检查
- [ ] 信号来自 Rust 注册名，或新增信号已在 `crates/czsc-signals` 实现
- [ ] `signals_config` 能被 `generate_czsc_signals` 正常调度
- [ ] 关键非 `其他` 信号经过分布统计和图形检查
- [ ] 事件字符串与信号 key/value 完整匹配
- [ ] `Position` 的开平仓、冷却、止损、超时配置明确
- [ ] Position 使用实例标的，参数单位及 `t0` 设置经过交易记录核验
- [ ] 回测和回放路径跑通
- [ ] holds 到权重表的转换、组合比例、收益时点和成本口径已核对
- [ ] 样本外参数已冻结，逐折结果和失败候选完整记录
- [ ] 未来数据泄漏与执行压力测试通过，或已明确标记未通过
- [ ] 配置、代码版本、数据清单、指标与产物可复现
- [ ] 结果文件和外部账号密钥没有误提交
- [ ] 新增 Rust 逻辑及 Python 公共调用路径均有聚焦测试覆盖；扩展和 stub 与源码一致
