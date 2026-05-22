# CTA 策略开发标准流程

本文档面向 CZSC 1.0 之后的项目结构。1.0 开始，缠论核心、信号注册、交易器和策略回测主路径已经迁移到 Rust，通过 `czsc._native` 暴露给 Python；Python 侧主要负责数据接入、策略配置、研究编排和结果分析。

核心流程仍然是：

```text
策略规格 -> 数据准备 -> 信号选择/开发 -> 信号验证 -> 事件/持仓配置 -> 回测/回放 -> 绩效分析
```

交易体系仍然遵循：

```text
Signal -> Factor -> Event -> Position -> Trader
```

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

示例：

```text
标的：000852.XSHG
基础周期：5分钟
开仓：RBreaker 突破价位触发趋势开仓
平仓：14:55 强制平仓
隔夜：不允许
单日重复开仓：不允许
```

## 2. 准备数据

项目标准输入是 `list[RawBar]`。从 DataFrame 转换时，优先使用顶层 API：

```python
from czsc import Freq, RawBar, format_standard_kline

bars = format_standard_kline(df, freq=Freq.F5)
```

如果使用模拟数据：

```python
from czsc import CZSC, Freq, format_standard_kline
from czsc.mock import generate_symbol_kines

df = generate_symbol_kines("000001", "30分钟", "20240101", "20240601", seed=42)
bars = format_standard_kline(df, freq=Freq.F30)
c = CZSC(bars)
```

如果使用聚宽 JQData，先安装可选依赖：

```bash
uv sync --extra jq
```

或临时运行：

```bash
uv run --extra jq python examples/signals_dev/backtest_rbreaker_000852_5m.py
```

账号密码通过环境变量读取：

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

查看当前信号模块速查：

```bash
uv run --no-sync python scripts/dump_signal_catalog.py
```

查看详细信号元信息：

```bash
uv run --no-sync python scripts/dump_signal_details.py --format json
```

## 4. 调用单个 K 线信号

K 线类信号可以通过统一分发器直接调用：

```python
from czsc import CZSC
from czsc._native.signals import call_signal

c = CZSC(bars)
sig = call_signal("bar_r_breaker_V230326", c, {})[0]
print(sig.key, sig.value)
```

带参数的例子：

```python
sig = call_signal("zdy_macd_bc_V230422", c, {"di": 1, "th": 50})[0]
print(sig.to_string())
```

注意：依赖 `CzscTrader` 或持仓状态的 Trader 类信号，不应通过 `call_signal` 直接调度，应放入 `signals_config` 由 `CzscSignals` / `CzscTrader` 计算。

## 5. 批量生成信号

研究和特征工程中，使用 `generate_czsc_signals`：

```python
from czsc.traders import generate_czsc_signals

signals_config = [
    {"name": "cxt_bi_status_V230101", "freq": "30分钟"},
    {"name": "bar_zdt_V230331", "freq": "30分钟", "di": 1},
]

df = generate_czsc_signals(
    bars,
    signals_config=signals_config,
    sdt="2024-01-01",
    df=True,
)
```

流式场景中，使用 `CzscSignals`：

```python
from czsc import BarGenerator
from czsc.traders import CzscSignals

bg = BarGenerator(base_freq="1分钟", freqs=["1分钟", "5分钟", "30分钟"], max_count=5000)
cs = CzscSignals(bg, signals_config=signals_config)

for bar in bars:
    cs.update_signals(bar)

print(cs.s)
```

## 6. 验证信号

信号验证重点：

- 非目标状态是否稳定返回 `其他`
- 目标状态是否不漏触发
- 触发频率是否合理
- 信号 key / value 是否符合事件匹配规则
- 多周期信号是否对齐正确

推荐验证方式：

```python
from czsc.traders import generate_czsc_signals

df = generate_czsc_signals(bars, signals_config=signals_config, df=True, sdt="2024-01-01")
sig_cols = [c for c in df.columns if len(c.split("_")) == 3]

for col in sig_cols:
    print(col, df[col].value_counts().head(10))
```

需要图形检查时，优先使用 lightweight HTML：

```python
from czsc.utils.plotting.lightweight import plot_czsc_signals

plot_czsc_signals(
    bars,
    signals_config=signals_config,
    file_html="signals_check.html",
)
```

## 7. 组合事件和持仓

事件由信号组合形成。信号完整字符串格式通常是：

```text
{k1}_{k2}_{k3}_{v1}_{v2}_{v3}_{score}
```

示例：

```python
from czsc import Event, Operate, Position

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

pos = Position(
    name="30分钟笔非多即空",
    symbol="000001",
    opens=[open_long, open_short],
    exits=[],
    interval=3600 * 4,
    timeout=16 * 30,
    stop_loss=500,
)
```

## 8. 组织策略类

推荐继承 `CzscStrategyBase`，只实现 `positions`：

```python
from czsc import CzscStrategyBase, Position


class MyStrategy(CzscStrategyBase):
    @property
    def positions(self) -> list[Position]:
        return [pos]
```

框架会自动派生：

- `base_freq`
- `freqs`
- `signals_config`
- `unique_signals`

完整示例见：

```text
docs/examples/07_strategy_backtest.py
examples/intraday_0935_direction_strategy.py
```

## 9. 回测和回放

内存回测：

```python
tactic = MyStrategy(symbol="000001")
res = tactic.backtest(bars, sdt="2024-01-01")
```

回放落盘：

```python
replay_res = tactic.replay(
    bars,
    res_path="results/my_strategy",
    sdt="2024-01-01",
    refresh=True,
)
```

1.0 之后不再使用旧的 `CTAResearch` 入口。研究入口统一看：

```python
from czsc import run_research, run_replay, run_optimize_batch
```

以及策略对象自身的：

```python
tactic.backtest(...)
tactic.replay(...)
```

## 10. 保存和复用策略

`CzscStrategyBase` 支持保存 / 加载持仓配置：

```python
tactic.save_positions("results/my_strategy/positions")
```

重新加载：

```python
from czsc import CzscJsonStrategy

tactic = CzscJsonStrategy(
    symbol="000001",
    files_position="results/my_strategy/positions",
)
```

该路径的序列化和校验逻辑已经下沉到 Rust，Python 侧只做透传。

## 11. 绩效分析

策略回测产物通常包括：

- `signals`：逐根 K 线信号
- `pairs`：完整开平仓交易对
- `holds`：持仓权重序列

可以用 `wbt` 或 `czsc.WeightBacktest` 分析权重序列：

```python
from czsc import WeightBacktest

wb = WeightBacktest(dfw, digits=2)
report = wb.stats
```

轻量研究脚本也可以直接从 `pairs` 或交易明细 CSV 计算净值曲线，例如 RBreaker 示例：

```text
examples/signals_dev/backtest_rbreaker_000852_5m.py
examples/signals_dev/backtest_rbreaker_000852_5m_s03_2020.py
```

## 12. 测试规范

测试文件放在 `tests/` 目录，使用 pytest。

推荐命令：

```bash
uv run --no-sync pytest tests/test_xxx.py -v
```

测试数据优先使用 `czsc.mock.generate_symbol_kines`：

```python
from czsc import Freq, format_standard_kline
from czsc.mock import generate_symbol_kines

df = generate_symbol_kines("000001", "30分钟", "20240101", "20240601", seed=42)
bars = format_standard_kline(df, freq=Freq.F30)
```

涉及外部数据源的测试不要默认进主测试集，避免依赖账号、网络和行情服务状态。

## 13. 旧文档迁移对照

| 旧路径 / 旧入口 | 1.0 后替代方式 |
|---|---|
| `from czsc.core import CZSC, RawBar, Freq` | `from czsc import CZSC, RawBar, Freq` |
| `from czsc.traders.base import generate_czsc_signals` | `from czsc.traders import generate_czsc_signals` |
| `czsc.signals.*` Python 信号函数 | `crates/czsc-signals/src/*.rs` + `czsc._native.signals` |
| `CTAResearch` | `CzscStrategyBase.backtest/replay` 或 `czsc.run_research/run_replay` |
| `czsc.sensors.*` | 按需求改用 `czsc.traders`、`czsc.research` 或本地研究脚本 |
| `test/` | `tests/` |

## 14. 开发检查清单

- [ ] 策略规格已经写清楚
- [ ] 数据函数返回 `list[RawBar]`
- [ ] 信号来自 Rust 注册名，或新增信号已在 `crates/czsc-signals` 实现
- [ ] `signals_config` 能被 `generate_czsc_signals` 正常调度
- [ ] 关键非 `其他` 信号经过分布统计和图形检查
- [ ] 事件字符串与信号 key/value 完整匹配
- [ ] `Position` 的开平仓、冷却、止损、超时配置明确
- [ ] 回测和回放路径跑通
- [ ] 结果文件和外部账号密钥没有误提交
- [ ] 新增逻辑有聚焦测试覆盖
