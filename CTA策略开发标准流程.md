# CTA 策略开发标准流程

本文档整理一套适用于本项目的 CTA 策略开发流程，核心思路是：

```text
信号开发 -> 信号验证 -> 信号缓存 / 入库 -> 因子组合 -> 事件触发 -> 策略回测 -> 策略回放 -> 绩效评估
```

该流程遵循项目的系统化交易框架：

```text
Signal -> Factor -> Event -> Position -> Trader
```

## 1. 明确策略规格

开发策略前，先把策略规则写清楚，不要直接进入编码。

需要明确以下内容：

- 交易标的：指数、ETF、股票、期货等
- 基础周期：`1分钟`、`5分钟`、`30分钟`、`日线` 等
- 是否多周期联立：例如日线定方向，5分钟找入场
- 开仓逻辑：什么条件开多 / 开空
- 平仓逻辑：反向信号、固定时间、止损、止盈、超时等
- 风控规则：止损比例、是否允许 T0、是否允许隔夜、单日最多交易次数
- 回测区间：`bar_sdt`、`sdt`、`edt`
- 数据源：聚宽、research、Tushare、天勤等

示例：

```text
标的：000852.XSHG
基础周期：1分钟
开仓：9:35 相对当日开盘价涨跌决定多空
平仓：14:55 或 1% 止损
隔夜：不允许
单日重复开仓：不允许
```

## 2. 准备数据接口

项目标准数据输入是 `List[RawBar]`。建议所有策略都提供统一的 `read_bars` 函数：

```python
def read_bars(symbol, freq, sdt, edt, fq="前复权", **kwargs):
    """读取 K 线数据，返回 List[RawBar]"""
    ...
    return bars
```

这个签名可以直接接入 `CTAResearch`：

```text
(symbol, freq, sdt, edt, fq='前复权', **kwargs) -> List[RawBar]
```

如果使用聚宽数据源，可以参考：

```python
from czsc.connectors.jq_connector import get_kline_period, freq_cn2jq

def read_bars(symbol, freq, sdt, edt, fq="前复权", **kwargs):
    jq_freq = freq_cn2jq[str(freq)]
    fq_flag = fq == "前复权"
    return get_kline_period(symbol=symbol, start_date=sdt, end_date=edt, freq=jq_freq, fq=fq_flag)
```

注意事项：

- 分钟数据建议传完整时间，例如 `2026-04-16 15:00:00`
- 指数观察可使用不复权
- 股票 / ETF 通常需要考虑前复权
- 聚宽 token 建议通过环境变量或 `set_token` 初始化，不要提交敏感信息

## 3. 开发信号 Signal

信号是最小判断单元，只回答“当前市场状态是什么”。

项目信号函数应放在 `czsc/signals/` 下，并按类别组织：

- `bar.py`：K线基础信号
- `cxt.py`：缠论上下文信号
- `tas.py`：技术指标信号
- `vol.py`：成交量信号
- `pos.py`：持仓管理信号

信号函数命名应带版本号，例如：

```text
bar_intraday_direction_V260424
bar_intraday_exit_V260424
```

信号字符串格式示例：

```text
1分钟_D1T0935_日内方向V260424_看多_任意_任意_0
1分钟_D1T0935_日内方向V260424_看空_任意_任意_0
1分钟_D1T1455_日内平仓V260424_平仓_任意_任意_0
```

信号开发完成后，需要验证：

- 是否只在目标条件下触发
- 非目标状态是否返回 `其他`
- 信号取值是否符合预期
- 边界条件是否正确

## 4. 生成信号

使用 `generate_czsc_signals` 批量生成信号序列。

```python
from czsc.traders.base import generate_czsc_signals

sigs = generate_czsc_signals(
    bars,
    signals_config=strategy.signals_config,
    sdt="20260101",
    df=True,
)
```

生成结果通常用于：

- 检查信号分布
- 定位信号触发时间
- 保存信号缓存
- 供 `DummyBacktest` 快速回测
- 后续因子和事件匹配分析

## 5. 验证信号

项目已有信号验证工具 `check_signals_acc`。

```python
from czsc.traders.base import check_signals_acc

check_signals_acc(
    bars,
    signals_config=strategy.signals_config,
    delta_days=1,
    height="780px",
)
```

它会：

- 生成信号
- 统计信号出现次数
- 找出非 `其他` 的有效信号
- 在信号触发时生成 HTML 快照
- 方便人工检查信号是否符合 K 线图形和策略逻辑

这一步重点检查：

- 信号是否误触发
- 信号是否漏触发
- 信号出现频率是否合理
- 信号触发时的图形是否符合预期

## 6. 信号缓存或入库

项目已有本地信号缓存能力，典型方式是保存为 parquet。

```python
sigs.to_parquet("signals/000852.XSHG.sigs")
```

`DummyBacktest` 中已经使用了这种模式：

```text
如果信号文件不存在：
    生成信号
    保存为 parquet
否则：
    读取 parquet
```

推荐目录结构：

```text
results/
  strategy_name/
    signals/
      000852.XSHG.sigs
    snapshots/
    event_match/
    backtest/
```

如果需要数据库入库，可以在此基础上增加自定义 `SignalStore`：

```python
def save_signals_to_db(sigs, table="czsc_signals"):
    ...
```

当前项目没有统一的“信号验证通过后自动入库数据库”的标准模块；已有能力主要是本地缓存和回测复用。

## 7. 组合因子 Factor

因子是信号组合，代表某个可解释的交易条件。

简单因子可以只包含一个信号：

```python
factor_long = {
    "name": "0935方向看多",
    "signals_all": ["1分钟_D1T0935_日内方向V260424_看多_任意_任意_0"],
    "signals_any": [],
    "signals_not": [],
}
```

复杂因子可以组合多个信号：

```python
factor_long = {
    "name": "0935方向看多且波动率过滤通过",
    "signals_all": [
        "1分钟_D1T0935_日内方向V260424_看多_任意_任意_0",
        "1分钟_D1波动率Vxxxx_高波动_任意_任意_0",
    ],
    "signals_any": [],
    "signals_not": [
        "1分钟_D1涨跌停V230331_涨停_任意_任意_0",
    ],
}
```

因子层适合放：

- 趋势过滤
- 波动率过滤
- 成交量过滤
- 缠论结构过滤
- 禁止交易条件

## 8. 定义事件 Event

事件是交易动作，由一个或多个因子触发。

常见事件：

- `开多`
- `开空`
- `平多`
- `平空`

事件配置示例：

```python
event_open_long = {
    "operate": "开多",
    "signals_all": [],
    "signals_any": [],
    "signals_not": [],
    "factors": [factor_long],
}
```

事件用于回答：

```text
当前是否应该执行某个交易动作？
```

## 9. 验证事件匹配

当信号组合成事件后，可以使用 `EventMatchSensor` 验证事件在历史数据中的触发情况。

```python
from czsc.sensors.event import EventMatchSensor

ems = EventMatchSensor(
    events=[event_open_long, event_open_short],
    symbols=["000852.XSHG"],
    read_bars=read_bars,
    results_path="results/event_match",
    bar_sdt="20250101",
    sdt="20260101",
    edt="20260424",
)
```

它会：

- 读取 K 线
- 生成信号
- 对每个事件执行 `event.is_match`
- 保存事件匹配结果
- 输出截面匹配次数 `cross_section_counts.csv`

这一步适合验证：

- 事件触发频率是否合理
- 开仓事件是否过多或过少
- 多空事件是否互斥
- 事件触发时间是否符合策略预期

## 10. 定义持仓 Position

`Position` 是策略的持仓规则单元。

示例：

```text
多头 Position:
- opens: 开多事件
- exits: 平多事件
- stop_loss: 100

空头 Position:
- opens: 开空事件
- exits: 平空事件
- stop_loss: 100
```

常用参数：

- `interval`：同类开仓间隔
- `timeout`：最多持仓 K 线数量
- `stop_loss`：止损，单位 BP
- `T0`：是否允许日内开平

单位说明：

```text
100 BP = 1%
300 BP = 3%
500 BP = 5%
```

## 11. 封装 Strategy

策略类继承 `CzscStrategyBase`。

标准职责：

- 定义 `positions`
- 定义或推导 `signals_config`
- 暴露策略参数

示例：

```python
from czsc.strategies import CzscStrategyBase

class MyStrategy(CzscStrategyBase):
    @property
    def positions(self):
        return [...]

    @property
    def signals_config(self):
        return [...]
```

如果信号函数文档模板完整，可以让框架从 `unique_signals` 自动解析；实验阶段也可以显式写 `signals_config`，方便控制和调试。

## 12. 单标的回测验证

不要一开始就批量回测。先做单标的验证。

```python
tactic = MyStrategy(symbol="000852.XSHG")
bars = read_bars("000852.XSHG", tactic.base_freq, "20250101", "20260424")
trader = tactic.init_trader(bars, sdt="20260101")
```

重点检查：

- 是否有交易
- 开仓时间是否正确
- 平仓时间是否正确
- 止损是否正确
- 是否出现重复开仓
- `pairs` 是否符合规则
- `holds` 是否符合持仓状态

常用检查：

```python
for pos in trader.positions:
    print(pos.pairs)
    print(pos.evaluate())
```

## 13. 策略回放 Replay

单标的交易记录正常后，用 `replay` 生成交易快照。

```python
tactic.replay(
    bars,
    res_path="examples/results/my_strategy_replay",
    sdt="20260416",
    refresh=True,
)
```

回放会在每次操作时生成 HTML 文件。

重点检查：

- 信号触发时 K 线是否正确
- 买卖点是否标在正确位置
- 进场价格和平仓价格是否合理
- 止损是否提前于尾盘平仓
- 多周期策略中，各周期图表是否一致

## 14. 批量回测 CTAResearch

单标的验证通过后，再使用 `CTAResearch` 做多标的回测。

```python
from czsc import CTAResearch

bot = CTAResearch(
    strategy=MyStrategy,
    read_bars=read_bars,
    results_path="examples/results/my_strategy",
    signals_module_name="czsc.signals",
)

bot.backtest(
    symbols=symbols,
    max_workers=3,
    bar_sdt="20250101",
    sdt="20260101",
    edt="20260424",
)
```

参数说明：

- `bar_sdt`：加载历史 K 线的起点，用于初始化
- `sdt`：正式统计开始时间
- `edt`：回测结束时间
- `max_workers`：多进程数量

## 15. 结果分析

至少分析以下指标：

- 交易次数
- 胜率
- 累计收益
- 平均单笔收益
- 盈亏比
- 最大回撤
- 持仓覆盖率
- 多头 / 空头分别表现
- 止损触发次数
- 按月表现
- 连续亏损次数

建议分别看：

```text
总表现
多头表现
空头表现
按标的表现
按年份 / 月份表现
样本内 / 样本外表现
```

## 16. 测试规范

测试文件放在 `test/` 目录，使用 `pytest`。

推荐至少写三类测试：

1. 信号测试
   - 验证特定 K 线时点输出正确的信号

2. 交易路径测试
   - 验证开仓、平仓、止损、禁止重复开仓

3. 数据接口测试
   - 验证 `read_bars` 返回 `RawBar` 且时间有序

运行：

```bash
uv run pytest test/test_xxx.py -v
```

测试数据优先使用 `czsc.mock.generate_symbol_kines`，对精确时点策略，可以在 mock 数据基础上定向修改关键 K 线。

## 17. 推荐迭代顺序

每轮迭代尽量只改一个变量。

推荐顺序：

1. 跑通最小策略
2. 验证信号
3. 缓存或入库信号
4. 组合因子
5. 验证事件匹配
6. 加止损
7. 加止盈
8. 加过滤条件
9. 加多周期
10. 加仓位管理
11. 批量回测
12. 样本外验证
13. 回放人工检查
14. 再考虑实盘化

## 18. 标准开发清单

```text
[ ] 写清楚策略规则
[ ] 确定 symbol / freq / sdt / edt / fq
[ ] 实现 read_bars
[ ] 在 czsc/signals/ 中实现信号函数
[ ] 写出 signal 字符串
[ ] 使用 generate_czsc_signals 生成信号
[ ] 使用 check_signals_acc 验证信号
[ ] 将验证后的信号保存为 parquet 或入库
[ ] 组合 factor
[ ] 定义 open / exit event
[ ] 使用 EventMatchSensor 验证事件匹配
[ ] 创建 position
[ ] 封装 CzscStrategyBase
[ ] 单标的 init_trader 回测
[ ] 检查 pairs / holds / evaluate
[ ] 生成 replay 快照
[ ] 写 pytest 测试
[ ] 使用 CTAResearch 批量回测
[ ] 汇总绩效并分析风险
[ ] 再进入策略优化
```

## 总结

本项目中 CTA 策略开发的关键不是写一个独立的 if-else 回测脚本，而是把交易逻辑拆解为：

```text
Signal -> Factor -> Event -> Position -> Trader
```

推荐的工程闭环是：

```text
信号开发
-> 信号验证
-> 信号缓存 / 入库
-> 因子组合
-> 事件匹配验证
-> 策略回测
-> 策略回放
-> 绩效评估
-> 策略迭代
```

