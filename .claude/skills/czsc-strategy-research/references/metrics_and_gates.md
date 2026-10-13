# CZSC 策略研究核心度量指标与数学公式

本文档为 `.claude/skills/czsc-strategy-research` 的配套参考文档，详细定义了 5 阶段工作流中使用的数学指标、统计检验方法及判定门禁。

---

## 1. 单信号事件研究（Event Study）指标

### 1.1 前向持仓收益（Forward Return）
对于在时间戳 $t$ 触发的信号事件，持有未来 $k$ 根 K 线（或 $k$ 个周期）的收益率定义为：
$$R_{t+k} = \frac{\text{Close}_{t+k} - \text{Close}_t}{\text{Close}_t}$$

### 1.2 胜率（Win Rate）
在所有 $N$ 次触发事件中，前向收益大于 0 的比率：
$$\text{WinRate}_k = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(R_{i, t+k} > 0)$$

### 1.3 95% 置信区间（95% Confidence Interval）
假定收益率均值在中心极限定理下渐近正态分布：
$$\text{CI}_{95\%} = \bar{R}_k \pm 1.96 \times \frac{s_k}{\sqrt{N}}$$
其中 $\bar{R}_k$ 为样本均值，$s_k$ 为样本标准差，$N$ 为触发事件总次数。
- **有效性检验判定**：若置信区间下界 $\text{Lower} = \bar{R}_k - 1.96 \times \frac{s_k}{\sqrt{N}} > 0$，则在 5% 显著性水平下拒绝无超额收益的原假设。

---

## 2. 截面并发度（Cross-Sectional Concurrency, CSC）

对于包含 $M$ 个标的（Symbol）的资产池，在同一时间步长 $t$ 统计同时触发某事件的资产数量：
$$\text{CSC}(t) = \sum_{m=1}^M \mathbb{I}(\text{Event triggered on symbol } m \text{ at } t)$$

- **并发率（Concurrency Rate）**：$\text{CR}(t) = \frac{\text{CSC}(t)}{M}$
- **高并发集中期**：当 $\text{CR}(t) > 50\%$ 时，通常代表系统性市场大幅波动（普涨/暴跌），单标的技术形态极易受宏观 beta 冲击而失真。
- **多周期共振增益**：
  $$\Delta \text{WinRate} = \text{WinRate}(\text{Event}_{\text{Resonance}}) - \text{WinRate}(\text{Event}_{\text{Baseline}})$$
  $$\Delta \text{MeanReturn} = \bar{R}(\text{Event}_{\text{Resonance}}) - \bar{R}(\text{Event}_{\text{Baseline}})$$

---

## 3. 持仓状态机与点对（Pairs）质量指标

每个闭环交易点对 $i$ 的收益率为 $r_i$（考虑开仓价 $P_{\text{open}}$ 与平仓价 $P_{\text{exit}}$）。

### 3.1 盈亏比（Profit-to-Loss Ratio, $R$）
所有盈利交易的平均收益与所有亏损交易的平均亏损绝对值之比：
$$R = \frac{\frac{1}{N_{\text{win}}} \sum_{r_i > 0} r_i}{\frac{1}{N_{\text{loss}}} \sum_{r_i < 0} |r_i|}$$

### 3.2 盈亏平衡胜率（Break-even Win Rate, $W_{\text{be}}$）
使单次交易数学期望为 0 所需的理论最低胜率：
$$\mathbb{E}[\text{Return}] = W_{\text{be}} \times \overline{\text{Gain}} - (1 - W_{\text{be}}) \times \overline{\text{Loss}} = 0$$
$$\Rightarrow W_{\text{be}} = \frac{1}{1 + R}$$

### 3.3 安全边际（Safety Margin, $\Delta W$）
实际历史胜率 $W$ 与盈亏平衡胜率之差：
$$\Delta W = W - W_{\text{be}} = W - \frac{1}{1 + R}$$
- $\Delta W > 0$：策略具有正期望收益；
- $\Delta W \ge +10\%$：策略具备充足抗滑点与未来胜率衰减的安全垫（生产准入线）。

---

## 4. 组合回测绩效指标（WeightBacktest）

对日度净值时序 $V_t$ 及日度收益率 $r_t$：

### 4.1 年化收益率（Annualized Return）
$$\text{AnnReturn} = \left(\frac{V_T}{V_0}\right)^{\frac{250}{T}} - 1 \quad \text{或} \quad \bar{r} \times 250$$

### 4.2 夏普比率（Sharpe Ratio, 无风险利率设为 0）
$$\text{Sharpe} = \frac{\bar{r}}{\sigma_r} \times \sqrt{250}$$

### 4.3 最大回撤（Max Drawdown）
$$\text{MaxDD} = \max_{t \in [1, T]} \left( \frac{\max_{s \le t} V_s - V_t}{\max_{s \le t} V_s} \right)$$

### 4.4 卡玛比率（Calmar Ratio）
$$\text{Calmar} = \frac{\text{AnnReturn}}{\text{MaxDD}}$$

---

## 5. 摩擦敏感性与衰减率

在不同单边费用率 $f \in \{0, 2\text{BP}, 5\text{BP}, 10\text{BP}\}$（$1\text{BP} = 0.01\% = 0.0001$）下执行回测：
$$\text{Friction Decay Rate} = \frac{\text{Sharpe}(f = 10\text{BP})}{\text{Sharpe}(f = 0)}$$
- 若 Decay Rate $> 50\%$，且在高摩擦下年化收益仍大于 0，说明策略交易频率合理，不是依赖微小价差的高频假 Alpha。
