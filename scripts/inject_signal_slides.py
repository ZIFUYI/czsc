#!/usr/bin/env python3
"""
Inject Slide 6 (Alpha Architecture & Factor Taxonomy) and Slide 7 (Native Signal Engine)
into docs/research/multitf_cta_research_presentation.html and sync to
examples/results/intraday_cta_ablation_20261017/multitf_cta_research_presentation.html.
"""

import re
from pathlib import Path

TARGET_DOCS_PATH = Path("docs/research/multitf_cta_research_presentation.html")
TARGET_EXAMPLES_PATH = Path("examples/results/intraday_cta_ablation_20261017/multitf_cta_research_presentation.html")

SLIDE_6_HTML = """<!-- ==================== SLIDE 6: 六大战略原型簇因子全貌与两阶段状态机 ==================== -->
<div aria-label="第 6 页，共 15 页" class="slide" data-cat="Alpha Architecture" data-title="06. 六大战略簇因子全貌与两阶段状态机" id="slide-6">
<div class="slide-header">
<div class="slide-header-left">
<div class="slide-category">ALPHA ARCHITECTURE &amp; FACTOR TAXONOMY</div>
<h2 class="slide-title">六大战略原型簇（A ~ F）因子全貌与两阶段状态机架构</h2>
<div class="slide-subtitle">从宏观趋势突破到微观截面强弱的全谱系因子库、两阶段状态机演化机制与连续结构强度 Q 分解码</div>
</div>
<span class="badge-pill">6大战略原型 · 137+ 候选因子 · 连续强度解码</span>
</div>
<div class="slide-content">
<!-- Top Bar: Two-Stage State Machine & Continuous Sizing Formulation -->
<div class="card" style="padding: 10px 14px; background: #fbfcfd; border: 1px solid var(--border-color); border-radius: 4px;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
<span style="font-weight: 700; color: var(--primary); font-size: 13.5px;">两阶段状态机 (Setup → Confirmation) 与波动率自适应衰减模型</span>
<span style="font-size: 11px; color: var(--text-muted);">严禁单根 K 线静态拼接开仓 · 100% 多空镜像对称架构</span>
</div>
<div class="grid-3" style="gap: 16px; font-size: 12.5px; line-height: 1.5;">
<div>
<strong style="color: var(--primary);">阶段一：Setup 势能蓄力</strong><br/>
大级别结构就位（中枢末端/均线排列/通道挤压），给出方向前置条件 <span class="math"><i>d</i><sub><i>i</i>,<i>t</i></sub> ∈ {−1, 0, +1}</span>。
</div>
<div>
<strong style="color: var(--success);">阶段二：Confirmation 微观确认</strong><br/>
次级别停顿分型、放量突破或资金流反包确认触发，读取 Rust 结构质量分 <span class="math"><i>Q</i><sub><i>i</i>,<i>t</i></sub> ∈ (0, 1]</span>。
</div>
<div>
<strong style="color: var(--purple);">连续强度与动态衰减机</strong><br/>
<div class="math-block" style="margin: 2px 0 0 0; padding: 2px 0;">
<span class="math"><i>S</i><sub><i>i</i>,<i>t</i></sub> = <i>d</i><sub><i>i</i>,<i>t</i></sub> · <i>Q</i><sub><i>i</i>,<i>t</i></sub> · <i>D</i><sub><i>i</i>,<i>t</i></sub>,  <i>D</i><sub><i>t</i></sub> = <i>D</i><sub><i>t</i>−1</sub> · exp(−<i>λ</i><sub>0</sub> · <i>ρ</i><sub><i>t</i></sub>)</span>
</div>
</div>
</div>
</div>

<!-- Middle: 6 Archetypes Taxonomy (3 columns x 2 rows) -->
<div class="grid-3" style="gap: 14px; flex: 1;">
<!-- Cluster A -->
<div class="card" style="padding: 10px 12px; border: 1px solid var(--border-color); border-left: 3.5px solid var(--primary); border-radius: 4px; display: flex; flex-direction: column; justify-content: space-between;">
<div>
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
<strong style="color: var(--primary); font-size: 13px;">A 簇 · 趋势突破与斜率动量簇</strong>
<span style="font-size: 10.5px; color: var(--text-muted); background: #f0f4f8; padding: 1px 6px; border-radius: 2px;">27 个</span>
</div>
<div style="font-size: 11.5px; color: var(--text-secondary); line-height: 1.45; margin-bottom: 6px;">
<strong>机理：</strong>大级别顺势中枢脱离爆发力与均线多空排列斜率加速度共振，捕捉单边趋势最丰厚波段。
</div>
</div>
<div style="font-size: 11px; background: #f8fafc; padding: 5px 7px; border-radius: 2px; color: var(--text-muted); line-height: 1.4;">
<strong>代表因子：</strong><code>AC01L/S</code> 三买卖脱离突破 · <code>AC02L/S</code> 均线斜率唐奇安 · <code>S1L/S</code> 中枢三买三卖 · <code>A04L/S</code> 宽幅唐奇安 · <code>CZ01/02</code> 颈线突破加速 · <code>D01/D03</code> 早午盘ORB
</div>
</div>

<!-- Cluster B -->
<div class="card" style="padding: 10px 12px; border: 1px solid var(--border-color); border-left: 3.5px solid var(--danger); border-radius: 4px; display: flex; flex-direction: column; justify-content: space-between;">
<div>
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
<strong style="color: var(--danger); font-size: 13px;">B 簇 · 背驰衰竭与极值反转簇</strong>
<span style="font-size: 10.5px; color: var(--text-muted); background: #fdf2f2; padding: 1px 6px; border-radius: 2px;">30 个</span>
</div>
<div style="font-size: 11.5px; color: var(--text-secondary); line-height: 1.45; margin-bottom: 6px;">
<strong>机理：</strong>动能竭尽与多空动能失衡，捕捉多杀多恐慌长下影或亢奋诱多反包，提供高盈亏比防御反转。
</div>
</div>
<div style="font-size: 11px; background: #fffaf9; padding: 5px 7px; border-radius: 2px; color: var(--text-muted); line-height: 1.4;">
<strong>代表因子：</strong><code>BC01L/S</code> 笔背驰衰竭分型 · <code>BC02L/S</code> TD九转极值2B · <code>TD01L/S</code> 蓄力三确认 · <code>CZ03/04</code> 一二买卖背驰反转 · <code>B01~05</code> 顶底背驰/双底顶 · <code>D04</code> VWAP反杀
</div>
</div>

<!-- Cluster C -->
<div class="card" style="padding: 10px 12px; border: 1px solid var(--border-color); border-left: 3.5px solid var(--accent); border-radius: 4px; display: flex; flex-direction: column; justify-content: space-between;">
<div>
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
<strong style="color: var(--accent); font-size: 13px;">C 簇 · 中枢震荡与锚定回归簇</strong>
<span style="font-size: 10.5px; color: var(--text-muted); background: #f0f7f5; padding: 1px 6px; border-radius: 2px;">24 个</span>
</div>
<div style="font-size: 11.5px; color: var(--text-secondary); line-height: 1.45; margin-bottom: 6px;">
<strong>机理：</strong>区间震荡箱体边界博弈，价格偏离价值中轴（中枢/均线）时的弹性张力拉回与高抛低吸。
</div>
</div>
<div style="font-size: 11px; background: #f4faf8; padding: 5px 7px; border-radius: 2px; color: var(--text-muted); line-height: 1.4;">
<strong>代表因子：</strong><code>CC01L/S</code> 中枢偏离拉回回归 · <code>CC02L/S</code> KC肯特纳轨道刺破收回 · <code>CZ05/06</code> 对称中枢/酒田强分型 · <code>C01~05</code> 布林箱体回归 · <code>D02/D05</code> VWAP回踩
</div>
</div>

<!-- Cluster D -->
<div class="card" style="padding: 10px 12px; border: 1px solid var(--border-color); border-left: 3.5px solid var(--amber); border-radius: 4px; display: flex; flex-direction: column; justify-content: space-between;">
<div>
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
<strong style="color: var(--amber); font-size: 13px;">D 簇 · 量价微结构与筹码冲击簇</strong>
<span style="font-size: 10.5px; color: var(--text-muted); background: #fbf7f0; padding: 1px 6px; border-radius: 2px;">6 个</span>
</div>
<div style="font-size: 11.5px; color: var(--text-secondary); line-height: 1.45; margin-bottom: 6px;">
<strong>机理：</strong>柱内收盘位置资金流（CLV）、多空主动买卖失衡度与能量潮增减仓，度量微观筹码真实动向。
</div>
</div>
<div style="font-size: 11px; background: #fcf9f2; padding: 5px 7px; border-radius: 2px; color: var(--text-muted); line-height: 1.4;">
<strong>代表因子：</strong><code>DC01L/S</code> 量价资金流CLV共振 · <code>DC02L/S</code> NTMDK-KCATR多空共振放量突破 · <code>A01L/S</code> OBV能量潮领先突破破位
</div>
</div>

<!-- Cluster E -->
<div class="card" style="padding: 10px 12px; border: 1px solid var(--border-color); border-left: 3.5px solid var(--purple); border-radius: 4px; display: flex; flex-direction: column; justify-content: space-between;">
<div>
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
<strong style="color: var(--purple); font-size: 13px;">E 簇 · 波动率压缩与分形弹性簇</strong>
<span style="font-size: 10.5px; color: var(--text-muted); background: #f6f4f8; padding: 1px 6px; border-radius: 2px;">8 个</span>
</div>
<div style="font-size: 11.5px; color: var(--text-secondary); line-height: 1.45; margin-bottom: 6px;">
<strong>机理：</strong>波动率进入临界收缩态（Squeeze）后的弹性释放，捕捉变盘点火与剧烈位移脉冲。
</div>
</div>
<div style="font-size: 11px; background: #faf9fb; padding: 5px 7px; border-radius: 2px; color: var(--text-muted); line-height: 1.4;">
<strong>代表因子：</strong><code>EC01L/S</code> 挤压后波动率扩张点火 · <code>EC02L/S</code> NR7窄幅极致收缩破局 · <code>A02L/S</code> 肯特纳挤压点火 · <code>A03L/S</code> 缩量星线后放量长阳破局
</div>
</div>

<!-- Cluster F -->
<div class="card" style="padding: 10px 12px; border: 1px solid var(--border-color); border-left: 3.5px solid #26333f; border-radius: 4px; display: flex; flex-direction: column; justify-content: space-between;">
<div>
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
<strong style="color: #26333f; font-size: 13px;">F 簇 · 多周期共振与截面强弱簇</strong>
<span style="font-size: 10.5px; color: var(--text-muted); background: #edf1f5; padding: 1px 6px; border-radius: 2px;">8 个</span>
</div>
<div style="font-size: 11.5px; color: var(--text-secondary); line-height: 1.45; margin-bottom: 6px;">
<strong>机理：</strong>跨周期大级别形态锚定小级别点火，以及跨资产截面板块龙头溢价与比价均值回归。
</div>
</div>
<div style="font-size: 11px; background: #f3f6f9; padding: 5px 7px; border-radius: 2px; color: var(--text-muted); line-height: 1.4;">
<strong>代表因子：</strong><code>FC01L/S</code> 大级别锚定小级别分型点火 · <code>FC02L/S</code> 板块龙头共振补涨补跌 · <code>FC03L/S</code> 截面比价背驰反包 · <code>FC04L/S</code> Top-K斜率时序共振
</div>
</div>
</div>

<!-- Bottom: Cross-Scale Heterogeneous Dedicated Factors Table -->
<div style="border-top: 1px solid var(--border-strong); padding-top: 8px;">
<div style="font-size: 12px; font-weight: 700; color: var(--primary); margin-bottom: 5px; display: flex; justify-content: space-between;">
<span>跨尺度异构专属因子库 (专为 1d、15m 与 5m 物理尺度量身定制，拒绝同质平移)</span>
<span style="font-size: 11px; color: var(--text-muted);">1d慢变量(6) + 15m低换手(6) + 5m订单流(6) = 18 个专属因子</span>
</div>
<div class="grid-3" style="gap: 12px; font-size: 11px; color: var(--text-secondary); line-height: 1.45;">
<div style="background: #f8fafc; padding: 6px 9px; border-radius: 3px; border: 1px solid #e2e8f0;">
<strong style="color: var(--primary);">1d 日线慢变量因子 (6个):</strong><br/>
<code>H1D_SP01</code> 板块价差离散回归 · <code>H1D_DV01</code> 日线乖离极值反转 · <code>H1D_RV01</code> 截面强弱均值回归<br/>
<span style="color: var(--text-muted);">持仓 2~8 交易日，与日内信号相关性 &lt; 0.20，平滑隔夜波动</span>
</div>
<div style="background: #f8fafc; padding: 6px 9px; border-radius: 3px; border: 1px solid #e2e8f0;">
<strong style="color: var(--accent);">15m 低换手波段因子 (6个):</strong><br/>
<code>H15M_EX01</code> 放量恐慌衰竭长影线反转 · <code>H15M_PB01</code> 顺势回踩MA60企稳 · <code>H15M_CL01</code> 波动率挤压脱离<br/>
<span style="color: var(--text-muted);">波段持有 24~80 根柱（1.5~5天），配置 32 根柱冷却期，拒绝过度交易</span>
</div>
<div style="background: #f8fafc; padding: 6px 9px; border-radius: 3px; border: 1px solid #e2e8f0;">
<strong style="color: var(--purple);">5m 微观成交流与持仓量因子 (6个):</strong><br/>
<code>H5M_CLV01</code> 尾盘抢筹放量回归 · <code>H5M_LQ01</code> 持仓量砍仓出清反转 · <code>H5M_OIB01</code> 持仓量激增顺势启动<br/>
<span style="color: var(--text-muted);">利用 TqSdk 持仓量(OI)与分笔主动买卖盘，捕获不可见增减仓资金流</span>
</div>
</div>
</div>
</div>
<div class="slide-footer">
<span>六大战略原型因子通过两阶段状态机与连续质量分，实现了从离散二值脉冲向连续风险权重的完整跨越</span>
<span>Slide 06 of 15</span>
</div>
</div>
"""

SLIDE_7_HTML = """<!-- ==================== SLIDE 7: 底层信号引擎：Rust 原生 240+ 信号分类与微观特征架构 ==================== -->
<div aria-label="第 7 页，共 15 页" class="slide" data-cat="Native Signal Engine" data-title="07. 底层信号引擎：Rust 原生 240+ 信号分类" id="slide-7">
<div class="slide-header">
<div class="slide-header-left">
<div class="slide-category">NATIVE SIGNAL ENGINE &amp; FEATURE TOPOLOGY</div>
<h2 class="slide-title">底层信号引擎：Rust 原生 240+ 信号分类与微观特征架构</h2>
<div class="slide-subtitle">基于 crates/czsc-signals 与 czsc-ta 的 17 大类原生技术算子矩阵，零 Python 回退的高性能微观信号池</div>
</div>
<span class="badge-pill">Rust 纯原语 · 17大类 · 240+ 原生信号</span>
</div>
<div class="slide-content">
<div class="grid-2" style="grid-template-columns: 1.15fr 1fr; gap: 18px; flex: 1;">
<!-- Left Column: 17 Categories Matrix Table -->
<div style="display: flex; flex-direction: column; justify-content: space-between;">
<div style="font-size: 13px; font-weight: 700; color: var(--primary); margin-bottom: 6px; display: flex; justify-content: space-between;">
<span>Rust 原生信号 17 大类全景矩阵 (czsc._native.signals)</span>
<span style="font-size: 11px; color: var(--text-muted);">共 240 个已注册原子算子</span>
</div>
<table class="table-academic" style="font-size: 11px; line-height: 1.45;">
<thead>
<tr>
<th style="width: 14%;">模块分类</th>
<th style="width: 11%;">信号数量</th>
<th style="width: 45%;">核心算法算子与技术指标</th>
<th style="width: 30%;">策略应用场景</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>tas</code></td>
<td><strong>61 个</strong></td>
<td>EMA, SMA, DEMA, MACD, BOLL, RSI, KDJ, ATR, CCI, CMO, Bias, Keltner, Donchian</td>
<td>趋势状态判断、均线穿越、通道边界</td>
</tr>
<tr>
<td><code>bar</code></td>
<td><strong>48 个</strong></td>
<td>三连阴阳, 吞没反包, 缺口跳空, 长影线锤子/流星, 内孕线, 十字星变盘</td>
<td>K线微观多空博弈、单/双柱强弱确认</td>
</tr>
<tr>
<td><code>cxt</code></td>
<td><strong>42 个</strong></td>
<td>分型确认(顶/底/强), 笔背驰/开始/结束/斜率, 线段端点, 中枢进出/三买三卖</td>
<td>缠论原生分形几何学与中枢形态定位</td>
</tr>
<tr>
<td><code>jcc</code></td>
<td><strong>19 个</strong></td>
<td>MACD零轴上下金叉死叉, 均线排列金叉死叉持续期, KDJ极值交叉</td>
<td>经典技术指标金叉死叉持续期状态机</td>
</tr>
<tr>
<td><code>zdy</code></td>
<td><strong>14 个</strong></td>
<td>双顺通道(大周期EMA斜率+通道共振), 均线缠绕发散, 动量加权突破</td>
<td>复合逻辑事件、多指标顺势过滤</td>
</tr>
<tr>
<td><code>strength</code></td>
<td><strong>12 个</strong></td>
<td><code>strength_fx</code>, <code>strength_bi_div</code>, <code>strength_zs_dev</code>, <code>strength_vol_flow</code></td>
<td><strong>连续质量分 Q∈(0,1]</strong>，专供上层连续定价</td>
</tr>
<tr>
<td><code>xl</code></td>
<td><strong>7 个</strong></td>
<td>笔线性回归斜率, 线段速度与动量加速度, 跨周期斜率加速比</td>
<td>趋势强弱度量与斜率动量因子核心输入</td>
</tr>
<tr>
<td><code>vol</code></td>
<td><strong>6 个</strong></td>
<td>倍量柱, 梯量柱(持续增量), 高量柱, 黄金柱, 地量极限萎缩</td>
<td>成交量异动、放量突破与缩量洗盘识别</td>
</tr>
<tr>
<td><code>byi / coo</code></td>
<td><strong>10 个</strong></td>
<td>摆动极值指标, 跨周期价格协整残差与滑动相关系数</td>
<td>超买超卖震荡与跨周期价差协整回归</td>
</tr>
<tr>
<td><code>pressure 等</code></td>
<td><strong>21 个</strong></td>
<td>支撑阻力位触碰, R-Breaker枢轴点, 能量潮(OBV), 筹码分布(CVOLP)</td>
<td>关键位破位与市场能量潮突破</td>
</tr>
</tbody>
</table>
</div>

<!-- Right Column: Key Operators & Architecture -->
<div style="display: flex; flex-direction: column; gap: 10px;">
<!-- Strength Operators Card -->
<div class="card" style="padding: 9px 12px; border: 1px solid var(--border-color); border-left: 3.5px solid var(--primary); border-radius: 4px;">
<div style="font-size: 12.5px; font-weight: 700; color: var(--primary); margin-bottom: 4px;">
核心机制 1：连续结构强度算子 (strength_*)
</div>
<div style="font-size: 11.5px; color: var(--text-secondary); line-height: 1.45;">
传统技术分析仅输出 {-1, 0, +1} 二值脉冲，本系统在 Rust 核心层构建连续强度解码：<br/>
• <strong>分型强度:</strong> <span class="math"><i>Q</i><sub>fx</sub> = Clip(100·|<i>C</i>−<i>O</i>|/ATR·(1−Overlap), 0, 100)</span> 衡量分型实体穿透力；<br/>
• <strong>笔背驰强度:</strong> <span class="math"><i>Q</i><sub>div</sub> = Round(100·Max(0, 1−MACD<sub>curr</sub>/MACD<sub>prev</sub>)·Δ<i>P</i><sub>ratio</sub>)</span>；<br/>
• <strong>中枢偏离度:</strong> <span class="math"><i>Z</i><sub>zs</sub> = (<i>P</i><sub><i>t</i></sub> − ZG)/ATR<sub>14</sub></span> 衡量价格脱离中枢的拉回张力。
</div>
</div>

<!-- Volume/Flow Operators Card -->
<div class="card" style="padding: 9px 12px; border: 1px solid var(--border-color); border-left: 3.5px solid var(--accent); border-radius: 4px;">
<div style="font-size: 12.5px; font-weight: 700; color: var(--accent); margin-bottom: 4px;">
核心机制 2：微观量价与通道挤压算子 (tas, vol, bar)
</div>
<div style="font-size: 11.5px; color: var(--text-secondary); line-height: 1.45;">
• <strong>柱内资金流向 (CLV):</strong> <span class="math">CLV = [(<i>C</i>−<i>L</i>) − (<i>H</i>−<i>C</i>)] / (<i>H</i>−<i>L</i>) ∈ [−1, 1]</span>，精准度量单 Bar 多空主动买卖意愿；<br/>
• <strong>肯特纳/布林挤压比 (Squeeze):</strong> <span class="math">SQZ = Bandwidth<sub>BOLL</sub> / Bandwidth<sub>KC</sub> &lt; 1.0</span> 识别波动率极致压抑临界态；<br/>
• <strong>量柱序列状态机:</strong> 严格按 Rust 原生向量化判定梯量柱（量价双升）与黄金柱。
</div>
</div>

<!-- Constitution Parity Card -->
<div class="card card-success" style="padding: 9px 12px; border: 1px solid var(--border-color); border-left: 3.5px solid var(--success); border-radius: 4px; background: #ffffff;">
<div style="font-size: 12.5px; font-weight: 700; color: var(--success); margin-bottom: 4px;">
架构铁律：Rust ↔ Python 行为绝对一致（宪法第一条）
</div>
<div style="font-size: 11.5px; color: var(--text-secondary); line-height: 1.45;">
• <strong>零 Python 回退:</strong> 240+ 个信号函数通过 <code>#[signal]</code> 过程宏在 Rust 编译期静态注册到 <code>SIGNAL_REGISTRY</code>，纳秒级增量计算；<br/>
• <strong>无任何适配层:</strong> Python 侧 <code>czsc._native</code> 纯透传调用，禁止参数二次补齐或错误码包装；<br/>
• <strong>Byte-for-byte 对齐:</strong> 保证 Rust 生产环境与 Python 研发回测环境完全同构，消灭任何执行漂移。
</div>
</div>
</div>
</div>
</div>
<div class="slide-footer">
<span>crates/czsc-signals 与 czsc-ta 构筑了业内最完备的高性能缠论与技术分析底层算子底座</span>
<span>Slide 07 of 15</span>
</div>
</div>
"""

def update_presentation():
    with open(TARGET_DOCS_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # Split slides
    parts = re.split(r"(<!-- ==================== SLIDE \d+:[^>]*==================== -->\s*<div[^>]*class=[\x27\"]slide(?:\s+active)?[\x27\"][^>]*id=[\x27\"]slide-\d+[\x27\"][^>]*>)", content)

    # If parts is not split well with comment, split by <div class="slide ...
    pattern = re.compile(r"(<div[^>]*class=[\x27\"]slide(?:\s+active)?[\x27\"][^>]*id=[\x27\"]slide-(\d+)[\x27\"][^>]*>)", re.DOTALL)
    
    # Let's find all slide boundaries
    slide_starts = list(pattern.finditer(content))
    print(f"Found {len(slide_starts)} existing slides.")
    assert len(slide_starts) == 13, f"Expected 13 slides, found {len(slide_starts)}"

    # Cut content into segments
    # prefix: before slide 1
    # slide 1 .. 5
    # slide 6 .. 13
    # suffix: after slide 13
    
    # Find start and end of each slide
    slide_spans = []
    for i in range(len(slide_starts)):
        start = slide_starts[i].start()
        # Find corresponding comment before start if any
        com_start = content.rfind("<!-- ==================== SLIDE", max(0, start - 200), start)
        if com_start != -1:
            start = com_start
        if i + 1 < len(slide_starts):
            # next start
            next_start = slide_starts[i+1].start()
            next_com = content.rfind("<!-- ==================== SLIDE", max(0, next_start - 200), next_start)
            if next_com != -1:
                end = next_com
            else:
                end = next_start
        else:
            # find end of last slide (before overview modal)
            mod_start = content.find("<!-- ==================== OVERVIEW MODAL", start)
            if mod_start == -1:
                mod_start = content.find("<div id=\"overview-modal\"", start)
            end = mod_start
        slide_spans.append((start, end))

    prefix = content[:slide_spans[0][0]]
    suffix = content[slide_spans[-1][1]:]

    old_slides = [content[s:e] for s, e in slide_spans]

    # Process old slides 1 to 5 (update 'of 13' to 'of 15', '共 13 页' to '共 15 页')
    new_slides = []
    for i in range(5):
        s_text = old_slides[i]
        s_text = s_text.replace("of 13", "of 15").replace("共 13 页", "共 15 页")
        new_slides.append(s_text)

    # Insert Slide 6 and Slide 7
    new_slides.append(SLIDE_6_HTML.strip() + "\n")
    new_slides.append(SLIDE_7_HTML.strip() + "\n")

    # Process old slides 6 to 13 -> become 8 to 15
    for old_idx in range(5, 13):
        new_idx = old_idx + 3  # old slide 6 (0-indexed 5) becomes slide 8 (1-indexed 8)
        s_text = old_slides[old_idx]
        
        # update comment
        s_text = re.sub(r"<!-- ==================== SLIDE \d+:", f"<!-- ==================== SLIDE {new_idx}:", s_text)
        # update id="slide-X"
        s_text = re.sub(r'id="slide-\d+"', f'id="slide-{new_idx}"', s_text)
        # update aria-label
        s_text = re.sub(r'aria-label="第 \d+ 页，共 \d+ 页"', f'aria-label="第 {new_idx} 页，共 15 页"', s_text)
        # update Slide XX of 13
        s_text = re.sub(r'Slide \d+ of \d+', f'Slide {new_idx:02d} of 15', s_text)
        # update data-title="XX. "
        def _repl_title(m):
            return f'data-title="{new_idx:02d}. '
        s_text = re.sub(r'data-title="\d+\.\s*', _repl_title, s_text)

        new_slides.append(s_text)

    full_updated_html = prefix + "\n".join(new_slides) + suffix

    with open(TARGET_DOCS_PATH, "w", encoding="utf-8") as f:
        f.write(full_updated_html)
    print(f"Updated {TARGET_DOCS_PATH} successfully! Total length: {len(full_updated_html)}")

    # Sync to examples/results/...
    TARGET_EXAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TARGET_EXAMPLES_PATH, "w", encoding="utf-8") as f:
        f.write(full_updated_html)
    print(f"Synced {TARGET_EXAMPLES_PATH} successfully!")

if __name__ == "__main__":
    update_presentation()
