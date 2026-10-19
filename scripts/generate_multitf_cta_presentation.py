"""
Generate an institutional-grade, academic clean-white HTML presentation slide deck
for the Multi-Timeframe (1d ~ 5m) × 6-Archetype Signals CTA research system.

Features:
- 100% strict alignment with empirical data and methodology from docs/research/multitf_5m_cta_research_report_20261018.md
- Accurate Walk-Forward protocol (Two-Fold Expanding Walk-Forward: 2024 IS -> 2025/2026 Stitched OOS)
- Accurate Gate 1/2/3 greedy forward admission criteria and exact Funnel counts
- Accurate Six Archetypes (A~F) and Hetero-factor descriptions
- Native 5m physical execution grid architecture with zero lookahead bias and friction accounting
- Clean, decoupled, deferred Plotly chart rendering (Slide 6 Funnel, Slide 8 NAV/DD, Slide 9 Symbol attribution)
  preventing zero-width collapse and title/legend overlap
- Crisp SVG diagrams for methodology, timeline, pyramid, funnel, grid execution, HRP, and production pipeline
"""

import json
import os
import shutil
import copy

REPORT_HTML_PATH = "examples/results/intraday_cta_ablation_20261017/multitf_cta_incremental_report.html"
TARGET_HTML_PATH = "examples/results/intraday_cta_ablation_20261017/multitf_cta_research_presentation.html"
DOCS_HTML_PATH = "docs/research/multitf_cta_research_presentation.html"


def extract_chart_data():
    with open(REPORT_HTML_PATH, "r", encoding="utf-8") as f:
        html = f.read()

    dec = json.JSONDecoder()

    # 1. Fig 1 (NAV & DD curves)
    idx1 = html.find('"b69fe4be')
    data1, end1 = dec.raw_decode(html, html.find("[", idx1))

    # 2. Fig 2 (Funnel + Symbol bars)
    idx2 = html.find('"04956e7d')
    data2, end2 = dec.raw_decode(html, html.find("[", idx2))

    funnel_data = data2[:4]
    symbol_data = copy.deepcopy(data2[4:])
    for t in symbol_data:
        t["xaxis"] = "x"
        t["yaxis"] = "y"

    return json.dumps(data1), json.dumps(funnel_data), json.dumps(symbol_data)


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>全周期跨尺度 CTA 策略研发与实证体系汇报 (PPT)</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<!-- Semantic HTML Math Typography (Native Chrome, Zero CDN Dependency) -->
<style>
  .math {
    font-family: 'Cambria Math', 'Latin Modern Math', 'STIX Two Math', 'Times New Roman', Georgia, serif;
    font-size: 1.05em;
    letter-spacing: 0.01em;
    white-space: nowrap;
    display: inline-block;
  }
  .math i {
    font-family: 'Cambria Math', 'Times New Roman', Georgia, serif;
    font-style: italic;
    padding: 0 1px;
  }
  .math sub, .math sup {
    font-size: 0.75em;
    line-height: 0;
    position: relative;
    vertical-align: baseline;
  }
  .math sub { bottom: -0.22em; }
  .math sup { top: -0.42em; }
  .math-block {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 3px solid #2563eb;
    border-radius: 6px;
    padding: 6px 12px;
    margin-top: 5px;
    font-family: 'Cambria Math', 'Times New Roman', Georgia, serif;
    font-size: 11.5px;
    color: #1e3a8a;
    text-align: center;
  }
  :root {
    --bg-page: #f8fafc;
    --bg-slide: #ffffff;
    --border-color: #e2e8f0;
    --border-strong: #cbd5e1;
    --text-main: #0f172a;
    --text-secondary: #334155;
    --text-muted: #64748b;
    --primary: #1e3a8a;
    --primary-light: #eff6ff;
    --primary-border: #bfdbfe;
    --accent: #2563eb;
    --success: #059669;
    --success-light: #ecfdf5;
    --success-border: #a7f3d0;
    --danger: #dc2626;
    --danger-light: #fef2f2;
    --danger-border: #fecaca;
    --purple: #7c3aed;
    --purple-light: #f5f3ff;
    --purple-border: #ddd6fe;
    --amber: #d97706;
    --amber-light: #fffbeb;
    --amber-border: #fde68a;
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.04);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.04);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.06), 0 4px 6px -4px rgba(0, 0, 0, 0.04);
  }

  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }

  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
    background-color: var(--bg-page);
    color: var(--text-main);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
    overflow: hidden;
    user-select: none;
    -webkit-font-smoothing: antialiased;
  }

  /* Top Navigation & Progress */
  #deck-header {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 48px;
    background: #ffffff;
    border-bottom: 1px solid var(--border-color);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 24px;
    z-index: 100;
    box-shadow: var(--shadow-sm);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .header-tag {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.05em;
    background: var(--primary-light);
    color: var(--primary);
    border: 1px solid var(--primary-border);
    padding: 2px 8px;
    border-radius: 4px;
    text-transform: uppercase;
  }

  .header-title {
    font-size: 13.5px;
    font-weight: 600;
    color: var(--text-secondary);
  }

  .header-right {
    display: flex;
    align-items: center;
    gap: 16px;
  }

  .slide-counter {
    font-size: 13px;
    font-weight: 700;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    color: var(--text-secondary);
    background: #f1f5f9;
    padding: 3px 10px;
    border-radius: 12px;
    border: 1px solid var(--border-color);
  }

  .header-btn {
    background: #ffffff;
    border: 1px solid var(--border-color);
    color: var(--text-secondary);
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 5px;
    transition: all 0.15s ease;
  }
  .header-btn:hover {
    background: #f8fafc;
    border-color: var(--border-strong);
    color: var(--text-main);
  }

  /* Top Progress Bar */
  #progress-bar {
    position: fixed;
    top: 48px;
    left: 0;
    height: 3px;
    background: linear-gradient(90deg, #3b82f6, #1d4ed8);
    width: 0%;
    transition: width 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    z-index: 101;
  }

  /* Main Slide Deck Canvas */
  #deck-container {
    width: 96vw;
    max-width: 1460px;
    height: calc(100vh - 108px);
    margin-top: 50px;
    background: var(--bg-slide);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    box-shadow: var(--shadow-lg);
    position: relative;
    overflow: hidden;
  }

  .slide {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    padding: 22px 32px;
    display: none;
    flex-direction: column;
    justify-content: space-between;
    overflow-y: auto;
    opacity: 0;
    transition: opacity 0.2s ease-in-out;
  }

  .slide.active {
    display: flex;
    opacity: 1;
  }

  /* Slide Typography & Structure */
  .slide-header {
    margin-bottom: 12px;
    border-bottom: 1px solid var(--border-color);
    padding-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
  }

  .slide-header-left {
    flex: 1;
  }

  .slide-category {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 3px;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .slide-category::before {
    content: "";
    display: inline-block;
    width: 6px;
    height: 6px;
    background: var(--accent);
    border-radius: 50%;
  }

  .slide-title {
    font-size: 20px;
    font-weight: 800;
    color: var(--text-main);
    letter-spacing: -0.02em;
    line-height: 1.25;
  }

  .slide-subtitle {
    font-size: 12px;
    color: var(--text-muted);
    margin-top: 2px;
  }

  .slide-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 12px;
    position: relative;
  }

  .slide-footer {
    margin-top: 8px;
    padding-top: 6px;
    border-top: 1px solid var(--border-color);
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 11px;
    color: var(--text-muted);
  }

  /* Bottom Control Bar */
  #deck-footer {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    height: 52px;
    background: #ffffff;
    border-top: 1px solid var(--border-color);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 24px;
    z-index: 100;
  }

  .footer-hints {
    font-size: 12px;
    color: var(--text-muted);
    display: flex;
    align-items: center;
    gap: 14px;
  }

  .key-hint {
    display: inline-flex;
    align-items: center;
    gap: 4px;
  }

  .kbd {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 11px;
    background: #f1f5f9;
    border: 1px solid var(--border-strong);
    border-radius: 4px;
    padding: 1px 6px;
    color: var(--text-secondary);
    box-shadow: 0 1px 0 rgba(0,0,0,0.1);
  }

  .dots-nav {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .dot-pill {
    width: 8px;
    height: 8px;
    border-radius: 4px;
    background: #cbd5e1;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .dot-pill.active {
    width: 24px;
    background: var(--accent);
  }

  .footer-controls {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .nav-btn {
    background: var(--primary-light);
    color: var(--primary);
    border: 1px solid var(--primary-border);
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.15s ease;
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }

  .nav-btn:hover:not(:disabled) {
    background: #dbeafe;
    color: #1e40af;
  }

  .nav-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  /* Card and Component Styles */
  .grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
  }

  .grid-3 {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 12px;
  }

  .grid-4 {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 11px;
  }

  .card {
    background: #ffffff;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 11px 14px;
    box-shadow: var(--shadow-sm);
  }

  .card-highlight {
    border-left: 4px solid var(--accent);
  }

  .card-success {
    border-left: 4px solid var(--success);
    background: #fbfdfc;
  }

  .card-purple {
    border-left: 4px solid var(--purple);
    background: #fdfcff;
  }

  .card-title {
    font-size: 12.5px;
    font-weight: 700;
    color: var(--text-main);
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .card-desc {
    font-size: 11px;
    color: var(--text-secondary);
    line-height: 1.48;
  }

  /* KPI Badges */
  .kpi-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 11px;
  }

  .kpi-box {
    background: #ffffff;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 9px 12px;
    box-shadow: var(--shadow-sm);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }

  .kpi-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
  }

  .kpi-value {
    font-size: 19px;
    font-weight: 800;
    color: var(--text-main);
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    margin: 2px 0 1px 0;
  }

  .kpi-badge {
    font-size: 10.5px;
    font-weight: 600;
    display: inline-block;
  }

  .badge-success {
    color: var(--success);
  }

  .badge-danger {
    color: var(--danger);
  }

  .badge-pill {
    display: inline-block;
    padding: 2px 7px;
    border-radius: 12px;
    font-size: 10.5px;
    font-weight: 600;
  }
  .pill-blue { background: var(--primary-light); color: var(--primary); border: 1px solid var(--primary-border); }
  .pill-green { background: var(--success-light); color: var(--success); border: 1px solid var(--success-border); }
  .pill-purple { background: var(--purple-light); color: var(--purple); border: 1px solid var(--purple-border); }
  .pill-amber { background: var(--amber-light); color: var(--amber); border: 1px solid var(--amber-border); }
  .pill-red { background: var(--danger-light); color: var(--danger); border: 1px solid var(--danger-border); }

  /* Academic Table */
  .table-academic {
    width: 100%;
    border-collapse: collapse;
    font-size: 11px;
    background: #ffffff;
    border-radius: 6px;
    overflow: hidden;
    border: 1px solid var(--border-color);
  }

  .table-academic th {
    background: #f8fafc;
    color: var(--text-secondary);
    font-weight: 700;
    text-align: left;
    padding: 5px 8px;
    border-bottom: 1.5px solid var(--border-strong);
    font-size: 10.5px;
    letter-spacing: 0.02em;
  }

  .table-academic td {
    padding: 5px 8px;
    border-bottom: 1px solid var(--border-color);
    color: var(--text-secondary);
  }

  .table-academic tr:hover td {
    background: #f8fafc;
  }

  .table-academic tr.highlight-row td {
    background: #eff6ff;
    font-weight: 700;
    color: #1e3a8a;
  }

  /* Overview Modal */
  #overview-modal {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(15, 23, 42, 0.7);
    backdrop-filter: blur(4px);
    z-index: 200;
    display: none;
    align-items: center;
    justify-content: center;
    padding: 40px;
  }

  #overview-modal.active {
    display: flex;
  }

  .overview-grid {
    background: #ffffff;
    border-radius: 12px;
    width: 90vw;
    max-width: 1200px;
    max-height: 85vh;
    overflow-y: auto;
    padding: 24px;
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    box-shadow: var(--shadow-lg);
  }

  .overview-item {
    border: 1.5px solid var(--border-color);
    border-radius: 8px;
    padding: 10px 12px;
    cursor: pointer;
    transition: all 0.2s ease;
    background: #ffffff;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 105px;
  }

  .overview-item:hover {
    border-color: var(--accent);
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
  }

  .overview-item.active {
    border-color: var(--accent);
    background: #eff6ff;
  }

  .ov-num {
    font-size: 11px;
    font-weight: 800;
    color: var(--accent);
    font-family: monospace;
  }

  .ov-title {
    font-size: 12px;
    font-weight: 700;
    color: var(--text-main);
    line-height: 1.35;
    margin-top: 3px;
  }

  .ov-cat {
    font-size: 9.5px;
    color: var(--text-muted);
    text-transform: uppercase;
  }

  /* Fullscreen adjustments */
  :fullscreen #deck-container {
    width: 98vw;
    height: calc(100vh - 100px);
    margin-top: 46px;
    max-width: none;
    max-height: none;
    border-radius: 0;
    border: none;
  }
</style>
</head>
<body>

<!-- Header Navigation -->
<div id="deck-header">
  <div class="header-left">
    <span class="header-tag">CZSC Quant Research</span>
    <span class="header-title" id="header-section-title">01. 封面与核心摘要</span>
  </div>
  <div class="header-right">
    <div class="slide-counter" id="slide-counter">01 / 12</div>
    <button class="header-btn" onclick="toggleOverview()" title="显示所有页面预览 (快捷键: O)">⊞ 幻灯片概览</button>
    <button class="header-btn" onclick="toggleFullscreen()" title="切换全屏模式 (快捷键: F)">⛶ 全屏演示</button>
  </div>
</div>

<!-- Progress Bar -->
<div id="progress-bar"></div>

<!-- Slide Deck Container -->
<div id="deck-container">

  <!-- ==================== SLIDE 1: 封面 ==================== -->
  <div class="slide active" id="slide-1" data-title="01. 封面与项目总览" data-cat="Overview">
    <div class="slide-header">
      <div class="slide-header-left">
        <div class="slide-category">RESEARCH PRESENTATION · 2026</div>
        <h1 class="slide-title">全周期跨尺度量化 CTA 策略研发与实证体系</h1>
        <div class="slide-subtitle">Multi-Timeframe (1d ~ 5m) × 6-Archetype Signals with 3-Stage Incremental Funnel & Native 5m Micro-Grid</div>
      </div>
      <div>
        <span class="pill-blue badge-pill">纯 Rust 计算引擎</span>
        <span class="pill-purple badge-pill">5级时间尺度</span>
        <span class="pill-green badge-pill">纯盲测 503 交易日</span>
      </div>
    </div>

    <div class="slide-content" style="justify-content: center; gap: 16px;">
      <!-- Core Out-of-Sample Performance Summary Strip (Clean Academic White) -->
      <div style="background: #ffffff; border: 1px solid var(--border-color); border-radius: 8px; padding: 14px 18px; box-shadow: var(--shadow-sm);">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #f1f5f9; padding-bottom: 8px; margin-bottom: 12px;">
          <div style="font-size: 11px; color: var(--text-muted); font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;">
            全样本外两折前向验证实证指标 (Stitched OOS: 2025.01 ~ 2026.09 · 503 结算日 / 21 个月)
          </div>
          <span class="pill-green badge-pill">纯盲测拼接 · 真实扣费 3.0 BP</span>
        </div>

        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px;">
          <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-left: 3.5px solid #2563eb; border-radius: 6px; padding: 10px 14px;">
            <div style="font-size: 11px; color: var(--text-secondary); font-weight: 600;">年化夏普比率 (Sharpe)</div>
            <div style="font-size: 30px; font-weight: 800; color: #1e3a8a; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.15; margin: 4px 0 2px 0;">3.03</div>
            <div style="font-size: 10.5px; color: #059669; font-weight: 600;">基准 ExpA (1.25) 提升 +142%</div>
          </div>

          <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-left: 3.5px solid #059669; border-radius: 6px; padding: 10px 14px;">
            <div style="font-size: 11px; color: var(--text-secondary); font-weight: 600;">卡玛比率 (Calmar Ratio)</div>
            <div style="font-size: 30px; font-weight: 800; color: #065f46; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.15; margin: 4px 0 2px 0;">5.64</div>
            <div style="font-size: 10.5px; color: var(--text-muted);">最大回撤仅 1.26% (年化 7.10%)</div>
          </div>

          <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-left: 3.5px solid #7c3aed; border-radius: 6px; padding: 10px 14px;">
            <div style="font-size: 11px; color: var(--text-secondary); font-weight: 600;">商品品种盲测胜率</div>
            <div style="font-size: 30px; font-weight: 800; color: #5b21b6; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.15; margin: 4px 0 2px 0;">70.0%</div>
            <div style="font-size: 10.5px; color: #059669; font-weight: 600;">20 大主流期货中 14 个正收益</div>
          </div>

          <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-left: 3.5px solid #d97706; border-radius: 6px; padding: 10px 14px;">
            <div style="font-size: 11px; color: var(--text-secondary); font-weight: 600;">实盘目标敞口版 (3.0x)</div>
            <div style="font-size: 30px; font-weight: 800; color: #92400e; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.15; margin: 4px 0 2px 0;">+42.5%</div>
            <div style="font-size: 10.5px; color: var(--text-muted);">年化 +21.30% · 最大回撤 3.58%</div>
          </div>
        </div>
      </div>

      <!-- Key Methodology & Paradigm Cards -->
      <div class="grid-3">
        <div class="card card-highlight">
          <div class="card-title">
            <span>🔬 演绎先验 × 归纳证伪</span>
            <span class="pill-blue badge-pill">方法论</span>
          </div>
          <div class="card-desc">
            拒绝盲目暴力数据挖掘与事后过拟合；以缠论几何学与市场微观博弈为严密演绎先验，解耦【势能蓄力】与【反包确认】两阶段状态机，严格多空对称。
          </div>
        </div>

        <div class="card card-purple">
          <div class="card-title">
            <span>🛡️ 三阶段增量信息漏斗</span>
            <span class="pill-purple badge-pill">信号门禁</span>
          </div>
          <div class="card-desc">
            从 5,880 个组合逐级净化至 622 个正交因子。Gate 1 周期自适应显著性，Gate 2 激活条件低共线排异与跨品种共振，Gate 3 扣费边际净夏普增益，摩擦压降 41%。
          </div>
        </div>

        <div class="card card-success">
          <div class="card-title">
            <span>⚡ 原生 5m 即时微观网格</span>
            <span class="pill-green badge-pill">执行架构</span>
          </div>
          <div class="card-desc">
            以原生 5m 物理网格为统一时钟底座，毫秒级捕获 5m 持仓量脉冲 (Micro OI) 与即时反包微观结构，夏普从 1.81 质跃至 3.03，卡玛比率达 5.64。
          </div>
        </div>
      </div>

      <!-- Scope and Specs -->
      <div style="background: #f8fafc; border: 1px solid var(--border-color); border-radius: 8px; padding: 9px 14px; display: flex; justify-content: space-between; align-items: center; font-size: 11px; color: var(--text-secondary);">
        <div><strong>覆盖标的:</strong> Core-20 主力商品期货（沪金、沪银、原油、铜、锌、铁矿、白糖、棕榈等）</div>
        <div><strong>验证时钟:</strong> 两折扩张窗口前向验证 (Two-Fold Expanding Walk-Forward, 2024~2026.09)</div>
        <div><strong>摩擦核算:</strong> 严格扣减手续费与滑点 (沪金单边 1.0 BP，其余 19 品种单边 1.5 BP / 双边 3.0 BP)</div>
      </div>
    </div>

    <div class="slide-footer">
      <span>CZSC Quantitative Strategy Research & Production Group · Institutional Grade</span>
      <span>Slide 01 of 12</span>
    </div>
  </div>

  <!-- ==================== SLIDE 2: 核心成果全景 ==================== -->
  <div class="slide" id="slide-2" data-title="02. 核心成果全景：三重跃迁" data-cat="Executive Summary">
    <div class="slide-header">
      <div class="slide-header-left">
        <div class="slide-category">EXECUTIVE SUMMARY</div>
        <h2 class="slide-title">核心成果全景：从基准到原生 5m 的三重跃迁</h2>
        <div class="slide-subtitle">从粗放盲堆全周期到三级漏斗治理，再到快慢协同与 5m 原生网格的阿尔法全面释放</div>
      </div>
      <span class="pill-green badge-pill">纯样本外 503 交易日严苛验证</span>
    </div>

    <div class="slide-content">
      <!-- 3 Paradigm Shifts SVG Diagram -->
      <div style="background: #ffffff; border: 1px solid var(--border-color); border-radius: 8px; padding: 10px 14px; box-shadow: var(--shadow-sm);">
        <svg viewBox="0 0 960 190" style="width: 100%; height: auto; display: block;" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif">
          <defs>
            <marker id="arrow-blue" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 1 L 10 5 L 0 9 z" fill="#2563eb" />
            </marker>
            <marker id="arrow-purple" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 1 L 10 5 L 0 9 z" fill="#7c3aed" />
            </marker>
            <marker id="arrow-green" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 1 L 10 5 L 0 9 z" fill="#059669" />
            </marker>
          </defs>

          <!-- Step 0: ExpA Baseline -->
          <g transform="translate(10, 15)">
            <rect width="180" height="155" rx="7" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
            <rect width="180" height="26" rx="7" fill="#64748b"/>
            <text x="90" y="18" fill="#ffffff" font-size="11.5" font-weight="700" text-anchor="middle">基准 ExpA (60m+30m)</text>
            <text x="14" y="50" fill="#475569" font-size="11">净收益: <tspan font-weight="700" fill="#0f172a">5.99%</tspan> (年化 3.0%)</text>
            <text x="14" y="72" fill="#475569" font-size="11">夏普比率: <tspan font-weight="700" fill="#0f172a">1.25</tspan></text>
            <text x="14" y="94" fill="#475569" font-size="11">最大回撤: <tspan font-weight="700" fill="#dc2626">1.89%</tspan></text>
            <text x="14" y="116" fill="#475569" font-size="11">卡玛比率: <tspan font-weight="700" fill="#0f172a">1.59</tspan></text>
            <text x="14" y="138" fill="#64748b" font-size="10.5">年化换手: 86.0x | 摩擦: 2.44%</text>
          </g>

          <!-- Arrow 1: Warning to ExpB -->
          <path d="M 195 85 L 225 85" stroke="#94a3b8" stroke-width="2" stroke-dasharray="3,3" marker-end="url(#arrow-blue)"/>
          <text x="210" y="75" fill="#dc2626" font-size="9.5" font-weight="700" text-anchor="middle">盲堆换手激增</text>

          <!-- Step 1: ExpC Funnel Leap -->
          <g transform="translate(235, 15)">
            <rect width="215" height="155" rx="7" fill="#eff6ff" stroke="#bfdbfe" stroke-width="1.5"/>
            <rect width="215" height="26" rx="7" fill="#2563eb"/>
            <text x="107" y="18" fill="#ffffff" font-size="11.5" font-weight="700" text-anchor="middle">跃迁一: 漏斗治理 (ExpC)</text>
            <text x="14" y="50" fill="#1e3a8a" font-size="11">门禁体系: <tspan font-weight="700">Gate 1/2/3 增量漏斗</tspan></text>
            <text x="14" y="72" fill="#1e3a8a" font-size="11">摩擦成本: <tspan font-weight="700" fill="#059669">1.98% (大幅下降 41%)</tspan></text>
            <text x="14" y="94" fill="#1e3a8a" font-size="11">净收益: <tspan font-weight="700">7.69%</tspan> | 换手: <tspan font-weight="700">70.7x</tspan></text>
            <text x="14" y="116" fill="#1e3a8a" font-size="11">夏普比率: <tspan font-weight="700" fill="#2563eb">1.72</tspan> (大幅跃升 +38%)</text>
            <text x="14" y="138" fill="#64748b" font-size="10.5">回撤收敛至 1.77% | Calmar 2.18</text>
          </g>

          <!-- Arrow 2: To ExpE -->
          <path d="M 455 85 L 490 85" stroke="#7c3aed" stroke-width="2.5" marker-end="url(#arrow-purple)"/>
          <text x="472" y="75" fill="#7c3aed" font-size="9.5" font-weight="700" text-anchor="middle">注入异构因子</text>

          <!-- Step 2: ExpE Hetero Synergy -->
          <g transform="translate(500, 15)">
            <rect width="205" height="155" rx="7" fill="#f5f3ff" stroke="#ddd6fe" stroke-width="1.5"/>
            <rect width="205" height="26" rx="7" fill="#7c3aed"/>
            <text x="102" y="18" fill="#ffffff" font-size="11.5" font-weight="700" text-anchor="middle">跃迁二: 快慢协同 (ExpE)</text>
            <text x="14" y="50" fill="#5b21b6" font-size="11">协同机制: <tspan font-weight="700">1d定势 × 15m微观反包</tspan></text>
            <text x="14" y="72" fill="#5b21b6" font-size="11">异构设计: <tspan font-weight="700">补充专属非重叠特征</tspan></text>
            <text x="14" y="94" fill="#5b21b6" font-size="11">净收益: <tspan font-weight="700">8.51%</tspan> (年化 4.27%)</text>
            <text x="14" y="116" fill="#5b21b6" font-size="11">夏普比率: <tspan font-weight="700" fill="#7c3aed">1.81</tspan> | 胜率: 50.3%</text>
            <text x="14" y="138" fill="#64748b" font-size="10.5">克服跨周期共振与盲目抵消</text>
          </g>

          <!-- Arrow 3: To ExpF -->
          <path d="M 710 85 L 745 85" stroke="#059669" stroke-width="3" marker-end="url(#arrow-green)"/>
          <text x="728" y="75" fill="#059669" font-size="9.5" font-weight="800" text-anchor="middle">原生5m网格</text>

          <!-- Step 3: ExpF Native 5M Grid -->
          <g transform="translate(755, 15)">
            <rect width="195" height="155" rx="7" fill="#ecfdf5" stroke="#a7f3d0" stroke-width="2"/>
            <rect width="195" height="26" rx="7" fill="#059669"/>
            <text x="97" y="18" fill="#ffffff" font-size="11.5" font-weight="800" text-anchor="middle">跃迁三: 原生5m网格 (ExpF)</text>
            <text x="12" y="50" fill="#065f46" font-size="11">物理网格: <tspan font-weight="800">原生5m时钟 + 5m微观OI</tspan></text>
            <text x="12" y="72" fill="#065f46" font-size="11">净收益: <tspan font-weight="800" fill="#059669">14.17% (+66.5%)</tspan></text>
            <text x="12" y="94" fill="#065f46" font-size="11">年化夏普: <tspan font-weight="800" fill="#059669">3.03 (+67.4%)</tspan></text>
            <text x="12" y="116" fill="#065f46" font-size="11">最大回撤: <tspan font-weight="800" fill="#059669">1.26% (-35.1%)</tspan></text>
            <text x="12" y="138" fill="#065f46" font-size="10.5">卡玛比率: <tspan font-weight="800">5.64</tspan> | 日胜率: <tspan font-weight="800">55.3%</tspan></text>
          </g>
        </svg>
      </div>

      <!-- Ablation Summary Table -->
      <div>
        <table class="table-academic">
          <thead>
            <tr>
              <th>实验版本</th>
              <th>核心特征与机制架构</th>
              <th>单标的因子数</th>
              <th>毛收益率</th>
              <th>摩擦成本</th>
              <th>纯净收益</th>
              <th>年化夏普</th>
              <th>最大回撤</th>
              <th>卡玛比率</th>
              <th>年化换手</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>ExpA</strong> (基准)</td>
              <td>仅日内 (60m+30m) + 标准 Gate 1 + Comb2 均权</td>
              <td>15.2</td>
              <td>8.43%</td>
              <td>2.44%</td>
              <td>5.99%</td>
              <td>1.25</td>
              <td>1.89%</td>
              <td>1.59</td>
              <td>86.0x</td>
            </tr>
            <tr style="color: #94a3b8;">
              <td><strong>ExpB</strong> (盲堆)</td>
              <td>全周期 (1d+60m+30m+15m) + 仅 Gate 1 筛因子 (无漏斗)</td>
              <td>25.2</td>
              <td>12.58%</td>
              <td>3.34%</td>
              <td>9.24%</td>
              <td>1.32</td>
              <td>2.38%</td>
              <td>1.95</td>
              <td>118.0x <span style="color:#dc2626;">(摩擦过载)</span></td>
            </tr>
            <tr>
              <td><strong>ExpC</strong> (漏斗)</td>
              <td>全周期 + Gate 1/2/3 增量信息漏斗 + 截面正交去冗余</td>
              <td>14.8</td>
              <td>9.67%</td>
              <td>1.98%</td>
              <td>7.69%</td>
              <td>1.72</td>
              <td>1.77%</td>
              <td>2.18</td>
              <td>70.7x <span style="color:#059669;">(-41%成本)</span></td>
            </tr>
            <tr>
              <td><strong>ExpD</strong> (协同)</td>
              <td>全周期 + Gate 1/2/3 漏斗 + 慢周期定势/快周期微调协同</td>
              <td>14.8</td>
              <td>10.17%</td>
              <td>2.09%</td>
              <td>8.08%</td>
              <td>1.80</td>
              <td>1.83%</td>
              <td>2.22</td>
              <td>74.5x</td>
            </tr>
            <tr>
              <td><strong>ExpE</strong> (异构)</td>
              <td>补充 1d/15m 异构专属特征 + Gate 1/2/3 增量漏斗 + 快慢协同</td>
              <td>15.2</td>
              <td>10.77%</td>
              <td>2.26%</td>
              <td>8.51%</td>
              <td>1.81</td>
              <td>1.94%</td>
              <td>2.19</td>
              <td>80.7x</td>
            </tr>
            <tr class="highlight-row">
              <td><strong>ExpF</strong> (原生5m)</td>
              <td><strong>增量融入 5m 异构持仓量因子 + 原生 5m 即时执行网格</strong></td>
              <td><strong>15.6</strong></td>
              <td><strong>16.52%</strong></td>
              <td><strong>2.35%</strong></td>
              <td><strong>14.17%</strong></td>
              <td><strong>3.03</strong></td>
              <td><strong>1.26%</strong></td>
              <td><strong>5.64</strong></td>
              <td><strong>83.9x</strong></td>
            </tr>
            <tr style="background: #fdf2f8; font-weight: 700; color: #9d174d;">
              <td><strong>ExpF (3.0x实盘)</strong></td>
              <td>3.0x 目标杠杆敞口版 (单标的上限 10% / 板块 30% / 保证金 40%)</td>
              <td>15.6</td>
              <td>49.55%</td>
              <td>7.04%</td>
              <td>42.51%</td>
              <td>3.03</td>
              <td>3.58%</td>
              <td>5.96</td>
              <td>251.6x</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="slide-footer">
      <span>注: 样本外测试区间为 2025.01.02 ~ 2026.09.30 (共 503 个真实结算日)，严格扣减手续费与滑点。</span>
      <span>Slide 02 of 12</span>
    </div>
  </div>

  <!-- ==================== SLIDE 3: 方法论体系 ==================== -->
  <div class="slide" id="slide-3" data-title="03. 方法论体系：双轮闭环" data-cat="Methodology">
    <div class="slide-header">
      <div class="slide-header-left">
        <div class="slide-category">METHODOLOGY & PHILOSOPHY</div>
        <h2 class="slide-title">方法论体系：演绎先验与归纳证伪的双轮闭环</h2>
        <div class="slide-subtitle">拒绝“黑盒暴力拟合”与“事后因果倒置”，坚持以严谨的物理与微观经济学逻辑构建防御堡垒</div>
      </div>
      <span class="pill-blue badge-pill">理论根基与工程纪律</span>
    </div>

    <div class="slide-content">
      <!-- Methodology SVG Diagram -->
      <div style="background: #ffffff; border: 1px solid var(--border-color); border-radius: 8px; padding: 10px 14px; box-shadow: var(--shadow-sm);">
        <svg viewBox="0 0 960 190" style="width: 100%; height: auto; display: block;" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif">
          <defs>
            <marker id="m-arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 1 L 10 5 L 0 9 z" fill="#1e3a8a" />
            </marker>
            <marker id="m-feedback" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 1 L 10 5 L 0 9 z" fill="#dc2626" />
            </marker>
          </defs>

          <!-- Box 1: Deductive Priors -->
          <g transform="translate(10, 10)">
            <rect width="275" height="142" rx="7" fill="#eff6ff" stroke="#bfdbfe" stroke-width="1.5"/>
            <text x="137" y="22" fill="#1e3a8a" font-size="12" font-weight="700" text-anchor="middle">Ⅰ. 演绎推理先验 (Deductive Priors)</text>
            <line x1="12" y1="30" x2="263" y2="30" stroke="#bfdbfe" stroke-width="1"/>
            <text x="16" y="48" fill="#1e293b" font-size="10.5">● <tspan font-weight="700">缠论自相似几何形态:</tspan> 分型、笔、中枢动态演化</text>
            <text x="16" y="67" fill="#1e293b" font-size="10.5">● <tspan font-weight="700">微观物理博弈:</tspan> 持仓量急降砍仓、主动买卖盘失衡</text>
            <text x="16" y="86" fill="#1e293b" font-size="10.5">● <tspan font-weight="700">两阶段状态机解耦:</tspan> Setup 势能蓄力 + 反包确认</text>
            <text x="16" y="105" fill="#1e293b" font-size="10.5">● <tspan font-weight="700">100% 对称做空架构:</tspan> 拒绝多头专属，防守单边熊市</text>
            <text x="16" y="125" fill="#2563eb" font-size="10" font-weight="600">→ 严禁无经济学先验的单根 K 线静态拼接</text>
          </g>

          <!-- Arrow 1 -> 2 -->
          <path d="M 290 81 L 330 81" stroke="#1e3a8a" stroke-width="2.5" marker-end="url(#m-arrow)"/>

          <!-- Box 2: Formalization & Feature Engine -->
          <g transform="translate(335, 10)">
            <rect width="275" height="142" rx="7" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
            <text x="137" y="22" fill="#0f172a" font-size="12" font-weight="700" text-anchor="middle">Ⅱ. 形式化与向量特征工程 (Engine)</text>
            <line x1="12" y1="30" x2="263" y2="30" stroke="#cbd5e1" stroke-width="1"/>
            <text x="16" y="48" fill="#334155" font-size="10.5">● <tspan font-weight="700">5级时间尺度:</tspan> 1d, 60m, 30m, 15m, 5m 原生网格</text>
            <text x="16" y="67" fill="#334155" font-size="10.5">● <tspan font-weight="700">6大正交战略簇:</tspan> 趋势、反转、锚定、微观、波动、截面</text>
            <text x="16" y="86" fill="#334155" font-size="10.5">● <tspan font-weight="700">147 个候选特征空间:</tspan> 含 1d/15m/5m 异构专属特征</text>
            <text x="16" y="105" fill="#334155" font-size="10.5">● <tspan font-weight="700">纯 Rust 算子计算:</tspan> 零开销抽象，微秒级跨品种计算</text>
            <text x="16" y="125" fill="#64748b" font-size="10" font-weight="600">→ Python 端纯透传，无额外适配层开销</text>
          </g>

          <!-- Arrow 2 -> 3 -->
          <path d="M 615 81 L 655 81" stroke="#1e3a8a" stroke-width="2.5" marker-end="url(#m-arrow)"/>

          <!-- Box 3: Inductive Falsification Gates -->
          <g transform="translate(660, 10)">
            <rect width="290" height="142" rx="7" fill="#fef2f2" stroke="#fecaca" stroke-width="1.5"/>
            <text x="145" y="22" fill="#991b1b" font-size="12" font-weight="700" text-anchor="middle">Ⅲ. 归纳证伪门禁 (Inductive Falsification)</text>
            <line x1="12" y1="30" x2="278" y2="30" stroke="#fecaca" stroke-width="1"/>
            <text x="16" y="48" fill="#7f1d1d" font-size="10.5">● <tspan font-weight="700">Gate 1 自适应显著性:</tspan> 单品种 Sharpe>0.15 (5m要求>=0.43)</text>
            <text x="16" y="67" fill="#7f1d1d" font-size="10.5">● <tspan font-weight="700">Gate 2 激活低共线:</tspan> |ρ_active|&lt;0.35 + 跨品种池化共振</text>
            <text x="16" y="86" fill="#7f1d1d" font-size="10.5">● <tspan font-weight="700">Gate 3 扣费边际净夏普:</tspan> 组合增量 ΔSharpe &ge; 0</text>
            <text x="16" y="105" fill="#7f1d1d" font-size="10.5">● <tspan font-weight="700">两折扩张 Walk-Forward:</tspan> 503 天绝对冻结盲测</text>
            <text x="16" y="125" fill="#b91c1c" font-size="10" font-weight="600">→ 最终留存 622 个高质量非共线核心因子</text>
          </g>

          <!-- Feedback Loop Arrow Bottom -->
          <path d="M 805 158 L 805 175 L 147 175 L 147 158" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="4,4" fill="none" marker-end="url(#m-feedback)"/>
          <text x="475" y="171" fill="#dc2626" font-size="10" font-weight="700" text-anchor="middle">失效假说证伪闭环: 任何无法通过 Gate 2/3 的特征必须无条件剔除，严禁暴力修改参数过拟合</text>
        </svg>
      </div>

      <!-- Core Research Principles 3 Cards -->
      <div class="grid-3">
        <div class="card card-highlight">
          <div class="card-title">
            <span>1. 两阶段状态机准则</span>
            <span class="pill-blue badge-pill">交易结构</span>
          </div>
          <div class="card-desc">
            开仓必须解耦为【阶段一：Setup 势能蓄力】（大级别趋势中继、日线极值收敛、持仓量持续堆叠）与【阶段二：Confirmation 微观反包确认】（小级别破位背驰、5m 持仓量急降砍仓）。严禁任何单根 K 线静态指标共振入场。
          </div>
        </div>

        <div class="card card-purple">
          <div class="card-title">
            <span>2. 100% 对称做空架构</span>
            <span class="pill-purple badge-pill">多空对称</span>
          </div>
          <div class="card-desc">
            商品期货具有典型的双向牛熊周期与负基差结构。本策略多头与空头规则严格 1:1 镜像对称，无多头专属偏置；在 2025~2026 年黑色系与有色金属宽幅下行和震荡行情中，保持了卓越的下行防御与超额捕获能力。
          </div>
        </div>

        <div class="card card-success">
          <div class="card-title">
            <span>3. 非对称赔率风控准则</span>
            <span class="pill-green badge-pill">风控纪律</span>
          </div>
          <div class="card-desc">
            初始止损点严格锚定结构破位点（次低点/次高点分型极值），单笔硬止损不超过 1.0%；一旦突破即启动 1.2%~1.5% 动态追踪止盈，确保单笔理论盈亏比 <span class="math"><i>R</i> &ge; 1.7</span>，单笔期望安全边际 <span class="math">&Delta;<i>W</i> = <i>W</i> &minus; <sup>1</sup>&frasl;<sub>(1+<i>R</i>)</sub> &gt; +10%</span>。
          </div>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <span>遵循 CZSC 机构级量化策略研发规范 (Constitution §2 SOP) · 严禁数据拟合与幸存者偏差</span>
      <span>Slide 03 of 12</span>
    </div>
  </div>

  <!-- ==================== SLIDE 4: 样本切分协议 ==================== -->
  <div class="slide" id="slide-4" data-title="04. 样本切分协议：严格防前视" data-cat="Data Protocol">
    <div class="slide-header">
      <div class="slide-header-left">
        <div class="slide-category">DATA INTEGRITY & SPLITTING PROTOCOL</div>
        <h2 class="slide-title">样本切分协议：严格防前视扩张时序时钟</h2>
        <div class="slide-subtitle">两折扩张窗口前向验证 (Two-Fold Expanding Walk-Forward)、真实物理摩擦与冻结标的池的三维防线</div>
      </div>
      <span class="pill-purple badge-pill">两折 Walk-Forward · 503天纯样本外</span>
    </div>

    <div class="slide-content">
      <!-- Accurate Gantt Timeline SVG -->
      <div style="background: #ffffff; border: 1px solid var(--border-color); border-radius: 8px; padding: 12px 16px; box-shadow: var(--shadow-sm);">
        <svg viewBox="0 0 960 140" style="width: 100%; height: auto; display: block;" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif">
          <!-- Timeline Axis -->
          <line x1="40" y1="95" x2="920" y2="95" stroke="#cbd5e1" stroke-width="2"/>
          <circle cx="40" cy="95" r="4" fill="#64748b"/>
          <circle cx="390" cy="95" r="4" fill="#2563eb"/>
          <circle cx="690" cy="95" r="4" fill="#7c3aed"/>
          <circle cx="920" cy="95" r="4" fill="#059669"/>

          <!-- Axis Labels -->
          <text x="40" y="115" fill="#64748b" font-size="10.5" font-weight="600" text-anchor="middle">2024-01-02</text>
          <text x="390" y="115" fill="#2563eb" font-size="10.5" font-weight="600" text-anchor="middle">2024-12-31 (Cutoff 1)</text>
          <text x="690" y="115" fill="#7c3aed" font-size="10.5" font-weight="600" text-anchor="middle">2025-12-31 (Cutoff 2)</text>
          <text x="920" y="115" fill="#059669" font-size="10.5" font-weight="600" text-anchor="middle">2026-09-30 (End)</text>

          <!-- Fold 1 Tracks -->
          <g transform="translate(0, 10)">
            <rect x="40" y="5" width="350" height="28" rx="5" fill="#eff6ff" stroke="#93c5fd" stroke-width="1.2"/>
            <text x="215" y="23" fill="#1e40af" font-size="10.5" font-weight="700" text-anchor="middle">Fold 1 训练集 (IS): 2024 全年 (12个月) · 执行 Gate 1~3 选出 2025 因子</text>

            <rect x="395" y="5" width="295" height="28" rx="5" fill="#ecfdf5" stroke="#6ee7b7" stroke-width="1.5"/>
            <text x="542" y="23" fill="#065f46" font-size="10.5" font-weight="700" text-anchor="middle">Fold 1 纯样本外 (OOS 1): 2025 全年盲测 (242 结算日)</text>
          </g>

          <!-- Fold 2 Tracks (Expanding) -->
          <g transform="translate(0, 45)">
            <rect x="40" y="5" width="650" height="28" rx="5" fill="#f5f3ff" stroke="#c4b5fd" stroke-width="1.2"/>
            <text x="365" y="23" fill="#6d28d9" font-size="10.5" font-weight="700" text-anchor="middle">Fold 2 扩张训练集 (IS): 2024 ~ 2025 (24个月) · 扩张窗口重跑 Gate 1~3 迭代 2026 因子</text>

            <rect x="695" y="5" width="225" height="28" rx="5" fill="#ecfdf5" stroke="#10b981" stroke-width="2"/>
            <text x="807" y="23" fill="#047857" font-size="10.5" font-weight="800" text-anchor="middle">Fold 2 纯样本外 (OOS 2): 2026.01~09 (9个月)</text>
          </g>

          <!-- Stitched Bracket Indicator -->
          <text x="655" y="132" fill="#059669" font-size="10.5" font-weight="800" text-anchor="middle">▲ Stitched OOS: 两折样本外无缝拼接共 21 个月 (503 交易日)，冷启动归零，绝对冻结</text>
        </svg>
      </div>

      <!-- 4 Isolation Gateway Cards -->
      <div class="grid-4">
        <div class="card card-highlight">
          <div class="card-title">
            <span>1. 扩张窗口单向时钟</span>
            <span class="pill-blue badge-pill">时序因果</span>
          </div>
          <div class="card-desc">
            因子得分、截面排序、波动率估算与协方差矩阵仅使用历史截面数据 <span class="math"><i>t</i> &le; <i>T</i></span>，严禁全时序统一求均值/标准差；计算时钟严格单向流逝，彻底杜绝未来信息偷窥。
          </div>
        </div>

        <div class="card card-purple">
          <div class="card-title">
            <span>2. 真实物理摩擦核算</span>
            <span class="pill-purple badge-pill">滑点约束</span>
          </div>
          <div class="card-desc">
            沪金单边 1.0 BP，其余 19 品种单边 1.5 BP (双边 3.0 BP)，底层 Rust `executed_weight_ledger` 逐柱扣除实际调仓名义本金（准确计入标的价格漂移导致的被动调仓量）。
          </div>
        </div>

        <div class="card card-success">
          <div class="card-title">
            <span>3. 冻结核心商品资产池</span>
            <span class="pill-green badge-pill">杜绝选择偏误</span>
          </div>
          <div class="card-desc">
            实验伊始即锁定国内流动性最强的 20 大主流期货连续主力合约（贵金属、能化、黑色、有色、农产品）；严禁在样本外回测表现出来后再反推删改资产池。
          </div>
        </div>

        <div class="card card-highlight" style="border-left-color: var(--amber);">
          <div class="card-title">
            <span>4. 离散网格因果对齐</span>
            <span class="pill-amber badge-pill">执行保真</span>
          </div>
          <div class="card-desc">
            第 <span class="math"><i>t</i></span> 根 K 线收盘生成目标权重，第 <span class="math"><i>t</i>+1</span> 根 K 线开盘撮合；高周期信号（1d/60m/30m/15m）通过后向阶梯对齐（`merge_asof backward`）映射至 5m 网格，杜绝透视未闭合 K 线。
          </div>
        </div>
      </div>

      <!-- Universe Breakdown -->
      <div style="background: #ffffff; border: 1px solid var(--border-color); border-radius: 8px; padding: 8px 12px; font-size: 11px; display: flex; justify-content: space-between;">
        <div><strong>贵金属 (2):</strong> au (沪金), ag (沪银)</div>
        <div><strong>能源化工 (5):</strong> sc (原油), ta (PTA), ma (甲醇), pp (聚丙烯), ru (橡胶)</div>
        <div><strong>有色金属 (4):</strong> cu (沪铜), zn (沪锌), ni (沪镍), al (沪铝)</div>
        <div><strong>黑色建材 (5):</strong> i (铁矿), fg (玻璃), hc (热卷), jm (焦煤), rb (螺纹)</div>
        <div><strong>农产品油脂 (4):</strong> sr (白糖), p (棕榈), oi (菜油), m (豆粕)</div>
      </div>
    </div>

    <div class="slide-footer">
      <span>两折扩张 Walk-Forward 验证 · 纯样本外 503 个结算日零未来函数泄露</span>
      <span>Slide 04 of 12</span>
    </div>
  </div>

  <!-- ==================== SLIDE 5: 全周期跨尺度信号矩阵 ==================== -->
  <div class="slide" id="slide-5" data-title="05. 全周期跨尺度信号矩阵" data-cat="Signal Engineering">
    <div class="slide-header">
      <div class="slide-header-left">
        <div class="slide-category">SIGNAL MATRIX & MULTI-TIMEFRAME ARCHITECTURE</div>
        <h2 class="slide-title">全周期跨尺度信号矩阵与六大战略原型</h2>
        <div class="slide-subtitle">从宏观日线定势到 5m 微观持仓量的 5 级离散金字塔，与 6 大正交战略原型深度融合</div>
      </div>
      <span class="pill-blue badge-pill">5大周期 × 6大原型 = 147 候选信号</span>
    </div>

    <div class="slide-content">
      <div class="grid-2" style="grid-template-columns: 1fr 1.25fr; gap: 14px;">
        <!-- Left: 5-Level Pyramid Diagram -->
        <div style="background: #ffffff; border: 1px solid var(--border-color); border-radius: 8px; padding: 12px 14px; box-shadow: var(--shadow-sm); display: flex; flex-direction: column; justify-content: space-between;">
          <div style="font-size: 12px; font-weight: 700; color: var(--primary); margin-bottom: 6px; display: flex; justify-content: space-between;">
            <span>跨尺度时间周期分层金字塔 (Hierarchical Multi-TF)</span>
            <span class="pill-blue badge-pill">快慢分工</span>
          </div>

          <svg viewBox="0 0 420 185" style="width: 100%; height: auto;" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif">
            <!-- Level 1: 1d (Top) -->
            <polygon points="150,8 270,8 296,40 124,40" fill="#1e3a8a" />
            <text x="210" y="23" fill="#ffffff" font-size="10.5" font-weight="800" text-anchor="middle">1d · 宏观定势慢变量 (Macro Trend)</text>
            <text x="210" y="35" fill="#93c5fd" font-size="8.5" text-anchor="middle">板块价差离散回归、日线乖离反转、大级别动量</text>

            <!-- Level 2: 60m -->
            <polygon points="122,44 298,44 324,76 96,76" fill="#2563eb" />
            <text x="210" y="59" fill="#ffffff" font-size="10.5" font-weight="800" text-anchor="middle">60m · 波段骨架层 (Swing Bone)</text>
            <text x="210" y="71" fill="#dbeafe" font-size="8.5" text-anchor="middle">1小时缠论笔延续、均线中继通道、核心波段主趋势</text>

            <!-- Level 3: 30m -->
            <polygon points="94,80 326,80 352,112 68,112" fill="#3b82f6" />
            <text x="210" y="95" fill="#ffffff" font-size="10.5" font-weight="800" text-anchor="middle">30m · 核心动量层 (Momentum Engine)</text>
            <text x="210" y="107" fill="#eff6ff" font-size="8.5" text-anchor="middle">经典能量潮、顺势突破共振、高留存阿尔法主干</text>

            <!-- Level 4: 15m -->
            <polygon points="66,116 354,116 380,148 40,148" fill="#7c3aed" />
            <text x="210" y="131" fill="#ffffff" font-size="10.5" font-weight="800" text-anchor="middle">15m · 微观确认层 (Micro Confirmation)</text>
            <text x="210" y="143" fill="#ede9fe" font-size="8.5" text-anchor="middle">放量恐慌衰竭反转、多日顺势回踩触发器、跨日持有</text>

            <!-- Level 5: 5m (Base) -->
            <polygon points="38,152 382,152 408,184 12,184" fill="#059669" />
            <text x="210" y="167" fill="#ffffff" font-size="10.5" font-weight="800" text-anchor="middle">5m · 即时触发与微观持仓量层 (Pulse & OI)</text>
            <text x="210" y="179" fill="#d1fae5" font-size="8.5" text-anchor="middle">持仓量急降砍仓出清 (H5M_LQ)、尾盘抢筹、增仓启动</text>
          </svg>

          <div style="font-size: 10.5px; color: var(--text-muted); margin-top: 4px; line-height: 1.4;">
            <strong>快慢协同调制公式:</strong>
            <div class="math-block">
              <span class="math"><i>W</i> = Clip(<i>W</i><sub>fast</sub> &middot; [1 + 0.25&middot;<b>I</b><sub>agree</sub> &minus; 0.35&middot;<b>I</b><sub>conflict</sub>] + 0.70&middot;<i>W</i><sub>1d</sub>, &minus;1, 1)</span>
            </div>
          </div>
        </div>

        <!-- Right: 6 Archetypes & Hetero Factors -->
        <div style="display: flex; flex-direction: column; gap: 7px;">
          <div class="card" style="padding: 8px 12px; border-left: 3.5px solid #2563eb;">
            <div style="font-size: 11.5px; font-weight: 700; color: #1e40af; display: flex; justify-content: space-between;">
              <span>A 簇：趋势动量 (Trend) & B 簇：极值反转 (Reversal)</span>
              <span class="pill-blue badge-pill">核心基石</span>
            </div>
            <div style="font-size: 10.5px; color: var(--text-secondary); margin-top: 2px;">
              唐奇安通道、均线排列、中枢离开段突破；RSI 极值超买超卖、底顶分型与线段背驰
            </div>
          </div>

          <div class="card" style="padding: 8px 12px; border-left: 3.5px solid #059669;">
            <div style="font-size: 11.5px; font-weight: 700; color: #065f46; display: flex; justify-content: space-between;">
              <span>C 簇：均线价差锚定 (Anchor) & E 簇：波动率突破 (Volatility)</span>
              <span class="pill-green badge-pill">结构突破</span>
            </div>
            <div style="font-size: 10.5px; color: var(--text-secondary); margin-top: 2px;">
              顺大势回踩 MA60 企稳触发；ATR 极窄幅波动率压缩 (Squeeze) 后的放量脱离
            </div>
          </div>

          <div class="card" style="padding: 8px 12px; border-left: 3.5px solid #7c3aed;">
            <div style="font-size: 11.5px; font-weight: 700; color: #5b21b6; display: flex; justify-content: space-between;">
              <span>1d / 15m 跨尺度异构专属特征 (6+6 个)</span>
              <span class="pill-purple badge-pill">ExpE 增量</span>
            </div>
            <div style="font-size: 10.5px; color: var(--text-secondary); margin-top: 2px;">
              <code>H1D_SP01</code> 板块价差离散回归、<code>H1D_DV01</code> 日线乖离反转；<code>H15M_EX01</code> 放量恐慌衰竭、<code>H15M_PB01</code> 顺势回踩企稳
            </div>
          </div>

          <div class="card" style="padding: 8px 12px; border-left: 3.5px solid #e11d48; background: #fff1f2;">
            <div style="font-size: 11.5px; font-weight: 700; color: #9f1239; display: flex; justify-content: space-between;">
              <span>5m 亚柱微观持仓量异构特征 (D 簇实验组 vs 对照组)</span>
              <span class="pill-red badge-pill">ExpF 质跃突破</span>
            </div>
            <div style="font-size: 10.5px; color: var(--text-secondary); margin-top: 2px;">
              <strong>实验组 (通过)</strong>: <code>H5M_LQ01</code> (持仓量急降砍仓出清反转, Shp 75%正)、<code>H5M_CLV01</code> (尾盘抢筹)、<code>H5M_OIB01</code> (增仓启动)<br/>
              <strong>对照组 (全灭)</strong>: <code>N5M_BRK/REV</code> (朴素短吵，换手 150x+，池化 t=-3.82 ~ -6.15，<strong>100% 淘汰</strong>)
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <span>147 个候选特征由 pure Rust (czsc-signals / czsc-ta) 实现，保障向量化计算精度与纳秒级性能</span>
      <span>Slide 05 of 12</span>
    </div>
  </div>

  <!-- ==================== SLIDE 6: 信号检验体系 ==================== -->
  <div class="slide" id="slide-6" data-title="06. 信号检验体系：三阶段漏斗" data-cat="Signal Validation">
    <div class="slide-header">
      <div class="slide-header-left">
        <div class="slide-category">FACTOR FORENSIC ADMISSION PIPELINE</div>
        <h2 class="slide-title">信号检验体系：三阶段增量准入漏斗</h2>
        <div class="slide-subtitle">从 5,880 个时序评估实例到 622 个高质量正交因子的贪心前向准入净化与留存分异</div>
      </div>
      <span class="pill-green badge-pill">Gate 1->3 留存率 52.1% · 交易摩擦降低 41%</span>
    </div>

    <div class="slide-content">
      <div class="grid-2" style="grid-template-columns: 1fr 1fr; gap: 14px;">
        <!-- Left: Funnel Flow & Exact Criteria -->
        <div style="display: flex; flex-direction: column; gap: 8px;">
          <div class="card card-highlight" style="padding: 8px 12px;">
            <div class="card-title">
              <span>Gate 1: 分周期自适应统计显著性准入</span>
              <span class="pill-blue badge-pill">初筛: 1,194 / 5,880 (20.3%)</span>
            </div>
            <div class="card-desc">
              ● 60m/30m: 单品种 n &ge; 10/15, 均值 > 0, 样本内净 Sharpe > 0.15<br/>
              ● 1d: n &ge; 2, 净 Sharpe > 0.10, 跨品种池化 total_n &ge; 25 且池化均值 > 0<br/>
              ● 15m 异构: n &ge; 6, 年化换手 &le; 26x, 池化 total_n &ge; 150, 净 Sharpe > 0.15<br/>
              ● 5m 异构: n &ge; 20, 胜率 &ge; 52.5%, 换手 &le; 56x, total_n &ge; 180, 净 Sharpe &ge; 0.43
            </div>
          </div>

          <div class="card card-purple" style="padding: 8px 12px;">
            <div class="card-title">
              <span>Gate 2: 激活条件低共线 + 池化共振 + 主干保护</span>
              <span class="pill-purple badge-pill">拦截: 557 个 | 通过: 637 个</span>
            </div>
            <div class="card-desc">
              ● <strong>激活条件低共线</strong>: 增量信号与已入库信号激活交集相关性 <span class="math">|<i>&rho;</i><sub>active</sub>| &lt; 0.35</span><br/>
              ● <strong>跨品种池化显著</strong>: 15m 要求 <span class="math"><i>t</i><sub>pooled</sub> &ge; 0.30</span>; 5m 要求 <span class="math"><i>t</i><sub>pooled</sub> &ge; 0.65</span><br/>
              ● <strong>5m 主干活性门禁</strong>: 仅允许 4 周期主干净 Sharpe &ge; 0.60 的活跃品种激活 5m 信号
            </div>
          </div>

          <div class="card card-success" style="padding: 8px 12px;">
            <div class="card-title">
              <span>Gate 3: 扣费后组合边际净夏普检验</span>
              <span class="pill-green badge-pill">最终入库: 622 个 (97.6%)</span>
            </div>
            <div class="card-desc">
              ● 候选特征按周期权重并入品种已选组合，重新核算<strong>扣费后净 Sharpe</strong><br/>
              ● 必须带来严格正增益: 1d/15m 要求 <span class="math">&Delta;Sharpe<sub>IS</sub> &ge; 0.00</span>; 5m 要求 <span class="math">&Delta;Sharpe<sub>IS</sub> &ge; +0.045</span><br/>
              ● 拦截 15 个边际负贡献与伪增量特征，最终单品种每折平均准入 15.55 个因子
            </div>
          </div>
        </div>

        <!-- Right: Funnel Grouped Bar Chart & Table -->
        <div style="background: #ffffff; border: 1px solid var(--border-color); border-radius: 8px; padding: 10px 12px; box-shadow: var(--shadow-sm); display: flex; flex-direction: column; justify-content: space-between;">
          <div id="chart-funnel" class="plotly-graph-div" style="height:210px; width:100%;"></div>

          <!-- Retention Table -->
          <table class="table-academic" style="margin-top: 6px;">
            <thead>
              <tr>
                <th>周期层级</th>
                <th>候选评估总数</th>
                <th>Gate 1 通过</th>
                <th>Gate 2 通过</th>
                <th>Gate 3 入选</th>
                <th>留存率 (G1&rarr;3)</th>
                <th>单标的每折均数</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>1d (慢变量)</strong></td>
                <td>1,480</td>
                <td>177</td>
                <td>18</td>
                <td><strong>15</strong></td>
                <td>8.5%</td>
                <td>0.38 个</td>
              </tr>
              <tr style="background: #f0fdf4;">
                <td><strong>60m (主干层)</strong></td>
                <td>1,320</td>
                <td>355</td>
                <td>294</td>
                <td><strong>287</strong></td>
                <td><strong style="color:#059669;">80.8%</strong></td>
                <td>7.17 个</td>
              </tr>
              <tr style="background: #f0fdf4;">
                <td><strong>30m (主干层)</strong></td>
                <td>1,320</td>
                <td>319</td>
                <td>297</td>
                <td><strong>297</strong></td>
                <td><strong style="color:#059669;">93.1%</strong></td>
                <td>7.42 个</td>
              </tr>
              <tr>
                <td><strong>15m (结构极值)</strong></td>
                <td>1,360</td>
                <td>316</td>
                <td>10</td>
                <td><strong>10</strong></td>
                <td>3.2%</td>
                <td>0.25 个</td>
              </tr>
              <tr style="background: #eff6ff;">
                <td><strong>5m (持仓量异构)</strong></td>
                <td>400</td>
                <td>27</td>
                <td>18</td>
                <td><strong>13</strong></td>
                <td><strong style="color:#2563eb;">48.1%</strong></td>
                <td>0.33 个</td>
              </tr>
              <tr class="highlight-row">
                <td><strong>全周期合计</strong></td>
                <td><strong>5,880</strong></td>
                <td><strong>1,194</strong></td>
                <td><strong>637</strong></td>
                <td><strong>622</strong></td>
                <td><strong>52.1%</strong></td>
                <td><strong>15.55 个</strong></td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <span>三阶段增量漏斗直接消除了 41% 摩擦成本，彻底打破了“盲目堆加高频信号导致收益被换手吞噬”的恶性循环</span>
      <span>Slide 06 of 12</span>
    </div>
  </div>

  <!-- ==================== SLIDE 7: 信号检验深层机理 ==================== -->
  <div class="slide" id="slide-7" data-title="07. 信号检验机理：共线性、池化与对照证伪" data-cat="Validation Forensics">
    <div class="slide-header">
      <div class="slide-header-left">
        <div class="slide-category">SIGNAL TESTING METHODOLOGY & FORENSICS</div>
        <h2 class="slide-title">信号检验深层机理：统计显著性、条件共线与对照组证伪</h2>
        <div class="slide-subtitle">四维检验闭环、非零激活交集去相关、母本资产门禁与微观异构 vs 朴素突破极限证伪</div>
      </div>
      <span class="pill-green badge-pill">四维检验闭环 · 对照组 100% 证伪 · 根除伪阿尔法</span>
    </div>

    <div class="slide-content">
      <!-- Top: 3 Core Validation Mechanics Cards -->
      <div class="grid-3" style="gap: 12px;">
        <div class="card card-highlight">
          <div class="card-title">
            <span>① 激活条件低共线排异</span>
            <span class="pill-blue badge-pill">去虚假相关</span>
          </div>
          <div class="card-desc">
            ● <strong>常规 Pearson 缺陷</strong>: 脉冲信号大部分时间为 0，全时序相关性被大量 (0,0) 虚假拉低至 0.05，无法识别同向撞车<br/>
            ● <strong>条件激活交集公式</strong>:
            <div style="background:#eff6ff; padding:4px 8px; border-radius:4px; margin:4px 0; text-align:center;">
              <span class="math"><i>&rho;</i><sub>active</sub> = Corr(<i>s</i><sub>1</sub>, <i>s</i><sub>2</sub> | |<i>s</i><sub>1</sub>|&gt;0 &or; |<i>s</i><sub>2</sub>|&gt;0) &lt; 0.35</span>
            </div>
            ● <strong>实证效果</strong>: 彻底切断极端行情下同向信号共振引发的组合多倍杠杆踩踏
          </div>
        </div>

        <div class="card card-purple">
          <div class="card-title">
            <span>② 全截面池化统计检验</span>
            <span class="pill-purple badge-pill">防数据窥探</span>
          </div>
          <div class="card-desc">
            ● <strong>单品种窥探陷阱</strong>: 单品种回测极易陷入局部最优与白噪声偶然拟合<br/>
            ● <strong>20品种联合池化</strong>:
            <div style="background:#f5f3ff; padding:4px 8px; border-radius:4px; margin:4px 0; text-align:center;">
              <span class="math"><i>t</i><sub>pooled</sub> = <i>r̄</i><sub>pooled</sub> / (<i>&sigma;</i><sub>pooled</sub> / &radic;<span style="border-top:1px solid;"><i>N</i><sub>total</sub></span>) &ge; 2.58</span>
            </div>
            ● <strong>跨资产普适性</strong>: 15m 要求 <span class="math"><i>t</i><sub>pooled</sub> &ge; 0.30</span>，5m 要求 <span class="math">&ge; 0.65</span>，确保信号在全市场具备跨周期稳健规律
          </div>
        </div>

        <div class="card card-success">
          <div class="card-title">
            <span>③ 母本资产主干门禁</span>
            <span class="pill-green badge-pill">长波控短波</span>
          </div>
          <div class="card-desc">
            ● <strong>高频微观噪声陷阱</strong>: 5m 信号信噪比脆弱，孤立高频开仓极易遭遇频繁打脸<br/>
            ● <strong>四周期主干准入前置条件</strong>:
            <div style="background:#ecfdf5; padding:4px 8px; border-radius:4px; margin:4px 0; text-align:center;">
              <span class="math">Sharpe<sub>IS</sub>(1d + 60m + 30m + 15m) &ge; 0.60</span>
            </div>
            ● <strong>动态权限激活</strong>: 仅允许主干表现卓越的品种挂载 5m 微观脉冲，其余品种 5m 权重置 0
          </div>
        </div>
      </div>

      <!-- Middle: Counterfactual Forensic Experiment Comparison Table -->
      <div>
        <table class="table-academic">
          <thead>
            <tr>
              <th style="width: 14%;">检验与实证维度</th>
              <th style="width: 25%; background: #fff1f2; color: #9f1239;">5m 朴素价格突破 / 反转 (对照组)</th>
              <th style="width: 27%; background: #ecfdf5; color: #065f46;">5m 微观持仓量异构特征 (实验组)</th>
              <th style="width: 34%;">实证机理深入剖析与工程结论</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>代表特征族</strong></td>
              <td><code>N5M_BRK01/02</code> (布林破位), <code>N5M_REV01</code> (RSI超买)</td>
              <td><code>H5M_LQ01</code> (持仓量急降砍仓出清), <code>H5M_CLV01</code> (尾盘抢筹)</td>
              <td>对照组仅依赖价格二阶导，实验组深挖<strong>资金微观博弈与筹码易手</strong></td>
            </tr>
            <tr>
              <td><strong>年化单向换手率</strong></td>
              <td><strong style="color: #dc2626;">150x ~ 220x</strong> (过度频繁日内翻转)</td>
              <td><strong style="color: #059669;">35x ~ 55x</strong> (关键微观拐点脉冲触发)</td>
              <td>异构特征依托分型与中枢结构平滑持有 6~18 小时，换手骤降 70%</td>
            </tr>
            <tr>
              <td><strong>池化 t 统计量</strong></td>
              <td><span class="math" style="color: #dc2626; font-weight: 700;"><i>t</i> = -3.82 &sim; -6.15</span> (显著负偏)</td>
              <td><span class="math" style="color: #059669; font-weight: 700;"><i>t</i> = +0.72 &sim; +1.28</span> (显著正偏)</td>
              <td>扣除双边 3.0 BP 摩擦后，朴素高频价格突破的期望收益在统计上<strong>显著为负</strong></td>
            </tr>
            <tr>
              <td><strong>三阶段漏斗通过率</strong></td>
              <td><strong style="color: #dc2626;">0.0% (Gate 1/2 阶段 100% 覆灭)</strong></td>
              <td><strong style="color: #059669;">72.7% (高保留率入模)</strong></td>
              <td>三阶段漏斗具备极高分辨率的“排毒能力”，彻底清洗伪阿尔法</td>
            </tr>
            <tr style="background: #f8fafc;">
              <td><strong>最终组合贡献</strong></td>
              <td>拖累组合夏普断崖下跌 <strong>-0.65</strong>，换手推高 42x</td>
              <td>贡献费后净年化 <strong>+3.38%</strong>，全组合夏普从 1.81 质跃至 <strong>3.03</strong></td>
              <td>实证铁律：<strong>“缺乏微观持仓量支持的纯价格高频突破，在真实摩擦下等于慢性自杀”</strong></td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- Bottom: Key Forensic Insights Cards -->
      <div class="grid-2">
        <div class="card card-highlight">
          <div class="card-title">
            <span>贪心前向逐步筛选机制 (Greedy Forward Stepwise Selection)</span>
            <span class="pill-blue badge-pill">边际增益</span>
          </div>
          <div class="card-desc">
            每个候选因子在并入资产池时，必须经过<strong>边际扣费增量检验</strong>：<span class="math">&Delta;Sharpe<sub>IS</sub> = Sharpe(<i>F</i><sub><i>k</i>-1</sub> &cup; {<i>f</i><sub><i>k</i></sub>}) &minus; Sharpe(<i>F</i><sub><i>k</i>-1</sub>) &ge; 0</span>。即便单因子单项表现优异，若其引起的换手成本增量超过了超额收益增量，Gate 3 将行使一票否决权，确保组合永不膨胀多余交易摩擦。
          </div>
        </div>

        <div class="card card-success">
          <div class="card-title">
            <span>分层统计治理：让长波做骨架、让微观做先锋</span>
            <span class="pill-green badge-pill">长短协同</span>
          </div>
          <div class="card-desc">
            漏斗筛选结果展现出惊人的尺度分工：<strong>60m 与 30m 留存率高达 80%~93%</strong>，构成了全策略 93% 的基础容量与底仓；而 <strong>15m 与 5m 仅保留极其严苛的 10~15 个异构极值特征</strong>，只在行情转折点输出瞬时不对称赔率。多周期联立不是无脑堆叠，而是由粗到精的严密金字塔结构。
          </div>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <span>严格的统计检验与对照组证伪证明：微观持仓量脉冲是打破高频换手摩擦困境的唯一有效钥匙</span>
      <span>Slide 07 of 12</span>
    </div>
  </div>\n\n  <!-- ==================== SLIDE 8: 原生 5m 物理执行网格架构 ==================== -->
  <div class="slide" id="slide-8" data-title="08. 核心执行架构：原生5m网格" data-cat="Execution Architecture">
    <div class="slide-header">
      <div class="slide-header-left">
        <div class="slide-category">EXECUTION GRID ARCHITECTURE</div>
        <h2 class="slide-title">核心执行架构：原生 5m 物理执行网格与微观调仓引擎</h2>
        <div class="slide-subtitle">统一 5m 物理主频时钟、多周期阶梯因果映射与微观低冲击调仓引擎</div>
      </div>
      <span class="pill-green badge-pill">原生 5m 主频 · 0 时滞因果闭环 · 年化摩擦仅 2.35%</span>
    </div>

    <div class="slide-content">
      <!-- Causal Grid Architecture SVG -->
      <div style="background: #ffffff; border: 1px solid var(--border-color); border-radius: 8px; padding: 10px 14px; box-shadow: var(--shadow-sm);">
        <svg viewBox="0 0 960 142" style="width: 100%; height: auto; display: block;" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif">
          <defs>
            <marker id="grid-arrow-blue" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 1 L 10 5 L 0 9 z" fill="#2563eb" />
            </marker>
            <marker id="grid-arrow-green" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 1 L 10 5 L 0 9 z" fill="#059669" />
            </marker>
            <marker id="grid-arrow-red" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 1 L 10 5 L 0 9 z" fill="#dc2626" />
            </marker>
          </defs>

          <!-- Step 1: Multi-scale Input Sources -->
          <g transform="translate(10, 8)">
            <rect width="260" height="126" rx="6" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
            <text x="14" y="22" fill="#1e3a8a" font-size="11" font-weight="800">全周期信号输入源 (Multi-Scale Inputs)</text>
            
            <rect x="14" y="32" width="232" height="20" rx="3" fill="#e0e7ff"/>
            <text x="22" y="46" fill="#3730a3" font-size="9.5" font-weight="700">1d 日线时钟: 日结算闭合后次日 09:00 注入底仓</text>

            <rect x="14" y="56" width="232" height="20" rx="3" fill="#ede9fe"/>
            <text x="22" y="70" fill="#5b21b6" font-size="9.5" font-weight="700">60m/30m/15m: 物理K线完成瞬间阶梯式更新</text>

            <rect x="14" y="80" width="232" height="20" rx="3" fill="#ecfdf5" stroke="#a7f3d0"/>
            <text x="22" y="94" fill="#065f46" font-size="9.5" font-weight="800">5m 原生物理时钟: 48 根/天，持仓量异动与反包</text>

            <text x="14" y="120" fill="#64748b" font-size="9">后向阶梯对齐 (merge_asof backward) 零前视泄露</text>
          </g>

          <!-- Arrow 1 -->
          <path d="M 278 71 L 308 71" stroke="#2563eb" stroke-width="2.5" marker-end="url(#grid-arrow-blue)"/>
          <text x="293" y="63" fill="#2563eb" font-size="9" font-weight="700" text-anchor="middle">汇聚</text>

          <!-- Step 2: Native 5m Master Clock Synchronization -->
          <g transform="translate(315, 8)">
            <rect width="330" height="126" rx="6" fill="#eff6ff" stroke="#bfdbfe" stroke-width="1.5"/>
            <text x="14" y="22" fill="#1e40af" font-size="11" font-weight="800">原生 5m 主频时钟与决策执行时序 (Master Clock)</text>

            <!-- Time Axis Demonstration -->
            <line x1="20" y1="52" x2="310" y2="52" stroke="#94a3b8" stroke-width="1.8"/>
            
            <!-- Bar t-1 -->
            <rect x="35" y="38" width="50" height="28" rx="3" fill="#ffffff" stroke="#94a3b8"/>
            <text x="60" y="55" fill="#475569" font-size="9.5" font-weight="600" text-anchor="middle">Bar t-1</text>

            <!-- Bar t -->
            <rect x="105" y="36" width="60" height="32" rx="4" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/>
            <text x="135" y="55" fill="#1e3a8a" font-size="10" font-weight="800" text-anchor="middle">Bar t (5m)</text>
            <text x="135" y="80" fill="#2563eb" font-size="8.5" font-weight="700" text-anchor="middle">收盘结算决策</text>

            <!-- Execution Arrow -->
            <path d="M 168 52 L 195 52" stroke="#dc2626" stroke-width="2.5" marker-end="url(#grid-arrow-red)"/>
            <text x="182" y="45" fill="#dc2626" font-size="8" font-weight="800" text-anchor="middle">0延迟</text>

            <!-- Bar t+1 -->
            <rect x="200" y="36" width="60" height="32" rx="4" fill="#fee2e2" stroke="#dc2626" stroke-width="1.5"/>
            <text x="230" y="55" fill="#991b1b" font-size="10" font-weight="800" text-anchor="middle">Bar t+1</text>
            <text x="230" y="80" fill="#dc2626" font-size="8.5" font-weight="700" text-anchor="middle">开盘市价撮合</text>

            <rect x="15" y="96" width="300" height="22" rx="4" fill="#ffffff" stroke="#bfdbfe"/>
            <text x="165" y="111" fill="#1e40af" font-size="9.5" font-weight="700" text-anchor="middle">严密因果律: t 闭合生成目标权重向量 → t+1 开盘真实撮合</text>
          </g>

          <!-- Arrow 2 -->
          <path d="M 653 71 L 683 71" stroke="#059669" stroke-width="2.5" marker-end="url(#grid-arrow-green)"/>
          <text x="668" y="63" fill="#059669" font-size="9" font-weight="700" text-anchor="middle">下发</text>

          <!-- Step 3: Rust Executed Weight Ledger & Friction -->
          <g transform="translate(690, 8)">
            <rect width="260" height="126" rx="6" fill="#ecfdf5" stroke="#a7f3d0" stroke-width="1.5"/>
            <text x="14" y="22" fill="#065f46" font-size="11" font-weight="800">Rust 调仓执行账本与摩擦核算</text>

            <rect x="14" y="32" width="232" height="20" rx="3" fill="#d1fae5"/>
            <text x="22" y="46" fill="#065f46" font-size="9.5" font-weight="700">死区滤波: |Δw| ≤ 0.08 抑制微观漂移抖动</text>

            <rect x="14" y="56" width="232" height="20" rx="3" fill="#d1fae5"/>
            <text x="22" y="70" fill="#065f46" font-size="9.5" font-weight="700">工业摩擦: AU 1.0 BP，其余 1.5 BP (双边 3.0 BP)</text>

            <rect x="14" y="80" width="232" height="20" rx="3" fill="#d1fae5"/>
            <text x="22" y="94" fill="#065f46" font-size="9.5" font-weight="700">实盘账本: executed_weight_ledger 逐笔对齐</text>

            <text x="14" y="120" fill="#047857" font-size="9" font-weight="800">年化换手仅 83.9x · 净夏普高达 3.03</text>
          </g>
        </svg>
      </div>

      <!-- Execution Performance & Technical Specs 4-Card Grid -->
      <div class="grid-4" style="gap: 10px;">
        <div class="card card-highlight">
          <div class="card-title">
            <span>物理时钟与决策密度</span>
            <span class="pill-blue badge-pill">高频对齐</span>
          </div>
          <div class="card-desc">
            ● <strong>计算频率</strong>: 单标的 48 根/天，全截面 960 节点/天<br/>
            ● <strong>执行时滞</strong>: 严格 0 延迟，<span class="math"><i>t</i></span> 收盘决策 <span class="math">&rarr; <i>t</i>+1</span> 开盘撮合<br/>
            ● <strong>防前视隔离</strong>: 高周期通过后向阶梯对齐，零未来泄露
          </div>
        </div>

        <div class="card card-success">
          <div class="card-title">
            <span>换手阻尼与平滑控制</span>
            <span class="pill-green badge-pill">控频滤波</span>
          </div>
          <div class="card-desc">
            ● <strong>年化换手</strong>: 单向换手仅 <strong>83.9x</strong> (极度温和)<br/>
            ● <strong>死区机制</strong>: <span class="math">Deadzone = 0.08</span> 屏蔽微小噪音调仓<br/>
            ● <strong>波段持有</strong>: 5m 异构特征平均持仓 6~18h，非频繁翻转
          </div>
        </div>

        <div class="card card-purple">
          <div class="card-title">
            <span>真实交易摩擦与收益留存</span>
            <span class="pill-purple badge-pill">工业记账</span>
          </div>
          <div class="card-desc">
            ● <strong>扣费标准</strong>: AU 单边 1.0 BP，其余 19 品种单边 1.5 BP<br/>
            ● <strong>年化摩擦</strong>: <strong>2.35%</strong> (占毛收益 16.52% 的 14.2%)<br/>
            ● <strong>费后留存</strong>: 净年化 <strong>14.17%</strong>，费后夏普 <strong>3.03</strong>
          </div>
        </div>

        <div class="card card-amber">
          <div class="card-title">
            <span>摩擦压力测试与容量</span>
            <span class="pill-amber badge-pill">极限鲁棒</span>
          </div>
          <div class="card-desc">
            ● <strong>滑点耐受</strong>: 盈亏平衡滑点上限高达 <strong>10.0 BP</strong><br/>
            ● <strong>成本加倍</strong>: 摩擦翻倍至 6.0 BP 时，净 Sharpe 仍达 2.30+<br/>
            ● <strong>策略容量</strong>: 基于日均成交量测算，单策略容量达 0.5~1 亿元
          </div>
        </div>
      </div>

      <!-- Core Architectural Insights -->
      <div class="grid-2">
        <div class="card card-success">
          <div class="card-title">
            <span>物理时钟与微观持仓量半衰期严格匹配</span>
            <span class="pill-green badge-pill">微观 Alpha 变现</span>
          </div>
          <div class="card-desc">
            5m 持仓量脉冲（如 <code>H5M_LQ01</code> 持仓量急降砍仓出清、<code>H5M_CLV01</code> 尾盘微观增仓）蕴含的定价信息有效半衰期仅 15~30 分钟。原生 5m 物理执行网格确保微观 Alpha 在信号产生的第一个物理时钟即可完成撮合变现，确保微观定价优势无损转化为策略净超额。
          </div>
        </div>

        <div class="card card-highlight">
          <div class="card-title">
            <span>为什么下沉至 5m 时钟没有发生换手爆炸？</span>
            <span class="pill-blue badge-pill">快慢分层架构</span>
          </div>
          <div class="card-desc">
            得益于严谨的架构分工：① 长周期信号（日线/60m）负责输出宏观趋势方向与核心底仓，5m 异构特征依托分型与中枢结构在关键拐点进行战术性增强与止损微调；② <strong><span class="math">Deadzone = 0.08</span> 调仓死区</strong>彻底屏蔽了 5m 级别微小的波动率权重漂移，实现“高频时钟感知、波段稳健持有”。
          </div>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <span>原生 5m 物理执行网格是多周期信号无损协同与微观持仓量 Alpha 变现的基石架构</span>
      <span>Slide 08 of 12</span>
    </div>
  </div>

  <!-- ==================== SLIDE 9: 六步递进消融实验 ==================== -->
  <div class="slide" id="slide-9" data-title="09. 消融实验与净值演进" data-cat="Empirical Results">
    <div class="slide-header">
      <div class="slide-header-left">
        <div class="slide-category">EMPIRICAL RESULTS & ABLATION STUDY</div>
        <h2 class="slide-title">六步递进消融实验与纯样本外净值演进</h2>
        <div class="slide-subtitle">Stitched OOS (2025.01 ~ 2026.09, 503 交易日) 绝对盲测区间下的净值与水下回撤全息联动图</div>
      </div>
      <span class="pill-green badge-pill">两折拼接纯样本外 · 交互联动</span>
    </div>

    <div class="slide-content">
      <!-- Plotly Nav & DD Chart Container -->
      <div style="background: #ffffff; border: 1px solid var(--border-color); border-radius: 8px; padding: 6px; box-shadow: var(--shadow-sm); overflow: hidden;">
        <div id="chart-nav" class="plotly-graph-div" style="height:360px; width:100%;"></div>
      </div>

      <!-- Ablation Takeaway Pills -->
      <div style="display: flex; gap: 8px; font-size: 10.5px; align-items: center; justify-content: space-between; background: #f8fafc; border: 1px solid var(--border-color); padding: 7px 12px; border-radius: 6px;">
        <div><strong style="color: #64748b;">ExpA 基准:</strong> 仅日内60m+30m，Shp 1.25，MaxDD 1.89%</div>
        <div><strong style="color: #dc2626;">ExpB 盲堆:</strong> 换手激增至 118x，摩擦过载，Shp 仅 1.32</div>
        <div><strong style="color: #2563eb;">ExpC 漏斗:</strong> 成本直降 41%，Shp 跃升至 1.72</div>
        <div><strong style="color: #7c3aed;">ExpE 异构:</strong> 注入1d/15m专属因子，Shp 升至 1.81</div>
        <div><strong style="color: #059669;">ExpF 原生5m:</strong> 突破至 <strong>Shp 3.03 | Cal 5.64 | MaxDD 1.26% | 3x实盘 +42.51%</strong></div>
      </div>
    </div>

    <div class="slide-footer">
      <span>可在图表中缩放、平移和查看各实验净值与水下回撤曲线细节 · 全程纯样本外绝对隔离</span>
      <span>Slide 09 of 12</span>
    </div>
  </div>

  <!-- ==================== SLIDE 9: 板块与品种归因全景 ==================== -->
  <div class="slide" id="slide-10" data-title="10. 板块与品种归因全景" data-cat="Attribution Analysis">
    <div class="slide-header">
      <div class="slide-header-left">
        <div class="slide-category">SECTOR & SYMBOL ATTRIBUTION</div>
        <h2 class="slide-title">板块与品种纯样本外归因全景</h2>
        <div class="slide-subtitle">20 大流动性商品纯样本外贡献：70% 品种胜率与 5 大板块结构化表现</div>
      </div>
      <span class="pill-green badge-pill">14 / 20 品种正收益 · 胜率 70.0%</span>
    </div>

    <div class="slide-content">
      <!-- Full-Width Symbol Attribution Chart -->
      <div style="background: #ffffff; border: 1px solid var(--border-color); border-radius: 8px; padding: 6px; box-shadow: var(--shadow-sm); overflow: hidden;">
        <div id="chart-symbol" class="plotly-graph-div" style="height:230px; width:100%;"></div>
      </div>

      <!-- 5 Sector Attribution Cards -->
      <div class="grid-4" style="grid-template-columns: repeat(5, 1fr); gap: 9px;">
        <div class="card card-success" style="padding: 8px 10px;">
          <div style="font-size: 11px; font-weight: 700; color: #065f46; display: flex; justify-content: space-between;">
            <span>贵金属 (au, ag)</span>
            <span class="pill-green badge-pill">71.1% 贡献</span>
          </div>
          <div style="font-size: 15px; font-weight: 800; color: #059669; margin: 2px 0 1px 0;">+10.08%</div>
          <div style="font-size: 10px; color: var(--text-secondary);">
            Sharpe <strong>3.14</strong> | 回撤 <strong>0.66%</strong><br/>
            au (+5.69%), ag (+4.39%)
          </div>
        </div>

        <div class="card card-highlight" style="padding: 8px 10px;">
          <div style="font-size: 11px; font-weight: 700; color: #1e40af; display: flex; justify-content: space-between;">
            <span>能源化工 (5标的)</span>
            <span class="pill-blue badge-pill">11.3% 贡献</span>
          </div>
          <div style="font-size: 15px; font-weight: 800; color: #2563eb; margin: 2px 0 1px 0;">+1.60%</div>
          <div style="font-size: 10px; color: var(--text-secondary);">
            Sharpe <strong>0.98</strong> | 回撤 <strong>0.56%</strong><br/>
            sc (+1.58%), ta (+0.28%)
          </div>
        </div>

        <div class="card card-highlight" style="padding: 8px 10px; border-left-color: #0284c7;">
          <div style="font-size: 11px; font-weight: 700; color: #0369a1; display: flex; justify-content: space-between;">
            <span>农产品油脂 (4标的)</span>
            <span class="pill-blue badge-pill">9.9% 贡献</span>
          </div>
          <div style="font-size: 15px; font-weight: 800; color: #0284c7; margin: 2px 0 1px 0;">+1.40%</div>
          <div style="font-size: 10px; color: var(--text-secondary);">
            Sharpe <strong>1.28</strong> | 回撤 <strong>0.45%</strong><br/>
            sr (+0.54%), p (+0.52%)
          </div>
        </div>

        <div class="card card-purple" style="padding: 8px 10px;">
          <div style="font-size: 11px; font-weight: 700; color: #5b21b6; display: flex; justify-content: space-between;">
            <span>有色金属 (4标的)</span>
            <span class="pill-purple badge-pill">8.3% 贡献</span>
          </div>
          <div style="font-size: 15px; font-weight: 800; color: #7c3aed; margin: 2px 0 1px 0;">+1.17%</div>
          <div style="font-size: 10px; color: var(--text-secondary);">
            Sharpe <strong>0.52</strong> | 回撤 <strong>1.07%</strong><br/>
            cu (+0.78%), zn (+0.61%)
          </div>
        </div>

        <div class="card" style="padding: 8px 10px; border-left: 3.5px solid #64748b;">
          <div style="font-size: 11px; font-weight: 700; color: #334155; display: flex; justify-content: space-between;">
            <span>黑色建材 (5标的)</span>
            <span class="pill-amber badge-pill">防守微亏</span>
          </div>
          <div style="font-size: 15px; font-weight: 800; color: #475569; margin: 2px 0 1px 0;">-0.09%</div>
          <div style="font-size: 10px; color: var(--text-secondary);">
            <strong>2026转正 (+0.19%)</strong><br/>
            i (+0.56%), fg (+0.29%)
          </div>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <span>各板块收益结构稳健，非单标的赌博；黑色系在 2024-2025 宽幅单边阴跌与震荡中实现稳健防守</span>
      <span>Slide 10 of 12</span>
    </div>
  </div>

  <!-- ==================== SLIDE 10: 组合分配层与风控 ==================== -->
  <div class="slide" id="slide-11" data-title="11. 组合分配与微观风控" data-cat="Risk & Portfolio">
    <div class="slide-header">
      <div class="slide-header-left">
        <div class="slide-category">PORTFOLIO SYNTHESIS & RISK CONTROL</div>
        <h2 class="slide-title">组合分配层与微观物理风控机制</h2>
        <div class="slide-subtitle">层次聚类风险平价 (Alloc3_SectorHRP) 与换手死区磁滞滤波 (Deadzone=0.08)</div>
      </div>
      <span class="pill-purple badge-pill">HRP 树状聚类 + 死区防抖</span>
    </div>

    <div class="slide-content">
      <div class="grid-2" style="grid-template-columns: 1fr 1fr; gap: 14px;">
        <!-- Left: Sector HRP Diagram -->
        <div style="background: #ffffff; border: 1px solid var(--border-color); border-radius: 8px; padding: 12px 16px; box-shadow: var(--shadow-sm);">
          <div style="font-size: 12.5px; font-weight: 700; color: var(--primary); margin-bottom: 5px;">
            层次聚类风险平价分配 (Alloc3_SectorHRP)
          </div>
          <div style="font-size: 10.5px; color: var(--text-muted); margin-bottom: 8px;">
            基于图论层次聚类，克服马科维茨均值-方差在非平稳时序下的权重奇异性与极化跳跃
          </div>

          <svg viewBox="0 0 440 170" style="width: 100%; height: auto; display: block;" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif">
            <!-- Root Portfolio -->
            <rect x="170" y="8" width="100" height="24" rx="4" fill="#0f172a" />
            <text x="220" y="24" fill="#ffffff" font-size="10.5" font-weight="700" text-anchor="middle">组合总资本 (1.0x)</text>

            <line x1="220" y1="32" x2="220" y2="48" stroke="#64748b" stroke-width="1.5"/>
            <line x1="60" y1="48" x2="380" y2="48" stroke="#64748b" stroke-width="1.5"/>

            <!-- 5 Sectors -->
            <line x1="60" y1="48" x2="60" y2="62" stroke="#64748b" stroke-width="1.5"/>
            <rect x="20" y="62" width="80" height="28" rx="4" fill="#eff6ff" stroke="#93c5fd" />
            <text x="60" y="76" fill="#1e40af" font-size="9" font-weight="700" text-anchor="middle">贵金属</text>
            <text x="60" y="86" fill="#3b82f6" font-size="8" text-anchor="middle">au, ag (20%)</text>

            <line x1="140" y1="48" x2="140" y2="62" stroke="#64748b" stroke-width="1.5"/>
            <rect x="105" y="62" width="70" height="28" rx="4" fill="#eff6ff" stroke="#93c5fd" />
            <text x="140" y="76" fill="#1e40af" font-size="9" font-weight="700" text-anchor="middle">能化</text>
            <text x="140" y="86" fill="#3b82f6" font-size="8" text-anchor="middle">sc,ta,ma..(20%)</text>

            <line x1="220" y1="48" x2="220" y2="62" stroke="#64748b" stroke-width="1.5"/>
            <rect x="185" y="62" width="70" height="28" rx="4" fill="#eff6ff" stroke="#93c5fd" />
            <text x="220" y="76" fill="#1e40af" font-size="9" font-weight="700" text-anchor="middle">农产品</text>
            <text x="220" y="86" fill="#3b82f6" font-size="8" text-anchor="middle">sr,p,oi,m (20%)</text>

            <line x1="300" y1="48" x2="300" y2="62" stroke="#64748b" stroke-width="1.5"/>
            <rect x="265" y="62" width="70" height="28" rx="4" fill="#eff6ff" stroke="#93c5fd" />
            <text x="300" y="76" fill="#1e40af" font-size="9" font-weight="700" text-anchor="middle">有色</text>
            <text x="300" y="86" fill="#3b82f6" font-size="8" text-anchor="middle">cu,zn,ni,al (20%)</text>

            <line x1="380" y1="48" x2="380" y2="62" stroke="#64748b" stroke-width="1.5"/>
            <rect x="345" y="62" width="70" height="28" rx="4" fill="#eff6ff" stroke="#93c5fd" />
            <text x="380" y="76" fill="#1e40af" font-size="9" font-weight="700" text-anchor="middle">黑色</text>
            <text x="380" y="86" fill="#3b82f6" font-size="8" text-anchor="middle">i,fg,hc,jm,rb</text>

            <!-- Bottom Intra-cluster HRP -->
            <rect x="20" y="102" width="400" height="52" rx="5" fill="#f8fafc" stroke="#cbd5e1" />
            <text x="220" y="118" fill="#334155" font-size="10" font-weight="700" text-anchor="middle">簇内逆方差加权 + 准对角化树遍历 (Quasi-Diagonalization)</text>
            <text x="220" y="133" fill="#64748b" font-size="9" text-anchor="middle">自适应切断极端板块风险扩散；板块内单标的权重上限 10%，板块总敞口上限 30%</text>
            <text x="220" y="146" fill="#059669" font-size="9" font-weight="600" text-anchor="middle">彻底规避传统优化器“满仓单一品种”的病态过拟合</text>
          </svg>
        </div>

        <!-- Right: Deadzone Hysteresis Filter SVG -->
        <div style="background: #ffffff; border: 1px solid var(--border-color); border-radius: 8px; padding: 12px 16px; box-shadow: var(--shadow-sm);">
          <div style="font-size: 12.5px; font-weight: 700; color: var(--purple); margin-bottom: 5px;">
            换手死区磁滞滤波机制 (Deadzone = 0.08)
          </div>
          <div style="font-size: 10.5px; color: var(--text-muted); margin-bottom: 8px;">
            过滤微小权重噪声漂移，换手率从 118x 锐降至 83.9x，节省 1.2%~1.5% 物理摩擦
          </div>

          <svg viewBox="0 0 440 170" style="width: 100%; height: auto; display: block;" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif">
            <!-- Axis -->
            <line x1="40" y1="85" x2="400" y2="85" stroke="#94a3b8" stroke-width="1.5"/>
            <line x1="220" y1="15" x2="220" y2="155" stroke="#94a3b8" stroke-width="1.5"/>
            <text x="400" y="80" fill="#64748b" font-size="9">权重目标差 Δw</text>
            <text x="225" y="25" fill="#64748b" font-size="9">执行调仓量</text>

            <!-- Deadzone Band -->
            <rect x="160" y="15" width="120" height="140" fill="rgba(220, 38, 38, 0.08)" stroke="#fca5a5" stroke-dasharray="3,3"/>
            <text x="220" y="65" fill="#dc2626" font-size="10.5" font-weight="800" text-anchor="middle">死区区间 (不交易)</text>
            <text x="220" y="80" fill="#dc2626" font-size="8.5" text-anchor="middle">|Δw| &le; 0.08</text>
            <text x="160" y="100" fill="#dc2626" font-size="8" text-anchor="middle">-0.08</text>
            <text x="280" y="100" fill="#dc2626" font-size="8" text-anchor="middle">+0.08</text>

            <!-- Active Response Lines -->
            <line x1="280" y1="85" x2="380" y2="35" stroke="#059669" stroke-width="3"/>
            <circle cx="280" cy="85" r="4" fill="#059669"/>
            <text x="340" y="50" fill="#059669" font-size="9" font-weight="700">实际执行买入</text>

            <line x1="160" y1="85" x2="60" y2="135" stroke="#059669" stroke-width="3"/>
            <circle cx="160" cy="85" r="4" fill="#059669"/>
            <text x="100" y="125" fill="#059669" font-size="9" font-weight="700">实际执行卖出</text>

            <!-- Bottom Remark -->
            <text x="220" y="162" fill="#475569" font-size="9" text-anchor="middle">当目标权重变化未超过 8% 门槛时，维持前一时刻物理持仓不变</text>
          </svg>
        </div>
      </div>

      <!-- 3 Physical Risk Bounds Cards -->
      <div class="grid-3" style="margin-top: 3px;">
        <div class="card card-highlight">
          <div class="card-title">
            <span>单品种目标敞口上限 10%</span>
            <span class="pill-blue badge-pill">集中度防线</span>
          </div>
          <div class="card-desc">
            强制限制单一商品名义头寸不得超过总资本的 10%（1x 基准下实际平均敞口 22.3%），即便黄金、原油等出现单边脉冲也绝不超限，杜绝黑天鹅穿仓。
          </div>
        </div>

        <div class="card card-purple">
          <div class="card-title">
            <span>单板块总敞口上限 30%</span>
            <span class="pill-purple badge-pill">行业隔离</span>
          </div>
          <div class="card-desc">
            限制同一板块内部（如黑色系 5 大品种）的多空净敞口与绝对敞口合计不得超过 30%，彻底阻断由于单一产业政策突变引发的全板块系统性踩踏。
          </div>
        </div>

        <div class="card card-success">
          <div class="card-title">
            <span>动态隔夜保证金上限 40%</span>
            <span class="pill-green badge-pill">资金冗余</span>
          </div>
          <div class="card-desc">
            在实盘目标敞口版本 (3.0x 名义杠杆) 下，全市场隔夜实际占用保证金严格锁定在总净值的 40% 以内，预留 60% 充裕现金抵御连续反向涨跌停极端压力。
          </div>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <span>组合层 Alloc3_SectorHRP 结合 Deadzone=0.08 实现了策略从单兵作战到机构级资产配置舰队的质跃</span>
      <span>Slide 11 of 12</span>
    </div>
  </div>

  <!-- ==================== SLIDE 11: 总结与生产落地 ==================== -->
  <div class="slide" id="slide-12" data-title="12. 总结与生产落地路线图" data-cat="Roadmap">
    <div class="slide-header">
      <div class="slide-header-left">
        <div class="slide-category">PRODUCTION ROADMAP & STRATEGIC TAKEAWAYS</div>
        <h2 class="slide-title">总结与工程化生产落地路线图</h2>
        <div class="slide-subtitle">从严谨投研验证到毫秒级生产实盘：低延迟行情总线、Rust 算子、执行器与全息风控</div>
      </div>
      <span class="pill-green badge-pill">生产上线就绪 · 工业级标准</span>
    </div>

    <div class="slide-content">
      <!-- Production Pipeline SVG -->
      <div style="background: #ffffff; border: 1px solid var(--border-color); border-radius: 8px; padding: 10px 14px; box-shadow: var(--shadow-sm);">
        <div style="font-size: 12.5px; font-weight: 700; color: var(--primary); margin-bottom: 5px;">
          四阶段工程化生产落地流水线 (Institutional Production Pipeline)
        </div>

        <svg viewBox="0 0 960 110" style="width: 100%; height: auto; display: block;" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif">
          <defs>
            <marker id="pipe-arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 1 L 10 5 L 0 9 z" fill="#2563eb" />
            </marker>
          </defs>

          <!-- Phase 1 -->
          <g transform="translate(10, 8)">
            <rect width="215" height="92" rx="6" fill="#eff6ff" stroke="#bfdbfe" stroke-width="1.5"/>
            <text x="107" y="23" fill="#1e40af" font-size="11" font-weight="800" text-anchor="middle">Phase 1: 微观行情总线</text>
            <text x="12" y="43" fill="#1e293b" font-size="9.5">● CTP / 天勤高频行情广播</text>
            <text x="12" y="59" fill="#1e293b" font-size="9.5">● 原生 5m K线生成与持仓增量</text>
            <text x="12" y="75" fill="#1e293b" font-size="9.5">● 截面微观特征流 (时延 &lt; 50ms)</text>
          </g>

          <path d="M 230 54 L 255 54" stroke="#2563eb" stroke-width="2.5" marker-end="url(#pipe-arrow)"/>

          <!-- Phase 2 -->
          <g transform="translate(260, 8)">
            <rect width="215" height="92" rx="6" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
            <text x="107" y="23" fill="#0f172a" font-size="11" font-weight="800" text-anchor="middle">Phase 2: 纯 Rust 信号计算</text>
            <text x="12" y="43" fill="#334155" font-size="9.5">● czsc-signals 纯算子并发评估</text>
            <text x="12" y="59" fill="#334155" font-size="9.5">● 622 个准入因子微秒级结算</text>
            <text x="12" y="75" fill="#334155" font-size="9.5">● 零 GC 停顿，全品种内存对齐</text>
          </g>

          <path d="M 480 54 L 505 54" stroke="#2563eb" stroke-width="2.5" marker-end="url(#pipe-arrow)"/>

          <!-- Phase 3 -->
          <g transform="translate(510, 8)">
            <rect width="215" height="92" rx="6" fill="#f5f3ff" stroke="#ddd6fe" stroke-width="1.5"/>
            <text x="107" y="23" fill="#6d28d9" font-size="11" font-weight="800" text-anchor="middle">Phase 3: 目标权重与死区执行</text>
            <text x="12" y="43" fill="#4c1d95" font-size="9.5">● Alloc3_SectorHRP 聚类分配</text>
            <text x="12" y="59" fill="#4c1d95" font-size="9.5">● Deadzone=0.08 磁滞滤波</text>
            <text x="12" y="75" fill="#4c1d95" font-size="9.5">● TWAP / VWAP 智能拆单算法</text>
          </g>

          <path d="M 730 54 L 755 54" stroke="#2563eb" stroke-width="2.5" marker-end="url(#pipe-arrow)"/>

          <!-- Phase 4 -->
          <g transform="translate(760, 8)">
            <rect width="190" height="92" rx="6" fill="#ecfdf5" stroke="#a7f3d0" stroke-width="1.5"/>
            <text x="95" y="23" fill="#065f46" font-size="11" font-weight="800" text-anchor="middle">Phase 4: 全息风控看门狗</text>
            <text x="10" y="43" fill="#064e3b" font-size="9.5">● 实时持仓对账与滑点监控</text>
            <text x="10" y="59" fill="#064e3b" font-size="9.5">● 40% 保证金熔断告警</text>
            <text x="10" y="75" fill="#064e3b" font-size="9.5">● 异常撤单率与延迟告警</text>
          </g>
        </svg>
      </div>

      <!-- 4 Strategic Takeaway Cards -->
      <div class="grid-4">
        <div class="card card-highlight">
          <div class="card-title">
            <span>1. 跨尺度全周期覆盖</span>
            <span class="pill-blue badge-pill">Alpha 根基</span>
          </div>
          <div class="card-desc">
            商品期货市场的流动性博弈在不同周期展现完全不同的物理规律。单周期策略必然经历漫长回撤；1d 定势 + 60m/30m 骨架 + 15m/5m 微观脉冲的跨尺度共振，是平滑收益曲线的唯一科学路径。
          </div>
        </div>

        <div class="card card-purple">
          <div class="card-title">
            <span>2. 增量漏斗制度化门禁</span>
            <span class="pill-purple badge-pill">治理法宝</span>
          </div>
          <div class="card-desc">
            没有门禁的因子堆砌是灾难（ExpB 换手 118x 吞噬超额）。三阶段漏斗（自适应显著性 Gate 1、激活低共线 Gate 2、边际信息贡献 Gate 3）将摩擦压降 41%，确保每一个入选因子都有真实的增量生命力。
          </div>
        </div>

        <div class="card card-success">
          <div class="card-title">
            <span>3. 原生 5m 物理执行网格底座</span>
            <span class="pill-green badge-pill">工程核心</span>
          </div>
          <div class="card-desc">
            执行架构必须与微观因子物理频率严格对齐。采用原生 5m 物理网格作为多周期统一执行时钟，即时捕获 5m 持仓量脉冲与反包，在换手仅 83.9x 的低冲击下实现费后夏普 3.03，确保微观 Alpha 零时滞全额兑现。
          </div>
        </div>

        <div class="card card-highlight" style="border-left-color: var(--amber);">
          <div class="card-title">
            <span>4. 严格防前视与实战韧性</span>
            <span class="pill-amber badge-pill">生产保障</span>
          </div>
          <div class="card-desc">
            503 个交易日的纯样本外两折拼接盲测、真实手续费与滑点扣减、70% 品种胜率以及 Alloc3 聚类风控，证明了本系统绝非纸面过拟合玩具，具备直接上线机构级实盘生产的充沛底气与强大韧性。
          </div>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <span>汇报完毕 · 感谢各位评委与投研同仁的聆听 · 敬请批评指正！</span>
      <span>Slide 12 of 12</span>
    </div>
  </div>

</div>

<!-- Bottom Deck Controls -->
<div id="deck-footer">
  <div class="footer-hints">
    <span class="key-hint"><span class="kbd">←</span> / <span class="kbd">→</span> 翻页</span>
    <span class="key-hint"><span class="kbd">Space</span> 下一页</span>
    <span class="key-hint"><span class="kbd">F</span> 全屏</span>
    <span class="key-hint"><span class="kbd">O</span> 概览</span>
  </div>

  <div class="dots-nav" id="dots-container"></div>

  <div class="footer-controls">
    <button class="nav-btn" id="btn-prev" onclick="prevSlide()">◀ 上一页</button>
    <button class="nav-btn" id="btn-next" onclick="nextSlide()">下一页 ▶</button>
  </div>
</div>

<!-- Overview Grid Modal -->
<div id="overview-modal" onclick="closeOverview(event)">
  <div class="overview-grid" id="overview-grid-content"></div>
</div>

<script>
  let currentSlide = 0;
  const slides = document.querySelectorAll('.slide');
  const totalSlides = slides.length;
  const progressBar = document.getElementById('progress-bar');
  const slideCounter = document.getElementById('slide-counter');
  const headerSectionTitle = document.getElementById('header-section-title');
  const dotsContainer = document.getElementById('dots-container');
  const btnPrev = document.getElementById('btn-prev');
  const btnNext = document.getElementById('btn-next');
  const overviewModal = document.getElementById('overview-modal');
  const overviewGridContent = document.getElementById('overview-grid-content');

  // Chart Data Injection
  const navData = {{NAV_DATA}};
  const funnelData = {{FUNNEL_DATA}};
  const symbolData = {{SYMBOL_DATA}};

  // Clean, overlap-free Layouts
  const funnelLayout = {
    title: {
      text: "5大周期候选信号在 Gate 1 -> Gate 2 -> Gate 3 漏斗中的留存演进 (ExpF)",
      font: { size: 12.5, color: "#0f172a" },
      x: 0.0,
      xanchor: "left"
    },
    barmode: "group",
    margin: { l: 40, r: 15, t: 42, b: 32 },
    height: 200,
    legend: { orientation: "h", y: 1.18, x: 0.0, font: { size: 9.5 } },
    xaxis: { tickfont: { size: 10 } },
    yaxis: { title: { text: "信号实例数", font: { size: 10 } }, tickfont: { size: 9.5 } },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff"
  };

  const navLayout = {
    title: {
      text: "Core-20 纯样本外累计净值曲线 (Stitched OOS: 2025.01 ~ 2026.09, 1x 基准上限)",
      font: { size: 13, color: "#0f172a" },
      x: 0.0,
      xanchor: "left"
    },
    hovermode: "x unified",
    margin: { l: 50, r: 25, t: 55, b: 35 },
    height: 350,
    legend: { orientation: "h", y: 1.15, x: 0.0, font: { size: 9.5 } },
    xaxis: { anchor: "y", domain: [0.0, 1.0], matches: "x2", showticklabels: false },
    xaxis2: { anchor: "y2", domain: [0.0, 1.0], tickfont: { size: 10 } },
    yaxis: { anchor: "x", domain: [0.38, 1.0], title: { text: "累计净值 (Base=100)", font: { size: 10.5 } }, tickfont: { size: 10 } },
    yaxis2: { anchor: "x2", domain: [0.0, 0.28], title: { text: "回撤 (%)", font: { size: 10.5 } }, tickfont: { size: 9.5 } },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff"
  };

  const symbolLayout = {
    title: {
      text: "Core-20 全品种纯样本外净收益贡献 vs 交易成本拖累 (%) · 70% 品种胜率 (14/20)",
      font: { size: 12.5, color: "#0f172a" },
      x: 0.0,
      xanchor: "left"
    },
    barmode: "group",
    margin: { l: 45, r: 20, t: 40, b: 35 },
    height: 220,
    legend: { orientation: "h", y: 1.18, x: 0.0, font: { size: 10 } },
    xaxis: { title: { text: "Core-20 商品主力连续 (按净收益降序)", font: { size: 10.5 } }, tickfont: { size: 10 } },
    yaxis: { title: { text: "样本外回报 (%)", font: { size: 10.5 } }, tickfont: { size: 10 } },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff"
  };

  const chartRendered = {};

  function renderSlideCharts(idx) {
    if (!window.Plotly) return;
    const currentSlideEl = slides[idx];
    if (!currentSlideEl) return;

    if (currentSlideEl.querySelector('#chart-funnel')) {
      if (!chartRendered['funnel']) {
        Plotly.newPlot('chart-funnel', funnelData, funnelLayout, {responsive: true});
        chartRendered['funnel'] = true;
      } else {
        Plotly.Plots.resize(document.getElementById('chart-funnel'));
      }
    }
    if (currentSlideEl.querySelector('#chart-nav')) {
      if (!chartRendered['nav']) {
        Plotly.newPlot('chart-nav', navData, navLayout, {responsive: true});
        chartRendered['nav'] = true;
      } else {
        Plotly.Plots.resize(document.getElementById('chart-nav'));
      }
    }
    if (currentSlideEl.querySelector('#chart-symbol')) {
      if (!chartRendered['symbol']) {
        Plotly.newPlot('chart-symbol', symbolData, symbolLayout, {responsive: true});
        chartRendered['symbol'] = true;
      } else {
        Plotly.Plots.resize(document.getElementById('chart-symbol'));
      }
    }
  }

  // Initialize Dots
  for (let i = 0; i < totalSlides; i++) {
    const dot = document.createElement('div');
    dot.className = 'dot-pill' + (i === 0 ? ' active' : '');
    dot.title = slides[i].getAttribute('data-title') || `Slide ${i + 1}`;
    dot.onclick = () => goToSlide(i);
    dotsContainer.appendChild(dot);
  }

  // Initialize Overview Items
  for (let i = 0; i < totalSlides; i++) {
    const item = document.createElement('div');
    item.className = 'overview-item' + (i === 0 ? ' active' : '');
    const num = (i + 1 < 10 ? '0' : '') + (i + 1);
    const title = slides[i].getAttribute('data-title') || `Slide ${i + 1}`;
    const cat = slides[i].getAttribute('data-cat') || 'Section';
    item.innerHTML = `
      <div>
        <div class="ov-cat">${cat}</div>
        <div class="ov-title">${title}</div>
      </div>
      <div class="ov-num">${num} / ${totalSlides}</div>
    `;
    item.onclick = () => {
      goToSlide(i);
      overviewModal.classList.remove('active');
    };
    overviewGridContent.appendChild(item);
  }

  function updateDeck() {
    slides.forEach((s, idx) => {
      if (idx === currentSlide) {
        s.classList.add('active');
      } else {
        s.classList.remove('active');
      }
    });

    // Update Progress
    const pct = ((currentSlide + 1) / totalSlides) * 100;
    progressBar.style.width = pct + '%';

    // Update Counter
    const numStr = (currentSlide + 1 < 10 ? '0' : '') + (currentSlide + 1);
    slideCounter.innerText = `${numStr} / ${totalSlides}`;

    // Update Header Section Title
    const title = slides[currentSlide].getAttribute('data-title') || '';
    headerSectionTitle.innerText = title;

    // Update Dots
    document.querySelectorAll('.dot-pill').forEach((d, idx) => {
      if (idx === currentSlide) d.classList.add('active');
      else d.classList.remove('active');
    });

    // Update Overview Items
    document.querySelectorAll('.overview-item').forEach((item, idx) => {
      if (idx === currentSlide) item.classList.add('active');
      else item.classList.remove('active');
    });

    // Update Button Disabled State
    btnPrev.disabled = currentSlide === 0;
    btnNext.disabled = currentSlide === totalSlides - 1;

    // Deferred chart rendering
    setTimeout(() => {
      renderSlideCharts(currentSlide);
      window.dispatchEvent(new Event('resize'));
    }, 40);


  }

  function nextSlide() {
    if (currentSlide < totalSlides - 1) {
      currentSlide++;
      updateDeck();
    }
  }

  function prevSlide() {
    if (currentSlide > 0) {
      currentSlide--;
      updateDeck();
    }
  }

  function goToSlide(idx) {
    if (idx >= 0 && idx < totalSlides) {
      currentSlide = idx;
      updateDeck();
    }
  }

  function toggleFullscreen() {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(err => {
        alert(`Error attempting to enable fullscreen: ${err.message}`);
      });
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
      }
    }
  }

  function toggleOverview() {
    overviewModal.classList.toggle('active');
  }

  function closeOverview(e) {
    if (e.target === overviewModal) {
      overviewModal.classList.remove('active');
    }
  }

  // Keyboard Navigation
  window.addEventListener('keydown', (e) => {
    if (overviewModal.classList.contains('active')) {
      if (e.key === 'Escape' || e.key === 'o' || e.key === 'O') {
        overviewModal.classList.remove('active');
      }
      return;
    }

    if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') {
      e.preventDefault();
      nextSlide();
    } else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {
      e.preventDefault();
      prevSlide();
    } else if (e.key === 'Home') {
      e.preventDefault();
      goToSlide(0);
    } else if (e.key === 'End') {
      e.preventDefault();
      goToSlide(totalSlides - 1);
    } else if (e.key === 'f' || e.key === 'F') {
      e.preventDefault();
      toggleFullscreen();
    } else if (e.key === 'o' || e.key === 'O') {
      e.preventDefault();
      toggleOverview();
    }
  });


  // Initial update
  updateDeck();
</script>
</body>
</html>
"""

def build_presentation_html():
    nav_json, funnel_json, symbol_json = extract_chart_data()

    html = (
        HTML_TEMPLATE.replace("{{NAV_DATA}}", nav_json)
        .replace("{{FUNNEL_DATA}}", funnel_json)
        .replace("{{SYMBOL_DATA}}", symbol_json)
    )

    with open(TARGET_HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Presentation successfully written to: {TARGET_HTML_PATH}")

    # Copy to docs/research/ for reference & persistence
    os.makedirs(os.path.dirname(DOCS_HTML_PATH), exist_ok=True)
    shutil.copy(TARGET_HTML_PATH, DOCS_HTML_PATH)
    print(f"Presentation also copied to: {DOCS_HTML_PATH}")

if __name__ == "__main__":
    build_presentation_html()
