"""
AlphaLab 投研台账与回测结果查询前端 UI 模板生成器
================================================================================
生成零外部依赖、同时支持「PostgreSQL 实时 API 直连」与「单文件离线快照」的响应式前端页面。
"""

from __future__ import annotations

import json
from typing import Any

_CSS_STYLES = """
:root {
  --bg: #f1f5f9;
  --card: #ffffff;
  --border: #e2e8f0;
  --text: #0f172a;
  --muted: #64748b;
  --primary: #2563eb;
  --primary-hover: #1d4ed8;
  --primary-soft: #eff6ff;
  --green: #059669;
  --green-soft: #ecfdf5;
  --red: #dc2626;
  --red-soft: #fef2f2;
  --amber: #d97706;
  --amber-soft: #fffbeb;
  --purple: #7c3aed;
  --purple-soft: #f5f3ff;
  --mono: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Microsoft YaHei", sans-serif;
  background: var(--bg);
  color: var(--text);
  font-size: 13.5px;
  line-height: 1.5;
}
.topbar {
  background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
  color: #f8fafc;
  padding: 14px 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 12px;
  box-shadow: 0 2px 8px rgba(15, 23, 42, 0.18);
  position: sticky;
  top: 0;
  z-index: 100;
}
.topbar-title { display: flex; align-items: center; gap: 12px; }
.topbar-title h1 { font-size: 17px; font-weight: 700; letter-spacing: 0.2px; }
.topbar-sub { font-size: 12px; color: #94a3b8; font-family: var(--mono); }
.topbar-actions { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.conn-badge {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 600;
  background: rgba(16, 185, 129, 0.16); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.35);
}
.conn-badge.offline {
  background: rgba(245, 158, 11, 0.16); color: #fbbf24; border-color: rgba(251, 191, 36, 0.35);
}
.btn {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 6px 12px; border-radius: 6px; font-size: 12.5px; font-weight: 600;
  cursor: pointer; border: 1px solid transparent; transition: all 0.15s;
  text-decoration: none;
}
.btn-primary { background: var(--primary); color: #fff; }
.btn-primary:hover { background: var(--primary-hover); }
.btn-outline { background: rgba(255,255,255,0.08); color: #e2e8f0; border-color: rgba(255,255,255,0.2); }
.btn-outline:hover { background: rgba(255,255,255,0.16); }
.btn-light { background: #fff; color: var(--text); border-color: var(--border); }
.btn-light:hover { background: #f8fafc; border-color: #cbd5e1; }
.btn-xs { padding: 3px 8px; font-size: 11.5px; border-radius: 4px; }
.container { max-width: 1600px; margin: 0 auto; padding: 18px 24px 48px; }
.kpi-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 12px; margin-bottom: 16px;
}
.kpi-card {
  background: var(--card); border: 1px solid var(--border); border-radius: 10px;
  padding: 14px 16px; box-shadow: 0 1px 2px rgba(15,23,42,0.03);
}
.kpi-label { font-size: 12px; color: var(--muted); font-weight: 600; margin-bottom: 4px; }
.kpi-val { font-size: 22px; font-weight: 800; font-family: var(--mono); color: var(--text); }
.kpi-sub { font-size: 11.5px; color: var(--muted); margin-top: 4px; }
"""

_CSS_STYLES_PART2 = """
.nav-tabs {
  display: flex; gap: 6px; border-bottom: 2px solid var(--border);
  margin-bottom: 16px; overflow-x: auto; padding-bottom: 2px;
}
.tab-btn {
  padding: 9px 16px; border-radius: 8px 8px 0 0; border: 1px solid transparent;
  background: transparent; color: var(--muted); font-size: 13.5px; font-weight: 600;
  cursor: pointer; white-space: nowrap; transition: all 0.15s;
}
.tab-btn:hover { color: var(--text); background: #e2e8f0; }
.tab-btn.active {
  background: var(--card); color: var(--primary);
  border-color: var(--border); border-bottom-color: var(--card);
  margin-bottom: -2px; box-shadow: 0 -2px 0 var(--primary) inset;
}
.tab-panel { display: none; }
.tab-panel.active { display: block; }
.panel-card {
  background: var(--card); border: 1px solid var(--border); border-radius: 10px;
  padding: 16px 18px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(15,23,42,0.03);
}
.panel-header {
  display: flex; align-items: center; justify-content: space-between;
  flex-wrap: wrap; gap: 10px; margin-bottom: 12px;
}
.panel-title { font-size: 15px; font-weight: 700; color: var(--text); display: flex; align-items: center; gap: 8px; }
.panel-desc { font-size: 12px; color: var(--muted); }
.filter-bar {
  display: flex; align-items: center; flex-wrap: wrap; gap: 10px;
  background: #f8fafc; border: 1px solid var(--border); border-radius: 8px;
  padding: 10px 14px; margin-bottom: 14px;
}
.filter-group { display: flex; align-items: center; gap: 6px; font-size: 12.5px; color: var(--muted); font-weight: 600; }
.filter-group select, .filter-group input[type="text"] {
  padding: 5px 10px; border: 1px solid #cbd5e1; border-radius: 6px;
  font-size: 12.5px; color: var(--text); background: #fff; outline: none;
}
.filter-group select:focus, .filter-group input[type="text"]:focus { border-color: var(--primary); }
.filter-check { display: inline-flex; align-items: center; gap: 5px; font-size: 12.5px; cursor: pointer; font-weight: 600; color: var(--text); }
.table-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 8px; max-height: 620px; overflow-y: auto; }
table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
th {
  position: sticky; top: 0; background: #f8fafc; color: #334155;
  font-weight: 700; text-align: left; padding: 9px 11px;
  border-bottom: 2px solid var(--border); white-space: nowrap; z-index: 10;
}
td { padding: 8px 11px; border-bottom: 1px solid #f1f5f9; vertical-align: middle; }
tr:hover td { background: #f8fafc; }
.num { font-family: var(--mono); text-align: right; white-space: nowrap; }
.mono { font-family: var(--mono); }
.pos { color: var(--green); font-weight: 600; }
.neg { color: var(--red); font-weight: 600; }
.badge {
  display: inline-block; padding: 2px 8px; border-radius: 999px;
  font-size: 11px; font-weight: 700; font-family: var(--mono); white-space: nowrap;
}
.badge-green { background: var(--green-soft); color: var(--green); border: 1px solid #a7f3d0; }
.badge-red { background: var(--red-soft); color: var(--red); border: 1px solid #fecaca; }
.badge-amber { background: var(--amber-soft); color: var(--amber); border: 1px solid #fde68a; }
.badge-blue { background: var(--primary-soft); color: var(--primary); border: 1px solid #bfdbfe; }
.badge-purple { background: var(--purple-soft); color: var(--purple); border: 1px solid #ddd6fe; }
.badge-gray { background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; }
.niche-cell {
  cursor: pointer; padding: 8px 10px; border-radius: 6px; border: 1px solid transparent;
  transition: all 0.15s; text-align: center;
}
.niche-cell:hover { border-color: var(--primary); transform: translateY(-1px); }
.niche-cell.selected { outline: 2px solid var(--primary); background: var(--primary-soft) !important; }
.sector-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(195px, 1fr)); gap: 10px; margin-bottom: 14px; }
.sector-card {
  background: #f8fafc; border: 1px solid var(--border); border-radius: 8px;
  padding: 10px 12px; cursor: pointer; transition: all 0.15s;
}
.sector-card:hover, .sector-card.active { border-color: var(--primary); background: var(--primary-soft); }
.sql-box {
  width: 100%; min-height: 105px; font-family: var(--mono); font-size: 12.5px;
  padding: 10px 12px; border: 1px solid #cbd5e1; border-radius: 8px;
  background: #0f172a; color: #e2e8f0; resize: vertical; margin-bottom: 10px;
}
.summary-strip {
  display: flex; gap: 18px; flex-wrap: wrap; align-items: center;
  padding: 8px 14px; background: var(--primary-soft); border: 1px solid #bfdbfe;
  border-radius: 8px; margin-bottom: 12px; font-size: 12.5px;
}
"""

_HTML_BODY_PART1 = """
<header class="topbar">
  <div class="topbar-title">
    <div>
      <h1>🏛️ CZSC AlphaLab · PostgreSQL 投研台账与回测查询工作台</h1>
      <div class="topbar-sub" id="db-meta-line">正在加载实验台账元数据...</div>
    </div>
  </div>
  <div class="topbar-actions">
    <span class="conn-badge" id="conn-badge">🟢 连接检测中...</span>
    <button class="btn btn-outline" onclick="refreshAllData()">🔄 刷新台账</button>
    <a class="btn btn-primary" id="export-html-btn" href="/api/export-html" onclick="return handleExportHtml(event)">📥 导出独立 HTML</a>
  </div>
</header>

<main class="container">
  <section class="kpi-grid" id="kpi-grid"></section>

  <nav class="nav-tabs">
    <button class="tab-btn active" data-tab="tab-ablation" onclick="switchTab('tab-ablation')">📈 1. 组合回测与 Stage 1~4 消融对比</button>
    <button class="tab-btn" data-tab="tab-pool" onclick="switchTab('tab-pool')">🏆 2. 因子池排行榜 & 6×5 生态位地图</button>
    <button class="tab-btn" data-tab="tab-assets" onclick="switchTab('tab-assets')">🔍 3. Full-60 分品种回测明细下钻</button>
    <button class="tab-btn" data-tab="tab-signals" onclick="switchTab('tab-signals')">⚡ 4. Rust 信号库、预检 & AST 血缘</button>
    <button class="tab-btn" data-tab="tab-graveyard" onclick="switchTab('tab-graveyard')">🪦 5. 失败实验墓地 & SQL 控制台</button>
  </nav>

  <!-- Tab 1: Ablation Runs -->
  <section class="tab-panel active" id="tab-ablation">
    <div class="panel-card">
      <div class="panel-header">
        <div>
          <div class="panel-title">📈 策略组合样本外净值曲线与水下回撤图 (`2025-01-02` ~ `2026-09-30` 全商品池 Full-60)</div>
          <div class="panel-desc">上方为累计净值/累计收益曲线，下方为同步水下回撤；支持鼠标悬停十字准星对比与下方表格勾选自定义叠加</div>
        </div>
        <div class="filter-group" style="flex-wrap:wrap;">
          <label>预设组合:</label>
          <select id="nav-preset" onchange="onNavPresetChange()">
            <option value="stage_evo">Stage 1 ~ Stage 5 核心演进对比</option>
            <option value="univ_funnel">品种池漏斗筛选对比 (N=28/30/32/35/40 vs Full-60)</option>
            <option value="arch_1x">Full-60 架构变体对比 (1x 标准波动)</option>
            <option value="arch_3x">Full-60 3倍目标波动组合 (3x Target)</option>
            <option value="custom">自定义表格勾选对比</option>
          </select>
          <label>权重口径:</label>
          <select id="nav-metric" onchange="renderNavChart()">
            <option value="inst">机构两层风险预算 (Institutional)</option>
            <option value="flat">Flat-60 等权组合 (Flat HRP)</option>
            <option value="both">同组合双口径对比 (实线=机构 / 虚线=等权)</option>
          </select>
          <label>纵轴:</label>
          <select id="nav-unit" onchange="renderNavChart()">
            <option value="nav">单位净值 (NAV, 初始=1.000)</option>
            <option value="pct">累计收益率 (%)</option>
          </select>
        </div>
      </div>
      <div id="nav-legend-bar" style="display:flex; gap:12px; flex-wrap:wrap; margin-bottom:8px; font-size:12px;"></div>
      <div id="nav-svg-wrap" style="position:relative; width:100%; background:#f8fafc; border:1px solid var(--border); border-radius:8px; padding:8px;"></div>
    </div>

    <div class="panel-card">
      <div class="panel-header">
        <div>
          <div class="panel-title">📊 Stage 1 ~ Stage 5 & 品种池漏斗消融明细表 (`alpha_lab.ablation_runs`)</div>
          <div class="panel-desc">勾选第一列「叠加净值」复选框可将任意策略组合曲线实时加入上方净值图对比</div>
        </div>
        <div class="filter-group">
          <label>阶段:</label>
          <select id="ab-stage-filter" onchange="renderAblationTab()"><option value="">全部阶段</option></select>
          <label>模式:</label>
          <select id="ab-mode-filter" onchange="renderAblationTab()">
            <option value="">全部模式</option>
            <option value="baseline">baseline (基线)</option>
            <option value="incremental">incremental (增量合成)</option>
            <option value="standalone">standalone (独立消融)</option>
            <option value="universe_funnel">universe_funnel (品种池漏斗 N=28~40)</option>
            <option value="architecture_variant">architecture_variant (架构变体)</option>
            <option value="3x_target">3x_target (3倍目标波动)</option>
          </select>
        </div>
      </div>
      <div id="ablation-chart-box" style="margin-bottom:14px; overflow-x:auto;"></div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th style="text-align:center;">叠加净值</th>
              <th>运行标识 (`run_id`)</th>
              <th>阶段 (`stage`)</th>
              <th>模式 (`mode`)</th>
              <th class="num">Flat-60 年化</th>
              <th class="num">Flat-60 Sharpe</th>
              <th class="num">Flat-60 最大回撤</th>
              <th class="num">Flat-60 Calmar</th>
              <th class="num">机构两层年化</th>
              <th class="num">机构两层 Sharpe</th>
              <th class="num">机构最大回撤</th>
              <th class="num">机构 Calmar</th>
              <th class="num">单品种均因子</th>
              <th class="num">OOS 天数</th>
            </tr>
          </thead>
          <tbody id="ablation-tbody"></tbody>
        </table>
      </div>
    </div>
  </section>

  <!-- Tab 2: Niche Matrix & Factor Pool Scores -->
  <section class="tab-panel" id="tab-pool">
    <div class="panel-card">
      <div class="panel-header">
        <div>
          <div class="panel-title">🗺️ 6×5 生态位饱和度矩阵 (Cluster A~F × Timeframe `1d`/`60m`/`30m`/`15m`/`5m`)</div>
          <div class="panel-desc">点击任意单元格即可联动过滤下方因子池排行榜（格式：`Gate1B通过折数 / 评估UID数`，括号内为平均 Sharpe）</div>
        </div>
        <button class="btn btn-light btn-xs" onclick="resetPoolNicheFilter()">重置矩阵选中</button>
      </div>
      <div class="table-wrap" style="max-height:none;">
        <table>
          <thead>
            <tr>
              <th>机制主簇 (Cluster A~F)</th>
              <th style="text-align:center">`1d`</th>
              <th style="text-align:center">`60m`</th>
              <th style="text-align:center">`30m`</th>
              <th style="text-align:center">`15m`</th>
              <th style="text-align:center">`5m`</th>
            </tr>
          </thead>
          <tbody id="niche-matrix-tbody"></tbody>
        </table>
      </div>
    </div>
"""

_HTML_BODY_PART2 = """
    <div class="panel-card">
      <div class="panel-header">
        <div>
          <div class="panel-title">🏆 Full-60 跨品种池化准入排行榜 (`alpha_lab.factor_pool_scores`)</div>
          <div class="panel-desc" id="pool-count-desc">正在加载池化评分记录...</div>
        </div>
      </div>
      <div class="filter-bar">
        <div class="filter-group"><label>窗口:</label><select id="pool-fold" onchange="renderPoolTable()"><option value="">全部 Fold</option></select></div>
        <div class="filter-group"><label>主簇:</label><select id="pool-cluster" onchange="renderPoolTable()"><option value="">全部主簇 (A~F)</option></select></div>
        <div class="filter-group"><label>周期:</label><select id="pool-tf" onchange="renderPoolTable()"><option value="">全部周期</option></select></div>
        <div class="filter-group"><label>阶段:</label><select id="pool-stage" onchange="renderPoolTable()"><option value="">全部阶段</option></select></div>
        <div class="filter-group"><label>排序:</label>
          <select id="pool-sort" onchange="renderPoolTable()">
            <option value="mean_sharpe">全池 Mean Sharpe ↓</option>
            <option value="pooled_t">Pooled-t 统计量 ↓</option>
            <option value="pos_asset_ratio">正收益品种占比 ↓</option>
            <option value="total_n">总成交笔数 N ↓</option>
          </select>
        </div>
        <label class="filter-check"><input type="checkbox" id="pool-gate1b" onchange="renderPoolTable()"> 仅看 Gate 1B 严格准入</label>
        <div class="filter-group"><input type="text" id="pool-q" placeholder="搜索 UID / 家族 / 名称..." oninput="renderPoolTable()"></div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>排名</th>
              <th>窗口 (`fold`)</th>
              <th>候选 (`uid`)</th>
              <th>家族 (`family_id`)</th>
              <th>因子名称</th>
              <th>周期</th>
              <th>阶段 (`stage`)</th>
              <th class="num">全池 Mean Sharpe</th>
              <th class="num">Pooled-t</th>
              <th class="num">正收益品种比</th>
              <th class="num">总笔数 N</th>
              <th>Gate 1B</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody id="pool-tbody"></tbody>
        </table>
      </div>
    </div>
  </section>

  <!-- Tab 3: Asset Scores Drill-Down -->
  <section class="tab-panel" id="tab-assets">
    <div class="panel-card">
      <div class="panel-header">
        <div>
          <div class="panel-title">🧭 六大商品板块回测表现总览（点击板块卡片可快速过滤）</div>
          <div class="panel-desc">覆盖黑色、贵金属、有色、能化、新能源、农产品共 60 个主力期货品种</div>
        </div>
        <button class="btn btn-light btn-xs" onclick="resetAssetFilters()">重置全部分品种筛选</button>
      </div>
      <div class="sector-grid" id="sector-cards-grid"></div>
      <div class="filter-bar">
        <div class="filter-group"><label>窗口:</label><select id="as-fold" onchange="queryAssetScores(0)"><option value="">全部 Fold</option></select></div>
        <div class="filter-group"><label>板块:</label><select id="as-sector" onchange="onAssetSectorChange()"><option value="">全部板块</option></select></div>
        <div class="filter-group"><label>品种:</label><select id="as-asset" onchange="queryAssetScores(0)"><option value="">全部 60 品种</option></select></div>
        <div class="filter-group"><label>周期:</label><select id="as-tf" onchange="queryAssetScores(0)"><option value="">全部周期</option></select></div>
        <div class="filter-group"><label>阶段:</label><select id="as-stage" onchange="queryAssetScores(0)"><option value="">全部阶段</option></select></div>
        <div class="filter-group"><label>排序:</label>
          <select id="as-sort" onchange="queryAssetScores(0)">
            <option value="is_net_sharpe">净夏普 (is_net_sharpe) ↓</option>
            <option value="t_stat">t 统计量 (t_stat) ↓</option>
            <option value="safety_margin">安全边际 (safety_margin) ↓</option>
            <option value="win_rate">胜率 (win_rate) ↓</option>
            <option value="pl_ratio">盈亏比 (pl_ratio) ↓</option>
            <option value="mean_net_pct">单笔均净利% (mean_net_pct) ↓</option>
            <option value="n">交易笔数 (n) ↓</option>
            <option value="ann_turnover">年化换手 (ann_turnover) ↓</option>
          </select>
        </div>
        <label class="filter-check"><input type="checkbox" id="as-canonical" checked onchange="queryAssetScores(0)"> 仅标准变体</label>
        <label class="filter-check"><input type="checkbox" id="as-localpass" onchange="queryAssetScores(0)"> 仅单品种达标</label>
        <div class="filter-group"><input type="text" id="as-q" placeholder="搜索 UID / 品种 / 家族..." oninput="debounceAssetQuery()"></div>
      </div>
      <div class="summary-strip" id="asset-summary-strip"></div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>窗口</th>
              <th>板块 (`sector`)</th>
              <th>品种 (`asset`)</th>
              <th>候选 (`uid`)</th>
              <th>家族</th>
              <th>周期</th>
              <th>阶段</th>
              <th class="num">笔数 N</th>
              <th class="num">胜率</th>
              <th class="num">盈亏比 R</th>
              <th class="num">安全边际 ΔW</th>
              <th class="num">单笔均净利%</th>
              <th class="num">t 统计量</th>
              <th class="num">样本内净 Sharpe</th>
              <th class="num">年化换手</th>
              <th>单品种达标</th>
            </tr>
          </thead>
          <tbody id="asset-tbody"></tbody>
        </table>
      </div>
      <div style="display:flex; justify-content:space-between; align-items:center; margin-top:12px;">
        <div id="asset-page-info" class="panel-desc"></div>
        <div style="display:flex; gap:8px;">
          <button class="btn btn-light btn-xs" id="as-prev-btn" onclick="stepAssetPage(-1)">⬅ 上一页</button>
          <button class="btn btn-light btn-xs" id="as-next-btn" onclick="stepAssetPage(1)">下一页 ➡</button>
        </div>
      </div>
    </div>
  </section>
"""

_HTML_BODY_PART3 = """
  <!-- Tab 4: Rust Signals, Preflight & AST Lineage -->
  <section class="tab-panel" id="tab-signals">
    <div class="panel-card">
      <div class="panel-header">
        <div>
          <div class="panel-title">⚡ Rust 原生 K 线信号目录与开采状态 (`alpha_lab.rust_signals`)</div>
          <div class="panel-desc" id="sig-count-desc">覆盖 17 个 Rust 信号子模块，支持快速筛选「精确镜像对称且尚未消费」的蓝海信号</div>
        </div>
      </div>
      <div class="filter-bar">
        <div class="filter-group"><label>子模块:</label><select id="sig-cat" onchange="renderSignalsTable()"><option value="">全部 17 个分类</option></select></div>
        <div class="filter-group"><label>状态筛选:</label>
          <select id="sig-mode" onchange="renderSignalsTable()">
            <option value="all">全部 Rust 信号 (234)</option>
            <option value="blue_ocean">💎 待开采对称蓝海 (exact_mirror=True & 未消费)</option>
            <option value="consumed">✅ 已被因子族消费 (consumed > 0)</option>
            <option value="mirror_all">🪞 全部镜像对称信号 (exact_mirror=True)</option>
          </select>
        </div>
        <div class="filter-group"><input type="text" id="sig-q" placeholder="搜索信号名 / state_key / 参数模板..." oninput="renderSignalsTable()"></div>
      </div>
      <div class="table-wrap" style="max-height:420px;">
        <table>
          <thead>
            <tr>
              <th>信号函数名 (`signal_name`)</th>
              <th>分类 (`category`)</th>
              <th>挂载状态键 (`state_key`)</th>
              <th>参数模板 (`params_template`)</th>
              <th>镜像对称</th>
              <th class="num">激活率</th>
              <th class="num">消费次数</th>
              <th>消费该信号的因子家族</th>
            </tr>
          </thead>
          <tbody id="sig-tbody"></tbody>
        </table>
      </div>
    </div>

    <div class="panel-card">
      <div class="panel-header">
        <div>
          <div class="panel-title">🧬 28 个 SSOT 原子因子族 AST 静态血缘与 <1s 预检审计 (`preflight_audits` + `factor_signal_lineage`)</div>
          <div class="panel-desc">自动提取五类特征依赖 (`state` / `strength` / `oi` / `cs` / `htf`) 并检验仿射反转对称性与零前视泄漏</div>
        </div>
        <div class="filter-group"><input type="text" id="lin-q" placeholder="搜索家族 / 信号 / 特征..." oninput="renderLineageTable()"></div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>家族 (`family_id`)</th>
              <th>名称</th>
              <th>阶段</th>
              <th>适用周期</th>
              <th>预检门禁</th>
              <th class="num">对称误差 (`fire/q`)</th>
              <th class="num">多/空触发</th>
              <th class="num">耗时(ms)</th>
              <th>AST 静态依赖血缘 (`state` / `strength` / `oi` / `cs` / `htf`)</th>
            </tr>
          </thead>
          <tbody id="lin-tbody"></tbody>
        </table>
      </div>
    </div>
  </section>

  <!-- Tab 5: Graveyard & Read-Only SQL Console -->
  <section class="tab-panel" id="tab-graveyard">
    <div class="panel-card">
      <div class="panel-header">
        <div>
          <div class="panel-title">💻 PostgreSQL (`alpha_lab` Schema) 在线只读 SQL 查询控制台</div>
          <div class="panel-desc">支持针对 8 张台账表直接编写 `SELECT` / `WITH` 聚合联表查询，并支持一键导出 CSV</div>
        </div>
        <div style="display:flex; gap:6px; flex-wrap:wrap;">
          <button class="btn btn-light btn-xs" onclick="loadPresetSql(0)">预设1: 组合消融演进</button>
          <button class="btn btn-light btn-xs" onclick="loadPresetSql(1)">预设2: Gate1B Top25因子</button>
          <button class="btn btn-light btn-xs" onclick="loadPresetSql(2)">预设3: 六大板块绩效汇总</button>
          <button class="btn btn-light btn-xs" onclick="loadPresetSql(3)">预设4: 待开采高激活对称信号</button>
          <button class="btn btn-light btn-xs" onclick="loadPresetSql(4)">预设5: 因子↔Rust信号血缘联表</button>
          <button class="btn btn-light btn-xs" onclick="loadPresetSql(5)">预设6: 墓地淘汰门禁分布</button>
        </div>
      </div>
      <textarea class="sql-box" id="sql-input" spellcheck="false"></textarea>
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; flex-wrap:wrap; gap:10px;">
        <div style="display:flex; gap:8px; align-items:center;">
          <button class="btn btn-primary" onclick="runCustomSql()">▶ 执行 SQL 查询</button>
          <button class="btn btn-light" onclick="exportSqlCsv()">📥 导出查询结果 CSV</button>
          <span id="sql-status-text" class="panel-desc"></span>
        </div>
      </div>
      <div class="table-wrap" id="sql-result-wrap" style="max-height:380px; display:none;">
        <table>
          <thead id="sql-thead"></thead>
          <tbody id="sql-tbody"></tbody>
        </table>
      </div>
    </div>

    <div class="panel-card">
      <div class="panel-header">
        <div>
          <div class="panel-title">🪦 失败实验与证伪候选墓地 (`alpha_lab.factor_graveyard`)</div>
          <div class="panel-desc">归档被 Gate 1B/2/3 或 Pre-flight 门禁拦截的候选因子，防止同质化重复试错</div>
        </div>
        <button class="btn btn-light btn-xs" onclick="resetGraveFilter()">显示全部淘汰记录</button>
      </div>
      <div id="grave-gate-badges" style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px;"></div>
      <div class="filter-bar">
        <div class="filter-group"><label>拦截门禁:</label><select id="gr-gate" onchange="renderGraveyardTable()"><option value="">全部门禁</option></select></div>
        <div class="filter-group"><label>阶段:</label><select id="gr-stage" onchange="renderGraveyardTable()"><option value="">全部阶段</option></select></div>
        <div class="filter-group"><input type="text" id="gr-q" placeholder="搜索 UID / 家族 / 淘汰原因..." oninput="renderGraveyardTable()"></div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>候选标识 (`uid`)</th>
              <th>家族</th>
              <th>周期</th>
              <th>阶段</th>
              <th>拦截门禁 (`failed_gate`)</th>
              <th class="num">Mean IS Sharpe</th>
              <th class="num">年化换手</th>
              <th>淘汰原因与证伪摘要 (`reason`)</th>
              <th>归档时间</th>
            </tr>
          </thead>
          <tbody id="grave-tbody"></tbody>
        </table>
      </div>
    </div>
  </section>
</main>
"""

_JS_SCRIPT_PART1 = """
const STATE = {
  liveMode: Boolean(window.__LIVE_MODE__) && window.location.protocol.startsWith('http'),
  snap: window.__INITIAL_SNAPSHOT__ || {},
  selectedNiche: { cluster: '', tf: '' },
  selectedNavRuns: ['Stage1_Base_Full60', 'Stage2_Rust_Incremental', 'Stage3_OI_Incremental', 'Stage4_CE_Incremental'],
  activeNavSeries: [],
  assetOffset: 0,
  assetLimit: 100,
  assetTimer: null,
  lastSqlResult: { columns: [], rows: [] }
};

const NAV_COLORS = ['#2563eb', '#059669', '#d97706', '#dc2626', '#7c3aed', '#0891b2', '#db2777', '#475569'];

const PRESET_SQLS = [
  `SELECT run_id, stage, mode, flat60_ann_ret_pct, flat60_sharpe, flat60_max_dd_pct,
       inst_ann_ret_pct, inst_sharpe, inst_max_dd_pct, inst_calmar, avg_factors_per_asset
FROM alpha_lab.ablation_runs
ORDER BY stage, inst_sharpe DESC`,
  `SELECT fold, uid, family_id, tf, stage, mean_sharpe, pooled_t, pos_asset_ratio, total_n
FROM alpha_lab.factor_pool_scores
WHERE gate1b_strict_pass = TRUE
ORDER BY mean_sharpe DESC
LIMIT 25`,
  `SELECT sector, COUNT(DISTINCT asset) AS assets, COUNT(*) AS eval_rows,
       ROUND(AVG(is_net_sharpe)::numeric, 4) AS avg_sharpe,
       ROUND(AVG(win_rate)::numeric, 4) AS avg_win_rate,
       SUM(CASE WHEN strict_local_pass THEN 1 ELSE 0 END) AS local_pass_cnt
FROM alpha_lab.factor_asset_scores
WHERE is_canonical = TRUE
GROUP BY sector
ORDER BY avg_sharpe DESC`,
  `SELECT signal_name, category, state_key, active_rate, params_template
FROM alpha_lab.rust_signals
WHERE exact_mirror = TRUE AND consumed_count = 0
ORDER BY active_rate DESC
LIMIT 30`,
  `SELECT l.family_id, l.stage, l.archetype, l.rust_signal_name, l.state_key, l.call_type,
       r.category, r.exact_mirror, r.active_rate
FROM alpha_lab.factor_signal_lineage l
LEFT JOIN alpha_lab.rust_signals r ON l.rust_signal_name = r.signal_name
ORDER BY l.stage, l.family_id, l.call_type`,
  `SELECT failed_gate, COUNT(*) AS rejected_cnt,
       ROUND(AVG(mean_is_sharpe)::numeric, 4) AS avg_is_sharpe,
       ROUND(AVG(mean_ann_turnover)::numeric, 2) AS avg_turnover
FROM alpha_lab.factor_graveyard
GROUP BY failed_gate
ORDER BY rejected_cnt DESC`
];

function esc(v) {
  if (v === null || v === undefined) return '';
  return String(v).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
function fmtNum(v, d = 2) {
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(d) : '-';
}
function fmtSigned(v, d = 2, suffix = '') {
  const n = Number(v);
  if (!Number.isFinite(n)) return '-';
  const cls = n > 0 ? 'pos' : (n < 0 ? 'neg' : '');
  const sign = n > 0 ? '+' : '';
  return `<span class="${cls}">${sign}${n.toFixed(d)}${suffix}</span>`;
}
function switchTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tabId));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.id === tabId));
}
function fillSelect(selId, items, defaultLabel) {
  const el = document.getElementById(selId);
  if (!el) return;
  const cur = el.value;
  el.innerHTML = `<option value="">${esc(defaultLabel)}</option>` +
    (items || []).map(x => {
      if (typeof x === 'object' && x !== null) {
        return `<option value="${esc(x.value)}">${esc(x.label)}</option>`;
      }
      return `<option value="${esc(x)}">${esc(x)}</option>`;
    }).join('');
  if (cur) el.value = cur;
}
"""

_JS_SCRIPT_PART2 = """
function initDashboard() {
  const ov = STATE.snap.overview || {};
  const badge = document.getElementById('conn-badge');
  if (STATE.liveMode) {
    badge.className = 'conn-badge';
    badge.textContent = `🟢 PostgreSQL 实时直连 (${ov.backend || 'postgres'})`;
  } else {
    badge.className = 'conn-badge offline';
    badge.textContent = `📦 离线静态快照 (${ov.backend || 'postgres'})`;
  }
  document.getElementById('db-meta-line').textContent =
    `后端: ${ov.backend || 'postgres'} | 连接: ${ov.db_url || ''} | 快照时间: ${ov.generated_at || ''}`;

  const tc = ov.table_counts || {};
  const sc = ov.signal_coverage || {};
  const abRows = STATE.snap.ablation || [];
  const poolRows = STATE.snap.pool_scores || [];
  const bestInst = abRows.reduce((m, r) => Math.max(m, Number(r.inst_sharpe || 0)), 0);
  const bestFlat = abRows.reduce((m, r) => Math.max(m, Number(r.flat60_sharpe || 0)), 0);
  const g1bPass = poolRows.filter(r => r.gate1b_strict_pass).length;

  const kpis = [
    { label: '组合最高 OOS Sharpe', val: bestInst.toFixed(2), sub: `Flat-60 等权最高: ${bestFlat.toFixed(2)} (${abRows.length} 组消融实验)` },
    { label: 'Gate 1B 严格准入折数', val: `${g1bPass} / ${poolRows.length}`, sub: `全品种池 PosRatio≥50% & Pooled-t≥1.96` },
    { label: 'Full-60 分品种评估明细', val: Number(tc.factor_asset_scores || 0).toLocaleString(), sub: `6 大商品板块 × 60 主力品种 × 双折 OOS` },
    { label: '登记因子与预检门禁', val: `${tc.factor_defs || 0} (${tc.preflight_audits || 0} 族)`, sub: `<1s 仿射镜像对称 + 零前视 100% 通过` },
    { label: 'Rust 原生信号覆盖率', val: `${sc.consumed_kline_signals || 0} / ${sc.total_kline_signals || 0}`, sub: `💎 待开采对称蓝海信号: ${ov.unconsumed_mirror_signals_count || 0} 个` },
    { label: '证伪淘汰墓地归档', val: `${tc.factor_graveyard || 0}`, sub: `自动拦截共线克隆与负边际候选` }
  ];
  document.getElementById('kpi-grid').innerHTML = kpis.map(k => `
    <div class="kpi-card">
      <div class="kpi-label">${esc(k.label)}</div>
      <div class="kpi-val">${esc(k.val)}</div>
      <div class="kpi-sub">${esc(k.sub)}</div>
    </div>
  `).join('');

  const fo = ov.filter_options || {};
  const cm = ov.cluster_meta || {};
  fillSelect('ab-stage-filter', fo.stages, '全部阶段');
  fillSelect('pool-fold', fo.folds, '全部 Fold');
  fillSelect('pool-cluster', (fo.clusters || []).map(c => ({ value: c, label: `${c} · ${cm[c] || c}` })), '全部主簇 (A~F)');
  fillSelect('pool-tf', fo.timeframes, '全部周期');
  fillSelect('pool-stage', fo.stages, '全部阶段');
  fillSelect('as-fold', fo.folds, '全部 Fold');
  fillSelect('as-sector', fo.sectors, '全部板块');
  fillSelect('as-tf', fo.timeframes, '全部周期');
  fillSelect('as-stage', fo.stages, '全部阶段');
  populateAssetDropdown('');
  fillSelect('sig-cat', fo.signal_categories, '全部 17 个分类');
  fillSelect('gr-gate', fo.failed_gates, '全部拦截门禁');
  fillSelect('gr-stage', fo.stages, '全部阶段');

  renderAblationTab();
  renderNicheMatrix();
  renderPoolTable();
  renderSectorCards();
  queryAssetScores(0);
  renderSignalsTable();
  renderLineageTable();
  renderGraveyardTable();
  loadPresetSql(0);
}

function populateAssetDropdown(sector) {
  const assets = ((STATE.snap.overview || {}).filter_options || {}).assets || [];
  const filtered = sector ? assets.filter(a => a.sector === sector) : assets;
  fillSelect('as-asset', filtered.map(a => ({ value: a.asset, label: `${a.asset} (${a.sector})` })), '全部品种');
}
"""

_JS_SCRIPT_PART3 = """
function renderAblationTab() {
  const stg = document.getElementById('ab-stage-filter').value;
  const mode = document.getElementById('ab-mode-filter').value;
  let rows = (STATE.snap.ablation || []).slice();
  if (stg) rows = rows.filter(r => r.stage === stg);
  if (mode) rows = rows.filter(r => r.mode === mode);

  const maxSh = Math.max(2.0, ...rows.map(r => Math.max(Number(r.flat60_sharpe || 0), Number(r.inst_sharpe || 0))));
  document.getElementById('ablation-chart-box').innerHTML = rows.map(r => {
    const fw = Math.max(2, Math.round((Math.max(0, Number(r.flat60_sharpe || 0)) / maxSh) * 180));
    const iw = Math.max(2, Math.round((Math.max(0, Number(r.inst_sharpe || 0)) / maxSh) * 180));
    return `
      <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px; font-size:12px;">
        <div class="mono" style="width:270px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-weight:600;" title="${esc(r.run_id)}">${esc(r.run_id)}</div>
        <span class="badge badge-blue" style="width:86px; text-align:center;">${esc(r.stage)}</span>
        <div style="display:flex; align-items:center; gap:6px; width:250px;">
          <div style="height:10px; width:${fw}px; background:#3b82f6; border-radius:3px;"></div>
          <span class="mono">Flat: ${fmtNum(r.flat60_sharpe, 2)}</span>
        </div>
        <div style="display:flex; align-items:center; gap:6px; width:250px;">
          <div style="height:10px; width:${iw}px; background:#059669; border-radius:3px;"></div>
          <span class="mono" style="font-weight:700; color:#059669;">Inst: ${fmtNum(r.inst_sharpe, 2)}</span>
        </div>
      </div>
    `;
  }).join('');

  document.getElementById('ablation-tbody').innerHTML = rows.map(r => {
    const checked = STATE.selectedNavRuns.includes(r.run_id) ? 'checked' : '';
    return `
    <tr>
      <td style="text-align:center;"><input type="checkbox" ${checked} onchange="toggleNavRun('${esc(r.run_id)}')"></td>
      <td class="mono" style="font-weight:700;">${esc(r.run_id)}</td>
      <td><span class="badge badge-blue">${esc(r.stage)}</span></td>
      <td><span class="badge badge-gray">${esc(r.mode)}</span></td>
      <td class="num">${fmtSigned(r.flat60_ann_ret_pct, 2, '%')}</td>
      <td class="num" style="font-weight:700;">${fmtSigned(r.flat60_sharpe, 3)}</td>
      <td class="num neg">${fmtNum(r.flat60_max_dd_pct, 2)}%</td>
      <td class="num">${fmtSigned(r.flat60_calmar, 3)}</td>
      <td class="num">${fmtSigned(r.inst_ann_ret_pct, 2, '%')}</td>
      <td class="num" style="font-weight:800;">${fmtSigned(r.inst_sharpe, 3)}</td>
      <td class="num neg">${fmtNum(r.inst_max_dd_pct, 2)}%</td>
      <td class="num">${fmtSigned(r.inst_calmar, 3)}</td>
      <td class="num">${fmtNum(r.avg_factors_per_asset, 1)}</td>
      <td class="num">${esc(r.oos_days)}</td>
    </tr>
  `;}).join('');
  renderNavChart();
}

function onNavPresetChange() {
  const preset = document.getElementById('nav-preset').value;
  if (preset === 'stage_evo') {
    STATE.selectedNavRuns = ['Stage1_Base_Full60', 'Stage2_Rust_Incremental', 'Stage3_OI_Incremental', 'Stage4_CE_Incremental', 'Stage5_FDB_Incremental'];
  } else if (preset === 'univ_funnel') {
    STATE.selectedNavRuns = [
      'Univ_0_Flat60_Baseline', 'Univ_1_Naive_Top32_Liquidity',
      'Univ_4_Inst_Funnel_N28', 'Univ_5_Inst_Funnel_N30',
      'Univ_6_Inst_Funnel_N32_SweetSpot', 'Univ_7_Inst_Funnel_N35_BestSharpe'
    ];
  } else if (preset === 'arch_1x') {
    STATE.selectedNavRuns = [
      'Full60_FlatHRP_StaticExit', 'Full60_FlatHRP_FullStructExit',
      'Full60_FlatHRP_HysteresisRunner', 'Full60_SecConserved_SymDZ_HR',
      'Full60_ZeroHardcode_SOPStage3', 'Full60_ZeroHardcode_ClusterConserved_Optimal'
    ];
  } else if (preset === 'arch_3x') {
    STATE.selectedNavRuns = [
      'Full60_FlatHRP_StaticExit_3x', 'Full60_FlatHRP_HysteresisRunner_3x',
      'Full60_SecConserved_SymDZ_HR_3x', 'Full60_ZeroHardcode_SOPStage3_3x',
      'Full60_ZeroHardcode_ClusterConserved_3x'
    ];
  }
  renderAblationTab();
}

function toggleNavRun(runId) {
  document.getElementById('nav-preset').value = 'custom';
  const idx = STATE.selectedNavRuns.indexOf(runId);
  if (idx >= 0) {
    STATE.selectedNavRuns.splice(idx, 1);
  } else {
    STATE.selectedNavRuns.push(runId);
  }
  renderNavChart();
}

function renderNavChart() {
  const dates = (STATE.snap.overview || {}).ablation_dates || [];
  const allRuns = STATE.snap.ablation || [];
  const metric = (document.getElementById('nav-metric') || {}).value || 'inst';
  const unit = (document.getElementById('nav-unit') || {}).value || 'nav';
  const wrap = document.getElementById('nav-svg-wrap');
  const legBar = document.getElementById('nav-legend-bar');
  if (!wrap || !dates.length) return;

  const picked = allRuns.filter(r => STATE.selectedNavRuns.includes(r.run_id));
  const series = [];
  picked.forEach((r, idx) => {
    const color = NAV_COLORS[idx % NAV_COLORS.length];
    if (metric === 'inst' || metric === 'both') {
      const cum = r.inst_cum_pct || [];
      const dd = r.inst_dd_pct || [];
      if (cum.length === dates.length) {
        series.push({
          label: metric === 'both' ? `${r.run_id} (机构)` : r.run_id,
          color, dashed: false, sharpe: Number(r.inst_sharpe || 0),
          vals: unit === 'nav' ? cum.map(v => 1.0 + Number(v) / 100.0) : cum.map(Number),
          dds: dd.map(Number)
        });
      }
    }
    if (metric === 'flat' || metric === 'both') {
      const cum = r.flat_cum_pct || [];
      const dd = r.flat_dd_pct || [];
      if (cum.length === dates.length) {
        series.push({
          label: metric === 'both' ? `${r.run_id} (Flat等权)` : r.run_id,
          color, dashed: metric === 'both', sharpe: Number(r.flat60_sharpe || 0),
          vals: unit === 'nav' ? cum.map(v => 1.0 + Number(v) / 100.0) : cum.map(Number),
          dds: dd.map(Number)
        });
      }
    }
  });
  STATE.activeNavSeries = series;
  _drawNavSvg(dates, series, unit, wrap, legBar);
}

function _drawNavSvg(dates, series, unit, wrap, legBar) {
  if (!series.length) {
    legBar.innerHTML = '<span style="color:#64748b;">请在下方表格中至少勾选一个策略组合以绘制净值曲线</span>';
    wrap.innerHTML = '';
    return;
  }
  legBar.innerHTML = series.map(s => {
    const endV = s.vals[s.vals.length - 1];
    const minDd = Math.min(0, ...s.dds);
    const valTxt = unit === 'nav' ? endV.toFixed(4) : `${endV >= 0 ? '+' : ''}${endV.toFixed(2)}%`;
    const dashStyle = s.dashed ? 'border-bottom:2px dashed ' + s.color : 'background:' + s.color + '; height:3px';
    return `<span style="display:inline-flex; align-items:center; gap:6px; padding:3px 9px; background:#fff; border:1px solid #e2e8f0; border-radius:999px;">
      <span style="display:inline-block; width:16px; ${dashStyle};"></span>
      <b class="mono">${esc(s.label)}</b>
      <span class="mono" style="color:${s.color}; font-weight:700;">${valTxt}</span>
      <span class="mono" style="color:#64748b;">(Sharpe ${s.sharpe.toFixed(2)} | DD ${minDd.toFixed(2)}%)</span>
    </span>`;
  }).join('');

  const W = 1160, H = 355, padL = 64, padR = 22, topY0 = 16, topY1 = 230, ddY0 = 252, ddY1 = 330;
  const n = dates.length;
  const allVals = series.flatMap(s => s.vals);
  const baseVal = unit === 'nav' ? 1.0 : 0.0;
  const minV = Math.min(baseVal, ...allVals);
  const maxV = Math.max(baseVal + (unit === 'nav' ? 0.02 : 2.0), ...allVals);
  const minDd = Math.min(-1.0, ...series.flatMap(s => s.dds));
  const xOf = i => padL + (i / Math.max(1, n - 1)) * (W - padL - padR);
  const yOf = v => topY1 - ((v - minV) / Math.max(1e-8, maxV - minV)) * (topY1 - topY0);
  const yDd = d => ddY0 + (Math.abs(d) / Math.abs(minDd)) * (ddY1 - ddY0);
  _renderSvgMarkup(dates, series, unit, wrap, W, H, padL, padR, topY0, topY1, ddY0, ddY1, minV, maxV, minDd, xOf, yOf, yDd);
}

function _renderSvgMarkup(dates, series, unit, wrap, W, H, padL, padR, topY0, topY1, ddY0, ddY1, minV, maxV, minDd, xOf, yOf, yDd) {
  const n = dates.length;
  let gridHtml = '';
  for (let k = 0; k <= 4; k++) {
    const v = minV + (k / 4) * (maxV - minV);
    const y = yOf(v).toFixed(1);
    const lbl = unit === 'nav' ? v.toFixed(3) : `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`;
    gridHtml += `<line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" stroke="#e2e8f0" stroke-dasharray="3,3"/>` +
                `<text x="${padL - 6}" y="${Number(y) + 4}" text-anchor="end" font-size="10.5" fill="#64748b" font-family="monospace">${lbl}</text>`;
  }
  gridHtml += `<line x1="${padL}" y1="${ddY0}" x2="${W - padR}" y2="${ddY0}" stroke="#cbd5e1"/>` +
              `<line x1="${padL}" y1="${ddY1}" x2="${W - padR}" y2="${ddY1}" stroke="#e2e8f0" stroke-dasharray="3,3"/>` +
              `<text x="${padL - 6}" y="${ddY0 + 8}" text-anchor="end" font-size="10" fill="#64748b" font-family="monospace">0.0%</text>` +
              `<text x="${padL - 6}" y="${ddY1}" text-anchor="end" font-size="10" fill="#dc2626" font-family="monospace">${minDd.toFixed(1)}%</text>`;

  const foldIdx = dates.findIndex(d => d >= '2026-01-01');
  if (foldIdx > 0) {
    const fx = xOf(foldIdx).toFixed(1);
    gridHtml += `<line x1="${fx}" y1="${topY0}" x2="${fx}" y2="${ddY1}" stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="4,4"/>` +
                `<text x="${Number(fx) + 5}" y="${topY0 + 12}" font-size="10.5" fill="#475569" font-weight="600">Fold2_2026 OOS ➔</text>`;
  }
  [0, Math.floor(n * 0.25), Math.floor(n * 0.5), Math.floor(n * 0.75), n - 1].forEach(idx => {
    gridHtml += `<text x="${xOf(idx).toFixed(1)}" y="${H - 6}" text-anchor="middle" font-size="10.5" fill="#64748b" font-family="monospace">${dates[idx]}</text>`;
  });

  const pathsHtml = series.map(s => {
    const pts = s.vals.map((v, i) => `${xOf(i).toFixed(1)},${yOf(v).toFixed(1)}`).join(' ');
    const ddPts = s.dds.map((d, i) => `${xOf(i).toFixed(1)},${yDd(d).toFixed(1)}`).join(' ');
    const dash = s.dashed ? 'stroke-dasharray="5,4"' : '';
    return `<polyline fill="none" stroke="${s.color}" stroke-width="2.2" ${dash} points="${pts}"/>` +
           `<polyline fill="none" stroke="${s.color}" stroke-width="1.4" stroke-opacity="0.75" ${dash} points="${ddPts}"/>`;
  }).join('');

  wrap.innerHTML = `
    <svg id="nav-svg-el" viewBox="0 0 ${W} ${H}" style="width:100%; height:auto; display:block; user-select:none;"
         onmousemove="onNavHover(event)" onmouseleave="hideNavTooltip()">
      ${gridHtml}
      ${pathsHtml}
      <line id="nav-crosshair" x1="0" y1="${topY0}" x2="0" y2="${ddY1}" stroke="#0f172a" stroke-width="1" stroke-dasharray="2,2" style="display:none;"/>
    </svg>
    <div id="nav-tooltip" style="display:none; position:absolute; top:14px; left:80px; background:rgba(15,23,42,0.92); color:#f8fafc; padding:8px 12px; border-radius:6px; font-size:11.5px; pointer-events:none; box-shadow:0 4px 12px rgba(0,0,0,0.2); z-index:20;"></div>
  `;
}

function onNavHover(evt) {
  const svg = document.getElementById('nav-svg-el');
  const ch = document.getElementById('nav-crosshair');
  const tip = document.getElementById('nav-tooltip');
  const dates = (STATE.snap.overview || {}).ablation_dates || [];
  const series = STATE.activeNavSeries || [];
  if (!svg || !ch || !tip || !dates.length || !series.length) return;

  const rect = svg.getBoundingClientRect();
  const relX = ((evt.clientX - rect.left) / Math.max(1, rect.width)) * 1160;
  const padL = 64, padR = 22, W = 1160;
  const ratio = Math.max(0, Math.min(1, (relX - padL) / (W - padL - padR)));
  const idx = Math.round(ratio * (dates.length - 1));
  const xPos = (padL + (idx / Math.max(1, dates.length - 1)) * (W - padL - padR)).toFixed(1);
  ch.setAttribute('x1', xPos);
  ch.setAttribute('x2', xPos);
  ch.style.display = 'block';

  const unit = (document.getElementById('nav-unit') || {}).value || 'nav';
  const lines = series.map(s => {
    const v = s.vals[idx];
    const dd = s.dds[idx];
    const vStr = unit === 'nav' ? Number(v).toFixed(4) : `${v >= 0 ? '+' : ''}${Number(v).toFixed(2)}%`;
    return `<div style="display:flex; justify-content:space-between; gap:14px; margin-top:2px;">
      <span><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${s.color};margin-right:5px;"></span>${esc(s.label)}</span>
      <span class="mono"><b>${vStr}</b> (DD ${Number(dd).toFixed(2)}%)</span>
    </div>`;
  }).join('');
  tip.innerHTML = `<div class="mono" style="font-weight:700; border-bottom:1px solid #334155; padding-bottom:3px; margin-bottom:3px;">📅 ${esc(dates[idx])}</div>${lines}`;
  tip.style.display = 'block';
  tip.style.left = (ratio > 0.62 ? Math.max(16, evt.clientX - rect.left - 310) : Math.min(rect.width - 280, evt.clientX - rect.left + 18)) + 'px';
}

function hideNavTooltip() {
  const ch = document.getElementById('nav-crosshair');
  const tip = document.getElementById('nav-tooltip');
  if (ch) ch.style.display = 'none';
  if (tip) tip.style.display = 'none';
}

function renderNicheMatrix() {
  const ov = STATE.snap.overview || {};
  const ns = ov.niche_saturation || {};
  const mat = ns.matrix || {};
  const cm = ov.cluster_meta || {};
  const clusters = ns.clusters || ['A', 'B', 'C', 'D', 'E', 'F'];
  const tfs = ns.timeframes || ['1d', '60m', '30m', '15m', '5m'];

  document.getElementById('niche-matrix-tbody').innerHTML = clusters.map(c => {
    const cellsHtml = tfs.map(tf => {
      const cell = (mat[c] && mat[c][tf]) || { candidate_uids: 0, gate1b_pass_folds: 0, avg_mean_sharpe: 0, scarcity_status: 'SCARCE' };
      const isSel = STATE.selectedNiche.cluster === c && STATE.selectedNiche.tf === tf;
      const st = cell.scarcity_status || 'SCARCE';
      const bg = st === 'SCARCE' ? '#ecfdf5' : (st === 'SATURATED' ? '#fef2f2' : '#fffbeb');
      const badge = st === 'SCARCE' ? '🟢 蓝海' : (st === 'SATURATED' ? '🔴 饱和' : '🟡 适中');
      return `
        <td>
          <div class="niche-cell ${isSel ? 'selected' : ''}" style="background:${bg};" onclick="selectNicheCell('${c}', '${tf}')">
            <div class="mono" style="font-weight:800; font-size:13px;">${cell.gate1b_pass_folds} / ${cell.candidate_uids}</div>
            <div class="mono" style="font-size:11px; color:#475569;">Sharpe: ${fmtSigned(cell.avg_mean_sharpe, 2)}</div>
            <div style="font-size:10.5px; margin-top:2px;">${badge}</div>
          </div>
        </td>
      `;
    }).join('');
    return `<tr><td style="font-weight:700;">${esc(c)} · ${esc(cm[c] || '')}</td>${cellsHtml}</tr>`;
  }).join('');
}

function selectNicheCell(c, tf) {
  if (STATE.selectedNiche.cluster === c && STATE.selectedNiche.tf === tf) {
    resetPoolNicheFilter();
    return;
  }
  STATE.selectedNiche = { cluster: c, tf: tf };
  document.getElementById('pool-cluster').value = c;
  document.getElementById('pool-tf').value = tf;
  renderNicheMatrix();
  renderPoolTable();
}

function resetPoolNicheFilter() {
  STATE.selectedNiche = { cluster: '', tf: '' };
  document.getElementById('pool-cluster').value = '';
  document.getElementById('pool-tf').value = '';
  renderNicheMatrix();
  renderPoolTable();
}
"""

_JS_SCRIPT_PART4 = """
function renderPoolTable() {
  const fold = document.getElementById('pool-fold').value;
  const cluster = document.getElementById('pool-cluster').value;
  const tf = document.getElementById('pool-tf').value;
  const stage = document.getElementById('pool-stage').value;
  const sortCol = document.getElementById('pool-sort').value || 'mean_sharpe';
  const g1bOnly = document.getElementById('pool-gate1b').checked;
  const q = (document.getElementById('pool-q').value || '').trim().toLowerCase();

  let rows = (STATE.snap.pool_scores || []).filter(r => {
    if (fold && r.fold !== fold) return false;
    if (cluster && r.cluster !== cluster) return false;
    if (tf && r.tf !== tf) return false;
    if (stage && r.stage !== stage) return false;
    if (g1bOnly && !r.gate1b_strict_pass) return false;
    if (q) {
      const text = `${r.uid} ${r.family_id} ${r.factor_name}`.toLowerCase();
      if (!text.includes(q)) return false;
    }
    return true;
  });

  rows.sort((a, b) => Number(b[sortCol] || 0) - Number(a[sortCol] || 0));
  document.getElementById('pool-count-desc').textContent =
    `当前匹配 ${rows.length} 条池化评估记录（Gate 1B 严格准入 ${rows.filter(r => r.gate1b_strict_pass).length} 条）`;

  document.getElementById('pool-tbody').innerHTML = rows.map((r, i) => `
    <tr>
      <td class="num">${i + 1}</td>
      <td class="mono">${esc(r.fold)}</td>
      <td class="mono" style="font-weight:700;">${esc(r.uid)}</td>
      <td><span class="badge badge-purple">${esc(r.family_id)}</span></td>
      <td>${esc(r.factor_name)}</td>
      <td class="mono">${esc(r.tf)}</td>
      <td><span class="badge badge-blue">${esc(r.stage)}</span></td>
      <td class="num" style="font-weight:700;">${fmtSigned(r.mean_sharpe, 4)}</td>
      <td class="num">${fmtSigned(r.pooled_t, 2)}</td>
      <td class="num">${(Number(r.pos_asset_ratio || 0) * 100).toFixed(1)}%</td>
      <td class="num">${Number(r.total_n || 0).toLocaleString()}</td>
      <td>${r.gate1b_strict_pass ? '<span class="badge badge-green">✅ PASS</span>' : '<span class="badge badge-gray">➖ 未过</span>'}</td>
      <td><button class="btn btn-light btn-xs" onclick="drillDownFactor('${esc(r.uid)}', '${esc(r.fold)}')">🔍 下钻60品种</button></td>
    </tr>
  `).join('');
}

function drillDownFactor(uid, fold) {
  switchTab('tab-assets');
  document.getElementById('as-fold').value = fold || '';
  document.getElementById('as-sector').value = '';
  populateAssetDropdown('');
  document.getElementById('as-asset').value = '';
  document.getElementById('as-tf').value = '';
  document.getElementById('as-stage').value = '';
  document.getElementById('as-q').value = uid || '';
  renderSectorCards();
  queryAssetScores(0);
}

function renderSectorCards() {
  const sectors = (STATE.snap.overview || {}).sector_summary || [];
  const curSec = document.getElementById('as-sector').value;
  document.getElementById('sector-cards-grid').innerHTML = sectors.map(s => `
    <div class="sector-card ${curSec === s.sector ? 'active' : ''}" onclick="selectSectorCard('${esc(s.sector)}')">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <span class="mono" style="font-weight:800; font-size:13px;">${esc(s.sector)}</span>
        <span class="badge badge-blue">${s.assets} 品种</span>
      </div>
      <div style="margin-top:6px; font-size:12px; display:flex; justify-content:space-between;">
        <span>均 Sharpe: ${fmtSigned(s.mean_sharpe, 3)}</span>
        <span>达标: <b>${Number(s.local_pass_count || 0).toLocaleString()}</b></span>
      </div>
    </div>
  `).join('');
}

function selectSectorCard(sec) {
  const sel = document.getElementById('as-sector');
  sel.value = (sel.value === sec) ? '' : sec;
  onAssetSectorChange();
}

function onAssetSectorChange() {
  const sec = document.getElementById('as-sector').value;
  populateAssetDropdown(sec);
  renderSectorCards();
  queryAssetScores(0);
}
"""

_JS_SCRIPT_PART5 = """
function resetAssetFilters() {
  document.getElementById('as-fold').value = '';
  document.getElementById('as-sector').value = '';
  populateAssetDropdown('');
  document.getElementById('as-asset').value = '';
  document.getElementById('as-tf').value = '';
  document.getElementById('as-stage').value = '';
  document.getElementById('as-q').value = '';
  document.getElementById('as-canonical').checked = true;
  document.getElementById('as-localpass').checked = false;
  renderSectorCards();
  queryAssetScores(0);
}

function debounceAssetQuery() {
  clearTimeout(STATE.assetTimer);
  STATE.assetTimer = setTimeout(() => queryAssetScores(0), 220);
}

function stepAssetPage(delta) {
  const nextOff = Math.max(0, STATE.assetOffset + delta * STATE.assetLimit);
  queryAssetScores(nextOff);
}

async function queryAssetScores(offset = 0) {
  STATE.assetOffset = offset;
  const fold = document.getElementById('as-fold').value;
  const sector = document.getElementById('as-sector').value;
  const asset = document.getElementById('as-asset').value;
  const tf = document.getElementById('as-tf').value;
  const stage = document.getElementById('as-stage').value;
  const sortBy = document.getElementById('as-sort').value || 'is_net_sharpe';
  const canonical = document.getElementById('as-canonical').checked;
  const localPass = document.getElementById('as-localpass').checked;
  const q = (document.getElementById('as-q').value || '').trim();

  if (STATE.liveMode) {
    try {
      const qs = new URLSearchParams({
        fold, sector, asset, tf, stage, q,
        canonical_only: canonical ? '1' : '0',
        local_pass_only: localPass ? '1' : '0',
        sort_by: sortBy, sort_dir: 'desc',
        limit: String(STATE.assetLimit), offset: String(offset)
      });
      const resp = await fetch(`/api/asset-scores?${qs.toString()}`);
      if (resp.ok) {
        const data = await resp.json();
        renderAssetScoresResult(data);
        return;
      }
    } catch (e) {}
  }

  const allRows = ((STATE.snap.asset_scores_initial || {}).rows || []).filter(r => {
    if (canonical && !r.is_canonical) return false;
    if (localPass && !r.strict_local_pass) return false;
    if (fold && r.fold !== fold) return false;
    if (sector && r.sector !== sector) return false;
    if (asset && r.asset !== asset) return false;
    if (tf && r.tf !== tf) return false;
    if (stage && r.stage !== stage) return false;
    if (q) {
      const kw = q.toLowerCase();
      if (!(`${r.uid} ${r.asset} ${r.family_id}`.toLowerCase().includes(kw))) return false;
    }
    return true;
  });
  allRows.sort((a, b) => Number(b[sortBy] || 0) - Number(a[sortBy] || 0));
  const total = allRows.length;
  const avgSh = total ? allRows.reduce((s, r) => s + Number(r.is_net_sharpe || 0), 0) / total : 0;
  const avgWr = total ? allRows.reduce((s, r) => s + Number(r.win_rate || 0), 0) / total : 0;
  const avgPl = total ? allRows.reduce((s, r) => s + Number(r.pl_ratio || 0), 0) / total : 0;
  const posRatio = total ? allRows.filter(r => Number(r.is_net_sharpe || 0) > 0).length / total : 0;
  const passCnt = allRows.filter(r => r.strict_local_pass).length;
  renderAssetScoresResult({
    summary: { total_rows: total, avg_sharpe: avgSh, avg_win_rate: avgWr, avg_pl_ratio: avgPl, pos_sharpe_ratio: posRatio, local_pass_count: passCnt },
    limit: STATE.assetLimit,
    offset,
    rows: allRows.slice(offset, offset + STATE.assetLimit)
  });
}

function renderAssetScoresResult(data) {
  const sum = data.summary || {};
  const total = Number(sum.total_rows || 0);
  const rows = data.rows || [];
  document.getElementById('asset-summary-strip').innerHTML = `
    <span>匹配行数: <b class="mono">${total.toLocaleString()}</b></span>
    <span>均值净 Sharpe: <b class="mono">${fmtSigned(sum.avg_sharpe || 0, 3)}</b></span>
    <span>正夏普占比: <b class="mono">${(Number(sum.pos_sharpe_ratio || 0) * 100).toFixed(1)}%</b></span>
    <span>平均胜率: <b class="mono">${(Number(sum.avg_win_rate || 0) * 100).toFixed(1)}%</b></span>
    <span>平均盈亏比: <b class="mono">${fmtNum(sum.avg_pl_ratio || 0, 2)}</b></span>
    <span>单品种严格达标数: <b class="mono">${Number(sum.local_pass_count || 0).toLocaleString()}</b></span>
  `;
  document.getElementById('asset-tbody').innerHTML = rows.map(r => `
    <tr>
      <td class="mono">${esc(r.fold)}</td>
      <td><span class="badge badge-gray">${esc(r.sector)}</span></td>
      <td class="mono" style="font-weight:800;">${esc(r.asset)}</td>
      <td class="mono" style="font-weight:700;">${esc(r.uid)}</td>
      <td><span class="badge badge-purple">${esc(r.family_id)}</span></td>
      <td class="mono">${esc(r.tf)}</td>
      <td><span class="badge badge-blue">${esc(r.stage)}</span></td>
      <td class="num">${Number(r.n || 0).toLocaleString()}</td>
      <td class="num">${(Number(r.win_rate || 0) * 100).toFixed(1)}%</td>
      <td class="num">${fmtNum(r.pl_ratio, 2)}</td>
      <td class="num">${fmtSigned((Number(r.safety_margin || 0) * 100), 1, '%')}</td>
      <td class="num">${fmtSigned(r.mean_net_pct, 3, '%')}</td>
      <td class="num">${fmtSigned(r.t_stat, 2)}</td>
      <td class="num" style="font-weight:700;">${fmtSigned(r.is_net_sharpe, 3)}</td>
      <td class="num">${fmtNum(r.ann_turnover, 1)}</td>
      <td>${r.strict_local_pass ? '<span class="badge badge-green">✅ 达标</span>' : '<span class="badge badge-gray">-</span>'}</td>
    </tr>
  `).join('');
  const startIdx = total === 0 ? 0 : STATE.assetOffset + 1;
  const endIdx = Math.min(total, STATE.assetOffset + rows.length);
  document.getElementById('asset-page-info').textContent = `显示第 ${startIdx} ~ ${endIdx} 条 / 共 ${total.toLocaleString()} 条`;
  document.getElementById('as-prev-btn').disabled = STATE.assetOffset <= 0;
  document.getElementById('as-next-btn').disabled = endIdx >= total;
}
"""

_JS_SCRIPT_PART6 = """
function renderSignalsTable() {
  const cat = document.getElementById('sig-cat').value;
  const mode = document.getElementById('sig-mode').value;
  const q = (document.getElementById('sig-q').value || '').trim().toLowerCase();
  const sigs = ((STATE.snap.signals_lineage || {}).signals || []).filter(s => {
    if (cat && s.category !== cat) return false;
    if (mode === 'blue_ocean' && (!s.exact_mirror || Number(s.consumed_count || 0) > 0)) return false;
    if (mode === 'consumed' && Number(s.consumed_count || 0) <= 0) return false;
    if (mode === 'mirror_all' && !s.exact_mirror) return false;
    if (q) {
      const txt = `${s.signal_name} ${s.state_key || ''} ${s.params_template || ''} ${s.consumed_by_families || ''}`.toLowerCase();
      if (!txt.includes(q)) return false;
    }
    return true;
  });
  document.getElementById('sig-count-desc').textContent = `当前筛选命中 ${sigs.length} 个 Rust 原生信号`;
  document.getElementById('sig-tbody').innerHTML = sigs.map(s => {
    let fams = [];
    try { fams = JSON.parse(s.consumed_by_families || '[]'); } catch (e) {}
    const famBadges = fams.length
      ? fams.map(f => `<span class="badge badge-purple" style="margin-right:3px;">${esc(f)}</span>`).join('')
      : '<span style="color:#94a3b8;">未消费 (待开采)</span>';
    return `
      <tr>
        <td class="mono" style="font-weight:700;">${esc(s.signal_name)}</td>
        <td><span class="badge badge-blue">${esc(s.category)}</span></td>
        <td class="mono">${s.state_key ? `<span class="badge badge-amber">${esc(s.state_key)}</span>` : '-'}</td>
        <td class="mono" style="font-size:11.5px; color:#475569;">${esc(s.params_template)}</td>
        <td>${s.exact_mirror ? '<span class="badge badge-green">🪞 对称</span>' : '<span class="badge badge-gray">非对称</span>'}</td>
        <td class="num">${(Number(s.active_rate || 0) * 100).toFixed(1)}%</td>
        <td class="num" style="font-weight:700;">${Number(s.consumed_count || 0)}</td>
        <td>${famBadges}</td>
      </tr>
    `;
  }).join('');
}

function renderLineageTable() {
  const q = (document.getElementById('lin-q').value || '').trim().toLowerCase();
  const rows = ((STATE.snap.signals_lineage || {}).families_lineage || []).filter(r => {
    if (!q) return true;
    return JSON.stringify(r).toLowerCase().includes(q);
  });
  document.getElementById('lin-tbody').innerHTML = rows.map(r => {
    const pf = r.preflight || {};
    const passBadge = pf.passed !== false ? '<span class="badge badge-green">✅ PASS</span>' : '<span class="badge badge-red">❌ FAIL</span>';
    const tags = [
      ...(r.states || []).map(x => `<span class="badge badge-blue" style="margin:1px;">state:${esc(x)}</span>`),
      ...(r.strengths || []).map(x => `<span class="badge badge-purple" style="margin:1px;">strength:${esc(x)}</span>`),
      ...(r.oi_features || []).map(x => `<span class="badge badge-amber" style="margin:1px;">oi:${esc(x)}</span>`),
      ...(r.cs_features || []).map(x => `<span class="badge badge-green" style="margin:1px;">cs:${esc(x)}</span>`),
      ...(r.htf_features || []).map(x => `<span class="badge badge-gray" style="margin:1px;">htf:${esc(x)}</span>`)
    ].join(' ');
    return `
      <tr>
        <td class="mono" style="font-weight:800;">${esc(r.family_id)}</td>
        <td>${esc(r.name)}</td>
        <td><span class="badge badge-blue">${esc(r.stage)}</span></td>
        <td class="mono">${esc((r.eligible_tfs || []).join(','))}</td>
        <td>${passBadge}</td>
        <td class="num">${pf.fire_diff || 0} / ${fmtNum(pf.q_max_err || 0, 4)}</td>
        <td class="num">${pf.long_triggers || 0} / ${pf.short_triggers || 0}</td>
        <td class="num">${fmtNum(pf.elapsed_ms || 0, 1)}</td>
        <td>${tags || '-'}</td>
      </tr>
    `;
  }).join('');
}

function renderGraveyardTable() {
  const gData = STATE.snap.graveyard || {};
  const gc = gData.gate_counts || {};
  const curGate = document.getElementById('gr-gate').value;
  const curStage = document.getElementById('gr-stage').value;
  const q = (document.getElementById('gr-q').value || '').trim().toLowerCase();

  document.getElementById('grave-gate-badges').innerHTML = Object.entries(gc).map(([g, cnt]) => `
    <button class="btn ${curGate === g ? 'btn-primary' : 'btn-light'} btn-xs" onclick="selectGraveGate('${esc(g)}')">
      ${esc(g)}: <b>${cnt}</b>
    </button>
  `).join('');

  const rows = (gData.rows || []).filter(r => {
    if (curGate && r.failed_gate !== curGate) return false;
    if (curStage && r.stage !== curStage) return false;
    if (q) {
      const txt = `${r.uid} ${r.family_id} ${r.reason || ''} ${r.formula_desc || ''}`.toLowerCase();
      if (!txt.includes(q)) return false;
    }
    return true;
  });

  document.getElementById('grave-tbody').innerHTML = rows.map(r => `
    <tr>
      <td class="mono" style="font-weight:700;">${esc(r.uid)}</td>
      <td><span class="badge badge-purple">${esc(r.family_id)}</span></td>
      <td class="mono">${esc(r.tf)}</td>
      <td><span class="badge badge-blue">${esc(r.stage)}</span></td>
      <td><span class="badge badge-red">${esc(r.failed_gate)}</span></td>
      <td class="num">${fmtSigned(r.mean_is_sharpe, 4)}</td>
      <td class="num">${fmtNum(r.mean_ann_turnover, 2)}</td>
      <td>${esc(r.reason)}</td>
      <td class="mono" style="font-size:11.5px; color:#64748b;">${esc((r.buried_at || '').slice(0, 19))}</td>
    </tr>
  `).join('');
}
"""

_JS_SCRIPT_PART7 = """
function selectGraveGate(g) {
  const sel = document.getElementById('gr-gate');
  sel.value = (sel.value === g) ? '' : g;
  renderGraveyardTable();
}
function resetGraveFilter() {
  document.getElementById('gr-gate').value = '';
  document.getElementById('gr-stage').value = '';
  document.getElementById('gr-q').value = '';
  renderGraveyardTable();
}

function loadPresetSql(idx) {
  document.getElementById('sql-input').value = PRESET_SQLS[idx] || PRESET_SQLS[0];
  if (STATE.liveMode) {
    runCustomSql();
  }
}

async function runCustomSql() {
  const sql = document.getElementById('sql-input').value.trim();
  const statusEl = document.getElementById('sql-status-text');
  const wrap = document.getElementById('sql-result-wrap');
  if (!sql) return;

  if (!STATE.liveMode) {
    statusEl.innerHTML = '<span class="neg">当前为离线静态 HTML 快照模式；请启动本地服务 (`python -m alpha_lab.cli dashboard`) 以执行实时 PostgreSQL SQL 查询。</span>';
    return;
  }
  statusEl.textContent = '⏳ 正在向 PostgreSQL 发送查询...';
  try {
    const resp = await fetch('/api/sql', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sql, max_rows: 500 })
    });
    const data = await resp.json();
    if (!resp.ok || data.error) {
      statusEl.innerHTML = `<span class="neg">❌ SQL 错误: ${esc(data.error || '查询失败')}</span>`;
      return;
    }
    STATE.lastSqlResult = { columns: data.columns || [], rows: data.rows || [] };
    statusEl.innerHTML = `<span class="pos">✅ 查询成功：返回 ${data.total_rows} 行（耗时 ${data.elapsed_ms} ms）</span>`;
    wrap.style.display = 'block';
    document.getElementById('sql-thead').innerHTML =
      '<tr>' + (data.columns || []).map(c => `<th>${esc(c)}</th>`).join('') + '</tr>';
    document.getElementById('sql-tbody').innerHTML = (data.rows || []).map(r =>
      '<tr>' + (data.columns || []).map(c => `<td class="mono">${esc(r[c])}</td>`).join('') + '</tr>'
    ).join('');
  } catch (err) {
    statusEl.innerHTML = `<span class="neg">❌ 请求异常: ${esc(err.message)}</span>`;
  }
}

function exportSqlCsv() {
  const { columns, rows } = STATE.lastSqlResult || {};
  if (!columns || !columns.length || !rows || !rows.length) return;
  const csvLines = [
    columns.join(','),
    ...rows.map(r => columns.map(c => `"${String(r[c] ?? '').replace(/"/g, '""')}"`).join(','))
  ];
  const blob = new Blob([csvLines.join('\\n')], { type: 'text/csv;charset=utf-8;' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'alpha_lab_query_result.csv';
  a.click();
}

async function refreshAllData() {
  if (!STATE.liveMode) { window.location.reload(); return; }
  const [ov, ab, pool, sig, gr] = await Promise.all([
    fetch('/api/overview').then(r => r.json()),
    fetch('/api/ablation').then(r => r.json()),
    fetch('/api/pool-scores').then(r => r.json()),
    fetch('/api/signals').then(r => r.json()),
    fetch('/api/graveyard').then(r => r.json())
  ]);
  STATE.snap.overview = ov;
  STATE.snap.ablation = ab.rows || [];
  STATE.snap.pool_scores = pool.rows || [];
  STATE.snap.signals_lineage = sig;
  STATE.snap.graveyard = gr;
  initDashboard();
}

function handleExportHtml(ev) {
  if (STATE.liveMode) return true;
  ev.preventDefault();
  const blob = new Blob([document.documentElement.outerHTML], { type: 'text/html;charset=utf-8' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'CZSC_AlphaLab_Dashboard.html';
  a.click();
  return false;
}

window.addEventListener('DOMContentLoaded', initDashboard);
"""


def render_dashboard_html(snapshot: dict[str, Any], *, live_mode: bool = True) -> str:
    """将实验台账快照与响应式前端组件渲染为完整的单页 HTML 字符串。"""
    snap_json = json.dumps(snapshot, ensure_ascii=False, default=str).replace("</", "<\\/")
    live_flag = "true" if live_mode else "false"
    css_all = _CSS_STYLES + _CSS_STYLES_PART2
    body_all = _HTML_BODY_PART1 + _HTML_BODY_PART2 + _HTML_BODY_PART3
    js_all = (
        _JS_SCRIPT_PART1
        + _JS_SCRIPT_PART2
        + _JS_SCRIPT_PART3
        + _JS_SCRIPT_PART4
        + _JS_SCRIPT_PART5
        + _JS_SCRIPT_PART6
        + _JS_SCRIPT_PART7
    )
    return (
        '<!DOCTYPE html>\n<html lang="zh-CN">\n<head>\n'
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        "<title>CZSC AlphaLab · PostgreSQL 投研台账与回测查询工作台</title>\n"
        f"<style>{css_all}</style>\n"
        "</head>\n<body>\n"
        f"{body_all}\n"
        f"<script>\nwindow.__LIVE_MODE__ = {live_flag};\nwindow.__INITIAL_SNAPSHOT__ = {snap_json};\n{js_all}\n</script>\n"
        "</body>\n</html>\n"
    )
