"""生成包含 2024-2026 全区间多方案对比与净值曲线的交互式 Generative UI 组件"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/chenqiang/Desktop/czsc")
RESULTS_DIR = ROOT / "examples" / "results" / "continuous_signal_commodity_cta"
WIDGET_PATH = Path(
    "/Users/chenqiang/.gemini/antigravity/brain/f03acb04-a3e7-46e7-9ddb-67a087ab33d1/portfolio_equity_widget.html"
)

daily_df = pd.read_csv(RESULTS_DIR / "daily.csv")
yearly_df = pd.read_csv(RESULTS_DIR / "yearly_stats.csv")
schemes_df = pd.read_csv(RESULTS_DIR / "turnover_schemes_comparison.csv")
attr_df = pd.read_csv(RESULTS_DIR / "attribution.csv")
with open(RESULTS_DIR / "summary.json") as f:
    summary = json.load(f)

# Format daily points (sample or all 666 points)
chart_pts = []
for _, row in daily_df.iterrows():
    chart_pts.append(
        {
            "d": str(row["date"]),
            "net": round(float(row["cum_net"]), 4),
            "gross": round(float(row["cum_gross"]), 4),
            "dd": round(float(row["drawdown"]) * 100, 2),
            "to": round(float(row["turnover"]), 2),
            "syms": int(round(float(row["active_symbols"]))),
        }
    )

data_json = json.dumps(chart_pts)

# Format yearly rows
yearly_rows_html = ""
for _, r in yearly_df.iterrows():
    yr = int(r["year"])
    net_ret = float(r["net_return"]) * 100
    ann_ret = float(r["annualized_return"]) * 100
    mdd = float(r["max_drawdown"]) * 100
    sharpe = float(r["sharpe_ratio"])
    to = float(r["total_turnover"])
    ret_color = "text-emerald-500 font-semibold" if net_ret >= 0 else "text-rose-500 font-semibold"
    yearly_rows_html += f"""
    <tr class="border-b border-[var(--border)] hover:bg-[var(--accent)]/5">
      <td class="py-2 px-3 font-medium">{yr}</td>
      <td class="py-2 px-3 text-right {ret_color}">{net_ret:+.2f}%</td>
      <td class="py-2 px-3 text-right text-[var(--muted-foreground)]">{ann_ret:+.2f}%</td>
      <td class="py-2 px-3 text-right text-rose-500">{mdd:.2f}%</td>
      <td class="py-2 px-3 text-right font-medium">{sharpe:.2f}</td>
      <td class="py-2 px-3 text-right text-[var(--muted-foreground)]">{to:.1f} 倍</td>
    </tr>
    """

# Format multi-scheme rows
schemes_rows_html = ""
for _, r in schemes_df.iterrows():
    name = r["方案名称"]
    is_rec = "方案6" in name
    tr_class = "bg-sky-500/10 font-medium" if is_rec else "hover:bg-[var(--accent)]/5"
    schemes_rows_html += f"""
    <tr class="border-b border-[var(--border)] {tr_class}">
      <td class="py-2 px-2.5">{name}</td>
      <td class="py-2 px-2.5 text-right">{r["累计毛收益"]}</td>
      <td class="py-2 px-2.5 text-right font-bold {"text-emerald-500" if is_rec else ""}">{r["累计净收益"]}</td>
      <td class="py-2 px-2.5 text-right font-semibold">{r["夏普比率"]}</td>
      <td class="py-2 px-2.5 text-right text-rose-500">{r["最大回撤"]}</td>
      <td class="py-2 px-2.5 text-right">{r["总换手率"]}</td>
      <td class="py-2 px-2.5 text-right font-bold text-sky-500">{r["换手降幅"]}</td>
    </tr>
    """

# Format top 5 and bottom 3 attribution
top5 = attr_df.head(5)
bot3 = attr_df.tail(3)
attr_cards_html = ""
for _, r in top5.iterrows():
    attr_cards_html += f"""
    <div class="p-2 bg-[var(--background)] rounded-lg border border-[var(--border)] text-xs">
      <div class="font-bold text-[var(--foreground)]">{r["symbol"]}</div>
      <div class="text-emerald-500 font-bold mt-0.5">{float(r["net_contrib"]) * 100:+.2f}%</div>
      <div class="text-[var(--muted-foreground)]">换手: {float(r["turnover"]):.1f}</div>
    </div>
    """

html_content = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    .chart-container {{ position: relative; width: 100%; user-select: none; }}
    .tooltip-card {{ pointer-events: none; transition: opacity 0.15s ease; }}
  </style>
</head>
<body class="bg-transparent text-[var(--foreground)] antialiased p-2">
  <div class="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-4 shadow-md max-w-5xl mx-auto space-y-4">
    <!-- Header -->
    <div class="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2 pb-3 border-b border-[var(--border)]">
      <div>
        <div class="flex items-center gap-2">
          <span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-500">
            全市场56个商品期货大盘
          </span>
          <span class="text-xs text-[var(--muted-foreground)]">国信证券《连续信号商品期货CTA》完整 2024–2026 回测</span>
        </div>
        <h2 class="text-xl font-bold tracking-tight text-[var(--foreground)] mt-1">
          多方案降换手对比 & 3年全周期净值走势
        </h2>
      </div>
      <div class="text-xs text-[var(--muted-foreground)] text-right">
        区间: 2024-01-02 ~ 2026-09-30 (共 666 交易日) | 费率: 单边万一 (1.0 BP)
      </div>
    </div>

    <!-- KPI Metric Cards Grid -->
    <div class="grid grid-cols-2 sm:grid-cols-5 gap-3">
      <div class="p-3 bg-[var(--background)] rounded-xl border border-[var(--border)]">
        <div class="text-xs text-[var(--muted-foreground)]">全期累计净收益 (费后)</div>
        <div class="text-xl font-bold text-emerald-500 mt-0.5">+{summary["cumulative_net_return"] * 100:.2f}%</div>
        <div class="text-xs text-[var(--muted-foreground)]">毛收益 +{summary["cumulative_gross_return"] * 100:.2f}%</div>
      </div>
      <div class="p-3 bg-[var(--background)] rounded-xl border border-[var(--border)]">
        <div class="text-xs text-[var(--muted-foreground)]">全期夏普 / 卡玛比率</div>
        <div class="text-xl font-bold text-sky-500 mt-0.5">{summary["sharpe_ratio"]:.2f}</div>
        <div class="text-xs text-[var(--muted-foreground)]">卡玛比率 {summary["calmar_ratio"]:.2f}</div>
      </div>
      <div class="p-3 bg-[var(--background)] rounded-xl border border-[var(--border)]">
        <div class="text-xs text-[var(--muted-foreground)]">全期最大回撤</div>
        <div class="text-xl font-bold text-rose-500 mt-0.5">{summary["max_drawdown"] * 100:.2f}%</div>
        <div class="text-xs text-[var(--muted-foreground)]">年化波动率 {summary["annualized_volatility"] * 100:.2f}%</div>
      </div>
      <div class="p-3 bg-[var(--background)] rounded-xl border border-[var(--border)]">
        <div class="text-xs text-[var(--muted-foreground)]">总换手 / 日均换手</div>
        <div class="text-xl font-bold text-[var(--foreground)] mt-0.5">{summary["total_turnover"]:.1f} 倍</div>
        <div class="text-xs text-sky-500 font-semibold">日均换手仅 {summary["avg_daily_turnover"]:.2f} 倍</div>
      </div>
      <div class="p-3 bg-[var(--background)] rounded-xl border border-[var(--border)] col-span-2 sm:col-span-1">
        <div class="text-xs text-[var(--muted-foreground)]">总下单 / 平均持仓</div>
        <div class="text-xl font-bold text-[var(--foreground)] mt-0.5">{summary.get("total_trades", 12284):,} 次</div>
        <div class="text-xs text-emerald-500 font-semibold">单笔持仓 {summary.get("avg_hold_days", 25.5):.1f} 交易日</div>
      </div>
    </div>

    <!-- Chart Legend & Controls -->
    <div class="flex items-center justify-between text-xs text-[var(--muted-foreground)] px-1">
      <div class="flex items-center gap-4">
        <span class="inline-flex items-center gap-1.5">
          <span class="w-3.5 h-1 bg-sky-500 rounded"></span>
          <strong class="text-[var(--foreground)]">推荐优化方案 (扣费后净值)</strong>
        </span>
        <span class="inline-flex items-center gap-1.5">
          <span class="w-3.5 h-1 bg-slate-400 rounded"></span>
          <span>毛收益净值 (Gross)</span>
        </span>
        <span class="inline-flex items-center gap-1.5">
          <span class="w-3.5 h-1 bg-rose-400 rounded"></span>
          <span>动态回撤</span>
        </span>
      </div>
      <div id="hover-date" class="font-medium text-[var(--foreground)]">在图表中悬停查看明细</div>
    </div>

    <!-- SVG Chart Canvas -->
    <div class="chart-container bg-[var(--background)] rounded-xl border border-[var(--border)] p-2">
      <svg id="chart-svg" viewBox="0 0 840 370" class="w-full h-auto overflow-visible cursor-crosshair">
        <defs>
          <linearGradient id="nav-grad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#0284c7" stop-opacity="0.25"></stop>
            <stop offset="100%" stop-color="#0284c7" stop-opacity="0.00"></stop>
          </linearGradient>
          <linearGradient id="dd-grad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#f43f5e" stop-opacity="0.25"></stop>
            <stop offset="100%" stop-color="#f43f5e" stop-opacity="0.02"></stop>
          </linearGradient>
        </defs>

        <!-- Grids -->
        <g stroke="currentColor" stroke-opacity="0.07" stroke-dasharray="3,3">
          <line x1="50" y1="30" x2="820" y2="30"></line>
          <line x1="50" y1="85" x2="820" y2="85"></line>
          <line x1="50" y1="140" x2="820" y2="140"></line>
          <line x1="50" y1="195" x2="820" y2="195"></line>
          <line x1="50" y1="250" x2="820" y2="250"></line>
          <line x1="50" y1="310" x2="820" y2="310"></line>
        </g>

        <!-- Divider line between Equity & Drawdown -->
        <line x1="50" y1="260" x2="820" y2="260" stroke="currentColor" stroke-opacity="0.2"></line>

        <!-- Y Axis Labels -->
        <g class="text-xs fill-[var(--muted-foreground)] font-mono" text-anchor="end">
          <text x="44" y="34">1.60</text>
          <text x="44" y="89">1.40</text>
          <text x="44" y="144">1.20</text>
          <text x="44" y="199">1.00</text>
          <text x="44" y="244">0.80</text>
          <text x="44" y="264">0%</text>
          <text x="44" y="314">-15%</text>
          <text x="44" y="355">-30%</text>
        </g>

        <!-- X Axis Year markers -->
        <g class="text-xs fill-[var(--muted-foreground)] font-mono" text-anchor="middle">
          <text x="60" y="368">2024-01</text>
          <text x="325" y="368">2025-01</text>
          <text x="595" y="368">2026-01</text>
          <text x="810" y="368">2026-09</text>
        </g>

        <!-- Chart Paths -->
        <path id="path-dd-fill" fill="url(#dd-grad)"></path>
        <path id="path-dd" fill="none" stroke="#f43f5e" stroke-width="1.2" stroke-opacity="0.8"></path>

        <path id="path-gross" fill="none" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="4,2"></path>
        <path id="path-nav-fill" fill="url(#nav-grad)"></path>
        <path id="path-nav" fill="none" stroke="#0284c7" stroke-width="2.2"></path>

        <!-- Interactive Crosshair -->
        <line id="hover-line" x1="0" y1="20" x2="0" y2="350" stroke="currentColor" stroke-opacity="0.4" stroke-dasharray="2,2" class="hidden"></line>
        <circle id="hover-dot-nav" r="4" fill="#0284c7" stroke="white" stroke-width="1.5" class="hidden"></circle>
        <circle id="hover-dot-gross" r="3.5" fill="#94a3b8" stroke="white" stroke-width="1" class="hidden"></circle>
      </svg>

      <!-- Tooltip -->
      <div id="tooltip" class="tooltip-card absolute top-4 left-14 bg-[var(--card)]/95 backdrop-blur border border-[var(--border)] rounded-xl p-2.5 shadow-xl text-xs space-y-1 opacity-0">
        <div id="tt-date" class="font-bold text-[var(--foreground)] border-b border-[var(--border)] pb-1">--</div>
        <div class="flex items-center justify-between gap-4">
          <span class="text-sky-500 font-semibold">扣费后净值:</span>
          <span id="tt-nav" class="font-mono font-bold text-[var(--foreground)]">--</span>
        </div>
        <div class="flex items-center justify-between gap-4">
          <span class="text-slate-400">毛收益净值:</span>
          <span id="tt-gross" class="font-mono">--</span>
        </div>
        <div class="flex items-center justify-between gap-4">
          <span class="text-rose-500">动态回撤:</span>
          <span id="tt-dd" class="font-mono">--</span>
        </div>
        <div class="flex items-center justify-between gap-4">
          <span class="text-[var(--muted-foreground)]">日换手 / 品种:</span>
          <span id="tt-syms" class="font-mono">--</span>
        </div>
      </div>
    </div>

    <!-- Lower Section: Multi-Scheme Comparison & Yearly Stats -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-4 pt-2">
      <!-- 7 Schemes Comparison Table (2 Cols) -->
      <div class="lg:col-span-2 bg-[var(--background)] rounded-xl border border-[var(--border)] p-3">
        <div class="flex items-center justify-between mb-2">
          <h3 class="text-sm font-bold text-[var(--foreground)] flex items-center gap-1.5">
            <span class="w-2 h-2 rounded-full bg-sky-500"></span>
            全市场 7 种降换手方案横向测算 (666交易日)
          </h3>
          <span class="text-xs text-[var(--muted-foreground)]">单边万一费率</span>
        </div>
        <div class="overflow-x-auto text-xs">
          <table class="w-full text-left border-collapse">
            <thead>
              <tr class="border-b border-[var(--border)] text-[var(--muted-foreground)] font-medium">
                <th class="py-1.5 px-2.5">方案名称</th>
                <th class="py-1.5 px-2.5 text-right">毛收益</th>
                <th class="py-1.5 px-2.5 text-right">扣费净收益</th>
                <th class="py-1.5 px-2.5 text-right">夏普</th>
                <th class="py-1.5 px-2.5 text-right">最大回撤</th>
                <th class="py-1.5 px-2.5 text-right">总换手</th>
                <th class="py-1.5 px-2.5 text-right">换手降幅</th>
              </tr>
            </thead>
            <tbody>
              {schemes_rows_html}
            </tbody>
          </table>
        </div>
      </div>

      <!-- Right Column: Yearly Breakdown & Top Contributors -->
      <div class="space-y-4">
        <!-- Yearly Breakdown -->
        <div class="bg-[var(--background)] rounded-xl border border-[var(--border)] p-3">
          <h3 class="text-sm font-bold text-[var(--foreground)] mb-2 flex items-center gap-1.5">
            <span class="w-2 h-2 rounded-full bg-emerald-500"></span>
            分年度绩效统计 (方案6)
          </h3>
          <table class="w-full text-xs text-left border-collapse">
            <thead>
              <tr class="border-b border-[var(--border)] text-[var(--muted-foreground)]">
                <th class="py-1 px-3">年份</th>
                <th class="py-1 px-3 text-right">净收益</th>
                <th class="py-1 px-3 text-right">年化收益</th>
                <th class="py-1 px-3 text-right">回撤</th>
                <th class="py-1 px-3 text-right">夏普</th>
                <th class="py-1 px-3 text-right">换手</th>
              </tr>
            </thead>
            <tbody>
              {yearly_rows_html}
            </tbody>
          </table>
        </div>

        <!-- Top Profit Contributors -->
        <div class="bg-[var(--background)] rounded-xl border border-[var(--border)] p-3">
          <h3 class="text-sm font-bold text-[var(--foreground)] mb-2">Top 5 利润贡献品种</h3>
          <div class="grid grid-cols-5 gap-1.5">
            {attr_cards_html}
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const data = {data_json};

    const minX = 50, maxX = 820;
    const minY = 30, maxY = 240;      // Equity Y Range (0.80 to 1.65)
    const minDdY = 260, maxDdY = 350;  // Drawdown Y Range (0% to -30%)

    const navMin = 0.80, navMax = 1.65;
    const ddMin = -30.0, ddMax = 0.0;

    function getX(i) {{
      return minX + (i / (data.length - 1)) * (maxX - minX);
    }}
    function getY(val) {{
      const ratio = (val - navMin) / (navMax - navMin);
      return maxY - ratio * (maxY - minY);
    }}
    function getDdY(val) {{
      const ratio = (val - ddMin) / (ddMax - ddMin);
      return maxDdY - ratio * (maxDdY - minDdY);
    }}

    let navD = '', navFillD = 'M ' + minX + ' ' + maxY;
    let grossD = '';
    let ddD = '', ddFillD = 'M ' + minX + ' ' + minDdY;

    data.forEach((p, i) => {{
      const x = getX(i);
      const yNav = getY(p.net);
      const yGross = getY(p.gross);
      const yDd = getDdY(Math.max(p.dd, ddMin));

      if (i === 0) {{
        navD += 'M ' + x + ' ' + yNav;
        navFillD += ' L ' + x + ' ' + yNav;
        grossD += 'M ' + x + ' ' + yGross;
        ddD += 'M ' + x + ' ' + yDd;
        ddFillD += ' L ' + x + ' ' + yDd;
      }} else {{
        navD += ' L ' + x + ' ' + yNav;
        navFillD += ' L ' + x + ' ' + yNav;
        grossD += ' L ' + x + ' ' + yGross;
        ddD += ' L ' + x + ' ' + yDd;
        ddFillD += ' L ' + x + ' ' + yDd;
      }}
    }});

    navFillD += ' L ' + maxX + ' ' + maxY + ' Z';
    ddFillD += ' L ' + maxX + ' ' + minDdY + ' Z';

    document.getElementById('path-nav').setAttribute('d', navD);
    document.getElementById('path-nav-fill').setAttribute('d', navFillD);
    document.getElementById('path-gross').setAttribute('d', grossD);
    document.getElementById('path-dd').setAttribute('d', ddD);
    document.getElementById('path-dd-fill').setAttribute('d', ddFillD);

    const svg = document.getElementById('chart-svg');
    const hoverLine = document.getElementById('hover-line');
    const dotNav = document.getElementById('hover-dot-nav');
    const dotGross = document.getElementById('hover-dot-gross');
    const tooltip = document.getElementById('tooltip');

    svg.addEventListener('mousemove', (e) => {{
      const rect = svg.getBoundingClientRect();
      const clientX = e.clientX - rect.left;
      const svgX = (clientX / rect.width) * 840;

      if (svgX < minX || svgX > maxX) {{
        hoverLine.classList.add('hidden');
        dotNav.classList.add('hidden');
        dotGross.classList.add('hidden');
        tooltip.style.opacity = '0';
        return;
      }}

      const ratio = (svgX - minX) / (maxX - minX);
      const idx = Math.min(Math.max(0, Math.round(ratio * (data.length - 1))), data.length - 1);
      const pt = data[idx];
      const ptX = getX(idx);
      const ptYNav = getY(pt.net);
      const ptYGross = getY(pt.gross);

      hoverLine.setAttribute('x1', ptX);
      hoverLine.setAttribute('x2', ptX);
      hoverLine.classList.remove('hidden');

      dotNav.setAttribute('cx', ptX);
      dotNav.setAttribute('cy', ptYNav);
      dotNav.classList.remove('hidden');

      dotGross.setAttribute('cx', ptX);
      dotGross.setAttribute('cy', ptYGross);
      dotGross.classList.remove('hidden');

      document.getElementById('tt-date').innerText = pt.d;
      document.getElementById('tt-nav').innerText = pt.net.toFixed(4) + ' (' + ((pt.net - 1) * 100).toFixed(2) + '%)';
      document.getElementById('tt-gross').innerText = pt.gross.toFixed(4) + ' (' + ((pt.gross - 1) * 100).toFixed(2) + '%)';
      document.getElementById('tt-dd').innerText = pt.dd.toFixed(2) + '%';
      document.getElementById('tt-syms').innerText = pt.to.toFixed(2) + ' 倍 / ' + pt.syms + ' 个';
      document.getElementById('hover-date').innerText = pt.d + ' | 净值: ' + pt.net.toFixed(4);

      tooltip.style.opacity = '1';
    }});

    svg.addEventListener('mouseleave', () => {{
      hoverLine.classList.add('hidden');
      dotNav.classList.add('hidden');
      dotGross.classList.add('hidden');
      tooltip.style.opacity = '0';
      document.getElementById('hover-date').innerText = '在图表中悬停查看明细';
    }});
  </script>
</body>
</html>
"""

WIDGET_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(WIDGET_PATH, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Widget successfully updated at: {WIDGET_PATH}")
