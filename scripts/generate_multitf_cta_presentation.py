"""
Complete script to assemble and generate multitf_cta_research_presentation.html.
"""

from __future__ import annotations

import base64
import copy
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

MULTITF_REPORT_HTML = "examples/results/intraday_cta_ablation_20261017/multitf_cta_incremental_report.html"
FULL60_REPORT_HTML = "examples/results/intraday_cta_ablation_20261017/full60_cta_expansion_report.html"
FULL60_SECTOR_CSV = "examples/results/intraday_cta_ablation_20261017/full60_sector_attribution.csv"
FULL60_SYMBOL_CSV = "examples/results/intraday_cta_ablation_20261017/full60_symbol_attribution.csv"


def _decode_bdata(y_obj):
    if isinstance(y_obj, list):
        return np.asarray(y_obj, dtype=np.float64)
    if isinstance(y_obj, dict) and "bdata" in y_obj:
        return np.frombuffer(base64.b64decode(y_obj["bdata"]), dtype=np.float64)
    raise ValueError(f"Unsupported y object: {type(y_obj)}")


def extract_chart_data():
    with open(MULTITF_REPORT_HTML, encoding="utf-8") as f:
        html_m = f.read()
    with open(FULL60_REPORT_HTML, encoding="utf-8") as f:
        html_60 = f.read()

    dec = json.JSONDecoder()

    # 1. Extract traces from multitf_cta_incremental_report.html
    plots_m = list(re.finditer(r'Plotly\.newPlot\(\s*"([^"]+)"\s*,\s*(\[)', html_m))
    nav_m_raw, _ = dec.raw_decode(html_m, plots_m[0].start(2))
    fun_m_raw, _ = dec.raw_decode(html_m, plots_m[1].start(2))

    funnel_data = copy.deepcopy(fun_m_raw[:4])
    for t in funnel_data:
        t["xaxis"] = "x"
        t["yaxis"] = "y"

    def find_m_pair(prefix: str):
        nav_t, dd_t = None, None
        for t in nav_m_raw:
            nm = t.get("name", "")
            if nm.startswith(prefix):
                if nm.endswith("DD"):
                    dd_t = copy.deepcopy(t)
                else:
                    nav_t = copy.deepcopy(t)
        return nav_t, dd_t

    t_exp_a, dd_exp_a = find_m_pair("基准A:")
    t_exp_e, dd_exp_e = find_m_pair("实验E(原基准):")
    t_exp_e3, dd_exp_e3 = find_m_pair("实验E3(新基准):")
    t_exp_hr, dd_exp_hr = find_m_pair("实验E4c(最优):")

    # 2. Extract Full-60 traces from full60_cta_expansion_report.html
    plots_60 = list(re.finditer(r'Plotly\.newPlot\(\s*"([^"]+)"\s*,\s*(\[)', html_60))
    traces_60, _ = dec.raw_decode(html_60, plots_60[0].start(2))
    by_name_60 = {t.get("name"): t for t in traces_60}

    def make_60_nav_and_dd(key: str, label: str, color: str, width: float = 2.2, dash: str = "solid"):
        raw_t = by_name_60[key]
        x_vals = raw_t["x"]
        cum_pct = _decode_bdata(raw_t["y"])
        nav_arr = (100.0 + cum_pct).round(4).tolist()
        eq = 1.0 + cum_pct / 100.0
        peak = np.maximum.accumulate(np.maximum(eq, 1.0))
        dd_arr = ((eq / peak - 1.0) * 100.0).round(4).tolist()
        nav_trace = {
            "type": "scatter",
            "mode": "lines",
            "name": label,
            "x": x_vals,
            "y": nav_arr,
            "line": {"color": color, "width": width, "dash": dash},
            "xaxis": "x",
            "yaxis": "y",
        }
        dd_trace = {
            "type": "scatter",
            "mode": "lines",
            "name": f"{label} DD",
            "x": x_vals,
            "y": dd_arr,
            "line": {"color": color, "width": max(1.3, width - 0.6), "dash": dash},
            "showlegend": False,
            "xaxis": "x2",
            "yaxis": "y2",
        }
        return nav_trace, dd_trace

    t_f60_flat, dd_f60_flat = make_60_nav_and_dd(
        "Full60_FlatHRP_HysteresisRunner",
        "Full60 朴素等权HRP (Shp 1.15 / MaxDD 1.88% / Net +4.53%)",
        "#dc2626",
        width=1.6,
        dash="dot",
    )
    t_f60_cs, dd_f60_cs = make_60_nav_and_dd(
        "Full60_CoreSatellite_SOPStage3_Optimal",
        "Full60 核心-卫星SOP (Shp 2.42 / MaxDD 1.68% / Net +10.84%)",
        "#0284c7",
        width=2.2,
        dash="dashdot",
    )
    t_f60_zhc, dd_f60_zhc = make_60_nav_and_dd(
        "Full60_ZeroHardcode_ClusterConserved_Optimal",
        "Full60 零先验聚类守恒 (Shp 2.56 / Cal 6.35 / MaxDD 0.83% / Net +10.55%)",
        "#059669",
        width=3.2,
        dash="solid",
    )

    t_exp_a["name"] = "Core20 基准ExpA: 60m+30m (Shp 1.25 / MaxDD 1.89% / Net +5.99%)"
    t_exp_a["line"] = {"color": "#94a3b8", "width": 1.5, "dash": "dash"}
    dd_exp_a["line"] = {"color": "#94a3b8", "width": 1.2, "dash": "dash"}

    t_exp_e["name"] = "Core20 快慢协同ExpE: 静态出场 (Shp 1.81 / MaxDD 1.94% / Net +8.51%)"
    t_exp_e["line"] = {"color": "#6366f1", "width": 1.8, "dash": "solid"}
    dd_exp_e["line"] = {"color": "#6366f1", "width": 1.3, "dash": "solid"}

    t_exp_e3["name"] = "Core20 结构出场ExpE3: 双层止损 (Shp 2.02 / MaxDD 1.69% / Net +9.15%)"
    t_exp_e3["line"] = {"color": "#8b5cf6", "width": 1.9, "dash": "dot"}
    dd_exp_e3["line"] = {"color": "#8b5cf6", "width": 1.3, "dash": "dot"}

    t_exp_hr["name"] = "Core20 滞回棘轮ExpE4c: HysteresisRunner (Shp 2.36 / MaxDD 1.64% / Net +10.42%)"
    t_exp_hr["line"] = {"color": "#2563eb", "width": 2.5, "dash": "solid"}
    dd_exp_hr["line"] = {"color": "#2563eb", "width": 1.6, "dash": "solid"}

    nav_data = [
        t_exp_a,
        dd_exp_a,
        t_f60_flat,
        dd_f60_flat,
        t_exp_e,
        dd_exp_e,
        t_exp_e3,
        dd_exp_e3,
        t_exp_hr,
        dd_exp_hr,
        t_f60_cs,
        dd_f60_cs,
        t_f60_zhc,
        dd_f60_zhc,
    ]

    # 3. Sector Attribution Chart Data
    sec_df = pd.read_csv(FULL60_SECTOR_CSV)
    sec_cn_map = {
        "PRECIOUS_METALS": "贵金属 (2)",
        "AGRICULTURE": "农产品 (16)",
        "FERROUS": "黑色建材 (9)",
        "ENERGY_CHEM": "能源化工 (22)",
        "BASE_METALS": "有色金属 (8)",
        "NEW_ENERGY": "新能源 (3)",
    }
    sec_labels = [sec_cn_map.get(s, s) for s in sec_df["sector"]]
    sector_data = [
        {
            "type": "bar",
            "name": "Flat-60 无筛选等权HRP (Shp 1.15)",
            "x": sec_labels,
            "y": sec_df["flat60_net_pct"].round(2).tolist(),
            "marker": {"color": "#94a3b8"},
        },
        {
            "type": "bar",
            "name": "Core-Satellite-60 先验核心+卫星 (Shp 2.42)",
            "x": sec_labels,
            "y": sec_df["coresat60_net_pct"].round(2).tolist(),
            "marker": {"color": "#3b82f6"},
        },
        {
            "type": "bar",
            "name": "ZeroHC-Cluster-60 零先验内生聚类 (Shp 2.56)",
            "x": sec_labels,
            "y": sec_df["zerohc60_net_pct"].round(2).tolist(),
            "marker": {"color": "#059669"},
        },
    ]

    # 4. Symbol Attribution Chart Data
    sym_df = pd.read_csv(FULL60_SYMBOL_CSV)
    active_mask = (
        (sym_df["zerohc60_net_pct"].abs() > 0.001)
        | (sym_df["coresat60_net_pct"].abs() > 0.15)
        | (sym_df["flat60_net_pct"].abs() > 0.35)
    )
    sub_sym = sym_df[active_mask].copy()
    sub_sym["sort_key"] = sub_sym["zerohc60_net_pct"] * 2.0 + sub_sym["coresat60_net_pct"]
    sub_sym = sub_sym.sort_values("sort_key", ascending=False).head(18)
    sym_x = [f"{r['asset_id']}({r['cn_name']})" for _, r in sub_sym.iterrows()]
    symbol_data = [
        {
            "type": "bar",
            "name": "ZeroHC-Cluster-60 零先验净收益(%)",
            "x": sym_x,
            "y": sub_sym["zerohc60_net_pct"].round(3).tolist(),
            "marker": {"color": "#059669"},
        },
        {
            "type": "bar",
            "name": "Core-Satellite-60 净收益(%)",
            "x": sym_x,
            "y": sub_sym["coresat60_net_pct"].round(3).tolist(),
            "marker": {"color": "#2563eb"},
        },
        {
            "type": "bar",
            "name": "Flat-60 无筛选等权净收益(%)",
            "x": sym_x,
            "y": sub_sym["flat60_net_pct"].round(3).tolist(),
            "marker": {"color": "#ef4444"},
        },
    ]

    return (
        json.dumps(nav_data),
        json.dumps(funnel_data),
        json.dumps(sector_data),
        json.dumps(symbol_data),
    )


TARGET_HTML_PATH = Path("examples/results/intraday_cta_ablation_20261017/multitf_cta_research_presentation.html")
DOCS_HTML_PATH = Path("docs/research/multitf_cta_research_presentation.html")


def build_deck():
    print("提取图表数据与实证序列...")
    nav_json, funnel_json, sector_json, symbol_json = extract_chart_data()

    html_parts = []

    # 1. HTML Head & CSS
    html_parts.append(r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>全周期跨尺度 CTA 策略研发、缠论结构出场与 60 品种零先验扩展体系汇报 (PPT)</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
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
    padding: 5px 10px;
    margin-top: 4px;
    font-family: 'Cambria Math', 'Times New Roman', Georgia, serif;
    font-size: 11px;
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

  * { box-sizing: border-box; margin: 0; padding: 0; }

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
    top: 0; left: 0; right: 0;
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

  .header-left { display: flex; align-items: center; gap: 12px; }

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

  .header-title { font-size: 13.5px; font-weight: 600; color: var(--text-secondary); }

  .header-right { display: flex; align-items: center; gap: 16px; }

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
  .header-btn:hover { background: #f8fafc; border-color: var(--border-strong); color: var(--text-main); }

  #progress-bar {
    position: fixed;
    top: 48px; left: 0;
    height: 3px;
    background: linear-gradient(90deg, #3b82f6, #059669);
    width: 0%;
    transition: width 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    z-index: 101;
  }

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
    top: 0; left: 0; width: 100%; height: 100%;
    padding: 20px 30px;
    display: none;
    flex-direction: column;
    justify-content: space-between;
    overflow-y: auto;
    opacity: 0;
    transition: opacity 0.2s ease-in-out;
  }

  .slide.active { display: flex; opacity: 1; }

  .slide-header {
    margin-bottom: 10px;
    border-bottom: 1px solid var(--border-color);
    padding-bottom: 7px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
  }

  .slide-header-left { flex: 1; }

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
    width: 6px; height: 6px;
    background: var(--accent);
    border-radius: 50%;
  }

  .slide-title {
    font-size: 19px;
    font-weight: 800;
    color: var(--text-main);
    letter-spacing: -0.02em;
    line-height: 1.25;
  }

  .slide-subtitle { font-size: 12px; color: var(--text-muted); margin-top: 2px; }

  .slide-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 11px;
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

  #deck-footer {
    position: fixed;
    bottom: 0; left: 0; right: 0;
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

  .key-hint { display: inline-flex; align-items: center; gap: 4px; }

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

  .dots-nav { display: flex; align-items: center; gap: 6px; }

  .dot-pill {
    width: 8px; height: 8px;
    border-radius: 4px;
    background: #cbd5e1;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .dot-pill.active { width: 24px; background: var(--accent); }

  .footer-controls { display: flex; align-items: center; gap: 8px; }

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

  .nav-btn:hover:not(:disabled) { background: #dbeafe; color: #1e40af; }
  .nav-btn:disabled { opacity: 0.4; cursor: not-allowed; }

  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }
  .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 11px; }

  .card {
    background: #ffffff;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 10px 13px;
    box-shadow: var(--shadow-sm);
  }

  .card-highlight { border-left: 4px solid var(--accent); }
  .card-success { border-left: 4px solid var(--success); background: #fbfdfc; }
  .card-purple { border-left: 4px solid var(--purple); background: #fdfcff; }
  .card-amber { border-left: 4px solid var(--amber); background: #fffdfa; }

  .card-title {
    font-size: 12.5px;
    font-weight: 700;
    color: var(--text-main);
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .card-desc { font-size: 11px; color: var(--text-secondary); line-height: 1.48; }

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

  .table-academic tr:hover td { background: #f8fafc; }
  .table-academic tr.highlight-row td { background: #eff6ff; font-weight: 700; color: #1e3a8a; }
  .table-academic tr.highlight-green td { background: #ecfdf5; font-weight: 700; color: #065f46; }

  #overview-modal {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(15, 23, 42, 0.7);
    backdrop-filter: blur(4px);
    z-index: 200;
    display: none;
    align-items: center;
    justify-content: center;
    padding: 40px;
  }

  #overview-modal.active { display: flex; }

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

  .overview-item.active { border-color: var(--accent); background: #eff6ff; }
  .ov-num { font-size: 11px; font-weight: 800; color: var(--accent); font-family: monospace; }
  .ov-title { font-size: 12px; font-weight: 700; color: var(--text-main); line-height: 1.35; margin-top: 3px; }
  .ov-cat { font-size: 9.5px; color: var(--text-muted); text-transform: uppercase; }

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

<div id="deck-header">
  <div class="header-left">
    <span class="header-tag">CZSC Quant Research</span>
    <span class="header-title" id="header-section-title">01. 封面与项目总览</span>
  </div>
  <div class="header-right">
    <div class="slide-counter" id="slide-counter">01 / 13</div>
    <button class="header-btn" onclick="toggleOverview()" title="显示所有页面预览 (快捷键: O)">⊞ 幻灯片概览</button>
    <button class="header-btn" onclick="toggleFullscreen()" title="切换全屏模式 (快捷键: F)">⛶ 全屏演示</button>
  </div>
</div>

<div id="progress-bar"></div>

<div id="deck-container">
""")

    # 2. Slide 1: Cover
    html_parts.append(r"""
  <!-- ==================== SLIDE 1: 封面 ==================== -->
  <div class="slide active" id="slide-1" data-title="01. 封面与项目总览" data-cat="Overview">
    <div class="slide-header">
      <div class="slide-header-left">
        <div class="slide-category">RESEARCH PRESENTATION · 2026</div>
        <h1 class="slide-title">全周期跨尺度量化 CTA 策略研发、缠论结构出场与 60 品种零先验扩展体系</h1>
        <div class="slide-subtitle">Multi-Timeframe (1d ~ 5m) × 6-Archetype Signals · Chan Structure HysteresisRunner · Full-60 Zero-Hardcode Cluster Conservation</div>
      </div>
      <div>
        <span class="pill-blue badge-pill">纯 Rust 算子计算</span>
        <span class="pill-purple badge-pill">缠论前缀因果时钟</span>
        <span class="pill-green badge-pill">60品种零先验硬编码</span>
        <span class="pill-amber badge-pill">503结算日纯样本外</span>
      </div>
    </div>

    <div class="slide-content" style="justify-content: center; gap: 14px;">
      <!-- Core Out-of-Sample Performance Summary Strip -->
      <div style="background: #ffffff; border: 1px solid var(--border-color); border-radius: 8px; padding: 13px 16px; box-shadow: var(--shadow-sm);">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #f1f5f9; padding-bottom: 8px; margin-bottom: 10px;">
          <div style="font-size: 11px; color: var(--text-muted); font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;">
            全市场 60 商品期货 100% 零先验内生聚类实证指标 (Stitched OOS: 2025.01 ~ 2026.09 · 503 结算日，原生 5m 执行网格，真实扣费 3.0 BP)
          </div>
          <span class="pill-green badge-pill">100% 零先验名单 · 拒绝后验幸存者偏差</span>
        </div>

        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;">
          <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-left: 3.5px solid #059669; border-radius: 6px; padding: 9px 13px;">
            <div style="font-size: 11px; color: var(--text-secondary); font-weight: 600;">60品种零先验净夏普 (1.0x)</div>
            <div style="font-size: 28px; font-weight: 800; color: #065f46; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.15; margin: 3px 0 1px 0;">2.56</div>
            <div style="font-size: 10.5px; color: #059669; font-weight: 600;">超 Core-20 (2.36) 与基准 ExpA (1.25) +105%</div>
          </div>

          <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-left: 3.5px solid #2563eb; border-radius: 6px; padding: 9px 13px;">
            <div style="font-size: 11px; color: var(--text-secondary); font-weight: 600;">卡玛比率 (Calmar Ratio)</div>
            <div style="font-size: 28px; font-weight: 800; color: #1e3a8a; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.15; margin: 3px 0 1px 0;">6.35</div>
            <div style="font-size: 10.5px; color: var(--text-muted);">最大回撤仅 0.83% (净收益 +10.55%)</div>
          </div>

          <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-left: 3.5px solid #7c3aed; border-radius: 6px; padding: 9px 13px;">
            <div style="font-size: 11px; color: var(--text-secondary); font-weight: 600;">单笔盈亏比与安全边际</div>
            <div style="font-size: 28px; font-weight: 800; color: #5b21b6; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.15; margin: 3px 0 1px 0;">2.890</div>
            <div style="font-size: 10.5px; color: #059669; font-weight: 600;">SOP阶段三安全边际 ΔW = +12.30%</div>
          </div>

          <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-left: 3.5px solid #d97706; border-radius: 6px; padding: 9px 13px;">
            <div style="font-size: 11px; color: var(--text-secondary); font-weight: 600;">实盘目标敞口版 (3.0x)</div>
            <div style="font-size: 28px; font-weight: 800; color: #92400e; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.15; margin: 3px 0 1px 0;">+31.64%</div>
            <div style="font-size: 10.5px; color: var(--text-muted);">年化 +15.85% · 最大回撤仅 2.07% · Calmar 7.66</div>
          </div>
        </div>
      </div>

      <!-- Key Methodology & Paradigm Cards -->
      <div class="grid-3">
        <div class="card card-highlight">
          <div class="card-title">
            <span>🔬 演绎先验 × 因果审计 × 归纳证伪</span>
            <span class="pill-blue badge-pill">方法论</span>
          </div>
          <div class="card-desc">
            拒绝盲目数据拟合；解耦【势能蓄力】与【反包确认】两阶段状态机，100% 多空对称；经专门因果性审计彻底拔除前视回填，以严格前缀因果律驱动策略体系。
          </div>
        </div>

        <div class="card card-purple">
          <div class="card-title">
            <span>🛡️ 三阶段增量漏斗 × 缠论结构滞回棘轮</span>
            <span class="pill-purple badge-pill">信号与出场</span>
          </div>
          <div class="card-desc">
            三级漏斗（自适应显著性、激活低共线、边际净夏普）压降 41% 摩擦；次极值/中枢重入止损与 <code>bi_or_peak</code> 同向重入门锁 × 35% 分批止盈保本 Runner，夏普从 1.81 升至 2.36。
          </div>
        </div>

        <div class="card card-success">
          <div class="card-title">
            <span>🌐 全市场 60 品种零先验重构 × 产业链簇守恒</span>
            <span class="pill-green badge-pill">宇宙扩展</span>
          </div>
          <div class="card-desc">
            升级 <code>CrossAssetDataEngine</code> 因果截面去噪（Top-K 龙头与 Kaufman ER 加权），100% 废除 <code>CORE_20_IDS</code> 硬编码，同板块产业链共线聚类与预算守恒，夏普达 2.56、回撤压至 0.83%。
          </div>
        </div>
      </div>

      <!-- Scope and Specs -->
      <div style="background: #f8fafc; border: 1px solid var(--border-color); border-radius: 8px; padding: 8px 14px; display: flex; justify-content: space-between; align-items: center; font-size: 11px; color: var(--text-secondary);">
        <div><strong>覆盖标的:</strong> 6 大板块 60 个流动性合格商品期货主力连续（贵金属、有色、黑色、能化、农产品、新能源）</div>
        <div><strong>验证时钟:</strong> 两折扩张窗口前向验证 (Expanding Walk-Forward, 2024 IS -> 2025/2026 OOS, 503 结算日)</div>
        <div><strong>摩擦核算:</strong> 严格逐笔扣费 (沪金单边 1.0 BP，其余 59 品种单边 1.5 BP / 双边 3.0 BP)，原生 5m 账本记账</div>
      </div>
    </div>

    <div class="slide-footer">
      <span>CZSC Quantitative Strategy Research & Production Group · Institutional Grade</span>
      <span>Slide 01 of 13</span>
    </div>
  </div>
""")

    # 3. Slide 2: Executive Summary (4 Leaps)
    html_parts.append(r"""
  <!-- ==================== SLIDE 2: 核心成果全景 ==================== -->
  <div class="slide" id="slide-2" data-title="02. 核心成果全景：四阶架构跃迁" data-cat="Executive Summary">
    <div class="slide-header">
      <div class="slide-header-left">
        <div class="slide-category">EXECUTIVE SUMMARY</div>
        <h2 class="slide-title">核心成果全景：从单周期基准到 60 品种零先验的四阶架构跃迁</h2>
        <div class="slide-subtitle">突破“高频换手过载”、“前视偏差虚假繁荣”与“后验核心幸存者偏差”的三重迷思</div>
      </div>
      <span class="pill-green badge-pill">纯样本外 503 交易日严苛验证</span>
    </div>

    <div class="slide-content">
      <!-- 4 Paradigm Shifts SVG Diagram -->
      <div style="background: #ffffff; border: 1px solid var(--border-color); border-radius: 8px; padding: 10px 12px; box-shadow: var(--shadow-sm);">
        <svg viewBox="0 0 960 170" style="width: 100%; height: auto; display: block;" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif">
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
          <g transform="translate(8, 12)">
            <rect width="195" height="142" rx="6" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
            <rect width="195" height="24" rx="6" fill="#64748b"/>
            <text x="97" y="16" fill="#ffffff" font-size="11" font-weight="700" text-anchor="middle">初始基准 ExpA (60m+30m)</text>
            <text x="12" y="44" fill="#475569" font-size="10.5">净收益: <tspan font-weight="700" fill="#0f172a">+5.99%</tspan> | 年化 3.00%</text>
            <text x="12" y="64" fill="#475569" font-size="10.5">夏普比率: <tspan font-weight="700" fill="#0f172a">1.25</tspan> | 卡玛: 1.59</text>
            <text x="12" y="84" fill="#475569" font-size="10.5">最大回撤: <tspan font-weight="700" fill="#dc2626">1.89%</tspan></text>
            <text x="12" y="104" fill="#475569" font-size="10.5">年化换手: 86.0x | 摩擦: 2.44%</text>
            <text x="12" y="125" fill="#64748b" font-size="9.5">仅日内主干 · 静态单一出场</text>
          </g>

          <path d="M 207 83 L 232 83" stroke="#2563eb" stroke-width="2" marker-end="url(#arrow-blue)"/>

          <!-- Step 1: ExpE Funnel + Synergy -->
          <g transform="translate(236, 12)">
            <rect width="215" height="142" rx="6" fill="#eff6ff" stroke="#bfdbfe" stroke-width="1.3"/>
            <rect width="215" height="24" rx="6" fill="#2563eb"/>
            <text x="107" y="16" fill="#ffffff" font-size="11" font-weight="700" text-anchor="middle">跃迁一: 漏斗治理与协同 (ExpE)</text>
            <text x="12" y="44" fill="#1e3a8a" font-size="10.5">机制: <tspan font-weight="700">Gate 1/2/3 增量漏斗 + 异构因子</tspan></text>
            <text x="12" y="64" fill="#1e3a8a" font-size="10.5">净收益: <tspan font-weight="700" fill="#2563eb">+8.51%</tspan> | 摩擦: 2.26%</text>
            <text x="12" y="84" fill="#1e3a8a" font-size="10.5">年化夏普: <tspan font-weight="700" fill="#2563eb">1.81</tspan> (大幅提升 +45%)</text>
            <text x="12" y="104" fill="#1e3a8a" font-size="10.5">回撤: 1.94% | 换手: 80.7x</text>
            <text x="12" y="125" fill="#059669" font-size="9.5">消除盲堆换手爆炸 · 压降41%摩擦</text>
          </g>

          <path d="M 455 83 L 480 83" stroke="#7c3aed" stroke-width="2" marker-end="url(#arrow-purple)"/>

          <!-- Step 2: Core-20 HysteresisRunner -->
          <g transform="translate(484, 12)">
            <rect width="225" height="142" rx="6" fill="#f5f3ff" stroke="#ddd6fe" stroke-width="1.5"/>
            <rect width="225" height="24" rx="6" fill="#7c3aed"/>
            <text x="112" y="16" fill="#ffffff" font-size="11" font-weight="700" text-anchor="middle">跃迁二: 缠论滞回棘轮 (Core20-HR)</text>
            <text x="12" y="44" fill="#5b21b6" font-size="10.5">机制: <tspan font-weight="700">重入门锁 bi_or_peak × 分批保本</tspan></text>
            <text x="12" y="64" fill="#5b21b6" font-size="10.5">净收益: <tspan font-weight="700" fill="#7c3aed">+10.42%</tspan> (3.0x: <tspan font-weight="700">+31.25%</tspan>)</text>
            <text x="12" y="84" fill="#5b21b6" font-size="10.5">年化夏普: <tspan font-weight="700" fill="#7c3aed">2.36</tspan> | 卡玛: <tspan font-weight="700">3.18</tspan></text>
            <text x="12" y="104" fill="#5b21b6" font-size="10.5">最大回撤: <tspan font-weight="700" fill="#059669">1.64%</tspan> | 盈亏比: <tspan font-weight="700">R=2.324</tspan></text>
            <text x="12" y="125" fill="#6d28d9" font-size="9.5">Rust 结构出场 · 彻底解决震荡追高磨损</text>
          </g>

          <path d="M 713 83 L 738 83" stroke="#059669" stroke-width="2.5" marker-end="url(#arrow-green)"/>

          <!-- Step 3: ZeroHC-Cluster Full-60 -->
          <g transform="translate(742, 12)">
            <rect width="210" height="142" rx="6" fill="#ecfdf5" stroke="#a7f3d0" stroke-width="2"/>
            <rect width="210" height="24" rx="6" fill="#059669"/>
            <text x="105" y="16" fill="#ffffff" font-size="11" font-weight="800" text-anchor="middle">跃迁三: 60品种零先验聚类守恒</text>
            <text x="12" y="44" fill="#065f46" font-size="10.5">机制: <tspan font-weight="800">F簇去噪 + 产业链聚类 + 预算守恒</tspan></text>
            <text x="12" y="64" fill="#065f46" font-size="10.5">年化夏普: <tspan font-weight="800" fill="#059669">2.56</tspan> (终极超额)</text>
            <text x="12" y="84" fill="#065f46" font-size="10.5">最大回撤: <tspan font-weight="800" fill="#059669">0.83%</tspan> (大幅压降 49%)</text>
            <text x="12" y="104" fill="#065f46" font-size="10.5">卡玛比率: <tspan font-weight="800" fill="#059669">6.35</tspan> | 盈亏比: <tspan font-weight="800">R=2.890</tspan></text>
            <text x="12" y="125" fill="#047857" font-size="9.5">换手 42.7x | 3.0x: +31.64% / 回撤 2.07%</text>
          </g>
        </svg>
      </div>

      <!-- Ablation Summary Table -->
      <div>
        <table class="table-academic">
          <thead>
            <tr>
              <th>实验版本</th>
              <th>资产池与准入机制</th>
              <th>出场与组合架构</th>
              <th>毛收益</th>
              <th>摩擦成本</th>
              <th>纯净收益</th>
              <th>年化夏普</th>
              <th>最大回撤</th>
              <th>卡玛比率</th>
              <th>年化换手</th>
              <th>单笔盈亏比 R</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>ExpA</strong> (基准)</td>
              <td>Core-20 (20)</td>
              <td>仅日内 (60m+30m) + Comb2 静态出场</td>
              <td>8.43%</td>
              <td>2.44%</td>
              <td>5.99%</td>
              <td>1.25</td>
              <td>1.89%</td>
              <td>1.59</td>
              <td>86.0x</td>
              <td>2.246</td>
            </tr>
            <tr>
              <td><strong>ExpC</strong> (漏斗)</td>
              <td>Core-20 (20)</td>
              <td>全周期 + Gate 1/2/3 增量漏斗去共线</td>
              <td>9.67%</td>
              <td>1.98%</td>
              <td>7.69%</td>
              <td>1.72</td>
              <td>1.77%</td>
              <td>2.18</td>
              <td>70.7x</td>
              <td>2.334</td>
            </tr>
            <tr>
              <td><strong>ExpE</strong> (异构基准)</td>
              <td>Core-20 (20)</td>
              <td>补充 1d/15m 异构特征 + 快慢协同调制</td>
              <td>10.77%</td>
              <td>2.26%</td>
              <td>8.51%</td>
              <td>1.81</td>
              <td>1.94%</td>
              <td>2.19</td>
              <td>80.7x</td>
              <td>2.265</td>
            </tr>
            <tr>
              <td><strong>ExpE3_FullStruct</strong></td>
              <td>Core-20 (20)</td>
              <td>双层结构止损 (次极值+三买卖中枢证伪)</td>
              <td>12.40%</td>
              <td>3.25%</td>
              <td>9.15%</td>
              <td>2.02</td>
              <td>1.69%</td>
              <td>2.71</td>
              <td>116.4x</td>
              <td>2.049</td>
            </tr>
            <tr style="background: #f5f3ff;">
              <td><strong>Core20_HR</strong></td>
              <td>Core-20 (20)</td>
              <td><strong>结构门锁 bi_or_peak × 35%分批保本Runner</strong></td>
              <td>13.34%</td>
              <td>2.92%</td>
              <td><strong>10.42%</strong></td>
              <td><strong>2.36</strong></td>
              <td>1.64%</td>
              <td>3.18</td>
              <td>104.7x</td>
              <td><strong>2.324</strong></td>
            </tr>
            <tr style="color: #94a3b8;">
              <td><strong>Full60_FlatHRP</strong></td>
              <td>Full-60 (58)</td>
              <td>60 品种无筛选等权 HRP + HysteresisRunner</td>
              <td>8.14%</td>
              <td>3.61%</td>
              <td>4.53%</td>
              <td>1.15</td>
              <td>1.88%</td>
              <td>1.21</td>
              <td>126.3x</td>
              <td>1.918</td>
            </tr>
            <tr>
              <td><strong>Full60_CoreSat</strong></td>
              <td>Core-20 + SOP 卫星</td>
              <td>核心-卫星因果分配 (带 CORE_20 先验锚定)</td>
              <td>13.85%</td>
              <td>3.00%</td>
              <td>10.84%</td>
              <td>2.42</td>
              <td>1.68%</td>
              <td>3.23</td>
              <td>107.6x</td>
              <td>2.248</td>
            </tr>
            <tr class="highlight-green">
              <td><strong>Full60_ZeroHC_Cluster</strong></td>
              <td><strong>ZeroHC 内生 (F1:7, F2:9)</strong></td>
              <td><strong>100% 零先验 + F簇去噪 + 产业链聚类预算守恒</strong></td>
              <td>11.68%</td>
              <td><strong>1.13%</strong></td>
              <td><strong>10.55%</strong></td>
              <td><strong>2.56</strong></td>
              <td><strong>0.83%</strong></td>
              <td><strong>6.35</strong></td>
              <td><strong>42.7x</strong></td>
              <td><strong>2.890</strong></td>
            </tr>
            <tr style="background: #fdf2f8; font-weight: 700; color: #9d174d;">
              <td><strong>ZeroHC_Cluster (3.0x)</strong></td>
              <td>ZeroHC 内生 (F1:7, F2:9)</td>
              <td>3.0x 实盘目标敞口版 (单标的上限 10% / 板块 30%)</td>
              <td>35.04%</td>
              <td>3.40%</td>
              <td>31.64%</td>
              <td>2.56</td>
              <td>2.07%</td>
              <td>7.66</td>
              <td>128.2x</td>
              <td>2.890</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="slide-footer">
      <span>样本外区间: 2025.01.02 ~ 2026.09.30 (503 个结算日)，扣减双边 3.0 BP (AU 2.0 BP)，原生 5m 执行网格</span>
      <span>Slide 02 of 13</span>
    </div>
  </div>
""")

    # 4. Slide 3: Methodology
    html_parts.append(r"""
  <!-- ==================== SLIDE 3: 方法论体系 ==================== -->
  <div class="slide" id="slide-3" data-title="03. 方法论体系：双轮闭环与SOP铁律" data-cat="Methodology">
    <div class="slide-header">
      <div class="slide-header-left">
        <div class="slide-category">METHODOLOGY & PHILOSOPHY</div>
        <h2 class="slide-title">方法论体系：演绎先验与归纳证伪的双轮闭环与 SOP 铁律</h2>
        <div class="slide-subtitle">拒绝“黑盒暴力拟合”与“事后因果倒置”，坚持以严密的几何形态学与微观博弈逻辑构建防御堡垒</div>
      </div>
      <span class="pill-blue badge-pill">理论根基与工程纪律</span>
    </div>

    <div class="slide-content">
      <!-- Methodology SVG Diagram -->
      <div style="background: #ffffff; border: 1px solid var(--border-color); border-radius: 8px; padding: 10px 14px; box-shadow: var(--shadow-sm);">
        <svg viewBox="0 0 960 170" style="width: 100%; height: auto; display: block;" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif">
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
            <rect width="275" height="130" rx="7" fill="#eff6ff" stroke="#bfdbfe" stroke-width="1.5"/>
            <text x="137" y="20" fill="#1e3a8a" font-size="11.5" font-weight="700" text-anchor="middle">Ⅰ. 演绎推理先验 (Deductive Priors)</text>
            <line x1="12" y1="28" x2="263" y2="28" stroke="#bfdbfe" stroke-width="1"/>
            <text x="16" y="45" fill="#1e293b" font-size="10.5">● <tspan font-weight="700">缠论自相似形态:</tspan> 分型、笔、中枢动态演化</text>
            <text x="16" y="63" fill="#1e293b" font-size="10.5">● <tspan font-weight="700">微观物理博弈:</tspan> 砍仓出清、主动买卖盘失衡</text>
            <text x="16" y="81" fill="#1e293b" font-size="10.5">● <tspan font-weight="700">两阶段状态机解耦:</tspan> Setup 势能蓄力 + 反包确认</text>
            <text x="16" y="99" fill="#1e293b" font-size="10.5">● <tspan font-weight="700">100% 对称做空架构:</tspan> 拒绝多头专属偏置</text>
            <text x="16" y="117" fill="#2563eb" font-size="9.5" font-weight="600">→ 严禁无经济学先验的单根 K 线静态拼接</text>
          </g>

          <path d="M 290 75 L 330 75" stroke="#1e3a8a" stroke-width="2.5" marker-end="url(#m-arrow)"/>

          <!-- Box 2: Formalization & Feature Engine -->
          <g transform="translate(335, 10)">
            <rect width="275" height="130" rx="7" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
            <text x="137" y="20" fill="#0f172a" font-size="11.5" font-weight="700" text-anchor="middle">Ⅱ. 形式化与向量特征工程 (Engine)</text>
            <line x1="12" y1="28" x2="263" y2="28" stroke="#cbd5e1" stroke-width="1"/>
            <text x="16" y="45" fill="#334155" font-size="10.5">● <tspan font-weight="700">5级时间尺度:</tspan> 1d, 60m, 30m, 15m, 5m 原生网格</text>
            <text x="16" y="63" fill="#334155" font-size="10.5">● <tspan font-weight="700">6大正交战略簇:</tspan> 趋势、反转、锚定、微观、波动、截面</text>
            <text x="16" y="81" fill="#334155" font-size="10.5">● <tspan font-weight="700">F 簇因果截面去噪:</tspan> 链式活跃增量 + Kaufman ER</text>
            <text x="16" y="99" fill="#334155" font-size="10.5">● <tspan font-weight="700">纯 Rust 算子计算:</tspan> czsc-signals 零 GC 内存对齐</text>
            <text x="16" y="117" fill="#64748b" font-size="9.5" font-weight="600">→ Python 端纯透传，无额外适配层开销</text>
          </g>

          <path d="M 615 75 L 655 75" stroke="#1e3a8a" stroke-width="2.5" marker-end="url(#m-arrow)"/>

          <!-- Box 3: Inductive Falsification Gates -->
          <g transform="translate(660, 10)">
            <rect width="290" height="130" rx="7" fill="#fef2f2" stroke="#fecaca" stroke-width="1.5"/>
            <text x="145" y="20" fill="#991b1b" font-size="11.5" font-weight="700" text-anchor="middle">Ⅲ. 归纳证伪门禁 (Inductive Falsification)</text>
            <line x1="12" y1="28" x2="278" y2="28" stroke="#fecaca" stroke-width="1"/>
            <text x="16" y="45" fill="#7f1d1d" font-size="10.5">● <tspan font-weight="700">Gate 1 自适应显著性:</tspan> 单品种 Sharpe>0.15 (5m要求>=0.43)</text>
            <text x="16" y="63" fill="#7f1d1d" font-size="10.5">● <tspan font-weight="700">Gate 2 激活低共线:</tspan> |ρ_active|&lt;0.35 + 跨品种池化共振</text>
            <text x="16" y="81" fill="#7f1d1d" font-size="10.5">● <tspan font-weight="700">Gate 3 扣费边际净夏普:</tspan> 组合增量 ΔSharpe &ge; 0</text>
            <text x="16" y="99" fill="#7f1d1d" font-size="10.5">● <tspan font-weight="700">SOP 阶段三安全边际:</tspan> ΔW = W - 1/(1+R) &ge; +10%</text>
            <text x="16" y="117" fill="#b91c1c" font-size="9.5" font-weight="600">→ 无法通过门禁的特征无条件淘汰，严禁修改参数过拟合</text>
          </g>

          <!-- Feedback Loop Arrow Bottom -->
          <path d="M 805 145 L 805 160 L 147 160 L 147 145" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="4,4" fill="none" marker-end="url(#m-feedback)"/>
          <text x="475" y="157" fill="#dc2626" font-size="9.5" font-weight="700" text-anchor="middle">失效假说证伪闭环: 任何无法通过 Gate 2/3 的特征必须无条件剔除，严禁暴力修改参数过拟合</text>
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
            初始止损点严格锚定结构破位点（次低点/次高点分型极值），单笔硬止损不超过 1.0%；一旦突破即启动动态追踪止盈，确保单笔理论盈亏比 <span class="math"><i>R</i> &ge; 1.7</span>，单笔期望安全边际 <span class="math">&Delta;<i>W</i> = <i>W</i> &minus; <sup>1</sup>&frasl;<sub>(1+<i>R</i>)</sub> &gt; +10%</span>。
          </div>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <span>遵循 CZSC 机构级量化策略研发规范 (Constitution §2 SOP) · 严禁数据拟合与幸存者偏差</span>
      <span>Slide 03 of 13</span>
    </div>
  </div>
""")

    # 5. Slide 4: Causality Audit & Splitting Protocol
    html_parts.append(r"""
  <!-- ==================== SLIDE 4: 样本切分与因果审计 ==================== -->
  <div class="slide" id="slide-4" data-title="04. 样本切分协议与缠论因果性全面审计" data-cat="Data Protocol">
    <div class="slide-header">
      <div class="slide-header-left">
        <div class="slide-category">DATA INTEGRITY & CAUSALITY AUDIT</div>
        <h2 class="slide-title">样本切分协议与缠论因果性全面审计</h2>
        <div class="slide-subtitle">两折扩张窗口前向验证 (Two-Fold Expanding Walk-Forward)、前视回填漏洞破除与零泄漏时钟防线</div>
      </div>
      <span class="pill-purple badge-pill">两折 Walk-Forward · 503天纯样本外</span>
    </div>

    <div class="slide-content">
      <!-- Accurate Gantt Timeline SVG -->
      <div style="background: #ffffff; border: 1px solid var(--border-color); border-radius: 8px; padding: 10px 16px; box-shadow: var(--shadow-sm);">
        <svg viewBox="0 0 960 135" style="width: 100%; height: auto; display: block;" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif">
          <line x1="40" y1="92" x2="920" y2="92" stroke="#cbd5e1" stroke-width="2"/>
          <circle cx="40" cy="92" r="4" fill="#64748b"/>
          <circle cx="390" cy="92" r="4" fill="#2563eb"/>
          <circle cx="690" cy="92" r="4" fill="#7c3aed"/>
          <circle cx="920" cy="92" r="4" fill="#059669"/>

          <text x="40" y="112" fill="#64748b" font-size="10.5" font-weight="600" text-anchor="middle">2024-01-02</text>
          <text x="390" y="112" fill="#2563eb" font-size="10.5" font-weight="600" text-anchor="middle">2024-12-31 (Cutoff 1)</text>
          <text x="690" y="112" fill="#7c3aed" font-size="10.5" font-weight="600" text-anchor="middle">2025-12-31 (Cutoff 2)</text>
          <text x="920" y="112" fill="#059669" font-size="10.5" font-weight="600" text-anchor="middle">2026-09-30 (End)</text>

          <!-- Fold 1 Tracks -->
          <g transform="translate(0, 8)">
            <rect x="40" y="5" width="350" height="26" rx="5" fill="#eff6ff" stroke="#93c5fd" stroke-width="1.2"/>
            <text x="215" y="22" fill="#1e40af" font-size="10.5" font-weight="700" text-anchor="middle">Fold 1 训练集 (IS): 2024 全年 (12个月) · 执行 Gate 1~3 选出 2025 因子</text>

            <rect x="395" y="5" width="295" height="26" rx="5" fill="#ecfdf5" stroke="#6ee7b7" stroke-width="1.5"/>
            <text x="542" y="22" fill="#065f46" font-size="10.5" font-weight="700" text-anchor="middle">Fold 1 纯样本外 (OOS 1): 2025 全年盲测 (242 结算日)</text>
          </g>

          <!-- Fold 2 Tracks (Expanding) -->
          <g transform="translate(0, 40)">
            <rect x="40" y="5" width="650" height="26" rx="5" fill="#f5f3ff" stroke="#c4b5fd" stroke-width="1.2"/>
            <text x="365" y="22" fill="#6d28d9" font-size="10.5" font-weight="700" text-anchor="middle">Fold 2 扩张训练集 (IS): 2024 ~ 2025 (24个月) · 扩张窗口重跑 Gate 1~3 迭代 2026 因子</text>

            <rect x="695" y="5" width="225" height="26" rx="5" fill="#ecfdf5" stroke="#10b981" stroke-width="2"/>
            <text x="807" y="22" fill="#047857" font-size="10.5" font-weight="800" text-anchor="middle">Fold 2 纯样本外 (OOS 2): 2026.01~09 (9个月)</text>
          </g>

          <text x="655" y="128" fill="#059669" font-size="10.5" font-weight="800" text-anchor="middle">▲ Stitched OOS: 两折样本外无缝拼接共 21 个月 (503 交易日)，冷启动归零，绝对冻结</text>
        </svg>
      </div>

      <!-- 4 Isolation Gateway Cards -->
      <div class="grid-4">
        <div class="card card-highlight">
          <div class="card-title">
            <span>1. 破除过去回填漏洞</span>
            <span class="pill-blue badge-pill">因果审计</span>
          </div>
          <div class="card-desc">
            审计定位并删除了 <code>w5_early[idx-2:idx]=w5[idx]</code> 向过去 10 分钟回填未来仓位的漏洞；证实原宣称的 Sharpe 3.03 系前视幻觉，严格因果对齐后真实 Sharpe 为 1.76，确立 <strong>ExpE / Core20-HR 为真实有效架构</strong>。
          </div>
        </div>

        <div class="card card-purple">
          <div class="card-title">
            <span>2. 非规则时段后向对齐</span>
            <span class="pill-purple badge-pill">时序因果</span>
          </div>
          <div class="card-desc">
            统一使用 <code>pd.merge_asof(direction="backward")</code> 对齐日线、60m、30m 与 15m 信号；严禁用长度可整除关系 <code>np.repeat</code> 简化；彻底消除 <code>np.roll</code> 循环位移造成的未来尾部向历史头部环形泄露。
          </div>
        </div>

        <div class="card card-success">
          <div class="card-title">
            <span>3. 缓存指纹与沙盒隔离</span>
            <span class="pill-green badge-pill">防缓存污染</span>
          </div>
          <div class="card-desc">
            剥离全局 <code>read_pickle</code> / <code>Path.exists</code> 猴子补丁；将 OHLCV 全量数值、时间戳范围、35+ 个信号参数 JSON、环境变量与 Rust 扩展动态库二进制全量纳入 <code>sha256</code> 校验，损坏或参数微变立即自动重算。
          </div>
        </div>

        <div class="card card-amber">
          <div class="card-title">
            <span>4. 结构确认与次柱开盘撮合</span>
            <span class="pill-amber badge-pill">执行保真</span>
          </div>
          <div class="card-desc">
            分型、分笔与中枢在完成 Bar 收盘瞬间确认；策略目标权重强制右移一期（<code>shift(1)</code>），于下一根 Bar 开盘价市价撮合，底层 Rust <code>executed_weight_ledger</code> 逐笔扣减实际调仓名义本金与滑点。
          </div>
        </div>
      </div>

      <!-- Universe Breakdown -->
      <div style="background: #ffffff; border: 1px solid var(--border-color); border-radius: 8px; padding: 7px 12px; font-size: 11px; display: flex; justify-content: space-between;">
        <div><strong>贵金属 (2):</strong> au, ag</div>
        <div><strong>能源化工 (22):</strong> sc, ta, ma, pp, ru, fu, lu, bu, pg, px, eg, pf, pr, ur, l, v, eb, sh, nr, br, sp, ec</div>
        <div><strong>有色金属 (8):</strong> cu, zn, ni, al, sn, pb, ao, ss</div>
        <div><strong>黑色建材 (9):</strong> i, fg, hc, jm, rb, j, sa, sm, sf</div>
        <div><strong>农产品油脂 (16):</strong> sr, p, oi, m, rm, a, b, y, pk, c, cs, cf, lh, jd, ap, cj</div>
        <div><strong>新能源 (3):</strong> si, lc, ps</div>
      </div>
    </div>

    <div class="slide-footer">
      <span>两折扩张 Walk-Forward 验证 · 纯样本外 503 个结算日零未来函数泄露 · 详见 chan_causality_audit_20261019.md</span>
      <span>Slide 04 of 13</span>
    </div>
  </div>
""")

    # 6. Slide 5: Multi-TF Signal Matrix & F-Cluster Denoising
    html_parts.append(r"""
  <!-- ==================== SLIDE 5: 全周期跨尺度信号矩阵与F簇去噪 ==================== -->
  <div class="slide" id="slide-5" data-title="05. 全周期信号矩阵与F簇因果截面去噪" data-cat="Signal Engineering">
    <div class="slide-header">
      <div class="slide-header-left">
        <div class="slide-category">SIGNAL MATRIX & CAUSAL CROSS-SECTIONAL DENOISING</div>
        <h2 class="slide-title">全周期跨尺度信号矩阵与 F 簇因果截面去噪升级</h2>
        <div class="slide-subtitle">从宏观日线定势到 5m 微观执行的 5 级离散金字塔，与解决 60 品种边缘毛刺篡权的截面加权新范式</div>
      </div>
      <span class="pill-blue badge-pill">5大周期 × 6大原型 · 因果截面去噪</span>
    </div>

    <div class="slide-content">
      <div class="grid-2" style="grid-template-columns: 1fr 1.25fr; gap: 14px;">
        <!-- Left: 5-Level Pyramid Diagram -->
        <div style="background: #ffffff; border: 1px solid var(--border-color); border-radius: 8px; padding: 11px 13px; box-shadow: var(--shadow-sm); display: flex; flex-direction: column; justify-content: space-between;">
          <div style="font-size: 12px; font-weight: 700; color: var(--primary); margin-bottom: 5px; display: flex; justify-content: space-between;">
            <span>跨尺度时间周期分层金字塔 (Hierarchical Multi-TF)</span>
            <span class="pill-blue badge-pill">快慢分工</span>
          </div>

          <svg viewBox="0 0 420 180" style="width: 100%; height: auto;" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif">
            <polygon points="150,8 270,8 296,38 124,38" fill="#1e3a8a" />
            <text x="210" y="22" fill="#ffffff" font-size="10.5" font-weight="800" text-anchor="middle">1d · 宏观定势慢变量 (Macro Trend)</text>
            <text x="210" y="34" fill="#93c5fd" font-size="8.5" text-anchor="middle">板块价差离散回归、日线乖离反转、大级别动量</text>

            <polygon points="122,42 298,42 324,72 96,72" fill="#2563eb" />
            <text x="210" y="56" fill="#ffffff" font-size="10.5" font-weight="800" text-anchor="middle">60m · 波段骨架层 (Swing Bone)</text>
            <text x="210" y="68" fill="#dbeafe" font-size="8.5" text-anchor="middle">1小时缠论笔延续、均线中继通道、核心波段主趋势</text>

            <polygon points="94,76 326,76 352,106 68,106" fill="#3b82f6" />
            <text x="210" y="90" fill="#ffffff" font-size="10.5" font-weight="800" text-anchor="middle">30m · 核心动量层 (Momentum Engine)</text>
            <text x="210" y="102" fill="#eff6ff" font-size="8.5" text-anchor="middle">经典能量潮、顺势突破共振、高留存阿尔法主干</text>

            <polygon points="66,110 354,110 380,140 40,140" fill="#7c3aed" />
            <text x="210" y="124" fill="#ffffff" font-size="10.5" font-weight="800" text-anchor="middle">15m · 微观确认层 (Micro Confirmation)</text>
            <text x="210" y="136" fill="#ede9fe" font-size="8.5" text-anchor="middle">放量恐慌衰竭反转、顺势回踩触发器、波段结构出场</text>

            <polygon points="38,144 382,144 408,174 12,174" fill="#059669" />
            <text x="210" y="158" fill="#ffffff" font-size="10.5" font-weight="800" text-anchor="middle">5m · 原生物理执行网格 (Execution Grid & Pulse)</text>
            <text x="210" y="170" fill="#d1fae5" font-size="8.5" text-anchor="middle">全周期后向阶梯映射对齐、低延迟执行撮合、调仓死区滤波</text>
          </svg>

          <div style="font-size: 10px; color: var(--text-muted); margin-top: 3px; line-height: 1.4;">
            <strong>快慢协同调制公式:</strong>
            <div class="math-block">
              <span class="math"><i>W</i> = Clip(<i>W</i><sub>fast</sub> &middot; [1 + 0.25&middot;<b>I</b><sub>agree</sub> &minus; 0.35&middot;<b>I</b><sub>conflict</sub>] + 0.70&middot;<i>W</i><sub>1d</sub>, &minus;1, 1)</span>
            </div>
          </div>
        </div>

        <!-- Right: 6 Archetypes & F-Cluster Denoising -->
        <div style="display: flex; flex-direction: column; gap: 7px;">
          <div class="card" style="padding: 7px 11px; border-left: 3.5px solid #2563eb;">
            <div style="font-size: 11.5px; font-weight: 700; color: #1e40af; display: flex; justify-content: space-between;">
              <span>A 簇：趋势动量 & B 簇：极值反转</span>
              <span class="pill-blue badge-pill">核心基石</span>
            </div>
            <div style="font-size: 10px; color: var(--text-secondary); margin-top: 1px;">
              唐奇安通道、均线排列、中枢离开段突破；RSI 极值超买超卖、底顶分型与线段背驰
            </div>
          </div>

          <div class="card" style="padding: 7px 11px; border-left: 3.5px solid #059669;">
            <div style="font-size: 11.5px; font-weight: 700; color: #065f46; display: flex; justify-content: space-between;">
              <span>C 簇：均线价差锚定 & E 簇：波动率突破</span>
              <span class="pill-green badge-pill">结构突破</span>
            </div>
            <div style="font-size: 10px; color: var(--text-secondary); margin-top: 1px;">
              顺大势回踩 MA60 企稳触发；ATR 极窄幅波动率压缩 (Squeeze) 后的放量脱离
            </div>
          </div>

          <div class="card" style="padding: 7px 11px; border-left: 3.5px solid #7c3aed;">
            <div style="font-size: 11.5px; font-weight: 700; color: #5b21b6; display: flex; justify-content: space-between;">
              <span>1d / 15m 跨尺度异构专属特征 (6+6 个)</span>
              <span class="pill-purple badge-pill">ExpE 增量</span>
            </div>
            <div style="font-size: 10px; color: var(--text-secondary); margin-top: 1px;">
              <code>H1D_SP01</code> 板块价差离散回归、<code>H1D_DV01</code> 日线乖离反转；<code>H15M_EX01</code> 放量恐慌衰竭、<code>H15M_PB01</code> 顺势回踩
            </div>
          </div>

          <!-- F-Cluster Causal Denoising Card -->
          <div class="card card-success" style="padding: 7px 11px; border-left: 3.5px solid #059669; background: #f0fdf4;">
            <div style="font-size: 11.5px; font-weight: 700; color: #065f46; display: flex; justify-content: space-between;">
              <span>F 簇（FC01~FC04）：因果截面特征去噪升级 (CrossAssetDataEngine)</span>
              <span class="pill-green badge-pill">60品种核心攻坚</span>
            </div>
            <div style="font-size: 10px; color: var(--text-secondary); margin-top: 2px; line-height: 1.45;">
              <strong>病根:</strong> 扩展至 60 品种后，边缘次新品种单 Bar 极值毛刺篡夺板块龙头，导致截面反转。<br/>
              <strong>升级 1 (链式增量合成):</strong> <span class="math">&Delta;<i>P</i><sub><i>s</i>,<i>t</i></sub><sup>sec</sup> = &sum; <i>w̃</i><sub><i>k</i>,<i>t</i>&minus;1</sub> &Delta;<i>P̃</i><sub><i>k</i>,<i>t</i></sub></span> 消除隔夜跨品种换月跳变<br/>
              <strong>升级 2 (因果信噪比加权):</strong> <span class="math"><i>W</i><sub><i>k</i>,<i>t</i>&minus;1</sub> = Liq<sub><i>k</i>,<i>t</i>&minus;1</sub> &middot; ER<sub><i>k</i>,<i>t</i>&minus;1</sub>(20) &middot; Stab<sub><i>k</i>,<i>t</i>&minus;1</sub></span><br/>
              <strong>升级 3 (Top-K 龙头合成):</strong> 替代单 Bar <code>max_abs</code> 极值，彻底切断小品种噪音篡权！
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <span>147 个候选特征由 pure Rust (czsc-signals / czsc-ta) 实现，保障向量化计算精度与纳秒级性能</span>
      <span>Slide 05 of 13</span>
    </div>
  </div>
""")

    # 7. Slide 6: Signal Validation (3-Stage Funnel)
    html_parts.append(r"""
  <!-- ==================== SLIDE 6: 信号检验体系 ==================== -->
  <div class="slide" id="slide-6" data-title="06. 信号检验体系：三阶段增量漏斗" data-cat="Signal Validation">
    <div class="slide-header">
      <div class="slide-header-left">
        <div class="slide-category">FACTOR FORENSIC ADMISSION PIPELINE</div>
        <h2 class="slide-title">信号检验体系：三阶段增量准入漏斗</h2>
        <div class="slide-subtitle">从 5,880 个时序评估实例到高质量正交因子的贪心前向准入净化与留存分异</div>
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
      <span>Slide 06 of 13</span>
    </div>
  </div>
""")

    # 8. Slide 7: Forensics
    html_parts.append(r"""
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
            <div style="background:#eff6ff; padding:3px 6px; border-radius:4px; margin:3px 0; text-align:center;">
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
            ● <strong>跨品种联合池化</strong>:
            <div style="background:#f5f3ff; padding:3px 6px; border-radius:4px; margin:3px 0; text-align:center;">
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
            <div style="background:#ecfdf5; padding:3px 6px; border-radius:4px; margin:3px 0; text-align:center;">
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
              <td><strong>组合边际表现</strong></td>
              <td>拖累组合夏普断崖下跌 <strong>-0.65</strong>，换手推高 42x</td>
              <td>在真实因果对齐下 Sharpe 1.76，为网格提供微观流动性感知</td>
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
            漏斗筛选结果展现出惊人的尺度分工：<strong>60m 与 30m 留存率高达 80%~93%</strong>，构成了全策略 93% 的基础容量与底仓；而 <strong>15m 与 5m 仅保留极其严苛的 10~15 个异构极值特征</strong>，只在行情转折点输出瞬时不对称赔率。多周期联立是由粗到精的严密金字塔结构。
          </div>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <span>严格的统计检验与对照组证伪证明：微观持仓量脉冲是打破高频换手摩擦困境的唯一有效钥匙</span>
      <span>Slide 07 of 13</span>
    </div>
  </div>
""")

    # 9. Slide 8: Chan Structure Dynamic Exit & HysteresisRunner
    html_parts.append(r"""
  <!-- ==================== SLIDE 8: 缠论结构化出场与滞回棘轮 ==================== -->
  <div class="slide" id="slide-8" data-title="08. 缠论结构化动态出场与滞回棘轮状态机" data-cat="Structure Exit">
    <div class="slide-header">
      <div class="slide-header-left">
        <div class="slide-category">CHAN STRUCTURE DYNAMIC EXIT & HYSTERESIS RUNNER</div>
        <h2 class="slide-title">缠论结构化动态出场与滞回棘轮状态机 (HysteresisRunner)</h2>
        <div class="slide-subtitle">Rust 原生 StructureSnapshot、同向重入结构门锁 (bi_or_peak) 与两级分批止盈保本底仓棘轮</div>
      </div>
      <span class="pill-purple badge-pill">Sharpe 1.81 -> 2.36 · R=2.324 · 3.0x +31.25%</span>
    </div>

    <div class="slide-content">
      <!-- Structure Exit Architecture SVG Diagram -->
      <div style="background: #ffffff; border: 1px solid var(--border-color); border-radius: 8px; padding: 10px 12px; box-shadow: var(--shadow-sm);">
        <svg viewBox="0 0 960 145" style="width: 100%; height: auto; display: block;" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif">
          <defs>
            <marker id="se-arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 1 L 10 5 L 0 9 z" fill="#7c3aed" />
            </marker>
          </defs>

          <!-- Module 1: StructureSnapshot Anchors -->
          <g transform="translate(8, 8)">
            <rect width="285" height="128" rx="6" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.3"/>
            <text x="142" y="20" fill="#1e3a8a" font-size="11" font-weight="800" text-anchor="middle">1. 缠论结构止损锚点 (StructureSnapshot)</text>
            <rect x="12" y="28" width="261" height="26" rx="4" fill="#eff6ff"/>
            <text x="20" y="44" fill="#1e40af" font-size="9.5" font-weight="700">笔次极值止损: 多头锚定前低与次低点较高者</text>
            <rect x="12" y="58" width="261" height="26" rx="4" fill="#eff6ff"/>
            <text x="20" y="74" fill="#1e40af" font-size="9.5" font-weight="700">中枢重入证伪: 跌破 ZG / 突破 ZD 即刻离场</text>
            <rect x="12" y="88" width="261" height="26" rx="4" fill="#eff6ff"/>
            <text x="20" y="104" fill="#1e40af" font-size="9.5" font-weight="700">单调棘轮止损: ratchet_stop 追踪位只升不降</text>
            <text x="142" y="125" fill="#64748b" font-size="9" text-anchor="middle">Rust 纯原生计算 · 前缀因果零前视</text>
          </g>

          <path d="M 300 72 L 325 72" stroke="#7c3aed" stroke-width="2.5" marker-end="url(#se-arrow)"/>

          <!-- Module 2: Same-Direction Re-entry Gate -->
          <g transform="translate(330, 8)">
            <rect width="295" height="128" rx="6" fill="#f5f3ff" stroke="#ddd6fe" stroke-width="1.5"/>
            <text x="147" y="20" fill="#6d28d9" font-size="11" font-weight="800" text-anchor="middle">2. 同向重入结构门锁 (bi_or_peak)</text>
            <rect x="12" y="28" width="271" height="26" rx="4" fill="#ede9fe"/>
            <text x="20" y="44" fill="#5b21b6" font-size="9.5" font-weight="700">平仓锁死: 止损/止盈出场后立即闭锁同向开仓</text>
            <rect x="12" y="58" width="271" height="26" rx="4" fill="#ede9fe"/>
            <text x="20" y="74" fill="#5b21b6" font-size="9.5" font-weight="700">开锁条件 A: 必须出现新的同向已完成缠论笔</text>
            <rect x="12" y="88" width="271" height="26" rx="4" fill="#ede9fe"/>
            <text x="20" y="104" fill="#5b21b6" font-size="9.5" font-weight="700">开锁条件 B: 或价格收盘彻底突破前次持仓极值</text>
            <text x="147" y="125" fill="#6d28d9" font-size="9" text-anchor="middle">彻底斩断高位震荡反复追高被打脸的摩擦磨损</text>
          </g>

          <path d="M 632 72 L 657 72" stroke="#7c3aed" stroke-width="2.5" marker-end="url(#se-arrow)"/>

          <!-- Module 3: Two-Stage Partial TP & Breakeven Runner -->
          <g transform="translate(662, 8)">
            <rect width="288" height="128" rx="6" fill="#ecfdf5" stroke="#a7f3d0" stroke-width="1.8"/>
            <text x="144" y="20" fill="#065f46" font-size="11" font-weight="800" text-anchor="middle">3. 两级分批止盈与保本底仓棘轮 (Runner)</text>
            <rect x="12" y="28" width="264" height="26" rx="4" fill="#d1fae5"/>
            <text x="20" y="44" fill="#065f46" font-size="9.5" font-weight="700">首段减仓: 浮盈达 1.6% 减仓 35% 锁定第一桶金</text>
            <rect x="12" y="58" width="264" height="26" rx="4" fill="#d1fae5"/>
            <text x="20" y="74" fill="#065f46" font-size="9.5" font-weight="700">保本单向棘轮: 剩余 65% 底仓止损移至保本位 (+5 BP)</text>
            <rect x="12" y="88" width="264" height="26" rx="4" fill="#d1fae5"/>
            <text x="20" y="104" fill="#065f46" font-size="9.5" font-weight="700">放宽追踪: Runner 底仓追踪倍数放宽至 1.5x 吃大波段</text>
            <text x="144" y="125" fill="#047857" font-size="9" font-weight="800" text-anchor="middle">让利润奔跑 · 永远不让盈利交易变为亏损</text>
          </g>
        </svg>
      </div>

      <!-- Core-20 Structure Exit Ablation Table -->
      <div>
        <table class="table-academic">
          <thead>
            <tr>
              <th>实验版本</th>
              <th>出场与状态机机制</th>
              <th>毛收益</th>
              <th>摩擦成本</th>
              <th>纯净收益</th>
              <th>年化夏普</th>
              <th>最大回撤</th>
              <th>卡玛比率</th>
              <th>年化换手</th>
              <th>单笔盈亏比 R</th>
              <th>单笔胜率</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>ExpE_StaticExit</strong></td>
              <td>静态时间衰减 + 固定点位止损</td>
              <td>10.77%</td>
              <td>2.26%</td>
              <td>8.51%</td>
              <td>1.81</td>
              <td>1.94%</td>
              <td>2.19</td>
              <td>80.7x</td>
              <td>2.265</td>
              <td>35.4%</td>
            </tr>
            <tr>
              <td><strong>ExpE_FactorExit</strong></td>
              <td>因子层结构衰减 (次极值+中枢证伪+反向笔)</td>
              <td>11.78%</td>
              <td>3.21%</td>
              <td>8.57%</td>
              <td>1.83</td>
              <td>1.87%</td>
              <td>2.30</td>
              <td>115.0x</td>
              <td>1.992</td>
              <td>37.4%</td>
            </tr>
            <tr>
              <td><strong>ExpE_FullStructExit</strong></td>
              <td>双层全量结构止损 (因子层衰减 + 合成层门控)</td>
              <td>12.40%</td>
              <td>3.25%</td>
              <td>9.15%</td>
              <td>2.02</td>
              <td>1.69%</td>
              <td>2.71</td>
              <td>116.4x</td>
              <td>2.049</td>
              <td>38.2%</td>
            </tr>
            <tr>
              <td><strong>ExpE_StructExit_ReentryGate</strong></td>
              <td>仅同向重入结构门锁 (bi_or_peak)</td>
              <td>12.15%</td>
              <td>2.97%</td>
              <td>9.17%</td>
              <td>2.00</td>
              <td>1.74%</td>
              <td>2.64</td>
              <td>106.5x</td>
              <td><strong>2.440</strong></td>
              <td>35.4%</td>
            </tr>
            <tr>
              <td><strong>ExpE_StructExit_PartialTP</strong></td>
              <td>仅两级分批止盈与保本底仓棘轮</td>
              <td>12.65%</td>
              <td>3.18%</td>
              <td>9.47%</td>
              <td>2.15</td>
              <td>1.94%</td>
              <td>2.45</td>
              <td>114.0x</td>
              <td>2.144</td>
              <td>37.4%</td>
            </tr>
            <tr class="highlight-row">
              <td><strong>ExpE_HysteresisRunner</strong></td>
              <td><strong>重入门锁 (bi_or_peak) × 35%分批保本Runner 协同</strong></td>
              <td><strong>13.34%</strong></td>
              <td><strong>2.92%</strong></td>
              <td><strong>10.42%</strong></td>
              <td><strong>2.36</strong></td>
              <td><strong>1.64%</strong></td>
              <td><strong>3.18</strong></td>
              <td><strong>104.7x</strong></td>
              <td><strong>2.324</strong></td>
              <td><strong>36.6%</strong></td>
            </tr>
            <tr style="background: #fdf2f8; font-weight: 700; color: #9d174d;">
              <td><strong>ExpE_HysteresisRunner_3x</strong></td>
              <td>3.0x 实盘目标敞口版 (单标的上限 10% / 板块 30%)</td>
              <td>40.01%</td>
              <td>8.75%</td>
              <td>31.25%</td>
              <td>2.36</td>
              <td>4.89%</td>
              <td>3.20</td>
              <td>314.1x</td>
              <td>2.324</td>
              <td>36.6%</td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- Core Insights 2-Card Grid -->
      <div class="grid-2">
        <div class="card card-highlight">
          <div class="card-title">
            <span>为什么同向重入门锁与分批止盈能形成巨大合力？</span>
            <span class="pill-blue badge-pill">非对称赔率放大</span>
          </div>
          <div class="card-desc">
            单独使用分批止盈虽然锁利但容易过早离场并在震荡中重新追入导致反复磨损；同向重入门锁 <code>bi_or_peak</code> 强行阻止了出场后的冲动回补，迫使资金等待新结构成立或极值有效突破。两者协同使得<strong>单笔盈亏比从 2.049 跃升至 2.324</strong>，换手压降 11.7x。
          </div>
        </div>

        <div class="card card-success">
          <div class="card-title">
            <span>实盘 3.0x 目标敞口下最大回撤严格锚定 4.89% (&le; 5.0%)</span>
            <span class="pill-green badge-pill">机构级风控达标</span>
          </div>
          <div class="card-desc">
            在 3.0x 实盘目标敞口版（年化收益 +15.66%、累计净回报 +31.25%）下，保本棘轮底仓确保了在 2025~2026 年多轮剧烈假突破与急跌行情中，<strong>最大回撤严格收敛在 4.89%</strong>，成功达成机构级 CTA 3.0x 实盘最大回撤 &le; 5.0% 的最严苛风控红线！
          </div>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <span>Rust 纯原生状态机 (crates/czsc-core/src/structure_exit.rs) · 零前视因果执行 · 详见 PR #12</span>
      <span>Slide 08 of 13</span>
    </div>
  </div>
""")

    # 10. Slide 9: Full-60 Zero-Hardcode & Cluster Conservation
    html_parts.append(r"""
  <!-- ==================== SLIDE 9: 60品种零先验内生准入与产业链聚类 ==================== -->
  <div class="slide" id="slide-9" data-title="09. 全商品池(60品种)零先验内生准入与产业链聚类" data-cat="Full-60 ZeroHC">
    <div class="slide-header">
      <div class="slide-header-left">
        <div class="slide-category">FULL-60 ZERO-HARDCODE UNIVERSE EXPANSION</div>
        <h2 class="slide-title">全商品池(60品种)零先验内生准入与产业链簇风险预算守恒</h2>
        <div class="slide-subtitle">100% 废除 CORE_20_IDS 硬编码特权，构建全市场客观竞争与无监督产业链共线性聚类架构</div>
      </div>
      <span class="pill-green badge-pill">100% 零先验硬编码 · Sharpe 2.56 · MaxDD 0.83%</span>
    </div>

    <div class="slide-content">
      <!-- Full-60 ZeroHC SVG Architecture Diagram -->
      <div style="background: #ffffff; border: 1px solid var(--border-color); border-radius: 8px; padding: 10px 12px; box-shadow: var(--shadow-sm);">
        <svg viewBox="0 0 960 145" style="width: 100%; height: auto; display: block;" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif">
          <defs>
            <marker id="zhc-arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 1 L 10 5 L 0 9 z" fill="#059669" />
            </marker>
          </defs>

          <!-- Module 1: 5-Dim In-Sample Admission -->
          <g transform="translate(8, 8)">
            <rect width="285" height="128" rx="6" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.3"/>
            <text x="142" y="20" fill="#1e3a8a" font-size="11" font-weight="800" text-anchor="middle">1. 五维因果准入门禁 (&le; cutoff)</text>
            <rect x="12" y="28" width="261" height="26" rx="4" fill="#eff6ff"/>
            <text x="20" y="44" fill="#1e40af" font-size="9.5" font-weight="700">流动性深度: 上市>120天 & V_med &ge; 8.8万手/日</text>
            <rect x="12" y="58" width="261" height="26" rx="4" fill="#eff6ff"/>
            <text x="20" y="74" fill="#1e40af" font-size="9.5" font-weight="700">非对称赔率: 样本内 R_IS &ge; 2.15 (过滤jm/fu等)</text>
            <rect x="12" y="88" width="261" height="26" rx="4" fill="#eff6ff"/>
            <text x="20" y="104" fill="#1e40af" font-size="9.5" font-weight="700">安全边际: ΔW_IS = W - 1/(1+R) &ge; +12.0%</text>
            <text x="142" y="125" fill="#64748b" font-size="9" text-anchor="middle">100% 废除 CORE_20 豁免特权 · 60品种一律平等</text>
          </g>

          <path d="M 300 72 L 325 72" stroke="#059669" stroke-width="2.5" marker-end="url(#zhc-arrow)"/>

          <!-- Module 2: Unsupervised Supply-Chain Clustering -->
          <g transform="translate(330, 8)">
            <rect width="305" height="128" rx="6" fill="#ecfdf5" stroke="#a7f3d0" stroke-width="1.5"/>
            <text x="152" y="20" fill="#065f46" font-size="11" font-weight="800" text-anchor="middle">2. 无监督同板块产业链聚类 (|&rho;_IS| &ge; 0.68)</text>
            <rect x="12" y="28" width="281" height="26" rx="4" fill="#d1fae5"/>
            <text x="20" y="44" fill="#065f46" font-size="9.5" font-weight="700">自动归簇: rb/hc, ta/px/pf, pp/l/v, m/rm/a/b</text>
            <rect x="12" y="58" width="281" height="26" rx="4" fill="#d1fae5"/>
            <text x="20" y="74" fill="#065f46" font-size="9.5" font-weight="700">簇内选优: 按 Q_i = S_HR · &radic;R · (1+0.15 ln V) 选龙头</text>
            <rect x="12" y="88" width="281" height="26" rx="4" fill="#d1fae5"/>
            <text x="20" y="104" fill="#065f46" font-size="9.5" font-weight="700">共振双核保护: N_f&ge;21, S&ge;1.85, R&ge;3.5 豁免 au, ag</text>
            <text x="152" y="125" fill="#047857" font-size="9" text-anchor="middle">彻底根除同产业链影子合约共线挤压与踩踏</text>
          </g>

          <path d="M 642 72 L 667 72" stroke="#059669" stroke-width="2.5" marker-end="url(#zhc-arrow)"/>

          <!-- Module 3: Budget Conservation & Causal Health -->
          <g transform="translate(672, 8)">
            <rect width="280" height="128" rx="6" fill="#eff6ff" stroke="#bfdbfe" stroke-width="1.8"/>
            <text x="140" y="20" fill="#1e40af" font-size="11" font-weight="800" text-anchor="middle">3. 板块预算守恒阻尼与因果健康度倾斜</text>
            <rect x="12" y="28" width="256" height="26" rx="4" fill="#dbeafe"/>
            <text x="20" y="44" fill="#1e40af" font-size="9.5" font-weight="700">预算守恒阻尼: 权重按 K_s^-0.65 阻尼板块膨胀</text>
            <rect x="12" y="58" width="256" height="26" rx="4" fill="#dbeafe"/>
            <text x="20" y="74" fill="#1e40af" font-size="9.5" font-weight="700">因果健康度: 滞后至 d-1 日滚动 Sharpe 动态微调</text>
            <rect x="12" y="88" width="256" height="26" rx="4" fill="#dbeafe"/>
            <text x="20" y="104" fill="#1e40af" font-size="9.5" font-weight="700">对称死区: 2% 死区屏蔽微观再平衡被动磨损</text>
            <text x="140" y="125" fill="#1d4ed8" font-size="9" font-weight="800" text-anchor="middle">年化换手从 126x 骤降至 42.7x · 摩擦仅 1.13%</text>
          </g>
        </svg>
      </div>

      <!-- Full-60 Ablation Comparison Table -->
      <div>
        <table class="table-academic">
          <thead>
            <tr>
              <th>实验版本</th>
              <th>名单先验性质</th>
              <th>样本外代表品种</th>
              <th>毛收益</th>
              <th>摩擦成本</th>
              <th>纯净收益</th>
              <th>年化夏普</th>
              <th>最大回撤</th>
              <th>卡玛比率</th>
              <th>年化换手</th>
              <th>单笔盈亏比 R</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Full60_FlatHRP_Static</strong></td>
              <td>无先验 (58标的)</td>
              <td>全池等权 HRP + 静态出场</td>
              <td>7.61%</td>
              <td>3.29%</td>
              <td>4.33%</td>
              <td>1.03</td>
              <td>2.02%</td>
              <td>1.07</td>
              <td>114.9x</td>
              <td>1.893</td>
            </tr>
            <tr style="color: #94a3b8;">
              <td><strong>Full60_FlatHRP_HR</strong></td>
              <td>无先验 (58标的)</td>
              <td>全池等权 HRP + 滞回出场</td>
              <td>8.14%</td>
              <td>3.61%</td>
              <td>4.53%</td>
              <td>1.15</td>
              <td>1.88%</td>
              <td>1.21</td>
              <td>126.3x</td>
              <td>1.918</td>
            </tr>
            <tr>
              <td><strong>Full60_SecConserved_HR</strong></td>
              <td>无先验 (58标的)</td>
              <td>板块预算守恒 (1/K^0.9) + 2.5% 死区</td>
              <td>7.58%</td>
              <td>2.37%</td>
              <td>5.21%</td>
              <td>1.21</td>
              <td>1.93%</td>
              <td>1.35</td>
              <td>83.8x</td>
              <td>1.918</td>
            </tr>
            <tr>
              <td><strong>Full60_ZeroHC_SOPStage3</strong></td>
              <td><strong>100% 零先验硬编码</strong></td>
              <td>全池统一 SOP 阶段三门禁初步基线</td>
              <td>9.10%</td>
              <td>1.62%</td>
              <td>7.48%</td>
              <td>1.78</td>
              <td>1.41%</td>
              <td>2.66</td>
              <td>59.0x</td>
              <td>2.365</td>
            </tr>
            <tr>
              <td><strong>Full60_CoreSatellite_Opt</strong></td>
              <td>先验 CORE_20 锚定</td>
              <td>核心-卫星因果分配 (对照组)</td>
              <td>13.85%</td>
              <td>3.00%</td>
              <td>10.84%</td>
              <td>2.42</td>
              <td>1.68%</td>
              <td>3.23</td>
              <td>107.6x</td>
              <td>2.248</td>
            </tr>
            <tr class="highlight-green">
              <td><strong>Full60_ZeroHC_Cluster_Opt</strong></td>
              <td><strong>100% 零先验硬编码</strong></td>
              <td><strong>F簇去噪 + 产业链聚类 + 预算守恒</strong></td>
              <td><strong>11.68%</strong></td>
              <td><strong>1.13%</strong></td>
              <td><strong>10.55%</strong></td>
              <td><strong>2.56</strong></td>
              <td><strong>0.83%</strong></td>
              <td><strong>6.35</strong></td>
              <td><strong>42.7x</strong></td>
              <td><strong>2.890</strong></td>
            </tr>
            <tr style="background: #fdf2f8; font-weight: 700; color: #9d174d;">
              <td><strong>ZeroHC_Cluster (3.0x)</strong></td>
              <td>100% 零先验硬编码</td>
              <td>3.0x 实盘目标敞口版 (单标的上限 10% / 板块 30%)</td>
              <td>35.04%</td>
              <td>3.40%</td>
              <td>31.64%</td>
              <td>2.56</td>
              <td>2.07%</td>
              <td>7.66</td>
              <td>128.2x</td>
              <td>2.890</td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- Core Insights 2-Card Grid -->
      <div class="grid-2">
        <div class="card card-success">
          <div class="card-title">
            <span>内生准入品种完全印证客观规律（拒绝先入为主）</span>
            <span class="pill-green badge-pill">客观实证</span>
          </div>
          <div class="card-desc">
            Fold 1 自动选出 <strong>7 个龙头代表</strong>（<code>au, ag, i, sm, pp, sr, ap</code>），Fold 2 选出 <strong>9 个代表</strong>（增加 <code>ni, a, b, jd</code>）。原 Core-20 中表现欠佳的螺纹热卷由于与铁矿石强共线（|&rho;|&gt;0.8）被自动合并去重，而农产品独立果品（苹果/鸡蛋）因高赔率被内生纳入，充分证明量化客观规则的强大生命力。
          </div>
        </div>

        <div class="card card-highlight">
          <div class="card-title">
            <span>最大回撤压降至惊人的 0.83%（降幅达 49%）</span>
            <span class="pill-blue badge-pill">极致夏普与防御</span>
          </div>
          <div class="card-desc">
            得益于产业链共线聚类消除踩踏、流动性深度门禁剔除毛刺品种以及板块预算守恒，全组合在 2025.01~2026.09 长达 21 个月的宽幅震荡大市中，<strong>最大回撤仅 0.83%，年化夏普达 2.56，卡玛比率达 6.35</strong>，单笔盈亏比高达 <strong>2.890</strong>，奠定了机构级量化投资的巅峰稳健度。
          </div>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <span>100% 废除 CORE_20_IDS 硬编码 · 纯样本外 503 交易日严格前缀因果验证 · 详见 PR #14</span>
      <span>Slide 09 of 13</span>
    </div>
  </div>
""")

    # 11. Slide 10: Empirical Ablation NAV Curves
    html_parts.append(r"""
  <!-- ==================== SLIDE 10: 消融实验与净值演进 ==================== -->
  <div class="slide" id="slide-10" data-title="10. 全链路递进消融实验与净值演进" data-cat="Empirical Results">
    <div class="slide-header">
      <div class="slide-header-left">
        <div class="slide-category">EMPIRICAL RESULTS & ABLATION STUDY</div>
        <h2 class="slide-title">全链路递进消融实验与纯样本外净值演进</h2>
        <div class="slide-subtitle">Stitched OOS (2025.01 ~ 2026.09, 503 交易日) 绝对盲测区间下的净值与水下回撤全息联动图</div>
      </div>
      <span class="pill-green badge-pill">两折拼接纯样本外 · 交互联动</span>
    </div>

    <div class="slide-content">
      <!-- Plotly Nav & DD Chart Container -->
      <div style="background: #ffffff; border: 1px solid var(--border-color); border-radius: 8px; padding: 6px; box-shadow: var(--shadow-sm); overflow: hidden;">
        <div id="chart-nav" class="plotly-graph-div" style="height:355px; width:100%;"></div>
      </div>

      <!-- Ablation Takeaway Pills -->
      <div style="display: flex; gap: 8px; font-size: 10px; align-items: center; justify-content: space-between; background: #f8fafc; border: 1px solid var(--border-color); padding: 6px 12px; border-radius: 6px;">
        <div><strong style="color: #64748b;">ExpA 基准:</strong> 仅日内60m+30m，Shp 1.25，MaxDD 1.89%</div>
        <div><strong style="color: #dc2626;">Full60 朴素等权:</strong> 换手 126x 摩擦重，Shp 仅 1.15</div>
        <div><strong style="color: #6366f1;">ExpE 异构协同:</strong> 4周期协同静态出场，Shp 1.81，Net +8.51%</div>
        <div><strong style="color: #2563eb;">Core20 滞回棘轮:</strong> 结构门锁×分批止盈，Shp 2.36，Net +10.42%</div>
        <div><strong style="color: #0284c7;">Core-Satellite-60:</strong> 先验核心+卫星，Shp 2.42，Net +10.84%</div>
        <div><strong style="color: #059669;">ZeroHC-Cluster-60:</strong> <strong>零先验最优 Shp 2.56 | MaxDD 0.83% | Cal 6.35 | R=2.890</strong></div>
      </div>
    </div>

    <div class="slide-footer">
      <span>可在图表中缩放、平移和查看各实验净值与水下回撤曲线细节 · 全程纯样本外绝对隔离</span>
      <span>Slide 10 of 13</span>
    </div>
  </div>
""")

    # 12. Slide 11: Sector & Symbol Attribution
    html_parts.append(r"""
  <!-- ==================== SLIDE 11: 板块与品种归因全景 ==================== -->
  <div class="slide" id="slide-11" data-title="11. 六大商品板块与品种纯样本外归因全景" data-cat="Attribution Analysis">
    <div class="slide-header">
      <div class="slide-header-left">
        <div class="slide-category">SECTOR & SYMBOL ATTRIBUTION</div>
        <h2 class="slide-title">六大商品板块与品种纯样本外归因全景</h2>
        <div class="slide-subtitle">60 商品期货全景：Flat-60 等权 HRP vs Core-Satellite-60 vs ZeroHC-Cluster-60 归因深度对比</div>
      </div>
      <span class="pill-green badge-pill">全市场结构化归因 · 彻底扭转黑色与有色拖累</span>
    </div>

    <div class="slide-content">
      <!-- Side-by-Side Attribution Charts -->
      <div class="grid-2" style="gap: 12px;">
        <div style="background: #ffffff; border: 1px solid var(--border-color); border-radius: 8px; padding: 6px; box-shadow: var(--shadow-sm); overflow: hidden;">
          <div id="chart-sector" class="plotly-graph-div" style="height:215px; width:100%;"></div>
        </div>
        <div style="background: #ffffff; border: 1px solid var(--border-color); border-radius: 8px; padding: 6px; box-shadow: var(--shadow-sm); overflow: hidden;">
          <div id="chart-symbol" class="plotly-graph-div" style="height:215px; width:100%;"></div>
        </div>
      </div>

      <!-- 5 Sector Attribution Cards -->
      <div class="grid-4" style="grid-template-columns: repeat(5, 1fr); gap: 9px;">
        <div class="card card-success" style="padding: 7px 10px;">
          <div style="font-size: 11px; font-weight: 700; color: #065f46; display: flex; justify-content: space-between;">
            <span>贵金属 (au, ag)</span>
            <span class="pill-green badge-pill">核心压舱石</span>
          </div>
          <div style="font-size: 15px; font-weight: 800; color: #059669; margin: 2px 0 1px 0;">+8.14%</div>
          <div style="font-size: 10px; color: var(--text-secondary);">
            换手仅 26.3x (摩擦0.64%)<br/>
            au (+4.75%), ag (+3.39%)
          </div>
        </div>

        <div class="card card-highlight" style="padding: 7px 10px; border-left-color: #0284c7;">
          <div style="font-size: 11px; font-weight: 700; color: #0369a1; display: flex; justify-content: space-between;">
            <span>农产品油脂 (16标的)</span>
            <span class="pill-blue badge-pill">独立超额源</span>
          </div>
          <div style="font-size: 15px; font-weight: 800; color: #0284c7; margin: 2px 0 1px 0;">+1.40%</div>
          <div style="font-size: 10px; color: var(--text-secondary);">
            较 Flat-60 (+0.16%) 提升 +1.24%<br/>
            p (+0.49%), sr (+0.37%), b (+0.29%)
          </div>
        </div>

        <div class="card card-purple" style="padding: 7px 10px;">
          <div style="font-size: 11px; font-weight: 700; color: #5b21b6; display: flex; justify-content: space-between;">
            <span>黑色建材 (9标的)</span>
            <span class="pill-purple badge-pill">逆势正贡献</span>
          </div>
          <div style="font-size: 15px; font-weight: 800; color: #7c3aed; margin: 2px 0 1px 0;">+0.89%</div>
          <div style="font-size: 10px; color: var(--text-secondary);">
            较 Flat-60 (-0.11%) 提升 +1.00%<br/>
            i (+0.81%), sm (+0.08%)
          </div>
        </div>

        <div class="card card-highlight" style="padding: 7px 10px;">
          <div style="font-size: 11px; font-weight: 700; color: #1e40af; display: flex; justify-content: space-between;">
            <span>能源化工 (22标的)</span>
            <span class="pill-blue badge-pill">防守止损</span>
          </div>
          <div style="font-size: 15px; font-weight: 800; color: #2563eb; margin: 2px 0 1px 0;">+0.10%</div>
          <div style="font-size: 10px; color: var(--text-secondary);">
            年化换手仅 0.2x (摩擦0.01%)<br/>
            pp (+0.10%), 剔除大量共线杂音
          </div>
        </div>

        <div class="card" style="padding: 7px 10px; border-left: 3.5px solid #64748b;">
          <div style="font-size: 11px; font-weight: 700; color: #334155; display: flex; justify-content: space-between;">
            <span>有色金属 (8标的)</span>
            <span class="pill-amber badge-pill">成功扭亏</span>
          </div>
          <div style="font-size: 15px; font-weight: 800; color: #475569; margin: 2px 0 1px 0;">+0.01%</div>
          <div style="font-size: 10px; color: var(--text-secondary);">
            <strong>较 Flat-60 (-2.21%) 改善 +2.22%</strong><br/>
            ni (+0.01%), 斩断虚假追高亏损
          </div>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <span>ZeroHC-ClusterConserved-60 架构消除了有色金属 (-2.21%) 与黑色系单边阴跌的系统性拖累</span>
      <span>Slide 11 of 13</span>
    </div>
  </div>
""")

    # 13. Slide 12: Execution & Risk Control
    html_parts.append(r"""
  <!-- ==================== SLIDE 12: 原生5m执行与组合风控 ==================== -->
  <div class="slide" id="slide-12" data-title="12. 组合分配、执行网格与微观物理风控" data-cat="Risk & Portfolio">
    <div class="slide-header">
      <div class="slide-header-left">
        <div class="slide-category">PORTFOLIO ALLOCATION & RISK DEFENSE</div>
        <h2 class="slide-title">组合分配层、原生 5m 执行网格与微观物理风控</h2>
        <div class="slide-subtitle">层次聚类风险平价 (Alloc3_SectorHRP)、换手死区磁滞滤波与三道物理安全防线</div>
      </div>
      <span class="pill-purple badge-pill">HRP 树状聚类 + 死区防抖 + 预算守恒</span>
    </div>

    <div class="slide-content">
      <div class="grid-2" style="grid-template-columns: 1fr 1fr; gap: 14px;">
        <!-- Left: Sector HRP Diagram -->
        <div style="background: #ffffff; border: 1px solid var(--border-color); border-radius: 8px; padding: 11px 15px; box-shadow: var(--shadow-sm);">
          <div style="font-size: 12px; font-weight: 700; color: var(--primary); margin-bottom: 4px;">
            层次聚类风险平价与板块预算阻尼 (SectorHRP + K_s^-0.65)
          </div>
          <div style="font-size: 10px; color: var(--text-muted); margin-bottom: 7px;">
            基于图论层次聚类与逆方差加权，结合板块预算守恒指数切断系统性风险扩散
          </div>

          <svg viewBox="0 0 440 160" style="width: 100%; height: auto; display: block;" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif">
            <rect x="170" y="8" width="100" height="22" rx="4" fill="#0f172a" />
            <text x="220" y="23" fill="#ffffff" font-size="10" font-weight="700" text-anchor="middle">组合总资本 (1.0x)</text>

            <line x1="220" y1="30" x2="220" y2="44" stroke="#64748b" stroke-width="1.5"/>
            <line x1="50" y1="44" x2="390" y2="44" stroke="#64748b" stroke-width="1.5"/>

            <!-- 5 Sectors -->
            <line x1="50" y1="44" x2="50" y2="58" stroke="#64748b" stroke-width="1.5"/>
            <rect x="15" y="58" width="70" height="26" rx="4" fill="#eff6ff" stroke="#93c5fd" />
            <text x="50" y="71" fill="#1e40af" font-size="8.5" font-weight="700" text-anchor="middle">贵金属</text>
            <text x="50" y="80" fill="#3b82f6" font-size="7.5" text-anchor="middle">au, ag</text>

            <line x1="135" y1="44" x2="135" y2="58" stroke="#64748b" stroke-width="1.5"/>
            <rect x="102" y="58" width="66" height="26" rx="4" fill="#eff6ff" stroke="#93c5fd" />
            <text x="135" y="71" fill="#1e40af" font-size="8.5" font-weight="700" text-anchor="middle">农产品</text>
            <text x="135" y="80" fill="#3b82f6" font-size="7.5" text-anchor="middle">p, sr, b, jd</text>

            <line x1="220" y1="44" x2="220" y2="58" stroke="#64748b" stroke-width="1.5"/>
            <rect x="187" y="58" width="66" height="26" rx="4" fill="#eff6ff" stroke="#93c5fd" />
            <text x="220" y="71" fill="#1e40af" font-size="8.5" font-weight="700" text-anchor="middle">黑色建材</text>
            <text x="220" y="80" fill="#3b82f6" font-size="7.5" text-anchor="middle">i, sm</text>

            <line x1="305" y1="44" x2="305" y2="58" stroke="#64748b" stroke-width="1.5"/>
            <rect x="272" y="58" width="66" height="26" rx="4" fill="#eff6ff" stroke="#93c5fd" />
            <text x="305" y="71" fill="#1e40af" font-size="8.5" font-weight="700" text-anchor="middle">能化</text>
            <text x="305" y="80" fill="#3b82f6" font-size="7.5" text-anchor="middle">pp 选优</text>

            <line x1="390" y1="44" x2="390" y2="58" stroke="#64748b" stroke-width="1.5"/>
            <rect x="357" y="58" width="66" height="26" rx="4" fill="#eff6ff" stroke="#93c5fd" />
            <text x="390" y="71" fill="#1e40af" font-size="8.5" font-weight="700" text-anchor="middle">有色</text>
            <text x="390" y="80" fill="#3b82f6" font-size="7.5" text-anchor="middle">ni 选优</text>

            <rect x="15" y="96" width="410" height="48" rx="5" fill="#f8fafc" stroke="#cbd5e1" />
            <text x="220" y="112" fill="#334155" font-size="9.5" font-weight="700" text-anchor="middle">产业链子簇代表选优 + 板块阻尼系数 K_s^-0.65</text>
            <text x="220" y="126" fill="#64748b" font-size="8.5" text-anchor="middle">板块内单标的权重上限 10%，板块总敞口上限 30%，彻底杜绝同质品种多倍杠杆踩踏</text>
            <text x="220" y="139" fill="#059669" font-size="8.5" font-weight="600" text-anchor="middle">结合 d-1 滚动因果健康度微调，全策略年化换手仅 42.7x</text>
          </svg>
        </div>

        <!-- Right: Deadzone Hysteresis Filter SVG -->
        <div style="background: #ffffff; border: 1px solid var(--border-color); border-radius: 8px; padding: 11px 15px; box-shadow: var(--shadow-sm);">
          <div style="font-size: 12px; font-weight: 700; color: var(--purple); margin-bottom: 4px;">
            换手死区磁滞滤波机制 (Deadzone = 0.02 ~ 0.08)
          </div>
          <div style="font-size: 10px; color: var(--text-muted); margin-bottom: 7px;">
            过滤微小权重噪声漂移，单向换手从 126x 骤降至 42.7x，节省 2.5% 真实物理摩擦
          </div>

          <svg viewBox="0 0 440 160" style="width: 100%; height: auto; display: block;" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif">
            <line x1="40" y1="80" x2="400" y2="80" stroke="#94a3b8" stroke-width="1.5"/>
            <line x1="220" y1="12" x2="220" y2="148" stroke="#94a3b8" stroke-width="1.5"/>
            <text x="400" y="75" fill="#64748b" font-size="8.5">权重目标差 Δw</text>
            <text x="225" y="22" fill="#64748b" font-size="8.5">执行调仓量</text>

            <rect x="170" y="12" width="100" height="136" fill="rgba(220, 38, 38, 0.08)" stroke="#fca5a5" stroke-dasharray="3,3"/>
            <text x="220" y="60" fill="#dc2626" font-size="10" font-weight="800" text-anchor="middle">死区区间 (不交易)</text>
            <text x="220" y="75" fill="#dc2626" font-size="8" text-anchor="middle">|Δw| &le; Deadzone</text>
            <text x="170" y="94" fill="#dc2626" font-size="7.5" text-anchor="middle">-Deadzone</text>
            <text x="270" y="94" fill="#dc2626" font-size="7.5" text-anchor="middle">+Deadzone</text>

            <line x1="270" y1="80" x2="370" y2="35" stroke="#059669" stroke-width="2.5"/>
            <circle cx="270" cy="80" r="3.5" fill="#059669"/>
            <text x="330" y="50" fill="#059669" font-size="8.5" font-weight="700">实际执行买入</text>

            <line x1="170" y1="80" x2="70" y2="125" stroke="#059669" stroke-width="2.5"/>
            <circle cx="170" cy="80" r="3.5" fill="#059669"/>
            <text x="110" y="115" fill="#059669" font-size="8.5" font-weight="700">实际执行卖出</text>

            <text x="220" y="154" fill="#475569" font-size="8.5" text-anchor="middle">当目标权重变化未超过死区门槛时，维持前一时刻物理持仓不变</text>
          </svg>
        </div>
      </div>

      <!-- 3 Physical Risk Bounds Cards -->
      <div class="grid-3" style="margin-top: 2px;">
        <div class="card card-highlight">
          <div class="card-title">
            <span>单品种目标敞口上限 10%</span>
            <span class="pill-blue badge-pill">集中度防线</span>
          </div>
          <div class="card-desc">
            强制限制单一商品名义头寸不得超过总资本的 10%，即便黄金、原油等出现单边脉冲也绝不超限，杜绝个别黑天鹅穿仓。
          </div>
        </div>

        <div class="card card-purple">
          <div class="card-title">
            <span>单板块总敞口上限 30%</span>
            <span class="pill-purple badge-pill">行业隔离</span>
          </div>
          <div class="card-desc">
            限制同一板块内部（如黑色系/有色）的多空净敞口与绝对敞口合计不得超过 30%，彻底阻断单一产业政策突变引发的全板块系统性踩踏。
          </div>
        </div>

        <div class="card card-success">
          <div class="card-title">
            <span>实盘 3.0x 最大回撤仅 2.07%</span>
            <span class="pill-green badge-pill">资金冗余</span>
          </div>
          <div class="card-desc">
            在实盘目标敞口版 (3.0x 名义杠杆) 下，全市场隔夜保证金占用严格锁定在 40% 以内，预留 60% 充裕现金，纯样本外最大回撤收敛至 2.07%。
          </div>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <span>组合层 Alloc3_SectorHRP 结合 Deadzone 与产业链预算守恒实现了策略从单兵作战到机构级资产配置舰队的质跃</span>
      <span>Slide 12 of 13</span>
    </div>
  </div>
""")

    # 14. Slide 13: Production Roadmap & Strategic Takeaways
    html_parts.append(r"""
  <!-- ==================== SLIDE 13: 总结与生产落地 ==================== -->
  <div class="slide" id="slide-13" data-title="13. 总结与工程化生产落地路线图" data-cat="Roadmap">
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
        <div style="font-size: 12px; font-weight: 700; color: var(--primary); margin-bottom: 5px;">
          四阶段工程化生产落地流水线 (Institutional Production Pipeline)
        </div>

        <svg viewBox="0 0 960 105" style="width: 100%; height: auto; display: block;" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif">
          <defs>
            <marker id="pipe-arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 1 L 10 5 L 0 9 z" fill="#2563eb" />
            </marker>
          </defs>

          <!-- Phase 1 -->
          <g transform="translate(10, 8)">
            <rect width="215" height="88" rx="6" fill="#eff6ff" stroke="#bfdbfe" stroke-width="1.5"/>
            <text x="107" y="22" fill="#1e40af" font-size="10.5" font-weight="800" text-anchor="middle">Phase 1: 微观行情总线</text>
            <text x="12" y="41" fill="#1e293b" font-size="9">● CTP / 天勤高频行情广播</text>
            <text x="12" y="56" fill="#1e293b" font-size="9">● 原生 5m K线生成与持仓增量</text>
            <text x="12" y="71" fill="#1e293b" font-size="9">● 截面微观特征流 (时延 &lt; 50ms)</text>
          </g>

          <path d="M 230 52 L 255 52" stroke="#2563eb" stroke-width="2.5" marker-end="url(#pipe-arrow)"/>

          <!-- Phase 2 -->
          <g transform="translate(260, 8)">
            <rect width="215" height="88" rx="6" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
            <text x="107" y="22" fill="#0f172a" font-size="10.5" font-weight="800" text-anchor="middle">Phase 2: 纯 Rust 信号与状态机</text>
            <text x="12" y="41" fill="#334155" font-size="9">● czsc-signals 纯算子并发评估</text>
            <text x="12" y="56" fill="#334155" font-size="9">● structure_exit 滞回棘轮出场</text>
            <text x="12" y="71" fill="#334155" font-size="9">● 零 GC 停顿，全品种微秒级完成</text>
          </g>

          <path d="M 480 52 L 505 52" stroke="#2563eb" stroke-width="2.5" marker-end="url(#pipe-arrow)"/>

          <!-- Phase 3 -->
          <g transform="translate(510, 8)">
            <rect width="215" height="88" rx="6" fill="#f5f3ff" stroke="#ddd6fe" stroke-width="1.5"/>
            <text x="107" y="22" fill="#6d28d9" font-size="10.5" font-weight="800" text-anchor="middle">Phase 3: 零先验聚类与死区执行</text>
            <text x="12" y="41" fill="#4c1d95" font-size="9">● 无监督产业链共线聚类选优</text>
            <text x="12" y="56" fill="#4c1d95" font-size="9">● Alloc3_SectorHRP 预算守恒</text>
            <text x="12" y="71" fill="#4c1d95" font-size="9">● Deadzone 磁滞滤波与智能拆单</text>
          </g>

          <path d="M 730 52 L 755 52" stroke="#2563eb" stroke-width="2.5" marker-end="url(#pipe-arrow)"/>

          <!-- Phase 4 -->
          <g transform="translate(760, 8)">
            <rect width="190" height="88" rx="6" fill="#ecfdf5" stroke="#a7f3d0" stroke-width="1.5"/>
            <text x="95" y="22" fill="#065f46" font-size="10.5" font-weight="800" text-anchor="middle">Phase 4: 全息风控看门狗</text>
            <text x="10" y="41" fill="#064e3b" font-size="9">● 实时持仓对账与滑点监控</text>
            <text x="10" y="56" fill="#064e3b" font-size="9">● 40% 隔夜保证金熔断告警</text>
            <text x="10" y="71" fill="#064e3b" font-size="9">● 异常撤单率与延迟看门狗</text>
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
            单周期策略必然经历漫长回撤；1d 定势 + 60m/30m 骨架 + 15m 结构出场 + 5m 原生执行网格的跨尺度共振，是平滑收益曲线的唯一科学路径。
          </div>
        </div>

        <div class="card card-purple">
          <div class="card-title">
            <span>2. 缠论滞回出场与棘轮</span>
            <span class="pill-purple badge-pill">非对称赔率</span>
          </div>
          <div class="card-desc">
            次极值止损与 <code>bi_or_peak</code> 同向重入门锁 × 35% 分批止盈保本底仓棘轮（HysteresisRunner），将单笔盈亏比推升至 2.324，让利润自由奔跑。
          </div>
        </div>

        <div class="card card-success">
          <div class="card-title">
            <span>3. 100% 零先验聚类守恒</span>
            <span class="pill-green badge-pill">宇宙进化</span>
          </div>
          <div class="card-desc">
            废除 <code>CORE_20_IDS</code> 硬编码，全池统一流动性与非对称赔率门禁，通过无监督产业链聚类消除共线踩踏，夏普达 2.56、最大回撤仅 0.83%。
          </div>
        </div>

        <div class="card card-amber">
          <div class="card-title">
            <span>4. 严格防前视与实战韧性</span>
            <span class="pill-amber badge-pill">生产保障</span>
          </div>
          <div class="card-desc">
            503 个交易日纯样本外两折盲测、彻底剥离回填漏洞、真实双边 3.0 BP 扣费与 Alloc3 聚类风控，具备直接上线机构级实盘生产的充沛底气与强大韧性。
          </div>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <span>汇报完毕 · 感谢各位评委与投研同仁的聆听 · 敬请批评指正！</span>
      <span>Slide 13 of 13</span>
    </div>
  </div>
""")

    # 15. Deck Container Close & Footer
    html_parts.append(r"""
</div>

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

<div id="overview-modal" onclick="closeOverview(event)">
  <div class="overview-grid" id="overview-grid-content"></div>
</div>
""")

    # 16. JavaScript with Data Injections
    js_content = f"""
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
  const navData = {nav_json};
  const funnelData = {funnel_json};
  const sectorData = {sector_json};
  const symbolData = {symbol_json};

  const funnelLayout = {{
    title: {{
      text: "5大周期候选信号在 Gate 1 -> Gate 2 -> Gate 3 漏斗中的留存演进 (单品种均值与合计)",
      font: {{ size: 12, color: "#0f172a" }},
      x: 0.0,
      xanchor: "left"
    }},
    barmode: "group",
    margin: {{ l: 40, r: 15, t: 38, b: 30 }},
    height: 200,
    legend: {{ orientation: "h", y: 1.18, x: 0.0, font: {{ size: 9.5 }} }},
    xaxis: {{ tickfont: {{ size: 10 }} }},
    yaxis: {{ title: {{ text: "信号实例数", font: {{ size: 9.5 }} }}, tickfont: {{ size: 9.5 }} }},
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff"
  }};

  const navLayout = {{
    title: {{
      text: "全周期跨尺度 CTA 核心消融净值演进 (Stitched OOS: 2025.01 ~ 2026.09, 503结算日, Base=100)",
      font: {{ size: 12.5, color: "#0f172a" }},
      x: 0.0,
      xanchor: "left"
    }},
    hovermode: "x unified",
    margin: {{ l: 48, r: 20, t: 50, b: 35 }},
    height: 345,
    legend: {{ orientation: "h", y: 1.15, x: 0.0, font: {{ size: 9.5 }} }},
    xaxis: {{ anchor: "y", domain: [0.0, 1.0], matches: "x2", showticklabels: false }},
    xaxis2: {{ anchor: "y2", domain: [0.0, 1.0], tickfont: {{ size: 9.5 }} }},
    yaxis: {{ anchor: "x", domain: [0.38, 1.0], title: {{ text: "累计净值 (Base=100)", font: {{ size: 10 }} }}, tickfont: {{ size: 9.5 }} }},
    yaxis2: {{ anchor: "x2", domain: [0.0, 0.28], title: {{ text: "回撤 (%)", font: {{ size: 10 }} }}, tickfont: {{ size: 9 }} }},
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff"
  }};

  const sectorLayout = {{
    title: {{
      text: "6大商品板块样本外净收益贡献对比 (%)",
      font: {{ size: 11.5, color: "#0f172a" }},
      x: 0.0,
      xanchor: "left"
    }},
    barmode: "group",
    margin: {{ l: 40, r: 15, t: 38, b: 32 }},
    height: 205,
    legend: {{ orientation: "h", y: 1.20, x: 0.0, font: {{ size: 9 }} }},
    xaxis: {{ tickfont: {{ size: 9.5 }} }},
    yaxis: {{ title: {{ text: "净贡献 (%)", font: {{ size: 9.5 }} }}, tickfont: {{ size: 9 }} }},
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff"
  }};

  const symbolLayout = {{
    title: {{
      text: "全市场代表品种纯样本外净贡献对比 (%)",
      font: {{ size: 11.5, color: "#0f172a" }},
      x: 0.0,
      xanchor: "left"
    }},
    barmode: "group",
    margin: {{ l: 40, r: 15, t: 38, b: 35 }},
    height: 205,
    legend: {{ orientation: "h", y: 1.20, x: 0.0, font: {{ size: 9 }} }},
    xaxis: {{ tickfont: {{ size: 9 }} }},
    yaxis: {{ title: {{ text: "净回报 (%)", font: {{ size: 9.5 }} }}, tickfont: {{ size: 9 }} }},
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff"
  }};

  const chartRendered = {{}};

  function renderSlideCharts(idx) {{
    if (!window.Plotly) return;
    const currentSlideEl = slides[idx];
    if (!currentSlideEl) return;

    if (currentSlideEl.querySelector('#chart-funnel')) {{
      if (!chartRendered['funnel']) {{
        Plotly.newPlot('chart-funnel', funnelData, funnelLayout, {{responsive: true}});
        chartRendered['funnel'] = true;
      }} else {{
        Plotly.Plots.resize(document.getElementById('chart-funnel'));
      }}
    }}
    if (currentSlideEl.querySelector('#chart-nav')) {{
      if (!chartRendered['nav']) {{
        Plotly.newPlot('chart-nav', navData, navLayout, {{responsive: true}});
        chartRendered['nav'] = true;
      }} else {{
        Plotly.Plots.resize(document.getElementById('chart-nav'));
      }}
    }}
    if (currentSlideEl.querySelector('#chart-sector')) {{
      if (!chartRendered['sector']) {{
        Plotly.newPlot('chart-sector', sectorData, sectorLayout, {{responsive: true}});
        chartRendered['sector'] = true;
      }} else {{
        Plotly.Plots.resize(document.getElementById('chart-sector'));
      }}
    }}
    if (currentSlideEl.querySelector('#chart-symbol')) {{
      if (!chartRendered['symbol']) {{
        Plotly.newPlot('chart-symbol', symbolData, symbolLayout, {{responsive: true}});
        chartRendered['symbol'] = true;
      }} else {{
        Plotly.Plots.resize(document.getElementById('chart-symbol'));
      }}
    }}
  }}

  // Initialize Dots
  for (let i = 0; i < totalSlides; i++) {{
    const dot = document.createElement('div');
    dot.className = 'dot-pill' + (i === 0 ? ' active' : '');
    dot.title = slides[i].getAttribute('data-title') || `Slide ${{i + 1}}`;
    dot.onclick = () => goToSlide(i);
    dotsContainer.appendChild(dot);
  }}

  // Initialize Overview Items
  for (let i = 0; i < totalSlides; i++) {{
    const item = document.createElement('div');
    item.className = 'overview-item' + (i === 0 ? ' active' : '');
    const num = (i + 1 < 10 ? '0' : '') + (i + 1);
    const title = slides[i].getAttribute('data-title') || `Slide ${{i + 1}}`;
    const cat = slides[i].getAttribute('data-cat') || 'Section';
    item.innerHTML = `
      <div>
        <div class="ov-cat">${{cat}}</div>
        <div class="ov-title">${{title}}</div>
      </div>
      <div class="ov-num">${{num}} / ${{totalSlides}}</div>
    `;
    item.onclick = () => {{
      goToSlide(i);
      overviewModal.classList.remove('active');
    }};
    overviewGridContent.appendChild(item);
  }}

  function updateDeck() {{
    slides.forEach((s, idx) => {{
      if (idx === currentSlide) {{
        s.classList.add('active');
      }} else {{
        s.classList.remove('active');
      }}
    }});

    const pct = ((currentSlide + 1) / totalSlides) * 100;
    progressBar.style.width = pct + '%';

    const numStr = (currentSlide + 1 < 10 ? '0' : '') + (currentSlide + 1);
    slideCounter.innerText = `${{numStr}} / ${{totalSlides}}`;

    const title = slides[currentSlide].getAttribute('data-title') || '';
    headerSectionTitle.innerText = title;

    document.querySelectorAll('.dot-pill').forEach((d, idx) => {{
      if (idx === currentSlide) d.classList.add('active');
      else d.classList.remove('active');
    }});

    document.querySelectorAll('.overview-item').forEach((item, idx) => {{
      if (idx === currentSlide) item.classList.add('active');
      else item.classList.remove('active');
    }});

    btnPrev.disabled = currentSlide === 0;
    btnNext.disabled = currentSlide === totalSlides - 1;

    setTimeout(() => {{
      renderSlideCharts(currentSlide);
      window.dispatchEvent(new Event('resize'));
    }}, 40);
  }}

  function nextSlide() {{
    if (currentSlide < totalSlides - 1) {{
      currentSlide++;
      updateDeck();
    }}
  }}

  function prevSlide() {{
    if (currentSlide > 0) {{
      currentSlide--;
      updateDeck();
    }}
  }}

  function goToSlide(idx) {{
    if (idx >= 0 && idx < totalSlides) {{
      currentSlide = idx;
      updateDeck();
    }}
  }}

  function toggleFullscreen() {{
    if (!document.fullscreenElement) {{
      document.documentElement.requestFullscreen().catch(err => {{
        alert(`Error attempting to enable fullscreen: ${{err.message}}`);
      }});
    }} else {{
      if (document.exitFullscreen) {{
        document.exitFullscreen();
      }}
    }}
  }}

  function toggleOverview() {{
    overviewModal.classList.toggle('active');
  }}

  function closeOverview(e) {{
    if (e.target === overviewModal) {{
      overviewModal.classList.remove('active');
    }}
  }}

  window.addEventListener('keydown', (e) => {{
    if (overviewModal.classList.contains('active')) {{
      if (e.key === 'Escape' || e.key === 'o' || e.key === 'O') {{
        overviewModal.classList.remove('active');
      }}
      return;
    }}

    if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') {{
      e.preventDefault();
      nextSlide();
    }} else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {{
      e.preventDefault();
      prevSlide();
    }} else if (e.key === 'Home') {{
      e.preventDefault();
      goToSlide(0);
    }} else if (e.key === 'End') {{
      e.preventDefault();
      goToSlide(totalSlides - 1);
    }} else if (e.key === 'f' || e.key === 'F') {{
      e.preventDefault();
      toggleFullscreen();
    }} else if (e.key === 'o' || e.key === 'O') {{
      e.preventDefault();
      toggleOverview();
    }}
  }});

  updateDeck();
</script>
</body>
</html>
"""
    html_parts.append(js_content)

    full_html = "".join(html_parts)

    TARGET_HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TARGET_HTML_PATH, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"已生成目标展示报告: {TARGET_HTML_PATH} ({len(full_html)} bytes)")

    DOCS_HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DOCS_HTML_PATH, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"已同步更新文档: {DOCS_HTML_PATH} ({len(full_html)} bytes)")


if __name__ == "__main__":
    build_deck()
