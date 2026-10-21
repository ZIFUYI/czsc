"""Recompute and render daily cross-sectional momentum with executable prices and fees."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_advanced_cta_momentum_strategy import (  # noqa: E402
    DATA_DIR,
    OUTPUT_DIR,
    load_all_daily_data,
    summarize_execution,
)


def build_momentum_targets(
    symbol_data: dict, k: int | None = 3, leverage: float = 1.0, warmup_days: int = 25
) -> pd.DataFrame:
    """Rank completed-day slopes; orders first become executable at each asset's next open."""
    slopes = pd.DataFrame({symbol: frame.slope for symbol, frame in symbol_data.items()}).sort_index()
    targets = pd.DataFrame(0.0, index=slopes.index[warmup_days:], columns=slopes.columns)
    for date in targets.index:
        values = slopes.loc[date].dropna().sort_values(ascending=False, kind="stable")
        if k is None:
            if len(values):
                targets.loc[date, values.index] = np.sign(values) * leverage / len(values)
        elif len(values) >= max(20, 2 * k):
            targets.loc[date, values.head(k).index] = leverage / (2 * k)
            targets.loc[date, values.tail(k).index] = -leverage / (2 * k)
    return targets


def run_momentum_backtest(
    symbol_data: dict, k: int | None = 3, leverage: float = 1.0, fee_rate: float = 0.0001, warmup_days: int = 25
) -> dict:
    """Run the original Top-6, Top-10 or all-market rules on the shared Rust ledger."""
    targets = build_momentum_targets(symbol_data, k, leverage, warmup_days)
    result, _ = summarize_execution(symbol_data, targets, fee_rate)
    result["parameters"] = {"k_per_side": k, "leverage": leverage, "warmup_days": warmup_days}
    return result


def render_dashboard(results: dict, output_path: Path) -> None:
    """Render only observed metrics; keep all output inside the caller's requested path."""
    figure = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], subplot_titles=["扣费后固定本金净值", "回撤（%）"]
    )
    rows = []
    for name, result in results.items():
        figure.add_trace(go.Scatter(x=result["dates"], y=result["nav"], name=name, mode="lines"), row=1, col=1)
        figure.add_trace(
            go.Scatter(x=result["dates"], y=result["dd"], name=name, mode="lines", showlegend=False), row=2, col=1
        )
        rows.append(
            {
                "方案": name,
                "累计净收益%": result["tot_ret"],
                "年化净收益%": result["ann_ret"],
                "夏普": result["sharpe"],
                "最大回撤%": result["max_dd"],
                "费用占初始本金%": result["fee_return_pct"],
                "交易日": result["trading_days"],
            }
        )
    figure.update_layout(height=850, template="plotly_white", title="商品期货日线截面动量：修正后历史重放")
    table = pd.DataFrame(rows).to_html(index=False, float_format=lambda value: f"{value:.3f}")
    document = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>商品截面动量审计重放</title>
<style>body{{font-family:system-ui;margin:28px;background:#f8fafc;color:#172033}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:12px;border:1px solid #d5dce5;text-align:right}}p{{line-height:1.7;max-width:1000px}}</style>
<h1>商品期货日线截面动量：修正后历史重放</h1>
<p>15分钟起点时间先转换为收盘时间，再按期货交易日聚合。每日收盘完成斜率排序，下一实际开盘执行；末端收盘清仓。
固定初始本金核算包含按成交额计收的单边1 BP费用，不把名义杠杆换算为保证金保证。
本页为历史连续合约参考价重放，未验证真实逐合约换月成交，也不是独立盲测结果。</p>
{table}{figure.to_html(full_html=False, include_plotlyjs=True)}
<p>输出：{html.escape(str(output_path))}</p></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    data = load_all_daily_data(args.data_dir)
    results = {
        "全市场方向等权1x": run_momentum_backtest(data, k=None, leverage=1),
        "Top6_1x": run_momentum_backtest(data, k=3, leverage=1),
        "Top6_3x": run_momentum_backtest(data, k=3, leverage=3),
        "Top10_3x": run_momentum_backtest(data, k=5, leverage=3),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "top_momentum_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    render_dashboard(results, args.output_dir / "top_momentum_dashboard.html")
    for name, result in results.items():
        print(
            f"{name}: return={result['tot_ret']:.3f}% Sharpe={result['sharpe']:.3f} maxDD={result['max_dd']:.3f}% fees={result['fee_return_pct']:.3f}% days={result['trading_days']}"
        )


if __name__ == "__main__":
    main()
