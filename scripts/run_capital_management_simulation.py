import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cta_execution import assign_trading_date, replay_allocated_execution  # noqa: E402
from scripts.cta_paths import research_output_path  # noqa: E402


def causal_capital_leverage(returns, target_vol=0.10):
    """Use completed daily observations; the first five decisions retain a fixed prior."""
    volatility = pd.Series(returns, dtype=float).rolling(20, min_periods=5).std().shift(1) * np.sqrt(252)
    return (target_vol / volatility.replace(0.0, np.nan)).clip(1.0, 7.0).fillna(4.0)


def run_simulation(*, input_path=None, output_path=None):
    # 读取已计算好的组合日度收益率序列
    weight_json_path = research_output_path("portfolio_weighting_comparison.json", input_path)
    with open(weight_json_path, encoding="utf-8") as f:
        wdata = json.load(f)

    if wdata.get("accounting") != "fixed_initial_capital_next_open" or "execution_ledger" not in wdata:
        raise ValueError("Regenerate portfolio weighting inputs with the audited executable ledger first")
    metadata = wdata["execution_ledger"]
    payload = Path(metadata["path"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != metadata["sha256"]:
        raise ValueError("The portfolio execution ledger checksum has changed")
    base_ledger = pickle.loads(payload)["hybrid_risk_parity"]

    dates = wdata["dates"]
    hybrid_stats = wdata["strategies"]["hybrid_risk_parity"]["stats"]
    rets_base = np.array(hybrid_stats["daily_returns"])

    ann_factor = 252.0 / len(dates)

    def evaluate_nav(multiplier, name, margin_ratio_desc, leverage_ratio):
        bundle = dict(base_ledger)
        days = pd.Index(assign_trading_date(pd.Series(bundle["weights_df"].index), True).astype(str))
        factors = pd.Series(multiplier, index=dates).reindex(days).to_numpy()
        target = bundle["weights_df"].mul(factors, axis=0)
        bundle["weights_df"] = target.where(bundle["tradable_mask"]).ffill().fillna(0.0)
        daily = replay_allocated_execution(bundle)["daily"].reindex(dates, fill_value=0.0)
        rets = daily.net.to_numpy()
        nav = 1.0 + np.cumsum(rets)
        peak = np.maximum.accumulate(np.maximum(nav, 1.0))
        dd = (nav - peak) / peak * 100.0
        tot_ret = (nav[-1] - 1.0) * 100.0
        ann_ret = tot_ret * ann_factor
        max_dd = float(np.min(dd))
        mean_r = float(np.mean(rets))
        std_r = float(np.std(rets))
        vol = std_r * np.sqrt(252) * 100.0
        sharpe = (mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0.0
        calmar = abs(tot_ret / max_dd) if max_dd != 0 else 0.0
        win_rate = float(np.mean(rets > 0) * 100.0)

        return {
            "name": name,
            "margin_desc": margin_ratio_desc,
            "leverage": leverage_ratio,
            "nav": [round(float(x), 4) for x in nav],
            "dd": [round(float(x), 2) for x in dd],
            "total_return": round(float(tot_ret), 2),
            "ann_return": round(float(ann_ret), 2),
            "volatility": round(float(vol), 2),
            "max_dd": round(float(max_dd), 2),
            "sharpe": round(float(sharpe), 2),
            "calmar": round(float(calmar), 2),
            "win_rate": round(float(win_rate), 1),
            "daily_returns": rets.tolist(),
            "gross_return_pct": float(daily.gross.sum() * 100),
            "fee_return_pct": float(daily.fee.sum() * 100),
        }

    # 1. 模拟不同杠杆层级
    models = []

    # 模式 0: 原始底仓 (1.0x 杠杆，保证金占用 ~2.5%)
    models.append(evaluate_nav(1.0, "原始底仓 (1.0x 杠杆)", "固定初始资金 1 倍名义敞口", 1.0))

    # 模式 1: 保守稳健型 (2.5x 杠杆，保证金占用 ~15%)
    models.append(evaluate_nav(2.5, "稳健型 CTA (2.5x 杠杆)", "固定初始资金 2.5 倍名义敞口", 2.5))

    # 模式 2: 标准平衡型 (4.0x 杠杆，保证金占用 ~25%)
    models.append(evaluate_nav(4.0, "平衡型 CTA (4.0x 杠杆)", "固定初始资金 4 倍名义敞口", 4.0))

    # 模式 3: 积极进取型 (6.0x 杠杆，保证金占用 ~35%)
    models.append(evaluate_nav(6.0, "进取型 CTA (6.0x 杠杆)", "固定初始资金 6 倍名义敞口", 6.0))

    # 模式 4: 动态目标波动率模型 (Target Volatility = 10%)
    # 滚动 20 日估计波动率，动态调整杠杆: L_t = clamp(TargetVol / RollingVol, 1.0, 7.0)
    dyn_leverage_lag = causal_capital_leverage(rets_base)
    models.append(
        evaluate_nav(
            dyn_leverage_lag.values,
            "动态目标波动率 (Target Vol 10%)",
            "自适应动态杠杆 (低波加仓/高波降仓)",
            round(float(np.mean(dyn_leverage_lag)), 1),
        )
    )

    # 打印对比结果
    print("====================== 资金管理与杠杆模型效果对比 ======================")
    for m in models:
        print(f"【{m['name']}】({m['margin_desc']})")
        print(
            f"  8个月累计收益: {m['total_return']:+.2f}% | 年化收益: {m['ann_return']:+.2f}% | 年化波动: {m['volatility']:.2f}%"
        )
        print(
            f"  动态最大回撤: {m['max_dd']:.2f}% | 夏普比率: {m['sharpe']:.2f} | 卡玛比率: {m['calmar']:.2f} | 日胜率: {m['win_rate']:.1f}%\n"
        )

    out_payload = {
        "accounting": "fixed_initial_capital_next_open",
        "source": str(Path(weight_json_path).resolve()),
        "source_sha256": hashlib.sha256(Path(weight_json_path).read_bytes()).hexdigest(),
        "dates": dates,
        "models": models,
    }

    out_file = research_output_path("capital_management_simulation.json", output_path)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, ensure_ascii=False)
    print(f"资金管理模拟数据已保存至: {out_file}")

    return out_payload


if __name__ == "__main__":
    run_simulation()
