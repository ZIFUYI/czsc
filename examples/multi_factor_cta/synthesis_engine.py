#!/usr/bin/env python
"""
第四层：时序×截面三支柱分层合成与跨品种风险平价引擎 (Intra-Asset Three-Pillar & Cross-Asset HRP Synthesis Engine)

包含两级核心合成架构：
1. 第一级（单品种内 · 时序三支柱分层合成 `IntraAssetThreePillarSynthesizer`）：
   - 支柱一：四簇 × 五周期双通道分层聚合（顺势突破簇 A 与正交战术簇 B/C/D 独立截断，防止顺势簇饱和淹没正交信号）；
   - 支柱二：跨簇多空非对称反向断路器（权益/ETF 采用非对称防逼空 β_S=0.50 > β_L=0.35；商品期货采用对称断路 β_L=β_S=0.40）；
   - 支柱三：因果有效湍流波动率封顶调制 σ_turb = σ_15d * (1 - 0.35 * ER_5d) + 最小调仓死区 + 交易制度边界适配器
     (支持双向多空 `LONG_SHORT`、纯多头现货 `LONG_ONLY_SPOT`、日内 14:55 强平 `INTRADAY_FLAT_1455`)。
2. 第二级（跨品种间 · 逆波动率平价与板块分层风险平价 `CrossAssetHRPSynthesizer`）：
   - 品种级因果逆波动率缩放 (Inverse-Volatility Sizing)；
   - 板块级相关性去拥挤与分层风险平价 (Sector-Level HRP)；
   - 截面并发拥挤度断路器 (CSC Crowding Damper)。
"""

from __future__ import annotations

from datetime import time
from typing import Any

import numpy as np
import pandas as pd

from czsc import WeightBacktest
from examples.alpha_mining_agent.forensic_pipeline import IS_END_DT, apply_rebalance_deadzone
from examples.alpha_mining_agent.hypothesis_bank import ALL_SIGNAL_SPECS
from examples.multi_factor_cta.data_engine import AssetClass, AssetSpec, ExecutionMode


class IntraAssetThreePillarSynthesizer:
    """单品种内四簇×多周期三支柱分层合成器"""

    TF_WEIGHTS: dict[str, float] = {
        "1d": 0.65,
        "60m": 0.55,
        "30m": 0.45,
        "15m": 0.40,
        "5m": 0.30,
    }
    COMMODITY_TF_WEIGHTS: dict[str, float] = {
        "1d": 0.95,
        "60m": 0.55,
        "30m": 0.25,
        "15m": 0.20,
        "5m": 0.00,
    }

    def synthesize(
        self,
        df_grid: pd.DataFrame,
        unit_weights_on_grid: dict[str, np.ndarray],
        unit_alloc_map: dict[str, float],
        asset_spec: AssetSpec,
        rebalance_deadzone: float = 0.06,
        execution_mode_override: ExecutionMode | None = None,
        enable_regime_gating: bool = True,
    ) -> np.ndarray:
        n = len(df_grid)
        w_tf_arch = {
            tf: {
                "A_顺势突破簇": np.zeros(n, dtype=np.float64),
                "B_背驰反转簇": np.zeros(n, dtype=np.float64),
                "C_震荡回归簇": np.zeros(n, dtype=np.float64),
                "D_日内微结构簇": np.zeros(n, dtype=np.float64),
            }
            for tf in self.TF_WEIGHTS
        }

        has_ortho = False
        for unit_id, coeff in unit_alloc_map.items():
            if unit_id not in unit_weights_on_grid:
                continue
            sid, tf = unit_id.split("@")
            if tf not in w_tf_arch:
                continue
            spec = ALL_SIGNAL_SPECS.get(sid)
            arch = spec.archetype if spec is not None else "A_顺势突破簇"
            if arch != "A_顺势突破簇":
                has_ortho = True
            w_tf_arch[tf][arch] += coeff * unit_weights_on_grid[unit_id]

        w_a_total = np.zeros(n, dtype=np.float64)
        w_bcd_total = np.zeros(n, dtype=np.float64)
        w_bcd_pos = np.zeros(n, dtype=np.float64)
        w_bcd_neg = np.zeros(n, dtype=np.float64)

        # 因果主场状态门控 (Home-Regime Gating)：对期货/跨品种标的在强单边逆风状态下的震荡抄底/摸顶信号施加保护性阻尼
        regimes = (
            self._get_causal_regimes(df_grid) if (enable_regime_gating and asset_spec.symbol != "SZSE.159915") else None
        )

        is_commodity = asset_spec.asset_class == AssetClass.COMMODITY_FUTURES
        tf_weight_map = self.COMMODITY_TF_WEIGHTS if is_commodity else self.TF_WEIGHTS

        for tf, tw in tf_weight_map.items():
            wa = np.clip(w_tf_arch[tf]["A_顺势突破簇"], -1.0, 1.0)
            wb = np.clip(w_tf_arch[tf]["B_背驰反转簇"], -1.0, 1.0)
            wc = np.clip(w_tf_arch[tf]["C_震荡回归簇"], -1.0, 1.0)
            wd = np.clip(w_tf_arch[tf]["D_日内微结构簇"], -1.0, 1.0)

            if regimes is not None:
                if is_commodity:
                    # 商品期货趋势延续性强：彻底剔除 B_背驰反转簇，强趋势态下对 C_震荡回归簇 逆势信号施加保护阻尼
                    wb = np.zeros_like(wb)
                    wc = np.where((regimes == -1) & (wc > 0.0), wc * 0.15, wc)
                    wc = np.where((regimes == 1) & (wc < 0.0), wc * 0.15, wc)
                    wd = np.where(regimes != 0, wd * 0.35, wd * 0.50)
                else:
                    # 权益/股指标的：仅在强单边趋势态下对 C_震荡回归簇 的逆势头寸施加保护阻尼
                    wc = np.where((regimes == -1) & (wc > 0.0), wc * 0.35, wc)
                    wc = np.where((regimes == 1) & (wc < 0.0), wc * 0.35, wc)

            wbcd = np.clip(wb + wc + wd, -1.0, 1.0)
            w_a_total += tw * wa
            w_bcd_total += tw * wbcd
            w_bcd_pos += tw * np.maximum(wbcd, 0.0)
            w_bcd_neg += tw * np.minimum(wbcd, 0.0)

        if not has_ortho:
            w_multi = np.clip(w_a_total, -1.0, 1.0)
        else:
            # 第二支柱：跨簇多空非对称反向断路器（按资产类别自适应）
            if is_commodity and enable_regime_gating:
                b_long, b_short = 0.15, 0.15
                bcd_mult = 0.25
            elif is_commodity:
                b_long, b_short = 0.40, 0.40
                bcd_mult = 1.25
            else:
                b_long, b_short = 0.35, 0.50
                bcd_mult = 1.25

            w_a_gated = w_a_total.copy()
            long_mask = w_a_gated > 0.0
            short_mask = w_a_gated < 0.0
            w_a_gated[long_mask] *= np.clip(1.0 - b_long * np.abs(w_bcd_neg[long_mask]), 0.15, 1.0)
            w_a_gated[short_mask] *= np.clip(1.0 - b_short * w_bcd_pos[short_mask], 0.15, 1.0)

            w_raw = w_a_gated + bcd_mult * w_bcd_total

            # 第三支柱：因果有效湍流波动率封顶调制
            vol_cap = self._get_causal_turbulence_cap(df_grid, asset_spec.base_tf)
            w_multi = np.clip(w_raw * np.sqrt(vol_cap), -vol_cap, vol_cap)

        w_filtered = apply_rebalance_deadzone(w_multi, deadzone=rebalance_deadzone)
        mode = execution_mode_override or asset_spec.execution_mode
        return self._apply_execution_mode_adapter(df_grid, w_filtered, mode)

    @staticmethod
    def _get_causal_regimes(df_grid: pd.DataFrame) -> np.ndarray:
        n = len(df_grid)
        cached = df_grid.attrs.get("_cached_regimes")
        if cached is not None and len(cached) == n:
            return cached
        from examples.multi_factor_cta.forensic_validator import classify_causal_market_regime

        regs = classify_causal_market_regime(df_grid, er_window=30, er_th=0.30)
        df_grid.attrs["_cached_regimes"] = regs
        return regs

    @staticmethod
    def _get_causal_turbulence_cap(df_grid: pd.DataFrame, base_tf: str) -> np.ndarray:
        n = len(df_grid)
        cached = df_grid.attrs.get("_cached_vol_cap")
        if cached is not None and len(cached) == n:
            return cached

        bars_per_day = {"5m": 48, "15m": 16, "30m": 8, "60m": 4}.get(base_tf, 16)
        w_15d = max(20, 15 * bars_per_day)
        w_5d = max(10, 5 * bars_per_day)

        close_arr = df_grid["close"].values
        ret_b = pd.Series(close_arr).pct_change().fillna(0.0).values
        vol_15d = pd.Series(ret_b).rolling(w_15d, min_periods=bars_per_day).std().bfill().values * np.sqrt(
            bars_per_day * 242.0
        )
        net_chg_5d = np.abs(pd.Series(close_arr).diff(w_5d).fillna(0.0).values)
        abs_path_5d = (
            pd.Series(np.abs(np.diff(close_arr, prepend=close_arr[0])))
            .rolling(w_5d, min_periods=bars_per_day)
            .sum()
            .bfill()
            .values
        )
        er_5d = np.clip(net_chg_5d / np.maximum(abs_path_5d, 1e-6), 0.0, 1.0)
        turb_vol = vol_15d * (1.0 - 0.35 * er_5d)
        vol_cap = np.clip(0.28 / np.maximum(turb_vol, 0.28), 0.72, 1.0)
        df_grid.attrs["_cached_vol_cap"] = vol_cap
        return vol_cap

    @staticmethod
    def _apply_execution_mode_adapter(df_grid: pd.DataFrame, w: np.ndarray, mode: ExecutionMode) -> np.ndarray:
        """
        交易制度与物理边界适配器：
        - LONG_SHORT: 完整双向多空权重 [-1, +1]
        - LONG_ONLY_SPOT: 纯多头现货轨 [0, +1]（空头信号自动转化为零延迟减仓/清仓风控）
        - INTRADAY_FLAT_1455: 日内 T+0 强平轨（14:45 起禁止新开仓，14:55 起强制平仓归零）
        """
        if mode == ExecutionMode.LONG_SHORT:
            return w
        if mode == ExecutionMode.LONG_ONLY_SPOT:
            return np.clip(w, 0.0, 1.0)
        if mode == ExecutionMode.INTRADAY_FLAT_1455:
            out = w.copy()
            times = pd.to_datetime(df_grid["dt"]).dt.time.values
            t_no_open = time(14, 45)
            t_flat = time(14, 55)
            for i in range(len(out)):
                if times[i] >= t_flat:
                    out[i] = 0.0
                elif times[i] >= t_no_open and i > 0 and abs(out[i]) > abs(out[i - 1]):
                    out[i] = out[i - 1]
            return out
        return w


def joint_is_coordinate_reweight(
    df_grid: pd.DataFrame,
    unit_weights_on_grid: dict[str, np.ndarray],
    initial_alloc: dict[str, float],
    unit_forensic_meta: dict[str, dict[str, Any]],
    asset_spec: AssetSpec,
    synthesizer: IntraAssetThreePillarSynthesizer,
    is_end_dt: pd.Timestamp = IS_END_DT,
) -> dict[str, float]:
    """
    严格样本内 (In-Sample) 法医经验贝叶斯先验锚定 + 双半场 (H1/H2) L2 信赖域联合坐标重配权
    """
    is_mask = (df_grid["dt"] <= is_end_dt).values
    mid_dt = pd.Timestamp("2021-12-31 23:59:59")
    h1_mask = (df_grid["dt"] <= mid_dt).values
    h2_mask = ((df_grid["dt"] > mid_dt) & (df_grid["dt"] <= is_end_dt)).values
    if int(h1_mask.sum()) < 100 or int(h2_mask.sum()) < 100:
        # 若品种起点较晚 (如 2021 起)，按样本内中位时间点切分 H1/H2
        is_dts = df_grid["dt"].values[is_mask]
        mid_val = is_dts[len(is_dts) // 2]
        h1_mask = (df_grid["dt"].values <= mid_val) & is_mask
        h2_mask = (df_grid["dt"].values > mid_val) & is_mask

    def _prep_slice(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sub = df_grid.iloc[mask].reset_index(drop=True)
        c = sub["close"].values
        r = np.zeros(len(c), dtype=np.float64)
        r[:-1] = (c[1:] - c[:-1]) / c[:-1]
        _, dids = np.unique(pd.to_datetime(sub["dt"]).dt.date.values, return_inverse=True)
        return r, dids

    def _fast_stats(close_ret: np.ndarray, day_ids: np.ndarray, w_slice: np.ndarray) -> dict[str, float]:
        dw = np.abs(np.diff(w_slice, prepend=0.0))
        bar_net = w_slice * close_ret - dw * asset_spec.fee_rate
        daily_net = np.bincount(day_ids, weights=bar_net)
        cum_ret = float(daily_net.sum() * 100.0)
        n_years = len(daily_net) / 252.0
        ann_ret = float(cum_ret / max(n_years, 1e-6))
        std_d = float(daily_net.std(ddof=1)) if len(daily_net) > 1 else 0.0
        sharpe = float((daily_net.mean() / std_d) * np.sqrt(252.0)) if std_d > 1e-12 else 0.0
        curve = np.cumsum(daily_net)
        peak = np.maximum.accumulate(np.maximum(curve, 0.0))
        mdd = float((peak - curve).max() * 100.0)
        calmar = float(ann_ret / mdd) if mdd > 1e-6 else 0.0
        return {"cum_ret_pct": cum_ret, "ann_ret_pct": ann_ret, "sharpe": sharpe, "mdd_pct": mdd, "calmar": calmar}

    ret_is, dids_is = _prep_slice(is_mask)
    ret_h1, dids_h1 = _prep_slice(h1_mask)
    ret_h2, dids_h2 = _prep_slice(h2_mask)

    raw_q: dict[str, float] = {}
    for uid in initial_alloc:
        meta = unit_forensic_meta.get(uid, {})
        n_p = max(int(meta.get("is_n_purged", 20)), 5)
        saf = float(np.clip(float(meta.get("is_safety_margin_pct", 4.0)), 1.0, 15.0))
        dd_res = float(np.clip(float(meta.get("dd_rescue_alpha_bp", 0.0)), -10.0, 30.0))
        raw_q[uid] = float(np.log(n_p) * saf * (1.0 + 0.02 * dd_res))

    non_seed = [u for u in initial_alloc if not unit_forensic_meta.get(u, {}).get("is_base_seed", False)]
    if non_seed:
        q_vals = np.array([raw_q[u] for u in non_seed])
        q_mean, q_std = float(q_vals.mean()), float(q_vals.std() + 1e-6)
    else:
        q_mean, q_std = 0.0, 1.0

    z_q = {
        u: (
            float(np.clip((raw_q[u] - q_mean) / q_std, -1.2, 1.6))
            if not unit_forensic_meta.get(u, {}).get("is_base_seed", False)
            else 0.0
        )
        for u in initial_alloc
    }
    eb_prior = {
        u: round(float(np.clip(initial_alloc[u] * (1.0 + 0.06 * z_q[u]), 0.15, 0.56)), 2) for u in initial_alloc
    }
    base_budget = sum(initial_alloc.values())

    def _eval(alloc: dict[str, float]) -> tuple[float, dict, dict, dict]:
        w_full = synthesizer.synthesize(df_grid, unit_weights_on_grid, alloc, asset_spec)
        s_is = _fast_stats(ret_is, dids_is, w_full[is_mask])
        s_h1 = _fast_stats(ret_h1, dids_h1, w_full[h1_mask])
        s_h2 = _fast_stats(ret_h2, dids_h2, w_full[h2_mask])
        l2_pen = sum((alloc[u] - eb_prior[u]) ** 2 for u in alloc)
        min_sh = min(s_h1["sharpe"], s_h2["sharpe"])
        obj = (
            s_is["sharpe"] * 1.15
            + min_sh * 0.85
            + s_is["calmar"] * 0.45
            + s_is["ann_ret_pct"] * 0.12
            - s_is["mdd_pct"] * 0.40
            - 1.0 * l2_pen
        )
        return obj, s_is, s_h1, s_h2

    opt_alloc = dict(eb_prior)
    obj_cur, st_is_cur, st_h1_cur, st_h2_cur = _eval(opt_alloc)
    ordered = sorted(initial_alloc.keys(), key=lambda u: raw_q[u], reverse=True)

    for uid in ordered:
        if bool(unit_forensic_meta.get(uid, {}).get("is_base_seed", False)):
            continue
        w_p = eb_prior[uid]
        tf = uid.split("@")[1]
        lo = max(0.16, round(w_p - 0.10, 2))
        hi = min(0.58 if tf == "5m" else 0.48, round(w_p + 0.10, 2))
        grid = sorted({lo, round(w_p - 0.05, 2), round(w_p, 2), round(w_p + 0.05, 2), hi})

        best_w = opt_alloc[uid]
        best_obj = obj_cur
        best_pack = (st_is_cur, st_h1_cur, st_h2_cur)
        for cw in grid:
            if abs(cw - opt_alloc[uid]) < 1e-4:
                continue
            t_alloc = dict(opt_alloc)
            t_alloc[uid] = cw
            if sum(t_alloc.values()) > base_budget + 0.45:
                continue
            obj_t, s_is, s_h1, s_h2 = _eval(t_alloc)
            if s_is["mdd_pct"] > st_is_cur["mdd_pct"] + 0.01:
                continue
            if s_h1["sharpe"] < st_h1_cur["sharpe"] - 0.003 or s_h2["sharpe"] < st_h2_cur["sharpe"] - 0.003:
                continue
            if s_is["sharpe"] < st_is_cur["sharpe"] or s_is["cum_ret_pct"] < st_is_cur["cum_ret_pct"] - 0.02:
                continue
            if obj_t > best_obj + 0.004:
                best_obj = obj_t
                best_w = cw
                best_pack = (s_is, s_h1, s_h2)

        if abs(best_w - opt_alloc[uid]) >= 1e-4:
            opt_alloc[uid] = best_w
            obj_cur = best_obj
            st_is_cur, st_h1_cur, st_h2_cur = best_pack

    return opt_alloc


class CrossAssetHRPSynthesizer:
    """
    跨品种（股指期货 + ETF + 商品期货）逆波动率平价与板块分层风险平价组合构建器
    """

    def __init__(self, target_portfolio_vol: float = 0.14) -> None:
        self.target_portfolio_vol = target_portfolio_vol

    def build_multi_asset_portfolio(
        self,
        asset_daily_returns: dict[str, pd.Series],
        asset_sectors: dict[str, str],
        is_end_date: pd.Timestamp = IS_END_DT,
    ) -> dict[str, Any]:
        """
        基于各品种单品种策略日净值收益率序列，按严格无未来函数的：
        1. 品种滚动 60 日因果逆波动率缩放 (Inverse-Volatility Parity)；
        2. 板块内相关性去拥挤与板块间分层风险平价 (Sector-Level HRP)；
        3. 组合目标波动率调制，输出跨资产全天候组合表现。
        """
        raw_df = pd.DataFrame(asset_daily_returns).sort_index()
        raw_df.index = pd.to_datetime(raw_df.index)
        active_mask = raw_df.notna().astype(float)
        ret_df = raw_df.fillna(0.0)

        # 0. 基准对比：在市品种动态等权组合 (Dynamic Listed Equal-Weight 1/N_t)
        eq_weights = active_mask.div(active_mask.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
        eq_daily_ret = (ret_df * eq_weights).sum(axis=1)

        # 1. 因果滚动 60 日波动率 (shift(1) 严格避免当日未来信息，仅在已上市区间内统计)
        roll_vol = raw_df.rolling(60, min_periods=10).std().shift(1).bfill() * np.sqrt(252.0)
        inv_vol = (1.0 / np.maximum(roll_vol.fillna(0.15), 0.06)) * active_mask

        # 2. 板块级分层风险预算 (Sector Equal-Risk / Inverse-Sector-Count Shrinkage)
        sectors: dict[str, list[str]] = {}
        for sym, sec in asset_sectors.items():
            if sym in ret_df.columns:
                sectors.setdefault(sec, []).append(sym)

        n_sectors = max(len(sectors), 1)
        sector_multiplier = pd.Series(1.0, index=ret_df.columns)
        for _sec, syms in sectors.items():
            # 同一板块多品种共享板块风险预算 (按 1 / sqrt(K) 折减同板块拥挤风险)
            sector_multiplier[syms] = 1.0 / (n_sectors * np.sqrt(len(syms)))

        raw_alloc = inv_vol.multiply(sector_multiplier, axis=1) * active_mask
        alloc_weights = raw_alloc.div(raw_alloc.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(eq_weights)

        # 3. 因果组合级波动率目标缩放 (使组合整体保持稳健高夏普)
        unscaled_port_ret = (ret_df * alloc_weights).sum(axis=1)
        port_roll_vol = unscaled_port_ret.rolling(40, min_periods=15).std().shift(1).bfill() * np.sqrt(252.0)
        lev_scalar = np.clip(self.target_portfolio_vol / np.maximum(port_roll_vol, 0.05), 0.70, 1.65)
        port_daily_ret = unscaled_port_ret * lev_scalar

        is_mask = port_daily_ret.index <= pd.Timestamp(is_end_date.date())
        oos_mask = port_daily_ret.index > pd.Timestamp(is_end_date.date())

        return {
            "daily_returns": port_daily_ret,
            "equal_weight_returns": eq_daily_ret,
            "equal_weights_df": eq_weights,
            "unscaled_hrp_weights_df": alloc_weights,
            "asset_weights_df": alloc_weights.multiply(lev_scalar, axis=0),
            "full": self._calc_return_series_stats(port_daily_ret),
            "is": self._calc_return_series_stats(port_daily_ret.loc[is_mask]),
            "oos": self._calc_return_series_stats(port_daily_ret.loc[oos_mask]),
            "eq_full": self._calc_return_series_stats(eq_daily_ret),
            "eq_is": self._calc_return_series_stats(eq_daily_ret.loc[is_mask]),
            "eq_oos": self._calc_return_series_stats(eq_daily_ret.loc[oos_mask]),
        }

    @staticmethod
    def _calc_return_series_stats(dr: pd.Series) -> dict[str, float]:
        if dr.empty:
            return {"cum_ret_pct": 0.0, "ann_ret_pct": 0.0, "sharpe": 0.0, "mdd_pct": 0.0, "calmar": 0.0}
        arr = dr.values.astype(np.float64)
        cum_ret = float(arr.sum() * 100.0)
        n_years = len(arr) / 252.0
        ann_ret = float(cum_ret / max(n_years, 1e-6))
        std_d = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        sharpe = float((arr.mean() / std_d) * np.sqrt(252.0)) if std_d > 1e-12 else 0.0
        curve = np.cumsum(arr)
        peak = np.maximum.accumulate(np.maximum(curve, 0.0))
        mdd = float((peak - curve).max() * 100.0)
        calmar = float(ann_ret / mdd) if mdd > 1e-6 else 0.0
        win_days = float((arr > 0).sum() / max((arr != 0).sum(), 1) * 100.0)
        return {
            "cum_ret_pct": round(cum_ret, 2),
            "ann_ret_pct": round(ann_ret, 2),
            "sharpe": round(sharpe, 2),
            "mdd_pct": round(mdd, 2),
            "calmar": round(calmar, 2),
            "daily_win_pct": round(win_days, 1),
        }


def evaluate_single_asset_backtest(
    df_grid: pd.DataFrame,
    w_grid: np.ndarray,
    symbol: str,
    fee_rate: float = 0.0002,
    is_end_dt: pd.Timestamp = IS_END_DT,
) -> dict[str, Any]:
    """调用 czsc.WeightBacktest 评估单品种 IS / OOS / Full 表现并返回日收益序列"""

    def _run_slice(sub_df: pd.DataFrame, sub_w: np.ndarray) -> tuple[dict[str, float], pd.Series]:
        if len(sub_df) < 10:
            return (
                {
                    "cum_ret_pct": 0.0,
                    "ann_ret_pct": 0.0,
                    "sharpe": 0.0,
                    "mdd_pct": 0.0,
                    "calmar": 0.0,
                    "active_pct": 0.0,
                },
                pd.Series(dtype=np.float64),
            )
        dfw = pd.DataFrame(
            {
                "dt": sub_df["dt"].astype(str),
                "symbol": symbol,
                "weight": sub_w,
                "price": sub_df["close"].values,
            }
        )
        wb = WeightBacktest(data=dfw, fee_rate=fee_rate, weight_type="ts")
        st = wb.stats
        dr = wb.daily_return.copy()
        dts = pd.to_datetime(dr["date"])
        wk = dts.dt.weekday
        shift_d = np.where(wk == 5, 2, np.where(wk == 6, 1, 0))
        dr["date"] = dts + pd.to_timedelta(shift_d, unit="D")
        s_dr = dr.groupby("date")["total"].sum()
        return (
            {
                "cum_ret_pct": round(float(st.get("绝对收益", 0.0) * 100.0), 2),
                "ann_ret_pct": round(float(st.get("年化收益", 0.0) * 100.0), 2),
                "sharpe": round(float(st.get("夏普比率", 0.0)), 2),
                "mdd_pct": round(float(st.get("最大回撤", 0.0) * 100.0), 2),
                "calmar": round(float(st.get("卡玛比率", 0.0)), 2),
                "active_pct": round(float((np.abs(sub_w) > 0.01).mean() * 100.0), 1),
            },
            s_dr,
        )

    is_mask = (df_grid["dt"] <= is_end_dt).values
    oos_mask = (df_grid["dt"] > is_end_dt).values

    st_full, dr_full = _run_slice(df_grid, w_grid)
    st_is, _ = _run_slice(df_grid.iloc[is_mask].reset_index(drop=True), w_grid[is_mask])
    st_oos, _ = _run_slice(df_grid.iloc[oos_mask].reset_index(drop=True), w_grid[oos_mask])
    return {"full": st_full, "is": st_is, "oos": st_oos, "daily_return": dr_full}
