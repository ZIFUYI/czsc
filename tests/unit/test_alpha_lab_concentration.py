"""
AlphaLab 动态稀疏聚焦 (Dynamic Sparse Concentration) M0 ~ M6 机制单元测试
================================================================================
覆盖：
1. 严格因果零前视检验（截断未来数据 t > T0 后，历史权重 w[t <= T0] 逐元素 100% 一致）；
2. 多空仿射镜像对称检验（输入信号与收益率同步取反后，wave_pos 精确取反、聚焦权重 100% 不变）；
3. 权重非负、有界与行和归一化合法性检验；
4. Full-60 真实因果消融产物（M0 ~ M6 双区间指标、14 品种波段/死水归因与看板真实日度曲线）完整性检验。
"""

from __future__ import annotations

import pickle

import numpy as np

from alpha_lab.concentration import (
    REAL_CURVES_CACHE_PATH,
    ConcentrationConfig,
    compute_dynamic_focused_optimal_weights,
    compute_probe_runner_pyramid_weights,
    compute_regime_ignition_weights,
    compute_rolling_tradability_weights,
    compute_sector_leader_weights,
    compute_topk_sparsity_weights,
    compute_wave_composite_signal,
)


def _make_synthetic_inputs(n_days: int = 160, n_assets: int = 18, seed: int = 42) -> dict[str, np.ndarray | list[str]]:
    rng = np.random.default_rng(seed)
    sig_1d = rng.normal(0.0, 0.12, size=(n_days, n_assets))
    sig_60m = rng.normal(0.0, 0.12, size=(n_days, n_assets))
    sig_30m = rng.normal(0.0, 0.12, size=(n_days, n_assets))
    ret_1d = rng.normal(0.0, 0.015, size=(n_days, n_assets))
    atr_pct = np.clip(rng.uniform(0.008, 0.032, size=(n_days, n_assets)), 0.004, 0.08)
    er8_reg = rng.uniform(0.15, 0.55, size=(n_days, n_assets))
    er_comp = rng.uniform(0.18, 0.50, size=(n_days, n_assets))
    er_struct = rng.uniform(0.20, 0.42, size=(n_days, n_assets))
    cs_90 = rng.uniform(0.35, 0.65, size=(n_days, n_assets))
    sqz_20 = rng.uniform(0.0, 0.40, size=(n_days, n_assets))
    vol_act = rng.uniform(0.65, 1.25, size=(n_days, n_assets))
    dw_struct = np.where(np.arange(n_assets)[None, :] % 4 == 0, 0.01, 1.0) * np.ones((n_days, n_assets))
    sectors = [f"SEC_{i % 6}" for i in range(n_assets)]
    return {
        "sig_1d": sig_1d,
        "sig_60m": sig_60m,
        "sig_30m": sig_30m,
        "ret_1d": ret_1d,
        "atr_pct": atr_pct,
        "er8_reg": er8_reg,
        "er_comp": er_comp,
        "er_struct": er_struct,
        "cs_90": cs_90,
        "sqz_20": sqz_20,
        "vol_act": vol_act,
        "dw_struct": dw_struct,
        "sectors": sectors,
    }


def _run_all_mechanisms(inp: dict[str, np.ndarray | list[str]], n_slice: int | None = None):
    sl = slice(None) if n_slice is None else slice(0, n_slice)
    s1 = np.asarray(inp["sig_1d"])[sl]
    s2 = np.asarray(inp["sig_60m"])[sl]
    s3 = np.asarray(inp["sig_30m"])[sl]
    r1 = np.asarray(inp["ret_1d"])[sl]
    atr = np.asarray(inp["atr_pct"])[sl]
    er8 = np.asarray(inp["er8_reg"])[sl]
    erc = np.asarray(inp["er_comp"])[sl]
    ers = np.asarray(inp["er_struct"])[sl]
    cs9 = np.asarray(inp["cs_90"])[sl]
    sqz = np.asarray(inp["sqz_20"])[sl]
    vact = np.asarray(inp["vol_act"])[sl]
    dws = np.asarray(inp["dw_struct"])[sl]
    sectors = list(inp["sectors"])
    cfg = ConcentrationConfig(k_min=5, k_max=7, k_base=6)

    w_raw, w_pos, er15 = compute_wave_composite_signal(s1, s2, s3, r1)
    w_sleeve = w_pos * r1
    w1, _ = compute_topk_sparsity_weights(w_pos, ers, erc, cs9, vact, dws, cfg)
    w2, g2 = compute_regime_ignition_weights(w_pos, er8, erc, sqz, vact, dws, cfg)
    w3, r3 = compute_probe_runner_pyramid_weights(w_pos, r1, atr, cs9, vact, dws, cfg)
    w4, l4 = compute_sector_leader_weights(w_pos, erc, cs9, vact, dws, sectors, cfg)
    w5, h5 = compute_rolling_tradability_weights(w_pos, w_sleeve, erc, vact, dws, cfg)
    w6 = compute_dynamic_focused_optimal_weights(w1, g2, r3, l4, h5, erc)
    return w_raw, w_pos, er15, (w1, w2, w3, w4, w5, w6)


def test_concentration_strict_causality_zero_lookahead():
    """验证截断未来数据 t > T0 后，M1 ~ M6 在 t <= T0 的历史权重与复合信号 100% 逐元素一致。"""
    inp = _make_synthetic_inputs(n_days=150, n_assets=18, seed=7)
    w_raw_full, w_pos_full, er15_full, weights_full = _run_all_mechanisms(inp, n_slice=None)

    for t0 in (65, 110):
        w_raw_sub, w_pos_sub, er15_sub, weights_sub = _run_all_mechanisms(inp, n_slice=t0)
        np.testing.assert_allclose(w_raw_full[:t0], w_raw_sub, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(w_pos_full[:t0], w_pos_sub, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(er15_full[:t0], er15_sub, atol=1e-12, rtol=0.0)
        for idx, (wf, ws) in enumerate(zip(weights_full, weights_sub, strict=True), start=1):
            np.testing.assert_allclose(
                wf[:t0],
                ws,
                atol=1e-12,
                rtol=0.0,
                err_msg=f"M{idx} weight violated strict causality at cutoff T0={t0}",
            )


def test_concentration_affine_long_short_mirror_symmetry_and_bounds():
    """验证多空信号与收益率同步取反后，wave_pos 精确取反，且 M1 ~ M6 截面聚焦权重保持 100% 对称不变与合法行归一化。"""
    inp_pos = _make_synthetic_inputs(n_days=140, n_assets=18, seed=19)
    inp_neg = dict(inp_pos)
    inp_neg["sig_1d"] = -np.asarray(inp_pos["sig_1d"])
    inp_neg["sig_60m"] = -np.asarray(inp_pos["sig_60m"])
    inp_neg["sig_30m"] = -np.asarray(inp_pos["sig_30m"])
    inp_neg["ret_1d"] = -np.asarray(inp_pos["ret_1d"])

    w_raw_p, w_pos_p, er15_p, weights_p = _run_all_mechanisms(inp_pos)
    w_raw_n, w_pos_n, er15_n, weights_n = _run_all_mechanisms(inp_neg)

    np.testing.assert_allclose(w_raw_p, -w_raw_n, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(w_pos_p, -w_pos_n, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(er15_p, er15_n, atol=1e-12, rtol=0.0)

    for idx, (wp, wn) in enumerate(zip(weights_p, weights_n, strict=True), start=1):
        np.testing.assert_allclose(wp, wn, atol=1e-12, rtol=0.0, err_msg=f"M{idx} violated mirror symmetry")
        assert np.all(np.isfinite(wp)) and np.all(wp >= 0.0)
        row_sums = wp.sum(axis=1)
        assert float(row_sums[0]) == 0.0
        assert np.all((row_sums[1:] >= 0.78 - 1e-9) & (row_sums[1:] <= 1.18 + 1e-9))


def test_full60_real_concentration_ablation_bundle_and_dashboard_curves():
    """验证 Full-60 真实因果消融产物 M0 ~ M6 双区间指标、14 品种归因及 Dashboard 100% 真实日度净值替换。"""
    from alpha_lab.dashboard import AlphaLabDashboardService

    assert REAL_CURVES_CACHE_PATH.exists(), "缺少真实因果日度净值缓存 real_ablation_equity_curves.pkl"
    bundle = pickle.loads(REAL_CURVES_CACHE_PATH.read_bytes())
    schemes = bundle["schemes"]
    expected_ids = [
        "M0_Static_Baseline",
        "M1_TopK_Sparsity",
        "M2_Regime_Ignition",
        "M3_Probe_Runner_Pyramid",
        "M4_Sector_Leader_Focus",
        "M5_Rolling_Tradability_Mem",
        "M6_Dynamic_Focused_Optimal",
    ]
    assert list(schemes.keys()) == expected_ids

    m0 = schemes["M0_Static_Baseline"]
    m6 = schemes["M6_Dynamic_Focused_Optimal"]
    assert m6["full_oos"]["net_sharpe"] > 1.20
    assert m6["apr_sep_2026"]["net_sharpe"] > 1.25
    assert m6["apr_sep_2026"]["annual_return_pct"] > 15.0
    assert m6["apr_sep_2026"]["active_breadth"] < 18.0
    assert m0["apr_sep_2026"]["net_sharpe"] < 0.0

    # 验证 2026-04 ~ 2026-09 下行波段、上行波段与死水箱体品种净贡献
    kc = m6["key_symbol_contrib_pct"]
    for s_down in ("lc", "ag", "au", "sa", "cj"):
        assert kc[s_down] > 0.0, f"Downward wave symbol {s_down} should have positive net contribution in M6"
    for s_up in ("sc", "bu", "pg", "ec", "j"):
        assert kc[s_up] > 0.0, f"Upward wave symbol {s_up} should have positive net contribution in M6"
    assert m6["down_wave_sum_pct"] > 1.0
    assert m6["up_wave_sum_pct"] > 7.5
    assert abs(m6["chop_dead_sum_pct"]) < 0.20

    # 验证 DashboardService 吐出真实逐日 OOS 曲线
    svc = AlphaLabDashboardService()
    ab_rows = {r["run_id"]: r for r in svc.get_ablation_runs()}
    for sid in expected_ids:
        assert sid in ab_rows
        assert ab_rows[sid]["is_real_causal_curve"] is True
        assert len(ab_rows[sid]["inst_cum_pct"]) == 424
    assert ab_rows["Stage6_ACE_Incremental"]["is_real_causal_curve"] is True
