"""Phase 4 engineering tests: rolling causality, signed exposure and full cost accounting.

Mock prices test deterministic behavior only. Profitability, PnL decorrelation and
10 BP economic acceptance require the separately declared real-data OOS study.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from czsc.mock import generate_symbol_kines
from examples.alpha_mining_agent.hypothesis_bank import (
    ALL_ARCHETYPES,
    ARCH_A,
    ARCH_B,
    ARCH_C,
    ARCH_D,
    ARCH_E,
    ARCH_F,
    attach_native_czsc_signals,
    enrich_kline_features,
)
from examples.multi_factor_cta.atomic_factors import (
    compute_atomic_signal,
    set_factor_context,
)
from examples.multi_factor_cta.data_engine import (
    AssetClass,
    AssetSpec,
    CrossAssetDataEngine,
    ExecutionMode,
)
from examples.multi_factor_cta.forensic_validator import (
    RollingEmpiricalBayesWeighter,
    compute_empirical_bayes_shrinkage_weight,
    compute_rolling_empirical_bayes_weights,
)
from examples.multi_factor_cta.synthesis_engine import (
    ContinuousSignalStateMachine,
    CrossAssetHRPSynthesizer,
    DynamicActivePoolAllocator,
    IntraAssetSixClusterSynthesizer,
    IntraAssetThreePillarSynthesizer,
    apply_cross_sectional_budget_tilt,
    build_asset_execution_inputs,
    gram_schmidt_cluster_orthogonalize,
)

BLACK_SECTOR_SYMBOLS = ("SHFE.RB9999", "DCE.I9999", "DCE.J9999", "DCE.JM9999")
METAL_SECTOR_SYMBOLS = ("SHFE.CU9999", "SHFE.AL9999", "SHFE.ZN9999", "SHFE.NI9999")
ALL_8_SYMBOLS = BLACK_SECTOR_SYMBOLS + METAL_SECTOR_SYMBOLS


def _build_8_asset_specs(fee_rate: float = 0.0002) -> dict[str, AssetSpec]:
    specs: dict[str, AssetSpec] = {}
    for sym in BLACK_SECTOR_SYMBOLS:
        specs[sym] = AssetSpec(
            symbol=sym,
            name=sym,
            asset_class=AssetClass.COMMODITY_FUTURES,
            sector="BLACK_METALS",
            base_tf="30m",
            eligible_tfs=("60m", "30m"),
            fee_rate=fee_rate,
            execution_mode=ExecutionMode.LONG_SHORT,
        )
    for sym in METAL_SECTOR_SYMBOLS:
        specs[sym] = AssetSpec(
            symbol=sym,
            name=sym,
            asset_class=AssetClass.COMMODITY_FUTURES,
            sector="NONFERROUS_METALS",
            base_tf="30m",
            eligible_tfs=("60m", "30m"),
            fee_rate=fee_rate,
            execution_mode=ExecutionMode.LONG_SHORT,
        )
    return specs


def _generate_mock_30m(symbol: str, seed: int, sdt: str = "20220101", edt: str = "20220930") -> pd.DataFrame:
    df = generate_symbol_kines(symbol, "30分钟", sdt=sdt, edt=edt, seed=seed)
    df = df.rename(columns={"vol": "volume"})
    df["dt"] = pd.to_datetime(df["dt"])
    df["date"] = df["dt"].dt.date
    for col in ("open", "high", "low", "close"):
        df[col] = np.round(df[col].astype(float) * 100.0, 2)
    df["volume"] = df["volume"].astype(float)
    return (
        df[["date", "dt", "symbol", "open", "high", "low", "close", "volume"]].sort_values("dt").reset_index(drop=True)
    )


# ===========================================================================
# Step 1: 滚动经验贝叶斯部分池化收缩配权零前视与先验锚定验证
# ===========================================================================
@pytest.mark.parametrize("lookback_bars", [60, None])
def test_step1_rolling_empirical_bayes_zero_lookahead_and_shrinkage(
    lookback_bars: int | None,
) -> None:
    """
    验证 RollingEmpiricalBayesWeighter / compute_rolling_empirical_bayes_weights:
    1. 保持原有 compute_empirical_bayes_shrinkage_weight 100% 向后兼容；
    2. 静态先验层按 λ_prior = 0.70 锚定跨品种池化先验 + 30% 单品种样本量自适应修正；
    3. 动态滚动/扩展更新层严格满足 w_t^{EB} = 0.70 * w^{prior} + 0.30 * w_{t-1}^{rolling_perf}；
    4. 任意截断时刻 t_cut 之前的权重序列与全量计算完全一致（最大绝对误差 < 1e-12）。
    """
    # 1) 旧版静态标量收缩函数向后兼容验证
    w_eb_scalar, lam_scalar = compute_empirical_bayes_shrinkage_weight(
        local_weight=0.50,
        cross_asset_prior_weight=0.30,
        is_n_purged=60,
        shrink_k=60,
    )
    assert lam_scalar == pytest.approx(0.50, abs=1e-3)
    assert w_eb_scalar == pytest.approx(0.40, abs=1e-3)

    # 2) 构造基于 czsc.mock 的单品种价格与 4 个跨簇单元持仓序列
    df_grid = _generate_mock_30m("SHFE.RB9999", seed=401)
    n = len(df_grid)
    c_ret = df_grid["close"].pct_change().shift(-1).fillna(0.0).to_numpy(dtype=np.float64)
    rng = np.random.default_rng(202610)

    # 构造 4 个单元：U1 持续高正夏普，U2 中性波动，U3 持续负夏普，U4 后期反转
    w_u1 = np.sign(c_ret + 0.0005) * 0.65
    w_u2 = rng.choice([-0.5, 0.0, 0.5], size=n)
    w_u3 = -np.sign(c_ret + 0.0004) * 0.60
    w_u4 = np.where(np.arange(n) < n // 2, -np.sign(c_ret) * 0.5, np.sign(c_ret) * 0.7)

    unit_weights_full = {
        "AC01L@30m": w_u1,
        "BC01S@30m": w_u2,
        "CC01L@30m": w_u3,
        "FC01L@30m": w_u4,
    }
    cross_asset_prior = {
        "AC01L@30m": 0.40,
        "BC01S@30m": 0.30,
        "CC01L@30m": 0.35,
        "FC01L@30m": 0.35,
    }
    local_quality = {
        "AC01L@30m": 0.60,
        "BC01S@30m": 0.25,
        "CC01L@30m": 0.20,
        "FC01L@30m": 0.45,
    }
    is_n_purged = {
        "AC01L@30m": 120,
        "BC01S@30m": 30,
        "CC01L@30m": 20,
        "FC01L@30m": 60,
    }

    weighter = RollingEmpiricalBayesWeighter(
        prior_anchor_lambda=0.70,
        lookback_bars=lookback_bars,
        min_bars=20,
        shrink_k=60,
    )
    res_full = weighter.compute_weights(
        cross_asset_prior=cross_asset_prior,
        local_quality=local_quality,
        is_n_purged=is_n_purged,
        df_grid=df_grid,
        unit_weights_on_grid=unit_weights_full,
        fee_rate=0.0002,
    )

    # 验证静态先验层 70% 池化先验 + 30% 样本量自适应局部权重公式
    for uid, w_pool in cross_asset_prior.items():
        n_obs = is_n_purged[uid]
        eta = n_obs / (n_obs + 60.0)
        w_loc_adj = (1.0 - eta) * w_pool + eta * local_quality[uid]
        expected_prior = 0.70 * w_pool + 0.30 * w_loc_adj
        assert res_full.prior_weights[uid] == pytest.approx(expected_prior, abs=1e-12)

    # 验证动态更新层严格满足 L2 收缩恒等式: w_t^{EB} == 0.70 * w^{prior} + 0.30 * w_{t-1}^{rolling_perf}
    for uid in cross_asset_prior:
        w_prior_u = res_full.prior_weights[uid]
        w_perf_u = res_full.rolling_perf_weights[uid]
        w_eb_u = res_full[uid]
        expected_eb = 0.70 * w_prior_u + 0.30 * w_perf_u
        assert np.max(np.abs(w_eb_u - expected_eb)) < 1e-12

    # 验证高夏普单元 AC01L@30m 获得正向配权增益，负夏普单元 CC01L@30m 被收缩压降但受 70% 先验保护不归零
    assert float(np.mean(res_full["AC01L@30m"][50:])) > res_full.prior_weights["AC01L@30m"]
    assert float(np.mean(res_full["CC01L@30m"][50:])) < res_full.prior_weights["CC01L@30m"]
    assert float(np.min(res_full["CC01L@30m"])) >= 0.70 * res_full.prior_weights["CC01L@30m"]

    # 3) 严格零前视截断不变性审计 (t_cut = 180, 420, 750)
    for t_cut in (180, 420, 750):
        df_sub = df_grid.iloc[:t_cut].reset_index(drop=True)
        w_sub = {u: arr[:t_cut].copy() for u, arr in unit_weights_full.items()}
        res_sub = compute_rolling_empirical_bayes_weights(
            cross_asset_prior=cross_asset_prior,
            local_quality=local_quality,
            is_n_purged=is_n_purged,
            df_grid=df_sub,
            unit_weights_on_grid=w_sub,
            prior_anchor_lambda=0.70,
            lookback_bars=lookback_bars,
            min_bars=20,
            shrink_k=60,
            fee_rate=0.0002,
        )
        for uid in cross_asset_prior:
            max_err = float(np.max(np.abs(res_sub[uid] - res_full[uid][:t_cut])))
            assert max_err < 1e-12, f"{uid} 在 t_cut={t_cut} 处最大绝对误差 {max_err} >= 1e-12"


# ===========================================================================
# Step 2: 六簇二级分层合成与跨簇施密特残差正交验证
# ===========================================================================
def test_step2_gram_schmidt_six_cluster_orthogonality() -> None:
    """Default and explicit rolling windows preserve every historical prefix."""
    df_grid = _generate_mock_30m("SHFE.CU9999", seed=402)
    n = len(df_grid)
    rng = np.random.default_rng(77)

    # 构造主趋势基向量 S^(A) 与另外 5 个同源正交噪音底座，再叠加高比例共线混叠
    t_axis = np.linspace(0.0, 12.0 * np.pi, n)
    base_a = np.clip(0.75 * np.sin(t_axis) + 0.20 * np.cos(0.5 * t_axis), -1.0, 1.0)
    indep_f = np.sin(3.3 * t_axis + 0.4) + 0.3 * rng.standard_normal(n)
    indep_e = np.cos(5.1 * t_axis + 0.9) + 0.3 * rng.standard_normal(n)
    indep_d = np.sin(7.7 * t_axis + 1.5) + 0.3 * rng.standard_normal(n)
    indep_b = np.cos(9.3 * t_axis + 2.1) + 0.3 * rng.standard_normal(n)
    indep_c = np.sin(11.5 * t_axis + 2.8) + 0.3 * rng.standard_normal(n)

    # 人为注入强共线性 (0.70 ~ 0.85 混入 ARCH_A 及前序簇)
    collinear_clusters = {
        ARCH_A: base_a,
        ARCH_F: np.clip(0.80 * base_a + 0.35 * indep_f, -1.0, 1.0),
        ARCH_E: np.clip(0.75 * base_a + 0.50 * indep_f + 0.35 * indep_e, -1.0, 1.0),
        ARCH_D: np.clip(0.70 * base_a + 0.45 * indep_e + 0.35 * indep_d, -1.0, 1.0),
        ARCH_B: np.clip(-0.72 * base_a + 0.40 * indep_d + 0.35 * indep_b, -1.0, 1.0),
        ARCH_C: np.clip(-0.68 * base_a + 0.55 * indep_b + 0.35 * indep_c, -1.0, 1.0),
    }

    # 确认正交化前存在严重共线
    raw_corr_af = abs(float(np.corrcoef(collinear_clusters[ARCH_A], collinear_clusters[ARCH_F])[0, 1]))
    raw_corr_ae = abs(float(np.corrcoef(collinear_clusters[ARCH_A], collinear_clusters[ARCH_E])[0, 1]))
    assert raw_corr_af > 0.70 and raw_corr_ae > 0.65

    # 1) 调用 gram_schmidt_cluster_orthogonalize
    ortho = gram_schmidt_cluster_orthogonalize(collinear_clusters)
    assert set(ortho.keys()) == set(ALL_ARCHETYPES)

    # 主基向量 ARCH_A 保持完整趋势形态
    assert np.allclose(ortho[ARCH_A], collinear_clusters[ARCH_A], atol=1e-12)
    assert float(np.corrcoef(ortho[ARCH_A], collinear_clusters[ARCH_A])[0, 1]) > 0.9999

    # Causal rolling residuals cannot promise an ex-post global correlation bound.
    arch_list = list(ALL_ARCHETYPES)
    for arch in arch_list:
        assert np.isfinite(ortho[arch]).all()
        assert np.max(np.abs(ortho[arch])) <= 1.0

    # 2) 验证 IntraAssetThreePillarSynthesizer / IntraAssetSixClusterSynthesizer.synthesize_cluster_orthogonals
    spec = _build_8_asset_specs()["SHFE.CU9999"]
    unit_weights_on_grid = {
        "AC01L@30m": collinear_clusters[ARCH_A],
        "FC01L@30m": collinear_clusters[ARCH_F],
        "EC01L@30m": collinear_clusters[ARCH_E],
        "DC01L@30m": collinear_clusters[ARCH_D],
        "BC01L@30m": collinear_clusters[ARCH_B],
        "CC01L@30m": collinear_clusters[ARCH_C],
    }
    unit_alloc = dict.fromkeys(unit_weights_on_grid, 0.80)

    synth = IntraAssetThreePillarSynthesizer()
    assert isinstance(IntraAssetSixClusterSynthesizer(), IntraAssetThreePillarSynthesizer)
    w_synth, ortho_from_synth = synth.synthesize_cluster_orthogonals(
        df_grid=df_grid,
        unit_weights_on_grid=unit_weights_on_grid,
        unit_alloc_map=unit_alloc,
        asset_spec=spec,
        soft_gate=False,
    )
    assert len(w_synth) == n
    assert np.all((w_synth >= -1.0) & (w_synth <= 1.0))
    cut = 360
    prefix_w, prefix_ortho = synth.synthesize_cluster_orthogonals(
        df_grid=df_grid.iloc[:cut].copy(),
        unit_weights_on_grid={k: v[:cut] for k, v in unit_weights_on_grid.items()},
        unit_alloc_map=unit_alloc,
        asset_spec=spec,
        soft_gate=False,
    )
    np.testing.assert_allclose(w_synth[:cut], prefix_w, atol=1e-12)
    for arch in arch_list:
        np.testing.assert_allclose(ortho_from_synth[arch][:cut], prefix_ortho[arch], atol=1e-12)

    # 3) 验证因果滚动窗口正交化模式 (causal_window=120) 的零前视截断不变性
    ortho_causal_full = gram_schmidt_cluster_orthogonalize(collinear_clusters, causal_window=120)
    t_cut = 360
    sub_clusters = {k: v[:t_cut].copy() for k, v in collinear_clusters.items()}
    ortho_causal_sub = gram_schmidt_cluster_orthogonalize(sub_clusters, causal_window=120)
    for k in ALL_ARCHETYPES:
        assert np.allclose(ortho_causal_sub[k], ortho_causal_full[k][:t_cut], atol=1e-12)


# ===========================================================================
# Step 3: 连续信号滞回状态机与截面强弱风险预算倾斜验证
# ===========================================================================
def test_step3_continuous_state_machine_hysteresis_and_cs_tilt() -> None:
    """
    验证：
    1. `ContinuousSignalStateMachine` 的非对称进出场滞回门限 (`entry_threshold=0.08`, `exit_threshold=0.03`)
       与同向调仓死区 (`rebalance_deadband=0.06`) 将高频微幅抖动换手率降低 > 40%，
       且在方向反转或低于 `exit_threshold` 清仓时零延迟执行；
    2. `apply_cross_sectional_budget_tilt` 严格保持原多空符号不变，且多头在截面龙头 (R > 0) 上获得更高预算、
       空头在截面弱势品种 (R < 0) 上获得更高预算。
    """
    sm = ContinuousSignalStateMachine(
        entry_threshold=0.08,
        exit_threshold=0.03,
        rebalance_deadband=0.06,
    )

    # 1) 验证状态机精确滞回转移与零延迟反转/清仓行为
    manual_seq = np.array(
        [
            0.00,
            0.05,  # < entry_threshold (0.08): 保持空仓 0.0
            0.075,  # < entry_threshold (0.08): 保持空仓 0.0
            0.12,  # >= 0.08: 触发入场 -> 0.12
            0.15,  # |0.15 - 0.12| = 0.03 < deadband (0.06): 保持 0.12
            0.05,  # >= exit_threshold (0.03) 但 |0.05 - 0.12| = 0.07 >= 0.06: 调仓至 0.05 (滞回区间维持持仓!)
            0.04,  # >= exit_threshold (0.03) 且 |0.04 - 0.05| = 0.01 < 0.06: 维持 0.05
            0.02,  # < exit_threshold (0.03): 零延迟清仓 -> 0.0
            0.06,  # 清仓后 < entry_threshold (0.08): 保持空仓 0.0
            0.25,  # >= 0.08: 重新入场 -> 0.25
            -0.30,  # 方向反转且 |-0.30| >= 0.08: 零延迟反手 -> -0.30
            0.04,  # 方向反转但 |0.04| < 0.08: 零延迟平空归零 -> 0.0
        ],
        dtype=np.float64,
    )
    expected_seq = np.array([0.0, 0.0, 0.0, 0.12, 0.12, 0.05, 0.05, 0.0, 0.0, 0.25, -0.30, 0.0], dtype=np.float64)
    out_manual = sm.transform(manual_seq)
    assert np.allclose(out_manual, expected_seq, atol=1e-12)

    # 2) 验证高频微幅抖动换手率压降 > 40%
    rng = np.random.default_rng(303)
    n_bars = 2000
    base_wave = 0.45 * np.sin(np.linspace(0.0, 8.0 * np.pi, n_bars))
    # 叠加幅度 ±0.035 的高频连续抖动噪声
    jitter_signal = np.clip(base_wave + rng.uniform(-0.035, 0.035, size=n_bars), -1.0, 1.0)
    filtered_signal = sm.transform(jitter_signal)

    raw_turnover = float(np.sum(np.abs(np.diff(jitter_signal, prepend=0.0))))
    sm_turnover = float(np.sum(np.abs(np.diff(filtered_signal, prepend=0.0))))
    turnover_reduction = 1.0 - (sm_turnover / raw_turnover)
    assert turnover_reduction > 0.40, f"状态机换手率下降比例 {turnover_reduction:.2%} 应 > 40%"

    # 3) 验证跨品种截面强弱风险预算倾斜 apply_cross_sectional_budget_tilt
    ts_w = np.array([0.60, 0.60, 0.60, -0.60, -0.60, -0.60, 0.0, 0.0], dtype=np.float64)
    # 分别对应: 强龙头(+0.9), 中性(0.0), 弱势(-0.9)
    cs_str = np.array([0.90, 0.00, -0.90, -0.90, 0.00, 0.90, 0.95, -0.95], dtype=np.float64)
    tilted_w = apply_cross_sectional_budget_tilt(ts_w, cs_str, tilt_Floor=0.25, tilt_Cap=1.50)

    # 严格不改变原时序信号的多空方向符号，且零仓位保持为零
    assert np.array_equal(np.sign(tilted_w), np.sign(ts_w))
    assert tilted_w[6] == 0.0 and tilted_w[7] == 0.0

    # 多头在截面龙头 (R=+0.90) 上预算放大，在截面弱势 (R=-0.90) 上预算收缩
    assert tilted_w[0] > tilted_w[1] > tilted_w[2] > 0.0
    assert tilted_w[0] == pytest.approx(0.60 * (0.5 + 0.5 * 1.90), abs=1e-12)
    assert tilted_w[2] == pytest.approx(0.60 * (0.5 + 0.5 * 0.10), abs=1e-12)

    # 空头在截面弱势品种 (R=-0.90) 上空头预算放大，在截面强龙头 (R=+0.90) 上空头预算收缩防逼空
    assert abs(tilted_w[3]) > abs(tilted_w[4]) > abs(tilted_w[5]) > 0.0
    assert tilted_w[3] == pytest.approx(-0.60 * (0.5 + 0.5 * 1.90), abs=1e-12)
    assert tilted_w[5] == pytest.approx(-0.60 * (0.5 + 0.5 * 0.10), abs=1e-12)


# ===========================================================================
# Step 4: 动态活跃池分配器 (1/N_active) 消除空仓品种现金拖累验证
# ===========================================================================
def test_step4_dynamic_active_pool_eliminates_cash_drag() -> None:
    """
    构造 8 品种跨板块组合（其中任意时刻仅 3~4 个品种有活跃信号，其余 4~5 个品种空仓静默），
    对比静态 1/N_total 分配与动态活跃池 1/N_active + 逆波动率 + Sector HRP 分配，验证：
    1. 单品种权重上限严格 <= 0.35；
    2. 当仅 1 个品种激活时，受 min_effective_assets=3 保底约束，单品种权重 <= 1/3 <= 0.35；
    3. 动态活跃池显著消除空仓品种造成的现金拖累，有效资金利用率与夏普比率大幅优于静态 1/N_total。
    """
    specs = _build_8_asset_specs()
    asset_sectors = {sym: specs[sym].sector for sym in ALL_8_SYMBOLS}

    # 从 czsc.mock 生成 8 个品种日线收益率底座
    dates = pd.date_range("2021-01-04", "2024-12-31", freq="B")
    n_days = len(dates)
    rng = np.random.default_rng(202604)

    daily_returns: dict[str, pd.Series] = {}
    daily_positions: dict[str, pd.Series] = {}
    daily_cs_strength: dict[str, pd.Series] = {}

    # 构造轮动激活表：每天从 8 个品种中轮流激活 3~4 个品种，其余品种持仓为 0 (收益为 0)
    active_matrix = np.zeros((n_days, len(ALL_8_SYMBOLS)), dtype=np.float64)
    for d_idx in range(n_days):
        if d_idx == 50:
            # 专门留出一天仅 1 个品种激活，用于检验 min_effective_assets=3 防单票满仓赌博
            active_matrix[d_idx, 0] = 0.80
        else:
            n_act = 3 if (d_idx % 2 == 0) else 4
            start_col = (d_idx // 5) % len(ALL_8_SYMBOLS)
            chosen = [(start_col + k * 2) % len(ALL_8_SYMBOLS) for k in range(n_act)]
            for c_i in chosen:
                active_matrix[d_idx, c_i] = 0.75

    for idx, sym in enumerate(ALL_8_SYMBOLS):
        df_d = generate_symbol_kines(sym, "日线", sdt="20210101", edt="20241231", seed=500 + idx)
        raw_r = df_d["close"].pct_change().fillna(0.0).to_numpy(dtype=np.float64)
        raw_r = np.resize(raw_r, n_days) if len(raw_r) < n_days else raw_r[:n_days]

        vol_scale = 0.008 if sym in BLACK_SECTOR_SYMBOLS else 0.011
        centered_noise = (raw_r - np.mean(raw_r)) / max(np.std(raw_r), 1e-6) * vol_scale
        act_col = active_matrix[:, idx]
        # Raw price moves are independent of activation; no invented alpha term.
        strat_ret = centered_noise
        cs_str = np.where(act_col > 0.0, 0.55 + 0.25 * rng.random(n_days), -0.30)

        daily_returns[sym] = pd.Series(strat_ret, index=dates)
        daily_positions[sym] = pd.Series(act_col, index=dates)
        daily_cs_strength[sym] = pd.Series(cs_str, index=dates)

    allocator = DynamicActivePoolAllocator(
        target_portfolio_vol=0.14,
        max_single_asset_weight=0.35,
        min_effective_assets=3,
    )
    res_dyn = allocator.allocate(
        asset_daily_returns=daily_returns,
        asset_sectors=asset_sectors,
        asset_positions_df=daily_positions,
        asset_cs_strength_df=daily_cs_strength,
        asset_fee_rates=dict.fromkeys(ALL_8_SYMBOLS, 0.0002),
        returns_kind="price",
        positions_timing="close",
    )

    # 1) 验证单品种权重上限严格 <= 0.35
    unscaled_w_df = res_dyn["unscaled_hrp_weights_df"]
    final_w_df = res_dyn["asset_weights_df"]
    assert float(unscaled_w_df.max().max()) <= 0.35 + 1e-9
    assert float(final_w_df.max().max()) <= 0.35 + 1e-9
    assert res_dyn["max_single_asset_weight_realized"] <= 0.35 + 1e-9

    # 验证未激活品种在当日分配到的权重严格为 0.0 (零资金闲置浪费)
    inactive_bool = res_dyn["active_mask_df"] <= 0.0
    assert float(unscaled_w_df.where(inactive_bool, 0.0).abs().max().max()) < 1e-12
    assert float(final_w_df.where(inactive_bool, 0.0).abs().max().max()) < 1e-12

    # 2) 验证仅 1 个品种激活时 (d_idx=50)，min_effective_assets=3 限制其基准权重 <= 1/3
    single_active_dt = dates[50]
    assert float(res_dyn["n_active_series"].loc[single_active_dt]) == 1.0
    assert float(unscaled_w_df.loc[single_active_dt].sum()) == pytest.approx(1.0 / 3.0, abs=1e-6)

    # 3) Compare utilization only; mock returns cannot certify higher performance.
    assert res_dyn["static_active_capital_utilization_ratio"] < 0.46  # (3.5 / 8 = 0.4375)
    assert res_dyn["active_capital_utilization_ratio"] > 0.95
    assert res_dyn["utilization_lift_ratio"] > 2.0
    assert (res_dyn["cost_returns_df"] >= 0.0).all().all()
    assert np.isfinite(res_dyn["daily_returns"]).all()


# ===========================================================================
# Step 5: 多品种端到端流水线 + 单边 10 BP 冲击成本压力审计
# ===========================================================================
def test_step5_end_to_end_multi_asset_portfolio_10bp_stress_audit(tmp_path: Path) -> None:
    """Exercise the multi-asset pipeline and assert 10 BP costs, not mock alpha."""
    stress_fee_rate = 0.0010  # 单边 10 BP (0.10%)，每笔实际成交名义额计费
    specs = _build_8_asset_specs(fee_rate=stress_fee_rate)
    is_end_dt = pd.Timestamp("2022-03-01 23:59:59")

    dfs_30m: dict[str, pd.DataFrame] = {}
    dfs_60m: dict[str, pd.DataFrame] = {}

    for idx, sym in enumerate(ALL_8_SYMBOLS):
        raw = _generate_mock_30m(sym, seed=600 + idx * 11, sdt="20220101", edt="20220430")
        sym_cache = tmp_path / sym.replace(".", "_")
        sym_cache.mkdir(parents=True, exist_ok=True)

        d30 = enrich_kline_features(raw.copy(), scale=1.0)
        d30 = attach_native_czsc_signals(d30, "30m", sym_cache)

        # 聚合已收盘 60m 大级别特征
        m = (len(raw) // 2) * 2
        sub = raw.iloc[:m]
        grp = np.arange(m) // 2
        htf = (
            sub.groupby(grp)
            .agg(
                dt=("dt", "last"),
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                volume=("volume", "sum"),
            )
            .reset_index(drop=True)
        )
        htf["symbol"] = sym
        htf["date"] = htf["dt"].dt.date
        d60 = enrich_kline_features(htf, scale=1.0)
        d60 = attach_native_czsc_signals(d60, "60m", sym_cache)

        set_factor_context(d30, htf=d60)
        dfs_30m[sym] = d30
        dfs_60m[sym] = d60

    # 计算并挂载 8 品种零前视截面特征
    cs_dfs_30m = CrossAssetDataEngine.compute_causal_cross_sectional_features(
        dfs_30m,
        specs=specs,
        tf="30m",
        slope_window=20,
        spread_window=20,
        lag_window=5,
        attach_to_dfs=True,
    )
    for sym in ALL_8_SYMBOLS:
        set_factor_context(cs_dfs_30m[sym], htf=dfs_60m[sym])

    synthesizer = IntraAssetSixClusterSynthesizer()
    state_machine = ContinuousSignalStateMachine(
        entry_threshold=0.08,
        exit_threshold=0.03,
        rebalance_deadband=0.08,
        min_holding_half_life_bars=2.0,
    )
    eb_weighter = RollingEmpiricalBayesWeighter(
        prior_anchor_lambda=0.70,
        lookback_bars=80,
        min_bars=20,
    )

    asset_daily_returns: dict[str, pd.Series] = {}
    asset_daily_positions: dict[str, pd.Series] = {}
    asset_daily_cs_strength: dict[str, pd.Series] = {}
    asset_sectors = {sym: specs[sym].sector for sym in ALL_8_SYMBOLS}

    selected_families = ("AC01", "BC01", "CC01", "DC01", "EC01", "FC01", "FC02")

    for sym in ALL_8_SYMBOLS:
        df_sym = cs_dfs_30m[sym]
        n = len(df_sym)
        unit_weights_on_grid: dict[str, np.ndarray] = {}
        prior_map: dict[str, float] = {}
        for fam in selected_families:
            _, w_l = compute_atomic_signal(df_sym, f"{fam}L")
            _, w_s = compute_atomic_signal(df_sym, f"{fam}S")
            uid_l = f"{fam}L@30m"
            uid_s = f"{fam}S@30m"
            unit_weights_on_grid[uid_l] = w_l
            unit_weights_on_grid[uid_s] = w_s
            prior_map[uid_l] = 0.35
            prior_map[uid_s] = 0.35

        eb_res = eb_weighter.compute_weights(
            cross_asset_prior=prior_map,
            df_grid=df_sym,
            unit_weights_on_grid=unit_weights_on_grid,
            fee_rate=stress_fee_rate,
        )

        cs_rank = df_sym["cs_slope_rank"].to_numpy(dtype=np.float64)
        cs_str = df_sym["cs_strength"].to_numpy(dtype=np.float64)

        w_pos, ortho_clusters = synthesizer.synthesize_cluster_orthogonals(
            df_grid=df_sym,
            unit_weights_on_grid=unit_weights_on_grid,
            unit_alloc_map=eb_res,
            asset_spec=specs[sym],
            cs_slope_rank=cs_rank,
            soft_gate=False,
            state_machine=state_machine,
        )
        assert len(w_pos) == n
        assert len(ortho_clusters) == 6

        price_ret, signed_pos = build_asset_execution_inputs(df_sym, w_pos)
        asset_daily_returns[sym] = price_ret
        asset_daily_positions[sym] = signed_pos
        asset_daily_cs_strength[sym] = (
            pd.Series(cs_str, index=pd.DatetimeIndex(pd.to_datetime(df_sym["dt"])))
            .shift(1)
            .reindex(price_ret.index)
            .ffill()
            .fillna(0.0)
        )

    hrp_allocator = CrossAssetHRPSynthesizer(
        target_portfolio_vol=0.14,
        max_single_asset_weight=0.35,
        min_effective_assets=3,
    )
    port_res = hrp_allocator.build_multi_asset_portfolio(
        asset_daily_returns=asset_daily_returns,
        asset_sectors=asset_sectors,
        is_end_date=is_end_dt,
        asset_positions_df=asset_daily_positions,
        asset_cs_strength_df=asset_daily_cs_strength,
        use_dynamic_active_pool=True,
        asset_fee_rates=dict.fromkeys(ALL_8_SYMBOLS, stress_fee_rate),
        returns_kind="price",
        positions_timing="execution",
        periods_per_year=252 * 8,
    )

    # Engineering accounting invariants, not evidence of profitable mock strategies.
    weights = port_res["asset_weights_df"]
    execution_prices = {}
    for symbol, bars in cs_dfs_30m.items():
        prices = pd.Series(bars["open"].to_numpy(), index=pd.DatetimeIndex(bars["dt"]))
        prices.loc[prices.index[-1] + pd.Timedelta(microseconds=1)] = float(bars["close"].iloc[-1])
        execution_prices[symbol] = prices
    prices_df = pd.DataFrame(execution_prices).reindex(index=weights.index, columns=weights.columns).ffill()
    capital = 250_000.0
    quantities = weights * capital / prices_df
    previous_quantities = quantities.shift(1).fillna(0.0)
    expected_gross = previous_quantities * prices_df.diff().fillna(0.0) / capital
    expected_cost = (quantities - previous_quantities).abs() * prices_df * stress_fee_rate / capital
    pd.testing.assert_frame_equal(port_res["gross_returns_df"], expected_gross)
    pd.testing.assert_frame_equal(port_res["cost_returns_df"], expected_cost)
    np.testing.assert_allclose(port_res["bar_net_returns"], (expected_gross - expected_cost).sum(axis=1))
    assert np.isfinite(port_res["daily_returns"]).all()
    assert port_res["max_single_asset_weight_realized"] <= 0.35 + 1e-9
