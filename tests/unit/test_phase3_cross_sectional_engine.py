"""Issue #6 · [Architecture][Phase 3] 跨品种截面强弱引擎：零前视因果截面特征、连续 Sigmoid 软门控与 F 簇共振因子专项测试。

使用 `czsc.mock.generate_symbol_kines` 构造多品种多板块测试池（2 个板块 × 各 4 个品种 = 8 个品种），完整验证：
1. 截面特征零前视审计（Zero-Lookahead Audit）：
   在任意时间截断点 t 对 8 个品种同步截断重算 `cs_slope_rank`、`sector_leader_dir`、`leader_lag_gap`、
   `sector_spread_zscore` 以及 `FC01~FC04` 因子，验证与全量计算的前 t 根结果严格一致（`np.allclose`）；
2. 截面多空镜像对称审计（Mirror Symmetry Audit）：
   将全市场 8 个品种的价格同时做镜像翻转 P'_{i,t} = M_i - P_{i,t}，验证 `cs_slope_rank` 翻转为 1 - R_{i,t}，
   `sector_leader_dir` 与 `sector_spread_zscore` 严格反号，且 `FC01~FC04` 的触发条件、质量与结构止损逐侧对应；
3. 连续 Sigmoid 软门控平滑性与防逼空验证：
   - 验证 `SoftCrossSectionalGate` 关于 R_{i,t} ∈ [0, 1] 单调、连续且可微（在 R=0.70 与 R=0.30 邻域微小扰动
     ΔR = 0.01 时门控输出无阶跃跳变，|Δg| < 0.05）；
   - 在实际 mock 价格路径上验证连续风险缩放保留全部触发事件，逆势做空亏损不会从准入样本中消失；
4. `F` 簇新因子接通 Phase 2 法医检验闭环：
   验证 `FC01~FC04`（及 `F01~F03` 别名）正常输出指标；mock 测试不证明经济准入通过。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from czsc.mock import generate_symbol_kines
from examples.alpha_mining_agent.hypothesis_bank import (
    ARCH_A,
    ARCH_B,
    ARCH_C,
    ARCH_F,
    attach_native_czsc_signals,
    enrich_kline_features,
)
from examples.multi_factor_cta.atomic_factors import (
    ATOMIC_FACTOR_DEFS,
    CS_FEATURE_COLS,
    DataFrameCrossSectionProvider,
    FactorContext,
    SoftCrossSectionalGate,
    compute_atomic_signal,
    set_factor_context,
    two_stage_trigger,
)
from examples.multi_factor_cta.data_engine import (
    AssetClass,
    AssetSpec,
    CrossAssetDataEngine,
    ExecutionMode,
    compute_causal_cross_sectional_features,
)
from examples.multi_factor_cta.factor_engine import RuleFactorRegistry
from examples.multi_factor_cta.forensic_validator import (
    RuleFactorForensicValidator,
    run_stratified_pooled_gate1b_audit,
)

PRICE_COLS = ("open", "high", "low", "close")
F_FAMILIES = ("FC01", "FC02", "FC03", "FC04")

# 2 个同质商品板块 × 各 4 个活跃品种 = 8 个品种
BLACK_SECTOR_SYMBOLS = ("SHFE.RB9999", "DCE.I9999", "DCE.J9999", "DCE.JM9999")
METAL_SECTOR_SYMBOLS = ("SHFE.CU9999", "SHFE.AL9999", "SHFE.ZN9999", "SHFE.NI9999")
ALL_8_SYMBOLS = BLACK_SECTOR_SYMBOLS + METAL_SECTOR_SYMBOLS


def _build_8_asset_specs() -> dict[str, AssetSpec]:
    specs: dict[str, AssetSpec] = {}
    for sym in BLACK_SECTOR_SYMBOLS:
        specs[sym] = AssetSpec(
            symbol=sym,
            name=sym,
            asset_class=AssetClass.COMMODITY_FUTURES,
            sector="BLACK_METALS",
            base_tf="30m",
            eligible_tfs=("60m", "30m"),
            fee_rate=0.0002,
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
            fee_rate=0.0002,
            execution_mode=ExecutionMode.LONG_SHORT,
        )
    return specs


def _generate_raw_30m(symbol: str, seed: int) -> pd.DataFrame:
    df = generate_symbol_kines(symbol, "30分钟", sdt="20220101", edt="20221031", seed=seed)
    df = df.rename(columns={"vol": "volume"})
    df["dt"] = pd.to_datetime(df["dt"])
    df = df[["dt", "symbol", *PRICE_COLS, "volume"]].sort_values("dt").reset_index(drop=True)
    # 量化为整数分值，使镜像翻转 M_i - P_{i,t} 消去浮点舍入尾差
    for col in PRICE_COLS:
        df[col] = np.round(df[col].astype(float) * 100.0)
    df["volume"] = df["volume"].astype(float)
    return df


def _mirror_df(raw: pd.DataFrame) -> pd.DataFrame:
    """对单品种价格序列做镜像翻转 P'_{i,t} = M_i - P_{i,t}，high/low 互换，成交量不变。"""
    m_i = float(raw["high"].max() + raw["low"].min())
    out = raw.copy()
    out["open"] = m_i - raw["open"].to_numpy(dtype=np.float64)
    out["close"] = m_i - raw["close"].to_numpy(dtype=np.float64)
    out["high"] = m_i - raw["low"].to_numpy(dtype=np.float64)
    out["low"] = m_i - raw["high"].to_numpy(dtype=np.float64)
    return out


def _aggregate_60m_from_30m(raw_30m: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """将相邻 2 根已收盘 30m 聚合为 60m K 线，仅保留完结分组以消除半成品前视。"""
    m = (len(raw_30m) // 2) * 2
    sub = raw_30m.iloc[:m]
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
    htf["symbol"] = symbol
    return htf


def _prepare_pool_pipeline(
    raw_pool_30m: dict[str, pd.DataFrame],
    specs: dict[str, AssetSpec],
    cache_root: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """对 8 个品种同时构建 30m 与 60m 特征 + 原生 Rust 信号 + 零前视截面特征并挂载上下文。"""
    dfs_30m: dict[str, pd.DataFrame] = {}
    dfs_60m: dict[str, pd.DataFrame] = {}
    for sym, raw in raw_pool_30m.items():
        sym_dir = cache_root / sym.replace(".", "_")
        sym_dir.mkdir(parents=True, exist_ok=True)

        d30 = raw.copy()
        d30["date"] = d30["dt"].dt.date
        d30 = enrich_kline_features(d30, scale=1.0)
        d30 = attach_native_czsc_signals(d30, "30m", sym_dir)

        raw60 = _aggregate_60m_from_30m(raw, sym)
        d60 = raw60.copy()
        d60["date"] = d60["dt"].dt.date
        d60 = enrich_kline_features(d60, scale=1.0)
        d60 = attach_native_czsc_signals(d60, "60m", sym_dir)

        set_factor_context(d30, htf=d60)
        dfs_30m[sym] = d30
        dfs_60m[sym] = d60

    # 在 30m 上计算零前视截面特征并自动挂载 DataFrameCrossSectionProvider
    cs_dfs_30m = CrossAssetDataEngine.compute_causal_cross_sectional_features(
        dfs_30m,
        specs=specs,
        tf="30m",
        slope_window=20,
        spread_window=20,
        lag_window=5,
        attach_to_dfs=True,
    )
    for sym in raw_pool_30m:
        set_factor_context(cs_dfs_30m[sym], htf=dfs_60m[sym])
    return cs_dfs_30m, dfs_60m


@pytest.fixture(scope="module")
def specs_8() -> dict[str, AssetSpec]:
    return _build_8_asset_specs()


@pytest.fixture(scope="module")
def raw_pool_8() -> dict[str, pd.DataFrame]:
    return {sym: _generate_raw_30m(sym, seed=300 + idx * 7) for idx, sym in enumerate(ALL_8_SYMBOLS)}


@pytest.fixture(scope="module")
def full_pool_built(
    raw_pool_8: dict[str, pd.DataFrame],
    specs_8: dict[str, AssetSpec],
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    cache_dir = tmp_path_factory.mktemp("phase3_full_pool")
    return _prepare_pool_pipeline(raw_pool_8, specs_8, cache_dir)


# ===========================================================================
# 1. 截面特征与 FC01~FC04 零前视审计 (Zero-Lookahead Audit)
# ===========================================================================
@pytest.mark.parametrize("t_cut", [420, 760])
def test_step1_cross_sectional_features_and_fc01_fc04_zero_lookahead(
    t_cut: int,
    raw_pool_8: dict[str, pd.DataFrame],
    specs_8: dict[str, AssetSpec],
    full_pool_built: tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]],
    tmp_path: Path,
) -> None:
    """在任意时间截断点 t 对 8 个品种同步截断重算截面特征与 FC01~FC04，验证与全量前 t 根严格一致。"""
    full_30m, _ = full_pool_built
    trunc_raw = {sym: df.iloc[:t_cut].reset_index(drop=True) for sym, df in raw_pool_8.items()}
    trunc_30m, _ = _prepare_pool_pipeline(trunc_raw, specs_8, tmp_path / f"trunc_{t_cut}")

    for sym in ALL_8_SYMBOLS:
        df_full = full_30m[sym]
        df_sub = trunc_30m[sym]
        assert len(df_sub) == t_cut

        # 1) 验证全部跨品种截面特征零前视
        for col in CS_FEATURE_COLS:
            assert np.allclose(
                df_sub[col].to_numpy(dtype=np.float64),
                df_full[col].to_numpy(dtype=np.float64)[:t_cut],
                atol=1e-12,
                rtol=0.0,
            ), f"{sym} 截面特征 {col} 在截断点 t={t_cut} 存在前视泄露"

        # 2) 验证 FC01~FC04 双向与单向连续因子零前视
        for fam in F_FAMILIES:
            for side_sfx in ("L", "S"):
                sid = f"{fam}{side_sfx}"
                trig_f, w_f = compute_atomic_signal(df_full, sid)
                trig_t, w_t = compute_atomic_signal(df_sub, sid)
                assert np.array_equal(trig_t, trig_f[:t_cut]), f"{sym} {sid} 触发脉冲在 t={t_cut} 存在前视"
                assert np.allclose(w_t, w_f[:t_cut], atol=1e-12, rtol=0.0), f"{sym} {sid} 连续强度在 t={t_cut} 存在前视"


def test_step1_cross_section_provider_asof_alignment_zero_lookahead(
    raw_pool_8: dict[str, pd.DataFrame],
    specs_8: dict[str, AssetSpec],
) -> None:
    """验证 60m 预计算截面特征通过 DataFrameCrossSectionProvider 向 30m 映射时严格满足 backward 因果对齐。"""
    raw_60m_pool = {sym: _aggregate_60m_from_30m(df, sym) for sym, df in raw_pool_8.items()}
    cs_60m = compute_causal_cross_sectional_features(raw_60m_pool, specs=specs_8, tf="60m", attach_to_dfs=True)

    t_cut_30m = 501  # 奇数截断点：最后一根 30m 处于未完结 60m 分组中间
    cut_dt = raw_pool_8[ALL_8_SYMBOLS[0]]["dt"].iloc[t_cut_30m - 1]
    trunc_60m_pool = {
        sym: _aggregate_60m_from_30m(df.iloc[:t_cut_30m].reset_index(drop=True), sym) for sym, df in raw_pool_8.items()
    }
    cs_60m_trunc = compute_causal_cross_sectional_features(trunc_60m_pool, specs=specs_8, tf="60m", attach_to_dfs=True)

    for sym in ALL_8_SYMBOLS:
        assert cs_60m_trunc[sym]["dt"].max() <= cut_dt
        prov_full = DataFrameCrossSectionProvider(cs_60m[sym], default_tf="60m")
        prov_trunc = DataFrameCrossSectionProvider(cs_60m_trunc[sym], default_tf="60m")

        feats_full = prov_full.features(raw_pool_8[sym])
        feats_trunc = prov_trunc.features(raw_pool_8[sym].iloc[:t_cut_30m].reset_index(drop=True))
        for col in CS_FEATURE_COLS:
            assert np.allclose(feats_trunc[col], feats_full[col][:t_cut_30m], atol=1e-12, rtol=0.0)


# ===========================================================================
# 2. 截面多空镜像对称审计 (Mirror Symmetry Audit)
# ===========================================================================
def test_step2_cross_sectional_features_and_fc01_fc04_mirror_symmetry(
    raw_pool_8: dict[str, pd.DataFrame],
    specs_8: dict[str, AssetSpec],
    full_pool_built: tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]],
    tmp_path: Path,
) -> None:
    """
    将全市场 8 个品种的价格同时做镜像翻转 P'_{i,t} = M_i - P_{i,t}，验证：
    1. cs_slope_rank 翻转为 1 - R_{i,t}；
    2. sector_leader_dir、leader_lag_gap 与 sector_spread_zscore 严格反号；
    3. FC01~FC04 的 Setup、Confirmation、质量与结构止损逐侧镜像一致。

    百分比跟踪止盈依赖价格分母，仿射镜像不保持百分比回撤；其两侧退出阈值由原生回归单独验证。
    """
    orig_30m, orig_60m = full_pool_built
    mirror_raw = {sym: _mirror_df(df) for sym, df in raw_pool_8.items()}
    mirror_30m, mirror_60m = _prepare_pool_pipeline(mirror_raw, specs_8, tmp_path / "mirror_pool")

    for sym in ALL_8_SYMBOLS:
        df_p = orig_30m[sym]
        df_m = mirror_30m[sym]

        # 1) cs_slope_rank 翻转为 1 - R_{i,t}，中心化 cs_strength 严格反号
        r_p = df_p["cs_slope_rank"].to_numpy(dtype=np.float64)
        r_m = df_m["cs_slope_rank"].to_numpy(dtype=np.float64)
        assert np.allclose(r_m, 1.0 - r_p, atol=1e-12), f"{sym} cs_slope_rank 镜像翻转不等于 1 - R"
        assert np.allclose(
            df_m["cs_strength"].to_numpy(dtype=np.float64),
            -df_p["cs_strength"].to_numpy(dtype=np.float64),
            atol=1e-12,
        )

        # 2) sector_leader_dir、leader_lag_gap、sector_spread_zscore 严格反号
        for signed_col in ("sector_leader_dir", "leader_lag_gap", "sector_spread_zscore"):
            v_p = df_p[signed_col].to_numpy(dtype=np.float64)
            v_m = df_m[signed_col].to_numpy(dtype=np.float64)
            assert np.allclose(v_m, -v_p, atol=1e-11), f"{sym} {signed_col} 在价格镜像下未严格反号"

    # 3) Check the symmetric setup/confirmation rule and structural invalidation itself.
    # Percentage trailing risk is intentionally tested with percentage paths in native tests.
    contexts = {
        sym: (
            FactorContext.from_df(orig_30m[sym], orig_60m[sym]),
            FactorContext.from_df(mirror_30m[sym], mirror_60m[sym]),
        )
        for sym in ALL_8_SYMBOLS
    }
    for fam in F_FAMILIES:
        definition = ATOMIC_FACTOR_DEFS[fam]
        candidates = 0
        for symbol, (ctx_p, ctx_m) in contexts.items():
            np.testing.assert_array_equal(ctx_p.bi_flip, -ctx_m.bi_flip)
            anchors_p = ctx_p.structure_anchors[definition.archetype]
            anchors_m = ctx_m.structure_anchors[definition.archetype]
            for side in (1, -1):
                setup_p, sq_p, confirm_p, cq_p, window_p = definition.build(ctx_p, side)
                setup_m, sq_m, confirm_m, cq_m, window_m = definition.build(ctx_m, -side)
                assert window_p == window_m
                np.testing.assert_array_equal(setup_p, setup_m, err_msg=f"{fam} {symbol} {side}: setup")
                np.testing.assert_array_equal(confirm_p, confirm_m, err_msg=f"{fam} {symbol} {side}: confirm")
                np.testing.assert_allclose(sq_p[setup_p], sq_m[setup_m], atol=0.011)
                np.testing.assert_allclose(cq_p[confirm_p], cq_m[confirm_m], atol=0.011)
                fire_p, q_p = two_stage_trigger(setup_p, sq_p, confirm_p, cq_p, window_p)
                fire_m, q_m = two_stage_trigger(setup_m, sq_m, confirm_m, cq_m, window_m)
                np.testing.assert_array_equal(fire_p, fire_m)
                np.testing.assert_allclose(q_p[fire_p], q_m[fire_m], atol=0.011)
                candidates += int(fire_p.sum())
                key_p, key_m = ("long_stop", "short_stop") if side > 0 else ("short_stop", "long_stop")
                np.testing.assert_allclose(
                    side * (ctx_p.c - np.asarray(anchors_p[key_p])),
                    -side * (ctx_m.c - np.asarray(anchors_m[key_m])),
                    atol=1e-9,
                )
        assert candidates >= 50, f"{fam} 在 8 品种池上应有足够的镜像候选"


# ===========================================================================
# 3. 连续 Sigmoid 软门控平滑性与防逼空验证
# ===========================================================================
def test_step3_soft_cross_sectional_gate_smoothness_and_monotonicity() -> None:
    """
    验证 SoftCrossSectionalGate 关于 R_{i,t} ∈ [0, 1] 单调、连续且可微：
    在 R=0.70 与 R=0.30 邻域微小扰动 ΔR = 0.01 时门控输出无阶跃跳变，|Δg| < 0.05。
    """
    gate = SoftCrossSectionalGate(k_b=12.0, b_short_center=0.70, b_long_center=0.30)
    r_grid = np.linspace(0.0, 1.0, 1001)  # 步长 ΔR = 0.001

    # 1) B 簇多空门控：严格单调、平滑且镜像对称
    g_b_short = gate.gate_multiplier(r_grid, er=0.5, archetype=ARCH_B, direction=-1)
    g_b_long = gate.gate_multiplier(r_grid, er=0.5, archetype=ARCH_B, direction=1)

    assert np.all((g_b_short > 0.0) & (g_b_short < 1.0))
    assert np.all((g_b_long > 0.0) & (g_b_long < 1.0))
    assert np.all(np.diff(g_b_short) < 0.0), "g_{B,short}(R) 必须关于 R 严格单调递减"
    assert np.all(np.diff(g_b_long) > 0.0), "g_{B,long}(R) 必须关于 R 严格单调递增"
    assert np.allclose(g_b_long, g_b_short[::-1], atol=1e-12), "g_{B,long}(1-R) 必须严格等于 g_{B,short}(R)"

    # 在临界点 R=0.70 与 R=0.30 处精确为 0.5，且在 R->1 时空头压制趋近 0，在 R->0 时多头压制趋近 0
    assert float(gate.gate_b_short(0.70)) == pytest.approx(0.5, abs=1e-12)
    assert float(gate.gate_b_long(0.30)) == pytest.approx(0.5, abs=1e-12)
    assert float(gate.gate_b_short(1.0)) < 0.03
    assert float(gate.gate_b_long(0.0)) < 0.03

    # 验证任意 ΔR = 0.01 微扰下无阶跃跳变 (|Δg| < 0.05)，重点检查 R=0.70 与 R=0.30 邻域
    for center in (0.30, 0.70):
        for arch in (ARCH_A, ARCH_B, ARCH_C):
            for d in (-1, 1):
                g_minus = float(gate.gate_multiplier(center - 0.005, er=0.5, archetype=arch, direction=d))
                g_plus = float(gate.gate_multiplier(center + 0.005, er=0.5, archetype=arch, direction=d))
                delta_g = abs(g_plus - g_minus)
                assert delta_g < 0.05, f"{arch} dir={d} 在 R={center} 处 ΔR=0.01 跳变过大: {delta_g:.4f}"

    # 全局步长 ΔR = 0.01 (10 个网格步长) 最大变化率验证
    for arch in (ARCH_A, ARCH_B, ARCH_C):
        for d in (-1, 1):
            g_curve = gate.gate_multiplier(r_grid, er=0.5, archetype=arch, direction=d)
            max_step_001 = float(np.max(np.abs(g_curve[10:] - g_curve[:-10])))
            assert max_step_001 < 0.05
            # 二阶差分连续微小，证明二阶可微无折点
            second_diff = np.max(np.abs(np.diff(g_curve, n=2)))
            assert second_diff < 1e-3

    # 2) C 簇居中度与低 ER 增益特性：中庸低效震荡区 (R=0.5, ER=0.1) 显著高于极端单边趋势区 (R=0.95, ER=0.9)
    g_c_choppy_mid = float(gate.gate_multiplier(0.50, er=0.10, archetype=ARCH_C, direction=-1))
    g_c_trend_top = float(gate.gate_multiplier(0.95, er=0.90, archetype=ARCH_C, direction=-1))
    assert g_c_choppy_mid > 0.80
    assert g_c_trend_top < 0.08


def test_step3_soft_gate_scales_risk_without_deleting_events(
    raw_pool_8: dict[str, pd.DataFrame],
    specs_8: dict[str, AssetSpec],
) -> None:
    """Continuous sizing retains adverse events; generated data cannot certify profitable admission."""
    pool = {}
    for symbol, raw in raw_pool_8.items():
        df = raw.copy()
        df["date"] = df["dt"].dt.date
        pool[symbol] = enrich_kline_features(df)
    cs_pool = compute_causal_cross_sectional_features(pool, specs=specs_8, tf="30m")
    gate = SoftCrossSectionalGate()
    triggers_by_sym, plain_weights, gated_weights = {}, {}, {}
    adverse_events = 0
    for symbol, df in cs_pool.items():
        triggers = np.arange(len(df)) % 17 == 0
        weights = -triggers.astype(float)
        rank = df["cs_slope_rank"].to_numpy()
        er = gate.extract_causal_er(df)
        gated_triggers, gated = gate.apply_to_triggers_and_weights(triggers, weights, rank, er, ARCH_B, -1)
        np.testing.assert_array_equal(gated_triggers, triggers)
        np.testing.assert_allclose(gated, weights * gate.gate_b_short(rank))
        assert np.all(np.abs(gated) <= np.abs(weights))
        # Evaluate the same adverse realized moves after sizing; losses are reduced, never erased.
        next_return = df["close"].pct_change().shift(-1).fillna(0).to_numpy()
        adverse = triggers & (next_return > 0) & (rank > 0.8)
        adverse_events += int(adverse.sum())
        assert np.all(gated[adverse] * next_return[adverse] < 0)
        assert np.all(gated[adverse] * next_return[adverse] > weights[adverse] * next_return[adverse])
        triggers_by_sym[symbol] = triggers
        plain_weights[symbol] = weights
        gated_weights[symbol] = gated
    assert adverse_events > 0, "fixture must contain actual adverse high-rank short events"

    validator = RuleFactorForensicValidator()
    plain = validator.run_stratified_pooled_gate1b_audit(
        archetype=ARCH_B,
        timeframe="30m",
        pool_tf_dfs=cs_pool,
        triggers_by_symbol=triggers_by_sym,
        weights_by_symbol=plain_weights,
        direction=-1,
        target_pool_group="commodity",
        soft_gate=False,
    )
    gated = validator.run_stratified_pooled_gate1b_audit(
        archetype=ARCH_B,
        timeframe="30m",
        pool_tf_dfs=cs_pool,
        triggers_by_symbol=triggers_by_sym,
        weights_by_symbol=gated_weights,
        direction=-1,
        target_pool_group="commodity",
        soft_gate=False,
    )
    assert set(plain["per_asset_audits"]) == set(gated["per_asset_audits"]) == set(cs_pool)
    for symbol in cs_pool:
        assert plain["per_asset_audits"][symbol]["is_n_purged"] == gated["per_asset_audits"][symbol]["is_n_purged"]


# ===========================================================================
# 4. F 簇新因子 (FC01~FC04 / F01~F03) 通过 Phase 2 法医检验闭环
# ===========================================================================
def test_step4_f_cluster_factors_forensic_validation_loop(
    full_pool_built: tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]],
) -> None:
    """验证 FC01~FC04（含 F01~F03 别名）与评价接口接通，不要求随机样本通过经济准入。"""
    cs_pool_30m, _ = full_pool_built
    registry = RuleFactorRegistry()
    validator = RuleFactorForensicValidator()

    # 1) 验证注册表元数据与 F01~F03 别名解析
    for fam in F_FAMILIES:
        assert fam in ATOMIC_FACTOR_DEFS
        for sfx, expected_dir in (("L", 1), ("S", -1)):
            fid = f"{fam}{sfx}"
            spec = registry.get(fid)
            assert spec.archetype == ARCH_F
            assert spec.direction == expected_dir
            assert spec.output_kind == "continuous"

    assert registry.get("F01L").factor_id == "FC02L"
    assert registry.get("F02S").factor_id == "FC03S"
    assert registry.get("F03L").factor_id == "FC04L"

    # 2) 在 8 品种池上通过 RuleFactorRegistry 评估 FC01~FC04，并送入 RuleFactorForensicValidator 闭环检验
    for fam in F_FAMILIES:
        sid = f"{fam}L"
        trig_by_sym: dict[str, np.ndarray] = {}
        w_by_sym: dict[str, np.ndarray] = {}
        total_triggers = 0

        for sym, df_sym in cs_pool_30m.items():
            trig, w = registry.evaluate_factor_on_tf(df_sym, sid, "30m", soft_gate=True)
            assert len(trig) == len(w) == len(df_sym)
            assert np.isfinite(w).all()
            assert (w >= 0.0).all() and (w <= 1.0 + 1e-12).all()
            trig_by_sym[sym] = trig
            w_by_sym[sym] = w
            total_triggers += int(trig.sum())

        assert total_triggers > 0, f"{sid} 在 8 品种池上应有真实触发"

        pooled_audit = run_stratified_pooled_gate1b_audit(
            archetype=ARCH_F,
            timeframe="30m",
            pool_tf_dfs=cs_pool_30m,
            triggers_by_symbol=trig_by_sym,
            weights_by_symbol=w_by_sym,
            direction=1,
            soft_gate=False,  # Registry has already applied the gate exactly once.
        )
        assert pooled_audit["pool_tier"] == "Tier1_CrossAsset"
        assert "universal_ratio" in pooled_audit and "pooled_t_stat" in pooled_audit

        # 同时验证单品种 Gate 1A 评估接口对连续强度 F 簇因子正常产出完整法医指标卡
        first_sym = ALL_8_SYMBOLS[0]
        g1a = validator.run_gate1_single_asset_audit(
            cs_pool_30m[first_sym],
            trig_by_sym[first_sym],
            direction=1,
            archetype=ARCH_F,
            timeframe="30m",
            weights_tf=w_by_sym[first_sym],
        )
        assert "passed" in g1a and "is_n_purged" in g1a
