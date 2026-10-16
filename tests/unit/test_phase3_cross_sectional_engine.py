"""Issue #6 · [Architecture][Phase 3] 跨品种截面强弱引擎：零前视因果截面特征、连续 Sigmoid 软门控与 F 簇共振因子专项测试。

使用 `czsc.mock.generate_symbol_kines` 构造多品种多板块测试池（2 个板块 × 各 4 个品种 = 8 个品种），完整验证：
1. 截面特征零前视审计（Zero-Lookahead Audit）：
   在任意时间截断点 t 对 8 个品种同步截断重算 `cs_slope_rank`、`sector_leader_dir`、`leader_lag_gap`、
   `sector_spread_zscore` 以及 `FC01~FC04` 因子，验证与全量计算的前 t 根结果严格一致（`np.allclose`）；
2. 截面多空镜像对称审计（Mirror Symmetry Audit）：
   将全市场 8 个品种的价格同时做镜像翻转 P'_{i,t} = M_i - P_{i,t}，验证 `cs_slope_rank` 翻转为 1 - R_{i,t}，
   `sector_leader_dir` 与 `sector_spread_zscore` 严格反号，且 `FC01~FC04` 的多空信号满足 S(P) ≈ -S(M(P))；
3. 连续 Sigmoid 软门控平滑性与防逼空验证：
   - 验证 `SoftCrossSectionalGate` 关于 R_{i,t} ∈ [0, 1] 单调、连续且可微（在 R=0.70 与 R=0.30 邻域微小扰动
     ΔR = 0.01 时门控输出无阶跃跳变，|Δg| < 0.05）；
   - 构造“强单边逼空上涨龙头品种上出现假顶背驰”的测试场景，验证绑定软门控后 `B_背驰衰竭簇` 的逆势做空亏损
     被大幅压降，跨商品池化正期望比例显著提升（无需硬编码 `wb = 0`）；
4. `F` 簇新因子通过 Phase 2 法医检验闭环：
   验证 `FC01~FC04`（及 `F01~F03` 别名）在多品种池上可正常通过 `RuleFactorRegistry` 与 `RuleFactorForensicValidator` 评估。
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
    compute_family,
    set_factor_context,
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
from examples.multi_factor_cta.synthesis_engine import IntraAssetThreePillarSynthesizer

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
    3. FC01~FC04 的多空信号满足 S(P) ≈ -S(M(P))，且 L(P) ≈ -S_short(M(P))。
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

    # 3) 验证 FC01~FC04 在 8 品种池上的多空信号满足 S(P) ≈ -S(M(P))
    for fam in F_FAMILIES:
        all_p = []
        all_m = []
        all_l_p = []
        all_s_m = []
        for sym in ALL_8_SYMBOLS:
            ctx_p = FactorContext.from_df(orig_30m[sym], orig_60m[sym])
            ctx_m = FactorContext.from_df(mirror_30m[sym], mirror_60m[sym])
            s_p, _ = compute_family(ctx_p, fam)
            s_m, _ = compute_family(ctx_m, fam)
            all_p.append(s_p)
            all_m.append(s_m)

            # 同时验证单向因子 L(P) 与 S(M(P)) 严格镜像反号
            _, w_l_p = compute_atomic_signal(orig_30m[sym], f"{fam}L")
            _, w_s_m = compute_atomic_signal(mirror_30m[sym], f"{fam}S")
            all_l_p.append(w_l_p)
            all_s_m.append(w_s_m)
            act_ls = (w_l_p != 0.0) | (w_s_m != 0.0)
            if np.any(act_ls):
                assert np.mean(np.abs(w_l_p[act_ls] + w_s_m[act_ls]) <= 0.03) >= 0.96

        vec_l_p = np.concatenate(all_l_p)
        vec_s_m = np.concatenate(all_s_m)
        act_pool_ls = (vec_l_p != 0.0) | (vec_s_m != 0.0)
        assert float(np.mean(np.abs(vec_l_p[act_pool_ls] + vec_s_m[act_pool_ls]) <= 0.03)) >= 0.99

        vec_p = np.concatenate(all_p)
        vec_m = np.concatenate(all_m)
        active = (vec_p != 0.0) | (vec_m != 0.0)
        assert int(active.sum()) >= 50, f"{fam} 在 8 品种池上激活样本应充足"
        diff = np.abs(vec_p + vec_m)
        assert float(np.mean(diff[active] <= 0.03)) >= 0.995, (
            f"{fam} 跨品种池镜像对称率不足: {np.mean(diff[active] <= 0.03):.4f}"
        )
        assert float(np.mean(np.sign(vec_p[active]) == -np.sign(vec_m[active]))) >= 0.995


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


def test_step3_soft_gate_prevents_short_squeeze_on_bull_leaders(
    raw_pool_8: dict[str, pd.DataFrame],
    specs_8: dict[str, AssetSpec],
) -> None:
    """
    构造“强单边逼空上涨龙头品种上出现假顶背驰”的测试场景：
    - 8 个商品品种中，2 个龙头品种 (SHFE.RB9999, SHFE.CU9999) 处于强单边逼空主升浪 (R_{i,t} > 0.85)，
      期间频繁出现假顶背驰信号，若不加软门控直接逆势做空将遭遇严重逼空亏损；
    - 其余 6 个品种在各自震荡/弱势区间 (R_{i,t} <= 0.60) 出现顶背驰做空信号均能稳定获利；
    - 验证绑定 SoftCrossSectionalGate 后：
      1) 龙头品种上的逆势摸顶做空亏损被压降 > 85%；
      2) 跨商品池化正期望比例 (universal_ratio) 从未达标提升至 >= 75%，顺利通过 Tier 2 Gate 1B；
      3) IntraAssetThreePillarSynthesizer 无需硬编码 `wb = 0`，即可在保留震荡品种反转正收益的同时防御逼空龙头亏损。
    """
    squeeze_leaders = {"SHFE.RB9999", "SHFE.CU9999"}
    pool_dfs: dict[str, pd.DataFrame] = {}
    triggers_by_sym: dict[str, np.ndarray] = {}
    weights_by_sym: dict[str, np.ndarray] = {}

    for sym_idx, sym in enumerate(ALL_8_SYMBOLS):
        df = raw_pool_8[sym].copy()
        df["date"] = df["dt"].dt.date
        n = len(df)
        c = df["close"].to_numpy(dtype=np.float64).copy()
        o = df["open"].to_numpy(dtype=np.float64).copy()
        h = df["high"].to_numpy(dtype=np.float64).copy()
        low_arr = df["low"].to_numpy(dtype=np.float64).copy()

        trig = np.zeros(n, dtype=bool)
        w_b = np.zeros(n, dtype=np.float64)
        event_bars = list(range(120, min(n - 35, 1400), 65))[:18]

        for k, idx in enumerate(event_bars):
            trig[idx] = True
            w_b[idx : idx + 6] = -0.85  # B 簇空头反转信号
            p0 = c[idx - 20]
            if sym in squeeze_leaders and k < 14:
                # 强单边逼空上涨：前 20 根持续大涨使截面排名稳居全市场 Top，触发假顶背驰后继续逼空暴涨 20 根 (做空大亏)
                for pre in range(-20, 1):
                    pos_i = idx + pre
                    boost = p0 * (1.0 + 0.0045 * (pre + 20))
                    c[pos_i] = boost
                    o[pos_i] = boost * 0.998
                    h[pos_i] = boost * 1.003
                    low_arr[pos_i] = boost * 0.996
                p_trig = c[idx]
                for step in range(1, 22):
                    squeeze_p = p_trig * (1.0 + 0.006 * step)
                    c[idx + step] = squeeze_p
                    o[idx + step] = p_trig if step == 1 else squeeze_p * 0.998
                    h[idx + step] = squeeze_p * 1.004
                    low_arr[idx + step] = p_trig * 0.999 if step == 1 else squeeze_p * 0.995
            else:
                # 中庸/弱势区真顶背驰：前 20 根温和走弱使截面排名回落至中低位，触发顶背驰后顺利下跌 20 根止盈 (做空盈利)
                pullback_rate = 0.0020 if sym in squeeze_leaders else 0.0008 * (1 + 0.1 * (sym_idx % 3))
                for pre in range(-20, 1):
                    pos_i = idx + pre
                    mild = p0 * (1.0 - pullback_rate * (pre + 20))
                    c[pos_i] = mild
                    o[pos_i] = mild * 1.001
                    h[pos_i] = mild * 1.002
                    low_arr[pos_i] = mild * 0.998
                p_trig = c[idx]
                for step in range(1, 22):
                    drop_p = p_trig * (1.0 - 0.0055 * step)
                    c[idx + step] = drop_p
                    o[idx + step] = p_trig if step == 1 else drop_p * 1.001
                    h[idx + step] = p_trig * 1.0005 if step == 1 else drop_p * 1.002
                    low_arr[idx + step] = drop_p * 0.996

        df["open"] = o
        df["high"] = h
        df["low"] = low_arr
        df["close"] = c
        df = enrich_kline_features(df, scale=1.0)
        pool_dfs[sym] = df
        triggers_by_sym[sym] = trig
        weights_by_sym[sym] = w_b

    # 计算并挂载 8 品种零前视截面特征
    cs_pool = compute_causal_cross_sectional_features(
        pool_dfs, specs=specs_8, tf="30m", slope_window=15, attach_to_dfs=True
    )

    validator = RuleFactorForensicValidator()

    # 对比 1：禁用软门控 (soft_gate=False) 时，逼空龙头品种严重亏损，拉低池化表现
    ungated_res = validator.run_stratified_pooled_gate1b_audit(
        archetype=ARCH_B,
        timeframe="30m",
        pool_tf_dfs=cs_pool,
        triggers_by_symbol=triggers_by_sym,
        weights_by_symbol=weights_by_sym,
        direction=-1,
        target_pool_group="commodity",
        soft_gate=False,
    )
    # 对比 2：启用连续 Sigmoid 软门控 (soft_gate=True)
    gated_res = validator.run_stratified_pooled_gate1b_audit(
        archetype=ARCH_B,
        timeframe="30m",
        pool_tf_dfs=cs_pool,
        triggers_by_symbol=triggers_by_sym,
        weights_by_symbol=weights_by_sym,
        direction=-1,
        target_pool_group="commodity",
        soft_gate=True,
    )

    for leader_sym in squeeze_leaders:
        loss_ungated = ungated_res["per_asset_audits"][leader_sym]["raw_mean_pct"]
        r_atr_ungated = ungated_res["per_asset_audits"][leader_sym]["norm_r_atr"]
        ret_gated = gated_res["per_asset_audits"][leader_sym]["raw_mean_pct"]
        r_atr_gated = gated_res["per_asset_audits"][leader_sym]["norm_r_atr"]
        assert loss_ungated < -0.35 and r_atr_ungated < -0.40, f"{leader_sym} 未门控时应体现逼空重亏"
        assert ret_gated > 0.0 and r_atr_gated > 0.50, (
            f"{leader_sym} 绑定软门控后逆势逼空触发被压制，期望应转正 (实际={ret_gated}, r_atr={r_atr_gated})"
        )

    assert gated_res["universal_ratio"] > ungated_res["universal_ratio"]
    assert gated_res["universal_ratio"] >= 0.85
    assert gated_res["passed"] is True

    # 对比 3：验证 IntraAssetThreePillarSynthesizer 使用 SoftCrossSectionalGate 替代硬编码 wb=0
    synth = IntraAssetThreePillarSynthesizer()
    leader_df = cs_pool["SHFE.RB9999"]
    weak_df = cs_pool["DCE.JM9999"]

    w_synth_leader = synth.synthesize(
        leader_df,
        unit_weights_on_grid={"BC01S@30m": weights_by_sym["SHFE.RB9999"]},
        unit_alloc_map={"BC01S@30m": 0.50},
        asset_spec=specs_8["SHFE.RB9999"],
        rebalance_deadzone=0.01,
    )
    w_synth_weak = synth.synthesize(
        weak_df,
        unit_weights_on_grid={"BC01S@30m": weights_by_sym["DCE.JM9999"]},
        unit_alloc_map={"BC01S@30m": 0.50},
        asset_spec=specs_8["DCE.JM9999"],
        rebalance_deadzone=0.01,
    )
    # 在弱势/中庸品种触发窗口上 B 簇信号得以保留（非全 0），而在强逼空龙头品种的 14 次逼空窗口上空头敞口被大幅压降
    squeeze_act_mask = np.zeros(len(leader_df), dtype=bool)
    for idx in list(range(120, min(len(leader_df) - 35, 1400), 65))[:14]:
        squeeze_act_mask[idx : idx + 6] = True
    assert float(np.mean(np.abs(w_synth_weak[squeeze_act_mask]))) > 0.025
    assert float(np.mean(np.abs(w_synth_leader[squeeze_act_mask]))) < (
        float(np.mean(np.abs(w_synth_weak[squeeze_act_mask]))) * 0.30
    )


# ===========================================================================
# 4. F 簇新因子 (FC01~FC04 / F01~F03) 通过 Phase 2 法医检验闭环
# ===========================================================================
def test_step4_f_cluster_factors_forensic_validation_loop(
    full_pool_built: tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]],
) -> None:
    """验证 FC01~FC04（含 F01~F03 别名）在多品种池上正常通过 RuleFactorRegistry 与 RuleFactorForensicValidator。"""
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
            soft_gate=True,
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
