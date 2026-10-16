"""Issue #5 · [Architecture][Phase 2] 法医准入检验体系（Gate 1~4）升级专项单元测试。

使用 `czsc.mock.generate_symbol_kines` 构造多标的测试池（5 个模拟宽基权益标的 + 5 个模拟商品标的），完整验证：
1. 因果性修复验证：包含存量 `D01~D05` 在内的全部候选信号及大级别未完结分组聚合在日内任意时刻 t 截断时，100% 通过零前视检验（`np.allclose`）；
2. 连续质量加权 Gate 1 与 MFE/MAE 门禁：验证高 Q 盈利、低 Q 亏损的合成信号在质量加权审计下显著优于等权审计，且高 MAE 扛单信号被 MFE/MAE 门禁精准拦截；
3. 参数平原检验：构造一个仅在单一精确点有效、邻域崩溃的“孤岛信号”，验证被 `run_parameter_smoothness_audit` 拦截；
4. 激活态条件相关性：构造两个稀疏度 95%（仅 5% 时间触发）但每次同时同向触发的同质信号，验证 `compute_active_union_correlation` 能精准识别出高相关（>0.85）并在 Gate 2 / Gate 3 拦截；
5. 三级多标的分层池化检验：构造“仅在单一标的赚钱、其余 4 个同池标的亏钱”的单品种过拟合信号，验证被 Stratified Pooling Gate 1B 精准淘汰入 `graveyard`。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from czsc.mock import generate_symbol_kines
from examples.alpha_mining_agent.forensic_pipeline import (
    compute_active_union_correlation,
    evaluate_triple_barrier,
    run_gate1_forensic_audit,
    run_gate2_3d_complementarity_audit,
    run_parameter_smoothness_audit,
)
from examples.alpha_mining_agent.hypothesis_bank import (
    ALL_SIGNAL_SPECS,
    ARCH_A,
    ARCH_B,
    ARCH_D,
    ARCHETYPE_RULERS,
    compute_signal_triggers_and_weights,
    enrich_kline_features,
)
from examples.alpha_mining_agent.run_evolution_loop import (
    evaluate_candidate_pooled_gate1b,
    load_all_four_timeframes,
)
from examples.alpha_mining_agent.state_ledger import AgentStateLedger
from examples.multi_factor_cta.data_engine import (
    AssetClass,
    AssetSpec,
    CrossAssetDataEngine,
)
from examples.multi_factor_cta.forensic_validator import (
    RuleFactorForensicValidator,
    run_stratified_pooled_gate1b_audit,
)

EQUITY_POOL_SYMBOLS = [
    "SZSE.159915",
    "SHSE.510300",
    "SHSE.510500",
    "SHSE.510050",
    "SHSE.588000",
]
COMMODITY_POOL_SYMBOLS = [
    "SHFE.RB9999",
    "DCE.I9999",
    "SHFE.CU9999",
    "ZCE.TA9999",
    "DCE.M9999",
]


def _make_mock_kline(
    symbol: str, freq: str = "5分钟", sdt: str = "20220101", edt: str = "20220315", seed: int = 42
) -> pd.DataFrame:
    """使用 czsc.mock.generate_symbol_kines 生成标准 K 线样本。"""
    df = generate_symbol_kines(symbol, freq, sdt=sdt, edt=edt, seed=seed)
    df = df.rename(columns={"vol": "volume"})
    df["dt"] = pd.to_datetime(df["dt"])
    df["date"] = df["dt"].dt.date
    for col in ("open", "high", "low", "close"):
        df[col] = np.round(df[col].astype(float) * 100.0, 2)
    df["volume"] = df["volume"].astype(float)
    return (
        df[["date", "dt", "symbol", "open", "high", "low", "close", "volume"]].sort_values("dt").reset_index(drop=True)
    )


@pytest.fixture(scope="module")
def multi_asset_mock_pool() -> dict[str, pd.DataFrame]:
    """构造 5 个模拟宽基权益标的 + 5 个模拟商品标的 (30分钟级别用于跨标的池化验证)。"""
    pool: dict[str, pd.DataFrame] = {}
    for idx, sym in enumerate(EQUITY_POOL_SYMBOLS):
        pool[sym] = _make_mock_kline(sym, freq="30分钟", sdt="20210101", edt="20231231", seed=100 + idx)
    for idx, sym in enumerate(COMMODITY_POOL_SYMBOLS):
        pool[sym] = _make_mock_kline(sym, freq="30分钟", sdt="20210101", edt="20231231", seed=200 + idx)
    return pool


# ===========================================================================
# 1. 因果性修复验证：enrich_kline_features、D01~D05 全信号与未完结大级别分组零前视
# ===========================================================================
def test_step1_intraday_causal_features_and_d01_d05_zero_lookahead(tmp_path: Path) -> None:
    """验证包含 D01~D05 在内的全部候选信号在日内任意时刻 t 截断时 100% 通过零前视检验。"""
    df_5m = _make_mock_kline("SZSE.159915", freq="5分钟", sdt="20220101", edt="20220125", seed=17)
    full_enriched = enrich_kline_features(df_5m.copy(), scale=6)

    # 选取早盘中段、早盘尾段、午盘中段三个日内截断点
    day_counts = df_5m.groupby("date").cumcount().values
    mid_morning_candidates = np.where((day_counts == 7) & (np.arange(len(df_5m)) > 200))[0]
    late_morning_candidates = np.where((day_counts == 19) & (np.arange(len(df_5m)) > 200))[0]
    afternoon_candidates = np.where((day_counts == 33) & (np.arange(len(df_5m)) > 200))[0]

    checkpoints = [
        int(mid_morning_candidates[0]),
        int(late_morning_candidates[0]),
        int(afternoon_candidates[0]),
    ]

    causal_cols = ["bars_in_day", "bar_idx_in_day", "orb_high", "orb_low", "am_high", "am_low", "intraday_vwap"]
    for t in checkpoints:
        sub_enriched = enrich_kline_features(df_5m.iloc[:t].copy(), scale=6)
        for col in causal_cols:
            assert np.allclose(
                sub_enriched[col].values,
                full_enriched[col].values[:t],
                rtol=1e-6,
                atol=1e-8,
            ), f"Column {col} leaked future intraday info at truncation t={t}"

    # 验证包含 D01~D05 在内的全部候选信号在日内截断时 100% 满足零前视
    full_tfs = load_all_four_timeframes(df_5m, symbol_name="159915", workspace_dir=tmp_path / "full")
    t_cut = checkpoints[0]  # 早盘第 8 根 5m K 线 (15m/30m/60m/1d 均处于未完结半成品分组中)
    sub_tfs = load_all_four_timeframes(df_5m.iloc[:t_cut], symbol_name="159915", workspace_dir=tmp_path / "sub")

    # 检查未完结高周期 K 线绝不会提前混入截断时刻 t_cut
    cut_dt = df_5m["dt"].iloc[t_cut - 1]
    for htf in ("15m", "30m", "60m", "1d"):
        if not sub_tfs[htf].empty:
            assert sub_tfs[htf]["dt"].max() <= cut_dt

    # 逐一验证全部候选信号 (含 D01~D05 及多周期共振 FC01L/S) 在截断前后的触发序列与连续权重序列完全一致
    tested_d_families = set()
    for sid in ALL_SIGNAL_SPECS:
        t_full, w_full = compute_signal_triggers_and_weights(full_tfs["5m"], sid, scale=6)
        t_sub, w_sub = compute_signal_triggers_and_weights(sub_tfs["5m"], sid, scale=6)
        assert np.array_equal(t_sub, t_full[:t_cut]), f"Signal {sid} trigger lookahead at t={t_cut}"
        assert np.allclose(w_sub, w_full[:t_cut], rtol=1e-6, atol=1e-7), f"Signal {sid} weight lookahead at t={t_cut}"
        if sid.startswith("D0"):
            tested_d_families.add(sid[:3])

    assert tested_d_families >= {"D01", "D02", "D03", "D04", "D05"}


def test_step1_cross_asset_data_engine_incomplete_htf_guard(tmp_path: Path) -> None:
    """验证 CrossAssetDataEngine.load_and_align_asset 在盘中任意时刻截断时剔除未完结高周期 K 线。"""
    from examples.multi_factor_cta.atomic_factors import _align_htf_state

    df_5m = _make_mock_kline("SZSE.159915", freq="5分钟", sdt="20220101", edt="20220125", seed=23)
    csv_full = tmp_path / "159915_full.csv"
    csv_sub = tmp_path / "159915_sub.csv"

    day_counts = df_5m.groupby("date").cumcount().values
    t_cut = int(np.where((day_counts == 10) & (np.arange(len(df_5m)) > 180))[0][0])

    df_5m.to_csv(csv_full, index=False)
    df_5m.iloc[:t_cut].to_csv(csv_sub, index=False)

    engine = CrossAssetDataEngine(cache_dir=tmp_path / "engine_cache")
    spec_full = AssetSpec(
        symbol="SZSE.159915",
        name="创业板ETF",
        asset_class=AssetClass.ETF_T0,
        sector="EQUITY_GROWTH",
        base_tf="5m",
        eligible_tfs=("1d", "60m", "30m", "15m", "5m"),
        fee_rate=0.0002,
        data_path=csv_full,
    )
    spec_sub = AssetSpec(
        symbol="SZSE.159915",
        name="创业板ETF",
        asset_class=AssetClass.ETF_T0,
        sector="EQUITY_GROWTH",
        base_tf="5m",
        eligible_tfs=("1d", "60m", "30m", "15m", "5m"),
        fee_rate=0.0002,
        data_path=csv_sub,
    )

    aligned_full = engine.load_and_align_asset(spec_full, attach_czsc=True)
    aligned_sub = engine.load_and_align_asset(spec_sub, attach_czsc=True)

    cut_dt = df_5m["dt"].iloc[t_cut - 1]
    for htf in ("15m", "30m", "60m", "1d"):
        if not aligned_sub[htf].empty:
            assert aligned_sub[htf]["dt"].max() <= cut_dt
        hdir_full, hq_full = _align_htf_state(aligned_full["5m"], aligned_full[htf])
        hdir_sub, hq_sub = _align_htf_state(aligned_sub["5m"], aligned_sub[htf])
        assert np.array_equal(hdir_sub, hdir_full[:t_cut]), f"HTF {htf} direction leaked at t={t_cut}"
        assert np.allclose(hq_sub, hq_full[:t_cut], rtol=1e-6, atol=1e-8), f"HTF {htf} quality leaked at t={t_cut}"


# ===========================================================================
# 2. 连续质量加权 Gate 1、Rank IC 衰减曲线与 MFE/MAE 路径偏离比门禁
# ===========================================================================
def test_step2_continuous_quality_weighted_gate1_and_mfe_mae_gate() -> None:
    """验证高 Q 盈利、低 Q 亏损信号在质量加权下显著优于等权，且高 MAE 扛单信号被 MFE/MAE 门禁拦截。"""
    df = _make_mock_kline("SZSE.159915", freq="30分钟", sdt="20210101", edt="20231231", seed=31)
    n = len(df)
    close = df["close"].values.copy()
    high = df["high"].values.copy()
    low = df["low"].values.copy()
    open_p = df["open"].values.copy()

    # 构造 36 个不重叠触发点：
    # k%3 == 0: 高 Q (q=0.98) 触发止盈大赢
    # k%3 == 1: 低 Q (q=0.12) 时间屏障微赢 (+0.05%)
    # k%3 == 2: 极低 Q (q=0.05) 亏损 (-1.2%)
    triggers = np.zeros(n, dtype=bool)
    weights = np.zeros(n, dtype=np.float64)
    trigger_indices = list(range(100, min(n - 30, 3700), 90))[:36]

    for k, idx in enumerate(trigger_indices):
        triggers[idx] = True
        p0 = close[idx]
        if k % 3 == 0:
            weights[idx : idx + 8] = np.linspace(0.98, 0.55, 8)
            for step in range(1, 18):
                drift = p0 * (1.0 + 0.0055 * step)
                close[idx + step] = drift
                open_p[idx + step] = p0 if step == 1 else drift * 0.999
                high[idx + step] = drift * 1.004
                low[idx + step] = p0 * 0.999 if step == 1 else drift * 0.998
        elif k % 3 == 1:
            weights[idx : idx + 6] = np.linspace(0.12, 0.06, 6)
            for step in range(1, 18):
                drift = p0 * (1.0 + 0.00004 * step)
                close[idx + step] = drift
                open_p[idx + step] = p0
                high[idx + step] = drift * 1.0002
                low[idx + step] = p0 * 0.9995
        else:
            weights[idx : idx + 6] = np.linspace(0.05, 0.02, 6)
            for step in range(1, 18):
                drift = p0 * (1.0 - 0.0012 * step)
                close[idx + step] = drift
                open_p[idx + step] = p0 if step == 1 else drift * 1.0005
                high[idx + step] = p0 * 1.0002 if step == 1 else drift * 1.001
                low[idx + step] = drift * 0.997

    df_synth = df.copy()
    df_synth["open"] = open_p
    df_synth["high"] = high
    df_synth["low"] = low
    df_synth["close"] = close
    df_synth = enrich_kline_features(df_synth, scale=1)

    ruler = ARCHETYPE_RULERS[ARCH_A]
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    g1_equal = run_gate1_forensic_audit(df_synth, triggers, 1, ruler, "30m", rng1, weights_tf=None)
    g1_weighted = run_gate1_forensic_audit(df_synth, triggers, 1, ruler, "30m", rng2, weights_tf=weights)

    assert g1_weighted["is_mean_pct"] > g1_equal["is_mean_pct"] + 0.30
    assert g1_weighted["is_win_rate_pct"] > g1_equal["is_win_rate_pct"] + 10.0
    assert g1_weighted["is_pl_ratio"] > g1_equal["is_pl_ratio"]
    assert g1_weighted["is_excursion_ratio"] >= ruler.min_excursion_ratio
    assert "rank_ic_5" in g1_weighted and "rank_ic_20" in g1_weighted
    assert "signal_turnover" in g1_weighted and "is_net_sharpe" in g1_weighted
    assert g1_weighted["passed"] is True

    # 构造“扛深幅浮亏换微利”的高 MAE 伪高胜率信号：先暴跌但未触及 2.5 ATR 止损，最后微涨 0.15% 收盘
    df_bag = df.copy()
    c_bag = df_bag["close"].values.copy()
    h_bag = df_bag["high"].values.copy()
    l_bag = df_bag["low"].values.copy()
    o_bag = df_bag["open"].values.copy()
    trig_bag = np.zeros(n, dtype=bool)
    w_bag = np.zeros(n, dtype=np.float64)

    for idx in trigger_indices:
        trig_bag[idx] = True
        w_bag[idx : idx + 10] = 0.80
        p0 = c_bag[idx]
        for step in range(1, 13):
            c_bag[idx + step] = p0 * 1.0015
            o_bag[idx + step] = p0 * 1.0005
            h_bag[idx + step] = p0 * 1.0020
            # 盘中深幅下探产生大 MAE，但控制在止损线之上
            l_bag[idx + step] = p0 * 0.985 if step == 2 else p0 * 0.998

    df_bag["open"] = o_bag
    df_bag["high"] = h_bag
    df_bag["low"] = l_bag
    df_bag["close"] = c_bag
    df_bag = enrich_kline_features(df_bag, scale=1)

    tdf_bag = evaluate_triple_barrier(
        df_bag,
        np.where(trig_bag)[0],
        1,
        max_bars=ruler.max_bars,
        tp_atr=ruler.tp_atr,
        sl_atr=ruler.sl_atr,
        weights_tf=w_bag,
    )
    assert not tdf_bag.empty
    assert float(tdf_bag["mae_atr"].mean()) > float(tdf_bag["mfe_atr"].mean())

    g1_bag = run_gate1_forensic_audit(df_bag, trig_bag, 1, ruler, "30m", np.random.default_rng(42), weights_tf=w_bag)
    assert g1_bag["is_excursion_ratio"] < ruler.min_excursion_ratio
    assert g1_bag["passed"] is False
    assert "MFE/MAE路径偏离比" in g1_bag["reason"]


# ===========================================================================
# 3. 参数平原检验 (Parameter Smoothness Gate)
# ===========================================================================
def test_step2_parameter_smoothness_gate_blocks_island_signal() -> None:
    """验证仅在单一精确点有效、邻域崩溃的“孤岛信号”被 run_parameter_smoothness_audit 拦截。"""

    # 孤岛信号：仅在 delta=0.0 时夏普为 1.8，邻域 (-20%, -10%, +10%, +20%) 崩溃为接近 0 或负数
    def _island_evaluator(delta: float) -> float:
        if abs(delta) < 1e-6:
            return 1.80
        return -0.20 if delta < 0 else 0.05

    audit_island = run_parameter_smoothness_audit(
        evaluate_fn=_island_evaluator,
        deltas=(-0.20, -0.10, 0.0, 0.10, 0.20),
        min_retention_ratio=0.70,
        max_cv=0.40,
    )
    assert audit_island["passed"] is False
    assert audit_island["retention_ratio"] < 0.70
    assert "孤岛参数" in audit_island["reason"] or "参数" in audit_island["reason"]

    # 平原信号：在 ±20% 邻域微扰下保持稳健正收益
    def _plateau_evaluator(delta: float) -> float:
        return 1.50 * (1.0 - 0.35 * (delta**2))

    audit_plateau = run_parameter_smoothness_audit(
        evaluate_fn=_plateau_evaluator,
        deltas=(-0.20, -0.10, 0.0, 0.10, 0.20),
        min_retention_ratio=0.70,
        max_cv=0.40,
    )
    assert audit_plateau["passed"] is True
    assert audit_plateau["retention_ratio"] >= 0.70
    assert audit_plateau["cv"] <= 0.40

    # 同时验证 RuleFactorForensicValidator 上的接口一致性
    validator = RuleFactorForensicValidator()
    val_island = validator.run_parameter_smoothness_audit(
        evaluate_fn=_island_evaluator, deltas=(-0.20, -0.10, 0.0, 0.10, 0.20)
    )
    assert val_island["passed"] is False


# ===========================================================================
# 4. 激活态条件相关性 (Active-Union Correlation) 识别稀疏同质信号
# ===========================================================================
def test_step2_active_union_correlation_catches_sparse_homogeneous_signals() -> None:
    """构造两个稀疏度 95%（仅 5% 时间触发）但同时同向触发的同质信号，验证激活态条件相关性 > 0.85 并在 Gate 2 拦截。"""
    df = _make_mock_kline("SZSE.159915", freq="30分钟", sdt="20210101", edt="20231231", seed=55)
    n = len(df)
    rng = np.random.default_rng(123)

    w1 = np.zeros(n, dtype=np.float64)
    w2 = np.zeros(n, dtype=np.float64)

    # 仅 5% 时间处于激活态：w1 恒定为 0.80，w2 在同一时刻激活但振幅轻微波动
    n_active = int(n * 0.05)
    active_idx = rng.choice(np.arange(50, n - 50), size=n_active, replace=False)
    w1[active_idx] = 0.80
    w2[active_idx] = 0.78 + 0.04 * rng.random(n_active)

    # 在激活集内部去均值的传统相关系数因 std(w1[active]) == 0 退化为 0.0 (被稀释/掩盖)
    naive_active_demean_corr = (
        0.0 if np.std(w1[active_idx]) < 1e-12 else float(np.corrcoef(w1[active_idx], w2[active_idx])[0, 1])
    )
    active_union_corr = compute_active_union_correlation(w1, w2, eps=1e-3, min_active_bars=30)

    assert naive_active_demean_corr == 0.0
    assert active_union_corr > 0.85

    # 验证 Gate 2 (run_gate2_3d_complementarity_audit) 精准识别高相关并拦截
    ruler = ARCHETYPE_RULERS[ARCH_A]
    g2 = run_gate2_3d_complementarity_audit(
        df_30m=df,
        cand_w_on_30m=w2,
        zoo_w_on_30m={"A01@30m": w1},
        current_portfolio_w_30m=w1,
        archetype=ARCH_A,
        current_archetype_new_count=1,
        ruler=ruler,
        current_arch_same_dir_new_count=1,
        max_corr_threshold=0.32,
    )
    assert g2["max_corr"] > 0.85
    assert g2["passed"] is False
    assert "相关性" in g2["reason"] or "同质" in g2["reason"]

    # 验证 Gate 3 (RuleFactorForensicValidator.run_gate3_orthogonality_and_marginal_audit) 同样拦截
    validator = RuleFactorForensicValidator(max_pairwise_corr=0.35)
    g3 = validator.run_gate3_orthogonality_and_marginal_audit(
        df_30m=df,
        cand_w_30m=w2,
        zoo_weights_30m={"A01@30m": w1},
        curr_portfolio_w_30m=w1,
    )
    assert g3["max_corr"] > 0.85
    assert g3["passed"] is False


# ===========================================================================
# 5. 三级多标的分层池化检验 (Stratified Multi-Asset Pooling Gate 1B)
# ===========================================================================
def test_step3_stratified_multi_asset_pooling_rejects_single_asset_overfit(
    multi_asset_mock_pool: dict[str, pd.DataFrame],
    tmp_path: Path,
) -> None:
    """构造“仅在单一标的赚钱、其余 4 个同池标的亏钱”的单品种过拟合信号，验证被 Gate 1B 精准淘汰入 graveyard。"""
    rng = np.random.default_rng(2026)
    ruler_b = ARCHETYPE_RULERS[ARCH_B]

    per_symbol_g1: dict[str, dict] = {}
    pool_tf_dfs: dict[str, dict[str, pd.DataFrame]] = {}

    for sym, raw_df in multi_asset_mock_pool.items():
        df_sym = raw_df.copy()
        n = len(df_sym)
        c = df_sym["close"].values.copy()
        h = df_sym["high"].values.copy()
        low_arr = df_sym["low"].values.copy()
        o = df_sym["open"].values.copy()

        trig = np.zeros(n, dtype=bool)
        w = np.zeros(n, dtype=np.float64)
        t_idxs = list(range(80, min(n - 25, 2800), 85))[:28]

        for idx in t_idxs:
            trig[idx] = True
            w[idx : idx + 6] = 0.85
            p0 = c[idx]
            if sym == "SZSE.159915":
                # 仅在单一标的 159915 上赚钱 (+3.5%)
                for s in range(1, 9):
                    drift = p0 * (1.0 + 0.0045 * s)
                    c[idx + s] = drift
                    o[idx + s] = drift * 0.999
                    h[idx + s] = drift * 1.002
                    low_arr[idx + s] = drift * 0.998
            else:
                # 在同池其余 4 个宽基权益标的及 5 个商品标的上全部亏钱 (-2.5%)
                for s in range(1, 9):
                    drift = p0 * (1.0 - 0.0035 * s)
                    c[idx + s] = drift
                    o[idx + s] = drift * 1.001
                    h[idx + s] = drift * 1.0015
                    low_arr[idx + s] = drift * 0.997

        df_sym["open"] = o
        df_sym["high"] = h
        df_sym["low"] = low_arr
        df_sym["close"] = c
        df_sym = enrich_kline_features(df_sym, scale=1)
        pool_tf_dfs[sym] = {"30m": df_sym}
        per_symbol_g1[sym] = run_gate1_forensic_audit(df_sym, trig, 1, ruler_b, "30m", rng, weights_tf=w)

    # 确认在单标的 159915 上通过了 Gate 1
    assert per_symbol_g1["SZSE.159915"]["passed"] is True

    # 验证 Tier 2 (同质资产类别特征簇 B_背驰衰竭)：在 5 个宽基权益池中仅 1/5 = 20% 正期望 (< 75% 门槛)，被拦截
    g1b_tier2 = run_stratified_pooled_gate1b_audit(
        per_symbol_gate1=per_symbol_g1,
        archetype=ARCH_B,
        timeframe="30m",
        allow_single_asset_degrade=False,
    )
    assert g1b_tier2["pool_tier"] == "Tier2_AssetClass"
    assert g1b_tier2["pool_size"] == 5  # 仅在 5 个同质权益标的上池化，不混入商品
    assert g1b_tier2["universal_ratio"] == pytest.approx(0.20)
    assert g1b_tier2["passed"] is False
    assert "Tier2(Equity-Pool)未达标" in g1b_tier2["reason"]

    # 验证 Tier 1 (跨资产普适簇 A_趋势突破)：在 10 个跨资产标的中仅 1/10 = 10% 正期望 (< 60% 门槛)，被拦截
    g1b_tier1 = run_stratified_pooled_gate1b_audit(
        per_symbol_gate1=per_symbol_g1,
        archetype=ARCH_A,
        timeframe="30m",
        allow_single_asset_degrade=False,
    )
    assert g1b_tier1["pool_tier"] == "Tier1_CrossAsset"
    assert g1b_tier1["pool_size"] == 10
    assert g1b_tier1["universal_ratio"] == pytest.approx(0.10)
    assert g1b_tier1["passed"] is False

    # 验证 Tier 3 (微结构簇 D_量价筹码)：占比 < 70% 且扣费池化夏普 < 0，被拦截
    g1b_tier3 = run_stratified_pooled_gate1b_audit(
        per_symbol_gate1=per_symbol_g1,
        archetype=ARCH_D,
        timeframe="15m",
        allow_single_asset_degrade=False,
    )
    assert g1b_tier3["pool_tier"] == "Tier3_Microstructure"
    assert g1b_tier3["passed"] is False

    # 验证接入进化主循环后，单品种过拟合候选单元被 Pre-Admission Gate 1B 精准淘汰入 graveyard.jsonl
    ledger = AgentStateLedger(tmp_path / "workspace_gate1b")
    g1b_loop = evaluate_candidate_pooled_gate1b(
        sid="BC01L",
        tf="30m",
        primary_symbol="SZSE.159915",
        primary_g1=per_symbol_g1["SZSE.159915"],
        pool_tf_dfs=pool_tf_dfs,
        rng=rng,
        allow_single_asset_degrade=False,
        ledger=ledger,
        per_symbol_gate1_overrides=per_symbol_g1,
    )
    assert g1b_loop["passed"] is False
    grave_records = ledger.load_graveyard_records()
    assert len(grave_records) == 1
    assert grave_records[0]["unit_id"] == "BC01L@30m"
    assert grave_records[0]["failed_gate"] == "Gate 1B (多标的分层池化普适关)"
