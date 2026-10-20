"""Issue #3 · 6 大正交战略簇 × 连续强度 `[-1, 1]` 原子因子专项测试。

验证：
1. 6 簇完整性：A/B/C/D/E/F 因子均可产出，值域严格 `[-1, 1]`，无 NaN/Inf，且不止 `{-1, 0, 1}` 三个离散值；
2. 零前视审计：`bars[:t]` 重算的第 `t-1` 根信号值必须与全量 `bars[:T]` 在同一根上 `np.allclose`
   (同时覆盖 Rust `strength_*` 信号列、大级别因果对齐、两阶段状态机与动态衰减机)；
3. 多空对称：价格镜像 `p' = max(high) + min(low) - p` 后，`S(P) ≈ -S(M(P))`；
4. 动态结构衰减机：反向笔确认 / 证伪点跌破立即归零，波动率放大衰减加快；
5. 6 簇注册表、旧账本迁移与存量状态机回归。

测试数据统一来自 `czsc.mock.generate_symbol_kines`。价格量化到 0.01 (×100 取整的整数价)，使镜像运算在浮点上精确。
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from czsc.mock import generate_symbol_kines
from examples.alpha_mining_agent.hypothesis_bank import (
    ALL_ARCHETYPES,
    ALL_SIGNAL_SPECS,
    ARCH_A,
    ARCH_B,
    ARCH_C,
    ARCH_D,
    ARCH_E,
    ARCH_F,
    ARCHETYPE_RULERS,
    LEGACY_ARCHETYPE_ALIAS,
    LEGACY_ARCHETYPE_BY_SIG,
    STRENGTH_SIGNAL_KINDS,
    attach_native_czsc_signals,
    compute_signal_triggers_and_weights,
    decode_strength,
    enrich_kline_features,
    migrate_archetype,
)
from examples.alpha_mining_agent.state_ledger import AgentStateLedger
from examples.multi_factor_cta.atomic_factors import (
    ATOMIC_FACTOR_DEFS,
    ATOMIC_SIGNAL_IDS,
    DecayConfig,
    FactorContext,
    apply_dynamic_structure_decay,
    compute_family,
    link_higher_timeframes,
    two_stage_trigger,
)
from examples.multi_factor_cta.data_engine import compute_causal_cross_sectional_features
from examples.multi_factor_cta.factor_engine import RuleFactorRegistry

PRICE_COLS = ["open", "high", "low", "close"]
FAMILIES = list(ATOMIC_FACTOR_DEFS)


# ---------------------------------------------------------------------------
# 数据与流水线辅助
# ---------------------------------------------------------------------------
def _mock_raw(seed: int = 5) -> pd.DataFrame:
    df = generate_symbol_kines("000001", "30分钟", "20220101", "20221231", seed=seed)
    df = df.rename(columns={"vol": "volume"})
    df["dt"] = pd.to_datetime(df["dt"])
    df = df[["dt", "symbol", *PRICE_COLS, "volume"]].reset_index(drop=True)
    for col in PRICE_COLS:
        df[col] = np.round(df[col].astype(float) * 100.0)
    return df


def _mirror(raw: pd.DataFrame) -> pd.DataFrame:
    """价格镜像 p' = max(high) + min(low) - p；high/low 互换，成交量不变。"""
    m = raw["high"].max() + raw["low"].min()
    out = raw.copy()
    out["open"], out["close"] = m - raw["open"], m - raw["close"]
    out["high"], out["low"] = m - raw["low"], m - raw["high"]
    out.attrs["market_mirrored"] = True
    return out


def _make_htf(raw: pd.DataFrame) -> pd.DataFrame:
    """把相邻 2 根 30m 聚合成 60m，只保留完整的分组 (截断时不产生半成品大级别 K 线)。"""
    m = len(raw) // 2 * 2
    sub = raw.iloc[:m]
    grp = np.arange(m) // 2
    htf = sub.groupby(grp).agg(
        dt=("dt", "last"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    htf = htf.reset_index(drop=True)
    htf["symbol"] = "000001"
    return htf


def _prepare(raw: pd.DataFrame, tf: str, cache_dir) -> pd.DataFrame:
    df = raw.copy()
    df["date"] = df["dt"].dt.date
    df = enrich_kline_features(df)
    return attach_native_czsc_signals(df, tf, cache_dir)


def _build_ctx(raw: pd.DataFrame, cache_dir) -> tuple[pd.DataFrame, FactorContext]:
    base = _prepare(raw, "30m", cache_dir)
    htf = _prepare(_make_htf(raw), "60m", cache_dir)
    # Cross-asset factors require a genuine simultaneous peer pool, including in mirror/prefix checks.
    pool = {"000001": base}
    for seed in (6, 7, 8):
        peer = _mock_raw(seed).iloc[: len(raw)].copy()
        if raw.attrs.get("market_mirrored", False):
            peer = _mirror(peer)
        pool[f"PEER{seed}"] = peer
    compute_causal_cross_sectional_features(
        pool, specs=dict.fromkeys(pool, "test_sector"), tf="30m", attach_to_dfs=True
    )
    return base, FactorContext.from_df(base, htf)


def _all_signals(ctx: FactorContext) -> dict[str, np.ndarray]:
    return {fam: compute_family(ctx, fam)[0] for fam in FAMILIES}


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return _mock_raw()


@pytest.fixture(scope="module")
def built(raw, tmp_path_factory):
    cache = tmp_path_factory.mktemp("full")
    base, ctx = _build_ctx(raw, cache)
    return base, ctx, _all_signals(ctx)


# ---------------------------------------------------------------------------
# 1. 6 簇完整性与连续强度
# ---------------------------------------------------------------------------
def test_six_clusters_registered_and_orthogonal():
    assert len(ALL_ARCHETYPES) == 6
    assert set(ARCHETYPE_RULERS) == set(ALL_ARCHETYPES)
    clusters_with_atomic = {d.archetype for d in ATOMIC_FACTOR_DEFS.values()}
    assert clusters_with_atomic == set(ALL_ARCHETYPES), "A~F 每簇至少一个连续强度原子因子"
    # 旧 D_日内微结构簇 已消除：任何注册信号都不再属于旧簇名，也不再使用 intraday 离场
    for spec in ALL_SIGNAL_SPECS.values():
        assert spec.archetype in ALL_ARCHETYPES, f"{spec.sig_id} 簇名未迁移: {spec.archetype}"
        assert spec.exit_mode != "intraday"
    assert len(ATOMIC_SIGNAL_IDS) == 2 * len(ATOMIC_FACTOR_DEFS)


def test_atomic_metadata_matches_hypothesis_bank():
    """hypothesis_bank 与 atomic_factors 各登记一份元数据 (避免循环导入)，必须保持一致。"""
    for fam, fdef in ATOMIC_FACTOR_DEFS.items():
        for sfx, direction in (("L", 1), ("S", -1)):
            spec = ALL_SIGNAL_SPECS[f"{fam}{sfx}"]
            assert spec.archetype == fdef.archetype
            assert spec.direction == direction
            assert spec.eligible_tfs == fdef.eligible_tfs
            assert spec.exit_mode == "continuous"


def test_legacy_d_cluster_members_moved_by_mechanism():
    to_a = {"D01L", "D01S", "D03L", "D03S", "CZ07L", "CZ07S", "CZ08L", "CZ08S", "M504L", "M504S"}
    to_b = {"D04L", "D04S", "TD01L", "TD01S", "TD04L", "TD04S"}
    to_c = {"D02L", "D02S", "D05L", "D05S"}
    for sid in to_a:
        assert ALL_SIGNAL_SPECS[sid].archetype == ARCH_A and ALL_SIGNAL_SPECS[sid].exit_mode == "trend"
    for sid in to_b:
        assert ALL_SIGNAL_SPECS[sid].archetype == ARCH_B and ALL_SIGNAL_SPECS[sid].exit_mode == "reversal"
    for sid in to_c:
        assert ALL_SIGNAL_SPECS[sid].archetype == ARCH_C and ALL_SIGNAL_SPECS[sid].exit_mode == "range"
    assert ALL_SIGNAL_SPECS["A01L"].archetype == ARCH_D
    assert ALL_SIGNAL_SPECS["A02S"].archetype == ARCH_E
    # 基石真金 S1~S4 保持 A 簇，冠军名册与配额稳定
    for sid in ("S1L", "S1S", "S2L", "S3L", "S4S"):
        assert ALL_SIGNAL_SPECS[sid].archetype == ARCH_A
    assert set(LEGACY_ARCHETYPE_BY_SIG) == to_a | to_b | to_c | {"A01L", "A01S", "A02L", "A02S", "A03L", "A03S"}


@pytest.mark.parametrize("family", FAMILIES)
def test_cluster_factor_outputs_continuous_and_bounded(family, built):
    _, ctx, signals = built
    s = signals[family]
    assert s.shape == (ctx.n,)
    assert np.isfinite(s).all(), "不得出现 NaN / Inf"
    assert float(np.max(np.abs(s))) <= 1.0 + 1e-12
    assert (s > 0).any() and (s < 0).any(), "多空两侧都应有产出"
    uniq = np.unique(np.round(s[s != 0.0], 6))
    assert len(uniq) > 20, f"{family} 应为连续强度，而非 {{-1,0,1}} 离散值 (唯一值={len(uniq)})"
    assert not set(np.unique(s)).issubset({-1.0, 0.0, 1.0})


def test_every_cluster_has_output(built):
    _, _, signals = built
    for arch in ALL_ARCHETYPES:
        fams = [f for f, d in ATOMIC_FACTOR_DEFS.items() if d.archetype == arch]
        assert fams and any(np.any(signals[f] != 0.0) for f in fams), f"{arch} 无输出"


@pytest.mark.parametrize("kind", list(STRENGTH_SIGNAL_KINDS))
def test_rust_strength_score_decodes_to_unit_interval(kind, built):
    base, _, _ = built
    d, q = decode_strength(base, kind)
    assert set(np.unique(d)).issubset({-1, 0, 1})
    assert q.min() >= 0.0 and q.max() <= 1.0
    fired = d != 0
    assert fired.any()
    assert (q[fired] > 0).all(), "有方向的信号 score 必须 > 0 (Signal.score 已被激活)"
    assert len(np.unique(q[fired])) > 10, "score 应为连续质量分而非少数几档"


def test_registry_integration_and_output_kind(built):
    base, _, _ = built
    reg = RuleFactorRegistry()
    spec = reg.get("AC01L")
    assert spec.output_kind == "continuous" and spec.archetype == ARCH_A
    assert reg.get("S1L").output_kind == "binary"
    link_higher_timeframes({"30m": base})  # 不抛错
    trig, w = reg.evaluate_factor_on_tf(base, "AC01L", "30m")
    assert w.shape == (len(base),) and trig.dtype == bool
    assert (w >= 0).all() and (w > 0).any(), "L 因子只产出多头"
    _, w_s = reg.evaluate_factor_on_tf(base, "AC01S", "30m")
    assert (w_s <= 0).all() and (w_s < 0).any(), "S 因子只产出空头"


def test_legacy_state_machines_still_run_after_reclassification(built):
    base, _, _ = built
    for sid in ("D01L", "D02S", "D04L", "TD01L", "TD04S", "A01L", "A02S", "CZ07L", "S1L"):
        trig, w = compute_signal_triggers_and_weights(base, sid)
        assert len(trig) == len(w) == len(base)
        assert np.isfinite(w).all()


# ---------------------------------------------------------------------------
# 2. 零前视审计 (Zero-Lookahead Audit)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("t", [2300, 2900, 3400])
def test_zero_lookahead_truncation(t, raw, built, tmp_path):
    """`bars[:t]` 上重算的信号，必须与全量 `bars[:T]` 在前 t 根上逐值相等 (np.allclose)。"""
    base_full, ctx_full, sig_full = built
    base_t, ctx_t = _build_ctx(raw.iloc[:t].reset_index(drop=True), tmp_path)
    assert ctx_t.n == t

    # Rust strength_* 信号列 (方向 + Signal.score)
    for kind in STRENGTH_SIGNAL_KINDS:
        d_f, q_f = decode_strength(base_full, kind)
        d_t, q_t = decode_strength(base_t, kind)
        assert np.array_equal(d_f[:t], d_t), f"{kind} 方向在截断 t={t} 后不一致 (前视)"
        assert np.allclose(q_f[:t], q_t, atol=1e-12), f"{kind} score 在截断 t={t} 后不一致 (前视)"

    # 大级别因果对齐
    assert np.array_equal(ctx_full.htf_dir[:t], ctx_t.htf_dir)
    assert np.allclose(ctx_full.htf_q[:t], ctx_t.htf_q, atol=1e-12)

    # 两阶段状态机 + 动态衰减机的最终输出
    sig_t = _all_signals(ctx_t)
    for fam in FAMILIES:
        assert np.allclose(sig_full[fam][:t], sig_t[fam], atol=1e-12, rtol=0.0), f"{fam} 在 t={t} 存在前视偏差"


# ---------------------------------------------------------------------------
# 3. 多空对称
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def mirrored(raw, tmp_path_factory):
    cache = tmp_path_factory.mktemp("mirror")
    base, ctx = _build_ctx(_mirror(raw), cache)
    return base, ctx, _all_signals(ctx)


def test_rust_strength_signals_are_mirror_symmetric(built, mirrored):
    base_a, _, _ = built
    base_b, _, _ = mirrored
    for kind in STRENGTH_SIGNAL_KINDS:
        d_a, q_a = decode_strength(base_a, kind)
        d_b, q_b = decode_strength(base_b, kind)
        assert np.array_equal(d_a, -d_b), f"{kind} 方向在价格镜像下未翻转"
        assert np.allclose(q_a, q_b, atol=0.011), f"{kind} 质量分在价格镜像下不一致"


@pytest.mark.parametrize("family", FAMILIES)
def test_long_short_mirror_symmetry(family, built, mirrored):
    """仿射镜像保持结构/两阶段规则，百分比回撤不具仿射不变性，另测其多空1.3%边界。"""
    _, ctx_a, _ = built
    _, ctx_b, _ = mirrored
    definition = ATOMIC_FACTOR_DEFS[family]
    mirror_sum = ctx_a.c[0] + ctx_b.c[0]
    for side in (1, -1):
        setup_a, sq_a, confirm_a, cq_a, window_a = definition.build(ctx_a, side)
        setup_b, sq_b, confirm_b, cq_b, window_b = definition.build(ctx_b, -side)
        assert window_a == window_b
        np.testing.assert_array_equal(setup_a, setup_b)
        np.testing.assert_array_equal(confirm_a, confirm_b)
        np.testing.assert_allclose(sq_a[setup_a], sq_b[setup_b], atol=0.011, rtol=0)
        np.testing.assert_allclose(cq_a[confirm_a], cq_b[confirm_b], atol=0.011, rtol=0)
        fire_a, q_a = two_stage_trigger(setup_a, sq_a, confirm_a, cq_a, window_a)
        fire_b, q_b = two_stage_trigger(setup_b, sq_b, confirm_b, cq_b, window_b)
        np.testing.assert_array_equal(fire_a, fire_b)
        np.testing.assert_allclose(q_a, q_b, atol=0.011, rtol=0)
        stop_a = np.asarray(ctx_a.structure_anchors[definition.archetype]["long_stop" if side == 1 else "short_stop"])
        stop_b = np.asarray(ctx_b.structure_anchors[definition.archetype]["short_stop" if side == 1 else "long_stop"])
        np.testing.assert_allclose(stop_a, mirror_sum - stop_b, atol=1e-9, rtol=0)


# ---------------------------------------------------------------------------
# 4. 动态结构衰减机与两阶段状态机
# ---------------------------------------------------------------------------
def _decay_inputs(n: int = 30):
    z = np.zeros(n)
    direction = np.zeros(n, dtype=np.int8)
    quality = np.zeros(n)
    close = np.full(n, 100.0)
    stop = np.full(n, 95.0)
    return direction, quality, close, stop, np.zeros(n, dtype=np.int8), np.ones(n), np.ones(n), z


def test_decay_machine_bounded_and_monotone_without_events():
    d, q, c, stop, rev, af, asl, _ = _decay_inputs()
    d[3], q[3] = 1, 0.8
    s, trig = apply_dynamic_structure_decay(d, q, c, stop, rev, af, asl)
    assert trig[3] and trig.sum() == 1
    assert s[2] == 0.0 and s[3] == pytest.approx(0.8)
    tail = s[3:15]
    assert (np.diff(tail) < 0).all(), "无事件时强度应随结构老化单调衰减"
    assert np.abs(s).max() <= 1.0


def test_decay_machine_reverse_bi_truncates_to_zero():
    d, q, c, stop, rev, af, asl, _ = _decay_inputs()
    d[3], q[3] = 1, 0.9
    rev[8] = -1  # 反向笔确认 (多头遇到向下笔)
    s, _ = apply_dynamic_structure_decay(d, q, c, stop, rev, af, asl)
    assert s[7] > 0.3
    assert (s[8:] == 0.0).all(), "反向笔确认后原方向残余强度必须截断归零"
    # 同向 (向上) 笔翻转不应截断
    rev2 = np.zeros_like(rev)
    rev2[8] = 1
    s2, _ = apply_dynamic_structure_decay(d, q, c, stop, rev2, af, asl)
    assert s2[8] > 0.0


def test_decay_machine_structure_break_truncates_to_zero():
    d, q, c, stop, rev, af, asl, _ = _decay_inputs()
    d[3], q[3] = -1, 0.7  # 空头：证伪点在上方
    stop[:] = 105.0
    c[10] = 105.0  # 收盘上穿空头证伪点
    s, _ = apply_dynamic_structure_decay(d, q, c, stop, rev, af, asl)
    assert s[9] < 0.0
    assert (s[10:] == 0.0).all(), "收盘突破结构证伪点后必须归零"


def test_decay_rate_adapts_to_atr_ratio():
    d, q, c, stop, rev, _, _, _ = _decay_inputs()
    d[2], q[2] = 1, 1.0
    slow_decay, _ = apply_dynamic_structure_decay(d, q, c, stop, rev, np.full(30, 0.5), np.ones(30))
    fast_decay, _ = apply_dynamic_structure_decay(d, q, c, stop, rev, np.full(30, 2.0), np.ones(30))
    assert fast_decay[10] < slow_decay[10], "ATR_fast/ATR_slow 越大 (波动率放大)，衰减越快"
    cfg = DecayConfig(lam0=0.10, rho_min=0.5, rho_max=2.0)
    assert slow_decay[3] == pytest.approx(np.exp(-cfg.lam0 * 0.5))
    assert fast_decay[3] == pytest.approx(np.exp(-cfg.lam0 * 2.0))


def test_decay_machine_stronger_trigger_refreshes_and_opposite_replaces():
    d, q, c, stop, rev, af, asl, _ = _decay_inputs()
    d[2], q[2] = 1, 0.5
    d[6], q[6] = 1, 0.9  # 更强的同向触发刷新
    d[12], q[12] = -1, 0.4  # 反向触发直接替换
    stop[:] = 95.0
    stop[12] = 105.0  # 空头必须使用收盘价上方的有效证伪点。
    s, trig = apply_dynamic_structure_decay(d, q, c, stop, rev, af, asl)
    assert s[6] == pytest.approx(0.9)
    assert s[12] < 0.0 and trig[12]


def test_two_stage_requires_strictly_earlier_setup_within_window():
    n = 12
    setup = np.zeros(n, dtype=bool)
    confirm = np.zeros(n, dtype=bool)
    sq = np.full(n, 0.8)
    cq = np.full(n, 0.5)
    setup[3] = confirm[3] = True  # 同一根 K 线静态拼接：不允许开仓
    fire, _ = two_stage_trigger(setup, sq, confirm, cq, window=4)
    assert not fire[3]
    confirm[6] = True  # 之后窗口内确认：允许
    fire, q = two_stage_trigger(setup, sq, confirm, cq, window=4)
    assert fire[6] and q[6] == pytest.approx(0.8 * (0.6 + 0.4 * 0.5))
    setup2 = np.zeros(n, dtype=bool)
    setup2[1] = True
    confirm2 = np.zeros(n, dtype=bool)
    confirm2[9] = True  # 超出 window：Setup 已过期
    fire2, _ = two_stage_trigger(setup2, sq, confirm2, cq, window=4)
    assert not fire2.any()
    fire3, _ = two_stage_trigger(setup, sq, confirm, cq, window=4)
    assert fire3.sum() == 1, "Setup 一经确认消费即解除武装"


# ---------------------------------------------------------------------------
# 5. 旧账本迁移
# ---------------------------------------------------------------------------
def test_migrate_archetype_resolves_legacy_names_by_signal_id():
    assert migrate_archetype(ARCH_C) == ARCH_C  # 新名原样返回
    assert migrate_archetype("A_顺势突破簇") == ARCH_A
    assert migrate_archetype("B_背驰反转簇", "B01L") == ARCH_B
    # 旧 D_日内微结构簇 必须借助 sig_id 才能判定归属
    assert migrate_archetype("D_日内微结构簇", "D01L") == ARCH_A
    assert migrate_archetype("D_日内微结构簇", "D04S") == ARCH_B
    assert migrate_archetype("D_日内微结构簇", "D02L") == ARCH_C
    assert migrate_archetype("A_顺势突破簇", "A02L") == ARCH_E  # 旧 A 簇候选已迁到 E
    with pytest.raises(ValueError):
        migrate_archetype("D_日内微结构簇")
    with pytest.raises(ValueError):
        migrate_archetype("X_不存在簇")
    assert set(LEGACY_ARCHETYPE_ALIAS.values()) == {ARCH_A, ARCH_B, ARCH_C}
    assert ARCH_D in ALL_ARCHETYPES and ARCH_E in ALL_ARCHETYPES and ARCH_F in ALL_ARCHETYPES


def test_state_ledger_migrates_legacy_zoo_on_load(tmp_path):
    ledger = AgentStateLedger(tmp_path)
    legacy_zoo = {
        "D01L@30m": {"unit_id": "D01L@30m", "sig_id": "D01L", "archetype": "D_日内微结构簇"},
        "B01L@30m": {"unit_id": "B01L@30m", "sig_id": "B01L", "archetype": "B_背驰反转簇"},
        "C01L@15m": {"unit_id": "C01L@15m", "sig_id": "C01L", "archetype": "C_震荡回归簇"},
    }
    ledger.zoo_path.write_text(json.dumps(legacy_zoo, ensure_ascii=False), encoding="utf-8")
    zoo = ledger.load_signal_zoo()
    assert zoo["D01L@30m"]["archetype"] == ARCH_A
    assert zoo["B01L@30m"]["archetype"] == ARCH_B
    assert zoo["C01L@15m"]["archetype"] == ARCH_C
    # 磁盘文件保持原样 (只在内存中迁移)
    assert json.loads(ledger.zoo_path.read_text(encoding="utf-8"))["D01L@30m"]["archetype"] == "D_日内微结构簇"
