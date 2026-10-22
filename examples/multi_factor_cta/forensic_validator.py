#!/usr/bin/env python
"""
第三层：规则型因子专属三级法医检验管线 (Three-Gate Forensic Validator for Rule-Based Factors)

为什么规则型因子不能直接套用股票截面多因子的全样本线性 IC/IR？
1. 规则型因子（分型/笔/中枢/两阶段状态机）在 70%~90% 时间处于 0（静默等待），全样本 IC 会被大量空仓期严重扭曲；
2. 不同战略簇（趋势突破 vs 背驰反转 vs 震荡回归 vs 日内微结构）的持仓半衰期与止盈止损结构截然不同。

本模块实现专用于规则型因子的三级门禁：
- Gate 1 (内在边缘与跨品种普适关)：
  正式路径使用完整实际执行交易 + Q 有效样本门槛 + 状态归一化事件池化；固定 ATR 标尺保留为显式诊断模式
- Gate 2 (时序×截面三维互补与防拥挤关)：
  非零激活态交并比 (Signed Jaccard < 0.30) + 时序相关性 (|ρ| < 0.32) + 空仓补盲率 (≥35%) + 回撤期救援 Alpha (>0) + 横截面并发度 CSC
- Gate 3 (双半场稳健边际与留一法新陈代谢关)：
  样本内 H1/H2 双半场夏普一致性 + LOO 后向消融修剪 + MAP-Elites 同生态位 1-for-1 精英替换
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pandas as pd

from examples.alpha_mining_agent.forensic_pipeline import (
    IS_END_DT,
    compute_active_union_correlation,
    compute_execution_net_metrics,
    compute_single_factor_net_sharpe_and_turnover,
    evaluate_execution_events,
    evaluate_triple_barrier,
    run_gate1_forensic_audit,
    run_parameter_smoothness_audit,
)
from examples.alpha_mining_agent.hypothesis_bank import (
    ALL_SIGNAL_SPECS,
    ARCH_A,
    ARCH_B,
    ARCH_C,
    ARCH_D,
    ARCH_E,
    ARCH_F,
    ARCHETYPE_RULERS,
    TF_SCALE_MAP,
    ArchetypeRuler,
    compute_signal_triggers_and_weights,
)
from examples.multi_factor_cta.atomic_factors import SoftCrossSectionalGate

# 周期自适应最小独立样本数门槛（解决日线/60m 低频高质量信号被固定高样本门槛误杀的问题）
TF_MIN_PURGED_SAMPLES: dict[str, int] = {
    "1d": 30,
    "60m": 30,
    "30m": 35,
    "15m": 55,
    "5m": 80,
}

# 三级多标的分层池化检验层级映射 (Tier 1: 跨资产普适簇 | Tier 2: 同质资产类别特征簇 | Tier 3: 交易制度与微结构簇)
STRATIFIED_POOL_TIERS: dict[str, int] = {
    ARCH_A: 1,
    ARCH_E: 1,
    ARCH_F: 1,
    ARCH_B: 2,
    ARCH_C: 2,
    ARCH_D: 3,
}


def get_stratified_pool_tier(archetype: str, timeframe: str = "30m") -> int:
    """根据战略簇与周期返回对应的分层池化检验层级 (Tier 1 / 2 / 3)。"""
    ruler = ARCHETYPE_RULERS.get(archetype)
    base_tier = int(getattr(ruler, "pool_tier", STRATIFIED_POOL_TIERS.get(archetype, 1)))
    if base_tier == 3 or archetype == ARCH_D:
        return 3
    if timeframe in ("5m", "15m"):
        return 3
    return base_tier


def infer_asset_pool_group(symbol: str, explicit_group: str | None = None) -> str:
    """Return an explicit asset/sector group, or infer only the broad asset class."""
    if explicit_group:
        group = explicit_group.strip().lower()
        if group in {"equity", "etf", "index", "stock"}:
            return "equity"
        if group == "commodity" or group.startswith("commodity:"):
            return group
        return group
    code = symbol.upper()
    if any(tag in code for tag in ("SHFE", "DCE", "CZCE", "ZCE", "INE", "GFEX", "KQ.M@")):
        return "commodity"
    return "equity"


# Full-60 超大商品板块的物理同质子产业链（按供需链固定先验划分，不随因子/样本期调整）。
# 键为「宏板块族」中各板块的 pool_group；值为 {子产业链: 品种代码}。仅当宏板块族品种数 > COMMODITY_SUBCHAIN_MIN_FAMILY_SIZE
# 且每条子链在当前样本内均 >= COMMODITY_SUBCHAIN_FLOOR 个品种时才启用；金属族（10 品种）不拆分。
# 未收录的品种（如 ec 集运指数、无物理产业链归属者）保持板块级标签，因池规模不足而无法准入（不降级）。
COMMODITY_SUBCHAIN_MIN_FAMILY_SIZE = 10
COMMODITY_SUBCHAIN_FLOOR = 5
COMMODITY_SUBCHAIN_FAMILIES: dict[frozenset[str], dict[str, tuple[str, ...]]] = {
    frozenset({"commodity:energy_chem"}): {
        "crude_aromatics_polyester": ("sc", "lu", "fu", "bu", "ta", "px", "eg", "pf", "pr"),
        "coal_chlor_polyolefin_rubber": ("ma", "ur", "l", "pp", "v", "eb", "pg", "sh", "ru", "br", "nr", "sp"),
    },
    frozenset({"commodity:agriculture"}): {
        "oils_oilseeds": ("m", "y", "p", "oi", "rm", "a", "b"),
        "softs_grains_livestock": ("sr", "cf", "cy", "ap", "cj", "pk", "lh", "jd", "cs", "c"),
    },
    frozenset({"commodity:ferrous", "commodity:new_energy"}): {
        "ferrous_burden": ("rb", "hc", "i", "j", "jm"),
        "buildmat_newenergy": ("fg", "sa", "sm", "sf", "si", "lc", "ps"),
    },
}


def _extract_commodity_root_code(symbol: str) -> str:
    """Extract lowercase commodity root code from short ('rb') or full symbol ('RB9999.XSGE', 'KQ.m@SHFE.rb')."""
    import re

    s = str(symbol).strip().lower()
    if "@" in s:
        s = s.rsplit("@", 1)[-1]
    if "." in s:
        parts = s.split(".")
        s = parts[-1] if parts[0] in {"shfe", "dce", "czce", "zce", "ine", "gfex", "cffex"} else parts[0]
    m = re.match(r"^([a-z]+)", s)
    return m.group(1) if m else s


def refine_commodity_pool_groups(groups: dict[str, str]) -> dict[str, str]:
    """Split oversized commodity sector pools into fixed homogeneous sub-chain pools.

    ``groups`` maps asset code to its ``commodity:<sector>`` label. A macro family is refined only when it has more
    than ``COMMODITY_SUBCHAIN_MIN_FAMILY_SIZE`` assets and every sub-chain keeps at least ``COMMODITY_SUBCHAIN_FLOOR``
    assets present; otherwise the family keeps its legacy sector labels. Assets outside every sub-chain keep their
    sector label, so they can never borrow a larger, heterogeneous pool.
    """
    refined = {k: str(v).strip().lower() for k, v in groups.items()}
    for sectors, chains in COMMODITY_SUBCHAIN_FAMILIES.items():
        members = {asset: group for asset, group in refined.items() if group in sectors}
        if len(members) <= COMMODITY_SUBCHAIN_MIN_FAMILY_SIZE:
            continue
        present = {
            name: [asset for asset in members if _extract_commodity_root_code(asset) in codes]
            for name, codes in chains.items()
        }
        if any(len(assets) < COMMODITY_SUBCHAIN_FLOOR for assets in present.values()):
            continue
        for name, assets in present.items():
            label = f"commodity:chain_{name}"
            for asset in assets:
                refined[asset] = label
    return refined


def _resolve_commodity_peer_groups(groups: dict[str, str], target: str, floor: int = 5) -> set[str]:
    """Resolve homogeneous commodity sector/macro-sector peer groups when sub-sectors are small."""
    exact = {target}
    if sum(1 for g in groups.values() if g in exact) >= floor and target not in {
        "commodity:precious_metals",
        "commodity:base_metals",
    }:
        return exact
    macro_families = (
        {"commodity:precious_metals", "commodity:base_metals"},
        {"commodity:new_energy", "commodity:ferrous"},
        {"commodity:agriculture", "commodity:energy_chem"},
    )
    for family in macro_families:
        if target in family and sum(1 for g in groups.values() if g in family) >= floor:
            return family
    return exact


def compute_signed_jaccard(w_u: np.ndarray, w_v: np.ndarray, eps: float = 1e-4) -> float:
    """
    计算两个稀疏规则型因子在非零激活状态下的同向交并比 (Signed Jaccard Similarity)：
    J_signed(u, v) = sum(I(w_u * w_v > 0)) / sum(I(|w_u| > eps or |w_v| > eps))
    专门解决大量共同 0 值掩盖真实触发重叠的问题。
    """
    active_union = (np.abs(w_u) > eps) | (np.abs(w_v) > eps)
    denom = int(active_union.sum())
    if denom == 0:
        return 0.0
    same_sign_inter = (w_u * w_v) > (eps * eps)
    return float(same_sign_inter.sum() / denom)


def compute_cross_sectional_concurrency(
    asset_weights_dict: dict[str, pd.Series], eps: float = 1e-3
) -> dict[str, float]:
    """
    计算单个规则因子在多品种横截面上的并发拥挤度 (Cross-Sectional Concurrency, CSC)：
    对齐各品种时间戳后，统计同一时刻触发同向持仓的品种占比分布。
    """
    if len(asset_weights_dict) <= 1:
        return {"mean_csc_pct": 0.0, "p95_csc_pct": 0.0, "max_csc_pct": 0.0}

    panel = pd.DataFrame(asset_weights_dict).ffill().fillna(0.0)
    active_mat = (panel.abs() > eps).astype(np.float64)
    any_active = active_mat.sum(axis=1) > 0
    if not np.any(any_active):
        return {"mean_csc_pct": 0.0, "p95_csc_pct": 0.0, "max_csc_pct": 0.0}

    csc_series = (active_mat.loc[any_active].mean(axis=1)) * 100.0
    return {
        "mean_csc_pct": round(float(csc_series.mean()), 2),
        "p95_csc_pct": round(float(np.percentile(csc_series, 95)), 2),
        "max_csc_pct": round(float(csc_series.max()), 2),
    }


def _pooled_time_block_t_stat(frames: list[pd.DataFrame]) -> tuple[float, int]:
    """Test the pooled Q/IPW weighted mean with a time-block ratio-estimator HAC variance.

    Each block contributes numerator A_b=sum(w*r) and mass B_b=sum(w). The target
    is mu=sum(A_b)/sum(B_b), not the equally weighted mean of A_b/B_b. HAC uses
    the centered ratio influence (A_b-mu*B_b)/mean(B), preserving regime weights
    and invariance to duplicating simultaneous assets.
    """
    if not frames:
        return 0.0, 0
    events = pd.concat(frames, ignore_index=True)
    duration = (events["exit_dt"] - events["entry_dt"]).max()
    block_width = max(pd.Timedelta(days=1), duration + pd.Timedelta(nanoseconds=1))
    events["block"] = (events["entry_dt"] - events["entry_dt"].min()) // block_width
    events["weighted_return"] = events["r_atr"] * events["pool_weight"]
    grouped = events.groupby("block")[["weighted_return", "pool_weight"]].sum()
    grouped = grouped.loc[grouped["pool_weight"] > 0]
    n = len(grouped)
    if n < 3:
        return 0.0, n
    numerator = grouped["weighted_return"].to_numpy(dtype=float)
    mass = grouped["pool_weight"].to_numpy(dtype=float)
    mean = float(numerator.sum() / mass.sum())
    influence = (numerator - mean * mass) / mass.mean()
    lag_max = min(n - 1, max(1, int(np.ceil(n ** (1 / 3)))))
    variance = float(influence @ influence / n)
    for lag in range(1, lag_max + 1):
        variance += 2 * (1 - lag / (lag_max + 1)) * float(influence[lag:] @ influence[:-lag] / n)
    variance *= n / (n - 1)
    return (float(mean / np.sqrt(variance / n)) if variance > 1e-12 else 0.0), n


class RuleFactorForensicValidator:
    """规则型与连续强度因子四级法医检验器 (Gate 1A + Gate 1B 分层池化 + 参数平原 + 激活态正交 & 边际增益)"""

    def __init__(
        self,
        is_end_dt: pd.Timestamp = IS_END_DT,
        max_pairwise_corr: float = 0.32,
        soft_gate: SoftCrossSectionalGate | bool | None = False,
    ) -> None:
        self.is_end_dt = is_end_dt
        self.max_pairwise_corr = max_pairwise_corr
        self.soft_gate = soft_gate
        self.rng = np.random.default_rng(42)

    def _resolve_soft_gated_inputs(
        self,
        df_tf: pd.DataFrame,
        raw_triggers: np.ndarray,
        weights_tf: np.ndarray | None,
        archetype: str,
        direction: int,
        soft_gate: SoftCrossSectionalGate | bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        eff_gate = self.soft_gate if soft_gate is None else soft_gate
        trig_bool = np.asarray(raw_triggers, dtype=bool)
        if eff_gate is None or eff_gate is False:
            return trig_bool, weights_tf
        gate_obj = eff_gate if isinstance(eff_gate, SoftCrossSectionalGate) else SoftCrossSectionalGate()
        cs_rank = gate_obj.extract_cs_rank(df_tf)
        if cs_rank is None:
            return trig_bool, weights_tf
        er = gate_obj.extract_causal_er(df_tf)
        return gate_obj.apply_to_triggers_and_weights(
            trig_bool,
            weights_tf,
            cs_rank,
            er,
            archetype=archetype,
            direction=direction,
        )

    def run_gate1_single_asset_audit(
        self,
        df_tf: pd.DataFrame,
        raw_triggers: np.ndarray,
        direction: int,
        archetype: str,
        timeframe: str,
        fee_rate: float = 0.0002,
        weights_tf: np.ndarray | None = None,
        soft_gate: SoftCrossSectionalGate | bool | None = None,
    ) -> dict[str, Any]:
        """
        Gate 1A：单品种单周期法医标尺检验（严格样本内 IS）
        - Purged 去重叠独立事件抽样
        - 周期自适应最小样本数门槛
        - 分战略簇三重屏障标尺 (Next-Bar Open + 交易成本扣除 + 连续质量分 Q 加权)
        - MFE / MAE 路径偏离比门槛 + 单因子扣费 Net Sharpe + 前向多期 Rank IC / ICIR 衰减曲线
        """
        triggers, weights = self._resolve_soft_gated_inputs(
            df_tf, raw_triggers, weights_tf, archetype, direction, soft_gate
        )
        return run_gate1_forensic_audit(
            df_tf,
            triggers,
            direction,
            ARCHETYPE_RULERS[archetype],
            timeframe,
            weights_tf=weights,
            fee_rate=fee_rate,
            is_end_dt=self.is_end_dt,
        )

    def run_gate1_cross_asset_universality_audit(
        self,
        asset_audits: dict[str, dict[str, Any]],
        min_positive_ratio: float = 0.60,
    ) -> dict[str, Any]:
        """Compatibility entry point using the same strict, real-event Tier 1 admission contract."""
        result = self.run_stratified_pooled_gate1b_audit(archetype=ARCH_A, asset_audits=asset_audits)
        if result["universal_ratio"] < max(0.60, min_positive_ratio):
            result["passed"] = False
        return result

    def run_stratified_pooled_gate1b_audit(
        self,
        archetype: str = ARCH_A,
        timeframe: str = "30m",
        *,
        pool_tf_dfs: dict[str, pd.DataFrame] | None = None,
        triggers_by_symbol: dict[str, np.ndarray] | None = None,
        weights_by_symbol: dict[str, np.ndarray] | None = None,
        sig_id: str | None = None,
        direction: int = 1,
        eval_fn: Callable[[pd.DataFrame], tuple[np.ndarray, np.ndarray]] | None = None,
        asset_audits: dict[str, dict[str, Any]] | None = None,
        per_symbol_gate1: dict[str, dict[str, Any]] | None = None,
        asset_groups: dict[str, str] | None = None,
        target_pool_group: str = "equity",
        microstructure_groups: dict[str, str] | None = None,
        target_microstructure_group: str | None = None,
        min_pool_assets: int | None = None,
        fee_rate: float = 0.0002,
        allow_single_asset_fallback: bool = False,
        allow_single_asset_degrade: bool | None = None,
        soft_gate: SoftCrossSectionalGate | bool | None = None,
        allow_pooled_effective_n: bool = False,
        target_asset: str | None = None,
    ) -> dict[str, Any]:
        """Admit only real, training-only event evidence from the required asset universe.

        Tier 1 requires at least 15 assets and both equity and commodity classes.
        Tier 2 requires five assets from the exact equity or commodity:<sector> group.
        Tier 3 additionally requires an explicitly identical trading/liquidity regime.
        Legacy single-asset and pooled-effective-sample flags preserve call compatibility,
        never permission to lower the per-asset evidence floor or missing-regime requirements.
        When ``target_asset`` is given for an all-commodity pool, oversized sector pools are first split into the
        fixed homogeneous sub-chains of ``COMMODITY_SUBCHAIN_FAMILIES`` and the target is that asset's sub-chain.
        """
        raw_audits = dict(asset_audits or per_symbol_gate1 or {})
        if not raw_audits and pool_tf_dfs is not None:
            for symbol, df_symbol in pool_tf_dfs.items():
                if triggers_by_symbol is not None and symbol in triggers_by_symbol:
                    triggers = triggers_by_symbol[symbol]
                    weights = (weights_by_symbol or {}).get(symbol)
                elif eval_fn is not None:
                    triggers, weights = eval_fn(df_symbol)
                elif sig_id in ALL_SIGNAL_SPECS:
                    spec = ALL_SIGNAL_SPECS[sig_id]
                    direction = spec.direction
                    triggers, weights = compute_signal_triggers_and_weights(
                        df_symbol, sig_id, scale=TF_SCALE_MAP.get(timeframe, 1.0)
                    )
                else:
                    raise ValueError(f"Missing triggers/eval_fn/sig_id for {symbol}.")
                raw_audits[symbol] = self.run_vol_and_regime_normalized_audit(
                    df_symbol,
                    triggers,
                    direction,
                    archetype,
                    timeframe,
                    fee_rate=fee_rate,
                    weights_tf=weights,
                    soft_gate=soft_gate,
                )
        tier = get_stratified_pool_tier(archetype, timeframe)
        names = {1: "Tier1_CrossAsset", 2: "Tier2_AssetClass", 3: "Tier3_Microstructure"}
        groups = {symbol: infer_asset_pool_group(symbol, (asset_groups or {}).get(symbol)) for symbol in raw_audits}
        target = infer_asset_pool_group(target_pool_group, target_pool_group)
        broad_all = {groups[s].split(":", 1)[0] for s in raw_audits}
        commodity_sector_mode = broad_all == {"commodity"} and target.startswith("commodity:")
        if commodity_sector_mode and (target_asset in groups or target.startswith("commodity:chain_")):
            groups = refine_commodity_pool_groups(groups)
            if target_asset in groups:
                target = groups[target_asset]
        if commodity_sector_mode:
            peer_groups = _resolve_commodity_peer_groups(groups, target, floor=5)
            selected = [s for s in raw_audits if groups[s] in peer_groups]
        else:
            selected = list(raw_audits) if tier == 1 else [s for s in raw_audits if groups[s] == target]
        reasons = []
        if tier == 3:
            eff_micro_groups = dict(microstructure_groups or {})
            eff_target_micro = target_microstructure_group
            if not eff_target_micro and commodity_sector_mode:
                eff_target_micro = "commodity_futures"
                for s in selected:
                    eff_micro_groups.setdefault(s, eff_target_micro)
            if not eff_target_micro:
                reasons.append("Tier3缺少明确的交易制度/流动性组")
                selected = []
            else:
                selected = [s for s in selected if eff_micro_groups.get(s) == eff_target_micro]
        if tier == 2 and target == "commodity":
            reasons.append("Tier2商品池必须指定commodity:<sector>，不能混合所有商品")
        floor = 5 if commodity_sector_mode else (15 if tier == 1 else 5)
        required_assets = max(floor, min_pool_assets or floor)
        if len(selected) < required_assets:
            reasons.append(f"目标池规模={len(selected)} < {required_assets}，不得降级准入")
        if (
            tier == 1
            and not commodity_sector_mode
            and {groups[s].split(":", 1)[0] for s in selected} != {"equity", "commodity"}
        ):
            reasons.append("Tier1必须同时包含权益和商品资产")

        positive, event_frames, net_sharpes, ipw_means = [], [], [], []
        low_eff_reasons, missing_regime_reasons = [], []
        valid_audits = {}
        for symbol in selected:
            audit = raw_audits[symbol]
            records = audit.get("events")
            if not isinstance(records, list) or (not records and int(audit.get("is_n_purged", 0)) > 0):
                reasons.append(f"{symbol}: 缺少真实events，不得用摘要生成样本")
                continue
            if not records:
                low_eff_reasons.append(f"{symbol}: 缺少真实events (有效独立样本=0 < 30)")
                continue
            events = pd.DataFrame(records)
            required = {"trigger_dt", "entry_dt", "exit_dt", "r_atr", "net_ret", "quality_w", "regime"}
            if not required.issubset(events.columns):
                reasons.append(f"{symbol}: events字段不完整")
                continue
            for col in ("trigger_dt", "entry_dt", "exit_dt"):
                events[col] = pd.to_datetime(events[col], errors="coerce")
            finite = np.isfinite(events[["r_atr", "net_ret", "quality_w"]].to_numpy(dtype=float)).all()
            ordered = (events["trigger_dt"] < events["entry_dt"]) & (events["entry_dt"] <= events["exit_dt"])
            in_sample = events["exit_dt"] <= self.is_end_dt
            if (
                not finite
                or not ordered.all()
                or not in_sample.all()
                or (events["quality_w"] <= 0).any()
                or (events["quality_w"] > 1).any()
            ):
                reasons.append(f"{symbol}: events含无效值或跨训练边界标签")
                continue
            events = events.sort_values("entry_dt").reset_index(drop=True)
            starts = events["entry_dt"].iloc[1:].to_numpy()
            previous_exits = events["exit_dt"].iloc[:-1].to_numpy()
            # A real next-open reversal closes before opening its opposite leg.
            # Diagnostic intrabar exits do not establish that ordering.
            overlaps = starts < previous_exits if audit.get("event_source") == "execution" else starts <= previous_exits
            if overlaps.any():
                reasons.append(f"{symbol}: events持有期重叠，尚未purge")
                continue
            q = events["quality_w"].to_numpy(dtype=float)
            effective_n = float(q.sum() ** 2 / (q @ q))
            min_asset_eff_n = 30.0
            if effective_n < min_asset_eff_n - 1e-9:
                low_eff_reasons.append(f"{symbol}: 有效独立样本={effective_n:.2f} < {min_asset_eff_n:.0f}")
            # Recompute all moments from the immutable event table; summaries are display-only.
            norm_r = float(np.average(events["r_atr"], weights=q))
            raw_mean = float(np.average(events["net_ret"], weights=q))
            regime_means = []
            events["pool_weight"] = 0.0
            active_regimes = [regime for regime in (-1, 0, 1) if int((events["regime"] == regime).sum()) >= 2]
            for regime in active_regimes:
                mask = events["regime"] == regime
                regime_q = events.loc[mask, "quality_w"]
                regime_means.append(float(np.average(events.loc[mask, "r_atr"], weights=regime_q)))
                events.loc[mask, "pool_weight"] = regime_q / regime_q.sum() / float(len(active_regimes))
            require_three_regimes = audit.get("event_source") != "execution"
            if len(regime_means) == 0 or (require_three_regimes and len(regime_means) != 3):
                missing_regime_reasons.append(f"{symbol}: 三态IPW缺少事件支持，不能声称状态均衡")
                ipw_mean = 0.0
            else:
                ipw_mean = float(np.mean(regime_means))
            if effective_n >= min_asset_eff_n - 1e-9 and norm_r > 0 and raw_mean > 0 and ipw_mean > 0:
                positive.append(symbol)
            events["symbol"] = symbol
            event_frames.append(events)
            ipw_means.append(ipw_mean)
            net_sharpes.append(float(audit.get("is_net_sharpe", 0.0)))
            valid_audits[symbol] = audit
        ratio = len(positive) / len(selected) if selected else 0.0
        pooled_t, block_count = _pooled_time_block_t_stat(event_frames)
        min_ratio = 0.60 if tier == 1 else (0.70 if tier == 3 else (0.75 if target == "equity" else 0.65))
        min_t = (1.96 if commodity_sector_mode else 2.5) if tier == 1 else 1.96
        if ratio < min_ratio:
            reasons.extend(low_eff_reasons)
            reasons.extend(missing_regime_reasons)
            reasons.append(f"Tier{tier}同质池普适率={ratio:.1%} < {min_ratio:.0%}")
        if pooled_t < min_t:
            reasons.append(f"同步时间块t={pooled_t:.3f} < {min_t:.2f}")
        mean_sharpe = float(np.mean(net_sharpes)) if net_sharpes else 0.0
        if tier == 3 and mean_sharpe <= 0:
            reasons.append("Tier3扣费净Sharpe不大于0")
        return {
            "passed": not reasons,
            "reason": "; ".join(reasons) if reasons else f"通过Tier{tier}真实事件池化检验",
            "tier": tier,
            "pool_tier": names[tier],
            "pool_size": len(selected),
            "required_pool_size": required_assets,
            "degraded_single_asset": len(selected) == 1,
            "universal_ratio": round(ratio, 4),
            "universal_ratio_pct": round(ratio * 100, 1),
            "pooled_t_stat": round(pooled_t, 3),
            "independent_time_blocks": block_count,
            "mean_pool_net_sharpe": round(mean_sharpe, 3),
            "mean_ipw_r_atr": float(np.mean(ipw_means)) if ipw_means else 0.0,
            "total_pool_is_n": sum(len(frame) for frame in event_frames),
            "positive_assets": positive,
            "evaluated_assets": selected,
            "per_asset_audits": valid_audits,
        }

    def run_parameter_smoothness_audit(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """委托执行参数平原平滑度检验 (Parameter Smoothness Gate)。"""
        if "is_end_dt" not in kwargs:
            kwargs["is_end_dt"] = self.is_end_dt
        return run_parameter_smoothness_audit(*args, **kwargs)

    def run_gate2_complementarity_and_jaccard_audit(
        self,
        df_grid: pd.DataFrame | None = None,
        cand_w_on_grid: np.ndarray | None = None,
        existing_same_dir_weights: dict[str, np.ndarray] | None = None,
        current_portfolio_w: np.ndarray | None = None,
        max_corr_threshold: float | None = None,
        max_jaccard_threshold: float = 0.30,
        min_gap_fill_pct: float = 35.0,
        active_corr_eps: float = 1e-3,
        min_active_bars: int = 30,
        *,
        df_30m: pd.DataFrame | None = None,
        cand_w_30m: np.ndarray | None = None,
        cand_w_on_30m: np.ndarray | None = None,
        zoo_weights_30m: dict[str, np.ndarray] | None = None,
        curr_portfolio_w_30m: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Gate 2 / Gate 3：时序×截面三维真互补与激活态正交门禁（严格样本内 IS）
        1. 激活态并集条件相关性 |ρ_active| < 0.32 且 非零激活态交并比 Signed Jaccard < 0.30；
        2. 空仓填补率 >= 35%；
        3. 原组合回撤/亏损期的雪中送炭 Alpha (dd_rescue_alpha_bp > 0)。
        """
        if df_grid is None:
            df_grid = df_30m
        if cand_w_on_grid is None:
            cand_w_on_grid = cand_w_30m if cand_w_30m is not None else cand_w_on_30m
        if existing_same_dir_weights is None:
            existing_same_dir_weights = zoo_weights_30m or {}
        if current_portfolio_w is None:
            current_portfolio_w = curr_portfolio_w_30m
        if max_corr_threshold is None:
            max_corr_threshold = self.max_pairwise_corr
        assert df_grid is not None and cand_w_on_grid is not None and current_portfolio_w is not None

        is_mask = (df_grid["dt"] <= self.is_end_dt).values
        if int(is_mask.sum()) < 25:
            return {"passed": False, "reason": "样本内历史不足"}
        cw_is = cand_w_on_grid[is_mask]
        port_w_is = current_portfolio_w[is_mask]

        max_corr = 0.0
        max_full_corr = 0.0
        max_jaccard = 0.0
        max_peer = "None"
        for peer_id, pw in existing_same_dir_weights.items():
            pw_is = pw[is_mask]
            act_c = float(
                abs(
                    compute_active_union_correlation(cw_is, pw_is, eps=active_corr_eps, min_active_bars=min_active_bars)
                )
            )
            full_c = (
                float(abs(np.corrcoef(cw_is, pw_is)[0, 1])) if (np.std(cw_is) > 1e-6 and np.std(pw_is) > 1e-6) else 0.0
            )
            jac = compute_signed_jaccard(cw_is, pw_is)
            if full_c > max_full_corr:
                max_full_corr = full_c
            if act_c > max_corr or jac > max_jaccard:
                if act_c > max_corr:
                    max_corr = act_c
                    max_peer = peer_id
                if jac > max_jaccard:
                    max_jaccard = jac

        cand_active = np.abs(cw_is) > 0.01
        if int(cand_active.sum()) == 0:
            return {"passed": False, "reason": "样本内无有效持仓激活"}

        from examples.alpha_mining_agent.forensic_pipeline import _next_open_rescue_metrics

        gap_fill_ratio, dd_rescue_alpha_bp = _next_open_rescue_metrics(
            df_grid.loc[is_mask].reset_index(drop=True), cw_is, port_w_is
        )

        passed = (
            (max_corr < max_corr_threshold)
            and (max_jaccard < max_jaccard_threshold)
            and (gap_fill_ratio >= min_gap_fill_pct)
            and (dd_rescue_alpha_bp >= 0.5)
        )
        return {
            "passed": bool(passed),
            "max_corr": round(max_corr, 3),
            "max_full_sample_corr": round(max_full_corr, 3),
            "max_signed_jaccard": round(max_jaccard, 3),
            "max_corr_peer": max_peer,
            "gap_fill_ratio_pct": round(gap_fill_ratio, 1),
            "dd_rescue_alpha_bp": round(dd_rescue_alpha_bp, 2),
        }

    run_gate3_orthogonality_and_marginal_audit = run_gate2_complementarity_and_jaccard_audit

    def run_vol_and_regime_normalized_audit(
        self,
        df_tf: pd.DataFrame,
        raw_triggers: np.ndarray,
        direction: int,
        archetype: str,
        timeframe: str,
        fee_rate: float = 0.0002,
        weights_tf: np.ndarray | None = None,
        soft_gate: SoftCrossSectionalGate | bool | None = None,
        *,
        event_source: Literal["diagnostic", "execution"] = "diagnostic",
    ) -> dict[str, Any]:
        """
        解决【跨标的波动率不同 + 市场状态构成不同】的法医归一化检验。
        `event_source="execution"` uses completed actual Rust execution episodes;
        the default keeps fixed ATR ruler diagnostics for existing analysis callers.
        Execution mode consumes final weights unchanged and requires weights_tf.
        诊断步骤：
        1. 波动率无量纲化 (R-Multiple / ATR-Unit Edge)：
           将每笔交易净收益除以入场时刻因果相对波动率 `atr_pct = atr14 / close`，
           转化为“每承担 1 单位 ATR 波动捕获的净 ATR 倍数 (r_atr)”，消除高波/低波标的名义收益率不可比问题；
        2. 因果三态状态机切分 (Bull Trend / Bear Trend / Choppy Range)：
           利用 Kaufman 效率系数 ER(20) 与均线结构将每个标的样本内切分为【多头趋势态(+1) / 空头趋势态(-1) / 震荡盘整态(0)】；
        3. 逆概率重加权状态均衡期望 (IPW Regime-Balanced Edge) 与主场状态安全边际 (Home-Regime ΔW)：
           按标准均衡状态配比 (Bull:Bear:Range = 1/3:1/3:1/3) 计算状态重加权后的期望与安全边际，
           剥离“某标的恰巧在样本内走了三年震荡或三年熊市”的状态构成偏差 (Regime Compositional Bias)。
        """
        if event_source not in ("diagnostic", "execution"):
            raise ValueError("event_source must be diagnostic or execution")
        if event_source == "execution" and weights_tf is None:
            raise ValueError("Actual execution events require signed weights_tf")
        ruler: ArchetypeRuler = ARCHETYPE_RULERS[archetype]
        is_mask = (df_tf["dt"] <= self.is_end_dt).values
        if int(is_mask.sum()) < 25:
            return {
                "passed": False,
                "event_source": event_source,
                "is_n_purged": 0,
                "events": [],
                "reason": "样本内历史不足",
            }
        eff_max_bars = ruler.max_bars * 2 if timeframe == "5m" else ruler.max_bars
        eff_tp_atr = ruler.tp_atr * 1.35 if timeframe == "5m" else ruler.tp_atr
        eff_sl_atr = ruler.sl_atr * 1.20 if timeframe == "5m" else ruler.sl_atr

        regimes = classify_causal_market_regime(df_tf)
        reg_is = regimes[is_mask]
        bull_time_pct = float(np.mean(reg_is == 1) * 100.0)
        bear_time_pct = float(np.mean(reg_is == -1) * 100.0)
        range_time_pct = float(np.mean(reg_is == 0) * 100.0)

        c_arr = df_tf["close"].values
        atr_pct_arr = np.maximum(df_tf["atr14"].values / np.maximum(c_arr, 1e-6), 1e-4)
        median_atr_pct = float(np.nanmedian(atr_pct_arr[is_mask]) * 100.0)

        if event_source == "execution":
            # These are final factor decisions; applying another gate would change
            # the traded path being audited. Rust owns resizing, fills and costs.
            tb_is = evaluate_execution_events(df_tf, weights_tf, fee_rate, label_end_dt=self.is_end_dt)
        else:
            trig_bool, weights_tf = self._resolve_soft_gated_inputs(
                df_tf, raw_triggers, weights_tf, archetype, direction, soft_gate
            )
            is_raw_idx = np.where(trig_bool & is_mask)[0]
            tb_is = evaluate_triple_barrier(
                df_tf,
                is_raw_idx,
                direction=direction,
                max_bars=eff_max_bars,
                tp_atr=eff_tp_atr,
                sl_atr=eff_sl_atr,
                fee_rate=fee_rate,
                weights_tf=weights_tf,
                label_end_dt=self.is_end_dt,
                purge_overlaps=True,
            )
        if tb_is.empty:
            return {
                "event_source": event_source,
                "is_n_purged": 0,
                "events": [],
                "median_bar_atr_pct": round(median_atr_pct, 3),
            }

        t_idx = tb_is["trigger_idx"].values.astype(int)
        raw_nets = tb_is["net_ret"].values
        q_w = tb_is["quality_w"].values if "quality_w" in tb_is.columns else np.ones(len(tb_is), dtype=np.float64)
        use_q_w = weights_tf is not None and (np.std(q_w) > 1e-4 or np.mean(q_w) < 0.99)
        q_norm = (q_w / max(float(np.sum(q_w)), 1e-12)) if use_q_w else np.full(len(tb_is), 1.0 / len(tb_is))

        entry_atr_pct = tb_is["atr_pct"].to_numpy()
        # 核心变换 1：转化为无量纲 ATR 单位净收益 (R-Multiple in ATR units)
        r_atr = raw_nets / entry_atr_pct
        trig_regimes = regimes[t_idx]

        # 定义该因子战略簇对应的“目标主场状态 (Home Regime)”
        if archetype == ARCH_A:
            home_mask = (trig_regimes == 1) if direction == 1 else (trig_regimes == -1)
        elif archetype == ARCH_C:
            home_mask = trig_regimes == 0
        else:
            # 背驰衰竭 / 量价筹码 / 波动率压缩 / 多周期共振：震荡态或逆势衰竭态均为主场
            home_mask = (trig_regimes == 0) | (trig_regimes == -direction)

        def _calc_safety(rets: np.ndarray) -> float:
            if len(rets) < 2:
                return 0.0
            w = float(np.mean(rets > 0))
            pos = rets[rets > 0]
            neg = rets[rets <= 0]
            avg_w = float(np.mean(pos)) if len(pos) > 0 else 0.0
            avg_l = float(abs(np.mean(neg))) if len(neg) > 0 else 1e-6
            r = avg_w / max(avg_l, 1e-6)
            return (w - 1.0 / (1.0 + r)) * 100.0

        uncond_safety = _calc_safety(raw_nets)
        home_safety = _calc_safety(raw_nets[home_mask]) if int(home_mask.sum()) >= 3 else uncond_safety
        norm_r_atr_val = float(np.sum(q_norm * r_atr))
        home_r_atr = (
            float(np.sum(q_norm[home_mask] * r_atr[home_mask]) / max(float(np.sum(q_norm[home_mask])), 1e-12))
            if int(home_mask.sum()) >= 3
            else norm_r_atr_val
        )

        # 核心变换 2：三态逆概率重加权 (IPW Regime-Balanced Expectancy)
        state_r_atr = []
        state_safeties = []
        for st in (1, 0, -1):
            m = trig_regimes == st
            if int(m.sum()) >= 2:
                w_m = q_norm[m]
                state_r_atr.append(float(np.sum(w_m * r_atr[m]) / max(float(np.sum(w_m)), 1e-12)))
                state_safeties.append(_calc_safety(raw_nets[m]))
        ipw_r_atr = float(np.mean(state_r_atr)) if len(state_r_atr) == 3 else 0.0
        ipw_safety = float(np.mean(state_safeties)) if len(state_safeties) == 3 else 0.0

        if event_source == "execution":
            net_sh_pack = compute_execution_net_metrics(df_tf, weights_tf, fee_rate=fee_rate, is_end_dt=self.is_end_dt)
        elif weights_tf is not None:
            net_sh_pack = compute_single_factor_net_sharpe_and_turnover(
                df_tf, weights_tf, timeframe, fee_rate=fee_rate, is_end_dt=self.is_end_dt
            )
        else:
            synth_w = np.zeros(len(df_tf), dtype=np.float64)
            for _, r_row in tb_is.iterrows():
                st_i = int(r_row["trigger_idx"])
                ed_i = int(r_row["exit_idx"])
                synth_w[st_i:ed_i] = float(direction)
            net_sh_pack = compute_single_factor_net_sharpe_and_turnover(
                df_tf, synth_w, timeframe, fee_rate=fee_rate, is_end_dt=self.is_end_dt
            )

        std_r_atr = float(np.std(r_atr, ddof=1)) if len(r_atr) > 1 else 0.5
        effective_n = float(1.0 / np.sum(q_norm**2))
        variance = float(np.sum(q_norm * (r_atr - norm_r_atr_val) ** 2))
        t_stat_r_atr = float(norm_r_atr_val / np.sqrt(variance / max(effective_n - 1, 1))) if variance > 1e-9 else 0.0
        mfe_mean = float(np.sum(q_norm * tb_is["mfe_atr"].values))
        mae_mean = float(np.sum(q_norm * tb_is["mae_atr"].values))
        exc_ratio = float(mfe_mean / max(mae_mean, 1e-6))

        return {
            "event_source": event_source,
            "is_n_purged": int(len(tb_is)),
            "median_bar_atr_pct": round(median_atr_pct, 3),
            "regime_dist_str": f"牛{bull_time_pct:.0f}%/震{range_time_pct:.0f}%/熊{bear_time_pct:.0f}%",
            "raw_mean_pct": round(float(np.sum(q_norm * raw_nets)) * 100.0, 3),
            "norm_r_atr": round(norm_r_atr_val, 3),
            "std_r_atr": round(std_r_atr, 3),
            "t_stat_r_atr": round(t_stat_r_atr, 3),
            "r_atr_values": [float(x) for x in r_atr],
            "events": tb_is.assign(r_atr=r_atr, regime=trig_regimes).to_dict("records"),
            "is_effective_n": effective_n,
            "regime_support_complete": len(state_r_atr) == 3,
            "ipw_r_atr": round(ipw_r_atr, 3),
            "home_r_atr": round(home_r_atr, 3),
            "uncond_safety_pct": round(uncond_safety, 1),
            "ipw_safety_pct": round(ipw_safety, 1),
            "home_safety_pct": round(home_safety, 1),
            "excursion_ratio": round(exc_ratio, 3),
            "is_net_sharpe": round(float(net_sh_pack["is_net_sharpe"]), 3),
            "ann_turnover": round(float(net_sh_pack.get("ann_turnover", 0.0)), 2),
            "cost_to_atr_pct": round(float(2 * fee_rate / (median_atr_pct / 100.0) * 100.0), 1),
            "cost_to_atr_basis": "nominal_round_trip_estimate",
        }


def run_stratified_pooled_gate1b_audit(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """模块级函数入口：执行三级多标的分层池化检验 (Pre-Admission Gate 1B)。"""
    is_end_dt = kwargs.pop("is_end_dt", IS_END_DT)
    validator = RuleFactorForensicValidator(is_end_dt=is_end_dt)
    return validator.run_stratified_pooled_gate1b_audit(*args, **kwargs)


_EXECUTION_POOL_COLUMNS = [
    "fold",
    "uid",
    "sig_id",
    "tf",
    "archetype",
    "total_n",
    "min_asset_n",
    "median_asset_n",
    "pos_asset_ratio",
    "pooled_mean_pct",
    "mean_sharpe",
    "pooled_t",
    "independent_time_blocks",
    "pool_evidence_kind",
    "pool_passed_assets",
    "pool_audits",
]


def summarize_execution_pooled_scores(scores_df: pd.DataFrame) -> pd.DataFrame:
    """Pool completed execution evidence using the existing stratified time-block audit.

    Input rows identify ``fold/asset/uid/tf/archetype`` and carry the complete
    ``events`` list and ``cutoff`` returned by the study evaluator. Explicit
    ``pool_group`` and ``microstructure_group`` identify the applicable universe.
    Missing events or pool metadata never grant admission. Display summaries,
    including historical t statistics, are not evidence. One output row per
    fold/unit retains per-target audit results and the assets actually admitted.
    """
    if scores_df.empty:
        return pd.DataFrame(columns=_EXECUTION_POOL_COLUMNS)
    required = {"fold", "asset", "uid", "tf", "archetype"}
    if not required.issubset(scores_df.columns):
        raise ValueError(f"Missing candidate identity columns: {sorted(required - set(scores_df.columns))}")
    if scores_df.duplicated(["fold", "asset", "uid"]).any():
        raise ValueError("Duplicate fold/asset/unit evidence is not an independent sample")
    rows = []
    for (fold, uid), group in scores_df.groupby(["fold", "uid"], sort=True):
        if group["tf"].nunique() != 1 or group["archetype"].nunique() != 1:
            raise ValueError("One fold/unit cannot mix timeframes or archetype definitions")
        tf, archetype = str(group["tf"].iloc[0]), str(group["archetype"].iloc[0])
        audits, asset_groups, micro_groups = {}, {}, {}
        cutoffs = set()
        for record in group.to_dict("records"):
            asset = str(record["asset"])
            events = record.get("events")
            audits[asset] = {
                "events": events if isinstance(events, list) else [],
                "event_source": "execution",
                "is_net_sharpe": record.get("is_net_sharpe", 0.0),
            }
            # The broad commodity label deliberately fails the Tier 2 sector contract.
            pool_group = record.get("pool_group")
            asset_groups[asset] = pool_group if isinstance(pool_group, str) and pool_group else "unknown"
            micro = record.get("microstructure_group")
            if isinstance(micro, str) and micro:
                micro_groups[asset] = micro
            elif isinstance(pool_group, str) and pool_group.lower().startswith("commodity:"):
                micro_groups[asset] = "commodity_futures"
            cutoff = pd.to_datetime(record.get("cutoff"), errors="coerce")
            if pd.notna(cutoff):
                cutoffs.add(cutoff)
        # Oversized commodity sectors are pooled by fixed physical sub-chain, never as one heterogeneous sector.
        asset_groups = refine_commodity_pool_groups(asset_groups)
        cutoff_valid = len(cutoffs) == 1 and "cutoff" in group and group["cutoff"].notna().all()
        cutoff = next(iter(cutoffs)) if cutoff_valid else pd.Timestamp.min
        checker = RuleFactorForensicValidator(is_end_dt=cutoff, soft_gate=False)
        by_target, by_asset, passed_assets = {}, {}, []
        for asset in audits:
            target = (asset_groups[asset], micro_groups.get(asset))
            if target not in by_target:
                by_target[target] = checker.run_stratified_pooled_gate1b_audit(
                    archetype=archetype,
                    timeframe=tf,
                    asset_audits=audits,
                    asset_groups=asset_groups,
                    target_pool_group=target[0],
                    microstructure_groups=micro_groups,
                    target_microstructure_group=target[1],
                )
            result = by_target[target]
            by_asset[asset] = {key: value for key, value in result.items() if key != "per_asset_audits"}
            if result["passed"] and cutoff_valid and asset_groups[asset] != "unknown":
                passed_assets.append(asset)
        results = list(by_target.values())
        sizes = [len(audit["events"]) for audit in audits.values()]
        event_net = [
            float(event["net_ret"])
            for audit in audits.values()
            for event in audit["events"]
            if "net_ret" in event and np.isfinite(event["net_ret"])
        ]
        if len(results) == 1:
            pos_ratio = float(results[0]["universal_ratio"])
            pooled_t_val = float(results[0]["pooled_t_stat"])
            indep_blocks = int(results[0]["independent_time_blocks"])
        else:
            pos_ratio = round(float((group["is_net_sharpe"] > 0.0).mean()), 3) if "is_net_sharpe" in group else 0.0
            t_vals = []
            for audit in audits.values():
                rets = np.asarray(
                    [float(e["net_ret"]) for e in audit["events"] if "net_ret" in e and np.isfinite(e["net_ret"])],
                    dtype=float,
                )
                if len(rets) > 1:
                    sd = float(rets.std(ddof=1))
                    t_vals.append(float(rets.mean() / (sd / np.sqrt(len(rets)))) if sd > 1e-12 else 0.0)
            n_act = sum(1 for sz in sizes if sz > 0)
            pooled_t_val = round(float(np.sum(t_vals) / np.sqrt(max(n_act, 1))), 3) if t_vals else 0.0
            indep_blocks = int(sum(r["independent_time_blocks"] for r in results))
        rows.append(
            {
                "fold": fold,
                "uid": uid,
                "sig_id": str(uid).split("@")[0],
                "tf": tf,
                "archetype": archetype,
                "total_n": sum(sizes),
                "min_asset_n": min(sizes, default=0),
                "median_asset_n": float(np.median(sizes)) if sizes else 0.0,
                "pos_asset_ratio": pos_ratio,
                "pooled_mean_pct": round(float(np.mean(event_net) * 100), 4) if event_net else 0.0,
                "mean_sharpe": round(float(group["is_net_sharpe"].mean()), 3) if "is_net_sharpe" in group else 0.0,
                "pooled_t": pooled_t_val,
                "independent_time_blocks": indep_blocks,
                "pool_evidence_kind": "execution_time_block_hac",
                "pool_passed_assets": passed_assets,
                "pool_audits": by_asset,
            }
        )
    return pd.DataFrame(rows, columns=_EXECUTION_POOL_COLUMNS)


def select_pooled_multitf_gate1_units(
    f_scores: pd.DataFrame,
    f_pool: pd.DataFrame,
    asset_id: str,
    *,
    allow_lowturn_hetero_15m: bool = False,
) -> pd.DataFrame:
    """
    滚动样本内多周期跨品种池化自由度 Gate 1 统一准入接口：
    - 30m: 单品种 n >= 15, mean_net_pct > 0, is_net_sharpe > 0.15
    - 15m: 标准候选要求 n >= 15；若启用低换手异构 15m 跨日波段因子 (H15M_*)，允许 n >= 6 且年化换手 <= 26x 且池化 total_n >= 150
    - 60m: 单品种 n >= 10, mean_net_pct > 0, is_net_sharpe > 0.15
    - 1d: 单品种 n >= 2, mean_net_pct > 0, is_net_sharpe > 0.10 且池化 total_n >= 25、池化均值 > 0
    """
    pool_cols = ["total_n", "pooled_mean_pct", "pooled_t", "pos_asset_ratio"]
    empty = f_scores.iloc[:0].copy()
    for name in pool_cols:
        if name not in empty:
            empty[name] = pd.Series(dtype=float)
    if f_scores.empty or f_pool.empty:
        return empty
    join = ["fold", "uid"] if "fold" in f_scores.columns and "fold" in f_pool.columns else ["uid"]
    avail_pool_cols = [c for c in pool_cols if c in f_pool.columns]
    sub = f_scores[f_scores["asset"] == asset_id].merge(
        f_pool[[*join, *avail_pool_cols]].drop_duplicates(join),
        on=join,
        how="left",
    )
    c_30m = (sub["tf"] == "30m") & (sub["n"] >= 15) & (sub["mean_net_pct"] > 0.0) & (sub["is_net_sharpe"] > 0.15)
    if allow_lowturn_hetero_15m:
        c_15m = (
            (sub["tf"] == "15m")
            & (
                (sub["n"] >= 15)
                | (
                    sub["uid"].str.startswith("H15M_")
                    & (sub["n"] >= 6)
                    & (sub["ann_turnover"] <= 26.0)
                    & (sub["total_n"] >= 150)
                )
            )
            & (sub["mean_net_pct"] > 0.0)
            & (sub["is_net_sharpe"] > 0.15)
        )
    else:
        c_15m = (sub["tf"] == "15m") & (sub["n"] >= 15) & (sub["mean_net_pct"] > 0.0) & (sub["is_net_sharpe"] > 0.15)

    c_60m = (sub["tf"] == "60m") & (sub["n"] >= 10) & (sub["mean_net_pct"] > 0.0) & (sub["is_net_sharpe"] > 0.15)
    c_1d = (
        (sub["tf"] == "1d")
        & (sub["n"] >= 2)
        & (sub["mean_net_pct"] > 0.0)
        & (sub["is_net_sharpe"] > 0.10)
        & (sub["total_n"] >= 25)
        & (sub["pooled_mean_pct"] > 0.0)
    )
    return sub[c_30m | c_15m | c_60m | c_1d].copy()


def classify_causal_market_regime(df_tf: pd.DataFrame, er_window: int = 20, er_th: float = 0.22) -> np.ndarray:
    """
    严格因果的三态市场状态分类器 (Causal 3-State Market Regime Classifier)：
    基于时刻 t 已收盘的 Kaufman 效率系数 ER(20) 与 MA20/MA60 均线结构，将市场切分为：
    - `+1` (BULL_TREND 多头趋势态): ER_20 >= er_th 且 close > ma60 且 ma20 >= ma60
    - `-1` (BEAR_TREND 空头趋势态): ER_20 >= er_th 且 close < ma60 且 ma20 <= ma60
    - `0`  (CHOPPY_RANGE 震荡盘整态): 其余时间（路径效率低或均线缠绕）
    """
    c = df_tf["close"].values.astype(np.float64)
    c_series = pd.Series(c)
    ma20 = df_tf["ma20"].values if "ma20" in df_tf.columns else c_series.rolling(20, min_periods=1).mean().values
    ma60 = df_tf["ma60"].values if "ma60" in df_tf.columns else c_series.rolling(60, min_periods=1).mean().values

    net_chg = np.abs(c_series.diff(er_window).fillna(0.0).values)
    abs_path = pd.Series(np.abs(np.diff(c, prepend=c[0]))).rolling(er_window, min_periods=5).sum().fillna(np.inf).values
    er = net_chg / np.maximum(abs_path, 1e-6)

    regimes = np.zeros(len(c), dtype=np.int8)
    bull_mask = (er >= er_th) & (c > ma60) & (ma20 >= ma60 * 0.998)
    bear_mask = (er >= er_th) & (c < ma60) & (ma20 <= ma60 * 1.002)
    regimes[bull_mask] = 1
    regimes[bear_mask] = -1
    return regimes


def compute_empirical_bayes_shrinkage_weight(
    local_weight: float,
    cross_asset_prior_weight: float,
    is_n_purged: int,
    shrink_k: int = 60,
) -> tuple[float, float]:
    """
    分层经验贝叶斯部分池化收缩配权 (Hierarchical Empirical Bayes Shrinkage)：
    w_final = (1 - λ) * w_cross_asset_prior + λ * w_local_asset
    其中置信权重 λ = N_purged / (N_purged + κ_0)。
    - 当单品种样本量 N_purged 较小时 (如 60m/1d 低频因子)，自动向跨品种共性先验收缩，防单品种状态过拟合；
    - 当单品种样本量充足时 (如 5m/15m 日内因子)，赋予单品种微观结构个性更高权重。
    """
    lam = float(is_n_purged / max(is_n_purged + shrink_k, 1))
    w_eb = (1.0 - lam) * cross_asset_prior_weight + lam * local_weight
    return round(float(w_eb), 3), round(lam, 3)


class RollingEBWeightsResult(dict[str, np.ndarray]):
    """
    滚动经验贝叶斯配权结果容器：
    - 继承自 `dict[str, np.ndarray]`，可直接按 `res[unit_id]` 获取各单元/簇在时间轴上的因果 EB 权重序列；
    - 同时暴露 `.eb_weights`、`.prior_weights`、`.rolling_perf_weights`、`.rolling_ir` 与 `.prior_anchor_lambda` 属性。
    """

    def __init__(
        self,
        eb_weights: dict[str, np.ndarray],
        prior_weights: dict[str, float],
        rolling_perf_weights: dict[str, np.ndarray],
        rolling_ir: dict[str, np.ndarray],
        prior_anchor_lambda: float = 0.70,
    ) -> None:
        super().__init__(eb_weights)
        self.eb_weights = eb_weights
        self.prior_weights = prior_weights
        self.rolling_perf_weights = rolling_perf_weights
        self.rolling_ir = rolling_ir
        self.prior_anchor_lambda = float(prior_anchor_lambda)

    def __getitem__(self, key: str) -> Any:
        if super().__contains__(key):
            return super().__getitem__(key)
        meta_map = {
            "eb_weights": self.eb_weights,
            "prior_weights": self.prior_weights,
            "rolling_perf_weights": self.rolling_perf_weights,
            "rolling_ir": self.rolling_ir,
            "prior_anchor_lambda": self.prior_anchor_lambda,
        }
        if key in meta_map:
            return meta_map[key]
        raise KeyError(key)

    def to_dataframe(self, index: Any = None) -> pd.DataFrame:
        return pd.DataFrame(self.eb_weights, index=index)


class RollingEmpiricalBayesWeighter:
    """
    零前视滚动/扩展窗口经验贝叶斯部分池化收缩配权器 (Rolling Empirical Bayes Shrinkage Weighter, Issue #7 Phase 4)

    核心机制：
    1. 静态先验层 (Prior Anchor):
       支持结合跨品种池化先验得分/权重 Q_pool,k (如 Gate 1B 的 pooled_r_atr / safety_margin 或先验基准权重)
       与单品种样本内质量得分/权重 Q_local,i,k，按先验收缩比 λ_prior = 0.70
       (70% 锚定全品种池化先验 + 30% 锚定单品种样本量自适应调整) 生成无过拟合的基准权重先验 w_{i,k}^{prior}。
    2. 因果滚动动态更新层 (Causal Rolling / Expanding Update):
       在时间轴 t 上，仅使用严格截至 t-1 (shift(1) 零前视) 的历史已实现净收益序列
       (滚动窗口 lookback_bars 如 60~120，或扩展窗口 lookback_bars=None)，计算各单元/各簇的因果滚动信息比率 / 净夏普得分，
       映射为滚动表现权重 w_{i,k,t-1}^{rolling_perf}，并向基准先验执行 L2 经验贝叶斯收缩：
           w_{i,k,t}^{EB} = λ_prior * w_{i,k}^{prior} + (1 - λ_prior) * w_{i,k,t-1}^{rolling_perf},  λ_prior = 0.70
    """

    def __init__(
        self,
        prior_anchor_lambda: float = 0.70,
        lookback_bars: int | None = 90,
        min_bars: int = 15,
        shrink_k: int = 60,
        annualization_factor: float = 252.0,
        perf_tilt_scale: float = 0.55,
        min_weight: float = 0.05,
        max_weight: float = 1.00,
        normalize_budget: bool = False,
        *,
        lambda_prior: float | None = None,
    ) -> None:
        eff_lam = float(lambda_prior if lambda_prior is not None else prior_anchor_lambda)
        if not (0.0 <= eff_lam <= 1.0):
            raise ValueError(f"prior_anchor_lambda must be in [0, 1], got {eff_lam}")
        self.prior_anchor_lambda = eff_lam
        self.lookback_bars = lookback_bars
        self.min_bars = max(int(min_bars), 2)
        self.shrink_k = max(int(shrink_k), 1)
        self.annualization_factor = float(annualization_factor)
        self.perf_tilt_scale = float(perf_tilt_scale)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.normalize_budget = bool(normalize_budget)

    def compute_prior_anchor(
        self,
        cross_asset_prior: dict[str, float],
        local_quality: dict[str, float] | None = None,
        is_n_purged: dict[str, int] | int | None = None,
        unit_forensic_meta: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, float]:
        """
        计算静态先验锚定权重 w_{i,k}^{prior}：
        - 70% (λ_prior) 锚定跨品种池化先验权重 w_pool,k；
        - 30% (1 - λ_prior) 锚定单品种样本量自适应调整后的局部权重 w_local,i,k。
        """
        lam_p = self.prior_anchor_lambda
        priors: dict[str, float] = {}

        # 若提供 unit_forensic_meta 但未直接提供 local_quality，从法医元数据提取局部质量修正
        derived_local: dict[str, float] = {}
        if local_quality is None and unit_forensic_meta:
            raw_q: dict[str, float] = {}
            for uid in cross_asset_prior:
                meta = unit_forensic_meta.get(uid, {})
                n_p = max(int(meta.get("is_n_purged", 20)), 5)
                saf = float(np.clip(float(meta.get("is_safety_margin_pct", 4.0)), 0.5, 15.0))
                r_atr = float(np.clip(float(meta.get("pooled_r_atr", meta.get("norm_r_atr", 0.5))), -1.0, 3.0))
                raw_q[uid] = float(np.log(n_p) * saf * (1.0 + 0.15 * r_atr))
            if len(raw_q) > 1:
                q_vals = np.asarray(list(raw_q.values()), dtype=np.float64)
                q_mean, q_std = float(q_vals.mean()), float(q_vals.std() + 1e-6)
            else:
                q_mean, q_std = 0.0, 1.0
            for uid, w_pool in cross_asset_prior.items():
                z_u = float(np.clip((raw_q[uid] - q_mean) / q_std, -1.2, 1.6)) if len(raw_q) > 1 else 0.0
                derived_local[uid] = float(np.clip(w_pool * (1.0 + 0.25 * z_u), self.min_weight, self.max_weight))

        eff_local_map = local_quality if local_quality is not None else (derived_local if derived_local else None)

        for uid, w_pool_raw in cross_asset_prior.items():
            w_pool = float(w_pool_raw)
            if eff_local_map is not None and uid in eff_local_map:
                w_loc_raw = float(eff_local_map[uid])
                n_obs: int | None = None
                if isinstance(is_n_purged, dict) and uid in is_n_purged:
                    n_obs = int(is_n_purged[uid])
                elif isinstance(is_n_purged, int):
                    n_obs = int(is_n_purged)
                elif unit_forensic_meta and uid in unit_forensic_meta and "is_n_purged" in unit_forensic_meta[uid]:
                    n_obs = int(unit_forensic_meta[uid]["is_n_purged"])

                if n_obs is not None:
                    eta = float(max(n_obs, 0) / max(max(n_obs, 0) + self.shrink_k, 1))
                    w_loc_adj = (1.0 - eta) * w_pool + eta * w_loc_raw
                else:
                    w_loc_adj = w_loc_raw
                w_prior = lam_p * w_pool + (1.0 - lam_p) * w_loc_adj
            else:
                w_prior = w_pool
            priors[uid] = float(np.clip(w_prior, self.min_weight, self.max_weight))

        return priors

    @staticmethod
    def build_causal_unit_net_returns(
        df_grid: pd.DataFrame,
        unit_weights_on_grid: dict[str, np.ndarray],
        fee_rate: float = 0.0002,
    ) -> dict[str, np.ndarray]:
        """Build observed next-open PnL, charging turnover at its actual fill.

        A close decision becomes a fixed-initial-capital target at the next
        open. Rust marks the existing quantity and charges actual traded
        notional, including resizing needed to restore an unchanged weight.
        No terminal liquidation is injected: a rolling history cannot know its end.
        """
        from czsc import execution_net_returns

        if "open" not in df_grid:
            raise ValueError("Rolling allocation requires actual open execution prices")
        opens = df_grid["open"].to_numpy(dtype=np.float64)
        unit_returns = {}
        for uid, values in unit_weights_on_grid.items():
            decisions = np.asarray(values, dtype=np.float64)
            unit_returns[uid] = np.asarray(execution_net_returns(opens, decisions, fee_rate))
        return unit_returns

    def compute_weights(
        self,
        cross_asset_prior: dict[str, float],
        realized_net_returns: dict[str, np.ndarray] | pd.DataFrame | None = None,
        *,
        local_quality: dict[str, float] | None = None,
        is_n_purged: dict[str, int] | int | None = None,
        unit_forensic_meta: dict[str, dict[str, Any]] | None = None,
        df_grid: pd.DataFrame | None = None,
        unit_weights_on_grid: dict[str, np.ndarray] | None = None,
        fee_rate: float = 0.0002,
        rolling_perf_override: dict[str, np.ndarray] | None = None,
    ) -> RollingEBWeightsResult:
        """
        执行零前视因果滚动/扩展窗口经验贝叶斯收缩配权：
            w_{i,k,t}^{EB} = λ_prior * w_{i,k}^{prior} + (1 - λ_prior) * w_{i,k,t-1}^{rolling_perf}
        """
        prior_map = self.compute_prior_anchor(
            cross_asset_prior=cross_asset_prior,
            local_quality=local_quality,
            is_n_purged=is_n_purged,
            unit_forensic_meta=unit_forensic_meta,
        )
        uids = list(prior_map.keys())

        if realized_net_returns is None:
            if df_grid is not None and unit_weights_on_grid is not None:
                realized_net_returns = self.build_causal_unit_net_returns(
                    df_grid, unit_weights_on_grid, fee_rate=fee_rate
                )
            elif rolling_perf_override is not None:
                first_arr = next(iter(rolling_perf_override.values()))
                realized_net_returns = {u: np.zeros(len(first_arr), dtype=np.float64) for u in uids}
            else:
                raise ValueError(
                    "Must provide either realized_net_returns or (df_grid, unit_weights_on_grid) to compute_weights."
                )

        if isinstance(realized_net_returns, pd.DataFrame):
            ret_dict = {
                col: realized_net_returns[col].to_numpy(dtype=np.float64) for col in uids if col in realized_net_returns
            }
        else:
            ret_dict = {
                u: np.asarray(realized_net_returns[u], dtype=np.float64) for u in uids if u in realized_net_returns
            }

        if not ret_dict:
            raise ValueError("No matching unit keys found between cross_asset_prior and realized_net_returns.")

        n = len(next(iter(ret_dict.values())))
        lam_p = self.prior_anchor_lambda
        sqrt_ann = np.sqrt(self.annualization_factor)
        w_lb = self.lookback_bars if (self.lookback_bars is not None and self.lookback_bars > 0) else None

        rolling_ir_map: dict[str, np.ndarray] = {u: np.zeros(n, dtype=np.float64) for u in uids}
        rolling_perf_map: dict[str, np.ndarray] = {u: np.full(n, prior_map[u], dtype=np.float64) for u in uids}
        eb_weights_map: dict[str, np.ndarray] = {u: np.full(n, prior_map[u], dtype=np.float64) for u in uids}

        if rolling_perf_override is not None:
            # 若显式传入滚动表现目标序列 (调试或直接收缩验证)，严格执行 shift(1) 后按 L2 公式线性收缩
            for u in uids:
                if u in rolling_perf_override:
                    raw_perf = np.asarray(rolling_perf_override[u], dtype=np.float64)
                    shifted_perf = np.empty(n, dtype=np.float64)
                    shifted_perf[0] = prior_map[u]
                    if n > 1:
                        shifted_perf[1:] = np.clip(raw_perf[:-1], self.min_weight, self.max_weight)
                    rolling_perf_map[u] = shifted_perf
                    eb_weights_map[u] = lam_p * prior_map[u] + (1.0 - lam_p) * shifted_perf
            return RollingEBWeightsResult(
                eb_weights=eb_weights_map,
                prior_weights=prior_map,
                rolling_perf_weights=rolling_perf_map,
                rolling_ir=rolling_ir_map,
                prior_anchor_lambda=lam_p,
            )

        prior_sum = sum(prior_map.values())
        for t in range(self.min_bars, n):
            # 严格只取 [start_idx, t) 即截至 t-1 的历史已实现净收益 (shift(1) 零前视)
            start_idx = max(0, t - w_lb) if w_lb is not None else 0
            raw_perf_t: dict[str, float] = {}
            for u in uids:
                r_win = ret_dict[u][start_idx:t]
                if len(r_win) < self.min_bars:
                    ir_val = 0.0
                else:
                    std_v = float(np.std(r_win, ddof=1))
                    ir_val = float((np.mean(r_win) / std_v) * sqrt_ann) if std_v > 1e-10 else 0.0
                rolling_ir_map[u][t] = ir_val
                tilt = float(np.tanh(ir_val / 2.0))
                w_perf = float(
                    np.clip(prior_map[u] * (1.0 + self.perf_tilt_scale * tilt), self.min_weight, self.max_weight)
                )
                raw_perf_t[u] = w_perf

            if self.normalize_budget and prior_sum > 1e-9:
                cur_sum = sum(raw_perf_t.values())
                if cur_sum > 1e-9:
                    scale_f = prior_sum / cur_sum
                    for u in uids:
                        raw_perf_t[u] = float(np.clip(raw_perf_t[u] * scale_f, self.min_weight, self.max_weight))

            for u in uids:
                w_perf_u = raw_perf_t[u]
                rolling_perf_map[u][t] = w_perf_u
                eb_weights_map[u][t] = lam_p * prior_map[u] + (1.0 - lam_p) * w_perf_u

        return RollingEBWeightsResult(
            eb_weights=eb_weights_map,
            prior_weights=prior_map,
            rolling_perf_weights=rolling_perf_map,
            rolling_ir=rolling_ir_map,
            prior_anchor_lambda=lam_p,
        )


def compute_rolling_empirical_bayes_weights(
    cross_asset_prior: dict[str, float],
    realized_net_returns: dict[str, np.ndarray] | pd.DataFrame | None = None,
    *,
    local_quality: dict[str, float] | None = None,
    is_n_purged: dict[str, int] | int | None = None,
    unit_forensic_meta: dict[str, dict[str, Any]] | None = None,
    df_grid: pd.DataFrame | None = None,
    unit_weights_on_grid: dict[str, np.ndarray] | None = None,
    prior_anchor_lambda: float = 0.70,
    lookback_bars: int | None = 90,
    min_bars: int = 15,
    shrink_k: int = 60,
    annualization_factor: float = 252.0,
    perf_tilt_scale: float = 0.55,
    min_weight: float = 0.05,
    max_weight: float = 1.00,
    normalize_budget: bool = False,
    fee_rate: float = 0.0002,
    rolling_perf_override: dict[str, np.ndarray] | None = None,
) -> RollingEBWeightsResult:
    """
    模块级便捷函数：计算零前视滚动/扩展窗口经验贝叶斯收缩权重序列 w_{i,k,t}^{EB}。
    """
    weighter = RollingEmpiricalBayesWeighter(
        prior_anchor_lambda=prior_anchor_lambda,
        lookback_bars=lookback_bars,
        min_bars=min_bars,
        shrink_k=shrink_k,
        annualization_factor=annualization_factor,
        perf_tilt_scale=perf_tilt_scale,
        min_weight=min_weight,
        max_weight=max_weight,
        normalize_budget=normalize_budget,
    )
    return weighter.compute_weights(
        cross_asset_prior=cross_asset_prior,
        realized_net_returns=realized_net_returns,
        local_quality=local_quality,
        is_n_purged=is_n_purged,
        unit_forensic_meta=unit_forensic_meta,
        df_grid=df_grid,
        unit_weights_on_grid=unit_weights_on_grid,
        fee_rate=fee_rate,
        rolling_perf_override=rolling_perf_override,
    )
