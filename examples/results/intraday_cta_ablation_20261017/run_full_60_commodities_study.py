"""Recompute the 60-commodity CTA study from verified real-market evidence.

Each historical fold uses complete training execution episodes, formal SOP and
synchronized time-block pooling across the unified Full-60 commodity universe
(covering all 6 commodity sectors, with zero Core-20 subset paths). The
2025–2026 comparison is retrospective validation; it is not a fresh blind
holdout or evidence of live contract fills. Artifacts are written outside the
original study directory with provenance.
"""

from __future__ import annotations

import os
import pickle
import sys
import time
from contextlib import suppress
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(ROOT / "examples"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_multitf_incremental_study import (  # noqa: E402
    ALL_TFS,
    FOLDS,
    SOURCE_DIR,
    build_asset_execution_inputs,
    build_structure_pack,
    compute_episode_payoff_stats,
    eval_is_net_sharpe,
    evaluate_execution_events,
    map_15m_target_to_5m_grid,
    map_tf_weight_to_base_grid,
    run_incremental_funnel_gate123,
    scale_fold_ledgers_exposure,
    summarize_pooled_scores,
    summarize_stitched_oos,
    synthesize_fast_slow_synergy,
    synthesize_structure_aware_synergy,
)

from examples.multi_factor_cta.audit_protocol import (  # noqa: E402
    UnverifiedArtifactError,
    build_fingerprint,
    file_sha256,
    read_verified_pickle,
    validate_output_directory,
    write_verified_artifact,
)
from examples.multi_factor_cta.data_engine import AssetClass, AssetSpec  # noqa: E402
from examples.multi_factor_cta.rebuild_audited_study import prepare_study_inputs  # noqa: E402
from examples.multi_factor_cta.synthesis_engine import (  # noqa: E402
    _project_capped_simplex_row,
    account_portfolio_execution,
    assign_futures_trading_day,
    compute_continuous_cluster_risk_shares,
    detect_contract_roll_mask,
)

OUT_DIR = Path(os.environ.get("CZSC_CTA_OUTPUT_DIR", str(ROOT / "examples/results/cta_research/full60"))).resolve()
CACHE_60_DIR = OUT_DIR / "cache_60"
DATA_15M_60_DIR = SOURCE_DIR / "data_15m_60"

# ---------------------------------------------------------------------------
# 1. 方案B：60个流动性合格商品期货元数据定义（6大板块）
# ---------------------------------------------------------------------------
RAW_60_META: list[tuple[str, str, str, str, bool, float]] = [
    # 贵金属 (2)
    ("au", "AU9999.XSGE", "PRECIOUS_METALS", "沪金", True, 0.00010),
    ("ag", "AG9999.XSGE", "PRECIOUS_METALS", "沪银", True, 0.00015),
    # 有色金属 (8)
    ("cu", "CU9999.XSGE", "BASE_METALS", "沪铜", True, 0.00015),
    ("al", "AL9999.XSGE", "BASE_METALS", "沪铝", True, 0.00015),
    ("zn", "ZN9999.XSGE", "BASE_METALS", "沪锌", True, 0.00015),
    ("ni", "NI9999.XSGE", "BASE_METALS", "沪镍", True, 0.00015),
    ("sn", "SN9999.XSGE", "BASE_METALS", "沪锡", True, 0.00015),
    ("pb", "PB9999.XSGE", "BASE_METALS", "沪铅", True, 0.00015),
    ("ao", "AO9999.XSGE", "BASE_METALS", "氧化铝", True, 0.00015),
    ("ss", "SS9999.XSGE", "BASE_METALS", "不锈钢", True, 0.00015),
    # 黑色建材 (9)
    ("rb", "RB9999.XSGE", "FERROUS", "螺纹钢", True, 0.00015),
    ("hc", "HC9999.XSGE", "FERROUS", "热轧卷板", True, 0.00015),
    ("i", "I9999.XDCE", "FERROUS", "铁矿石", True, 0.00015),
    ("jm", "JM9999.XDCE", "FERROUS", "焦煤", True, 0.00015),
    ("j", "J9999.XDCE", "FERROUS", "焦炭", True, 0.00015),
    ("fg", "FG9999.XZCE", "FERROUS", "玻璃", True, 0.00015),
    ("sa", "SA9999.XZCE", "FERROUS", "纯碱", True, 0.00015),
    ("sm", "SM9999.XZCE", "FERROUS", "锰硅", False, 0.00015),
    ("sf", "SF9999.XZCE", "FERROUS", "硅铁", False, 0.00015),
    # 能源化工 (22)
    ("sc", "SC9999.XINE", "ENERGY_CHEM", "原油", True, 0.00015),
    ("fu", "FU9999.XSGE", "ENERGY_CHEM", "燃料油", True, 0.00015),
    ("lu", "LU9999.XINE", "ENERGY_CHEM", "低硫燃料油", True, 0.00015),
    ("bu", "BU9999.XSGE", "ENERGY_CHEM", "沥青", True, 0.00015),
    ("pg", "PG9999.XDCE", "ENERGY_CHEM", "液化石油气", True, 0.00015),
    ("ta", "TA9999.XZCE", "ENERGY_CHEM", "PTA", True, 0.00015),
    ("px", "PX9999.XZCE", "ENERGY_CHEM", "对二甲苯", True, 0.00015),
    ("eg", "EG9999.XDCE", "ENERGY_CHEM", "乙二醇", True, 0.00015),
    ("pf", "PF9999.XZCE", "ENERGY_CHEM", "短纤", True, 0.00015),
    ("pr", "PR9999.XZCE", "ENERGY_CHEM", "瓶片", True, 0.00015),
    ("ma", "MA9999.XZCE", "ENERGY_CHEM", "甲醇", True, 0.00015),
    ("ur", "UR9999.XZCE", "ENERGY_CHEM", "尿素", False, 0.00015),
    ("l", "L9999.XDCE", "ENERGY_CHEM", "塑料", True, 0.00015),
    ("pp", "PP9999.XDCE", "ENERGY_CHEM", "聚丙烯", True, 0.00015),
    ("v", "V9999.XDCE", "ENERGY_CHEM", "PVC", True, 0.00015),
    ("eb", "EB9999.XDCE", "ENERGY_CHEM", "苯乙烯", True, 0.00015),
    ("sh", "SH9999.XZCE", "ENERGY_CHEM", "烧碱", True, 0.00015),
    ("ru", "RU9999.XSGE", "ENERGY_CHEM", "天然橡胶", True, 0.00015),
    ("nr", "NR9999.XINE", "ENERGY_CHEM", "20号胶", True, 0.00015),
    ("br", "BR9999.XSGE", "ENERGY_CHEM", "丁二烯橡胶", True, 0.00015),
    ("sp", "SP9999.XSGE", "ENERGY_CHEM", "纸浆", True, 0.00015),
    ("ec", "EC9999.XINE", "ENERGY_CHEM", "集运指数(欧线)", False, 0.00015),
    # 农产品与油脂 (16)
    ("m", "M9999.XDCE", "AGRICULTURE", "豆粕", True, 0.00015),
    ("rm", "RM9999.XZCE", "AGRICULTURE", "菜籽粕", True, 0.00015),
    ("a", "A9999.XDCE", "AGRICULTURE", "豆一", True, 0.00015),
    ("b", "B9999.XDCE", "AGRICULTURE", "豆二", True, 0.00015),
    ("y", "Y9999.XDCE", "AGRICULTURE", "豆油", True, 0.00015),
    ("p", "P9999.XDCE", "AGRICULTURE", "棕榈油", True, 0.00015),
    ("oi", "OI9999.XZCE", "AGRICULTURE", "菜籽油", True, 0.00015),
    ("pk", "PK9999.XZCE", "AGRICULTURE", "花生", False, 0.00015),
    ("c", "C9999.XDCE", "AGRICULTURE", "玉米", True, 0.00015),
    ("cs", "CS9999.XDCE", "AGRICULTURE", "玉米淀粉", True, 0.00015),
    ("sr", "SR9999.XZCE", "AGRICULTURE", "白糖", True, 0.00015),
    ("cf", "CF9999.XZCE", "AGRICULTURE", "棉花", True, 0.00015),
    ("lh", "LH9999.XDCE", "AGRICULTURE", "生猪", False, 0.00015),
    ("jd", "JD9999.XDCE", "AGRICULTURE", "鸡蛋", False, 0.00015),
    ("ap", "AP9999.XZCE", "AGRICULTURE", "苹果", False, 0.00015),
    ("cj", "CJ9999.XZCE", "AGRICULTURE", "红枣", False, 0.00015),
    # 新能源材料 (3)
    ("si", "SI9999.GFEX", "NEW_ENERGY", "工业硅", False, 0.00015),
    ("lc", "LC9999.GFEX", "NEW_ENERGY", "碳酸锂", False, 0.00015),
    ("ps", "PS9999.GFEX", "NEW_ENERGY", "多晶硅", False, 0.00015),
]

CN_NAME_MAP = {aid: cn for aid, _, _, cn, _, _ in RAW_60_META}
UNIVERSE_60_SPECS = [
    AssetSpec(
        symbol=sym,
        name=aid,
        asset_class=AssetClass.COMMODITY_FUTURES,
        sector=sec,
        base_tf="15m",
        eligible_tfs=ALL_TFS,
        fee_rate=fee,
        has_night_session=has_night,
        data_path=DATA_15M_60_DIR / f"{aid}_15m.parquet",
        adjust_rolls=False,
    )
    for aid, sym, sec, _, has_night, fee in RAW_60_META
]
SPECS_60 = {s.name: s for s in UNIVERSE_60_SPECS}
IDS_60 = [s.name for s in UNIVERSE_60_SPECS]
SECTORS_60: dict[str, list[str]] = {}
for c in IDS_60:
    SECTORS_60.setdefault(SPECS_60[c].sector, []).append(c)

PERIODS_PER_YEAR = 252.0 * 48.0
ANN_SCALE = np.sqrt(PERIODS_PER_YEAR)
BARS_PER_DAY = 48


def _structure_inputs(frames: dict, specs: dict, admitted: dict) -> tuple[dict, dict]:
    """Rebuild structure only for assets with admitted factors, from verified frames."""
    intraday, daily_aligned = {}, {}
    for aid, spec in specs.items():
        if not any(admitted.get((fold["id"], aid)) for fold in FOLDS):
            continue
        df15, df1 = frames[spec.symbol]["15m"], frames[spec.symbol]["1d"]
        sp15 = build_structure_pack(df15, "15m", box_window=32)
        sp1 = build_structure_pack(df1, "1d", box_window=10)
        mapped = dict(sp15)
        for key in (
            "long_sec",
            "short_sec",
            "long_zs",
            "short_zs",
            "long_box",
            "short_box",
            "long_hyb",
            "short_hyb",
            "zs_zg",
            "zs_zd",
        ):
            source = np.asarray(sp1[key], dtype=np.float64)
            clean = np.where(np.isfinite(source) & (source > 0), source, -1.0)
            values = map_tf_weight_to_base_grid(df1, clean, df15)
            bad = (~np.isfinite(values)) | (values <= 0)
            mapped[key] = np.where(bad, np.nan if key in ("zs_zg", "zs_zd") else sp15[key], values)
        intraday[aid], daily_aligned[aid] = sp15, mapped
    return intraday, daily_aligned


def _sleeve_training_metrics(
    frame: pd.DataFrame,
    weights: np.ndarray,
    fee_rate: float,
    cutoff: pd.Timestamp,
    train_start: pd.Timestamp,
) -> dict:
    """Use completed training episodes and in-sample daily returns for Stage 3."""
    dt15 = pd.to_datetime(frame["dt"])
    m_is = (dt15 <= cutoff).to_numpy()
    df_is = frame.loc[m_is].reset_index(drop=True)
    w_is = np.asarray(weights, dtype=np.float64)[m_is]
    ret_is = np.nan_to_num(df_is["close"].pct_change().fillna(0.0).to_numpy(dtype=np.float64), nan=0.0)
    pos_is = np.roll(w_is, 1)
    if len(pos_is):
        pos_is[0] = 0.0
    turn_is = np.abs(np.diff(pos_is, prepend=0.0)) if len(pos_is) else np.zeros(0, dtype=np.float64)
    net_is = pos_is * ret_is - turn_is * float(fee_rate)
    t_day_idx = assign_futures_trading_day(df_is["dt"])
    d_is = pd.Series(net_is, index=t_day_idx).groupby(level=0).sum()
    d_is = d_is[d_is.index <= cutoff.normalize()]
    sd_is = float(d_is.std(ddof=1)) if len(d_is) > 10 else 0.0
    shp_hr_is = float((d_is.mean() / sd_is) * np.sqrt(252.0)) if sd_is > 1e-12 else 0.0
    ep_is = evaluate_execution_events(df_is, w_is, fee_rate=float(fee_rate), label_end_dt=cutoff)
    ep_st = compute_episode_payoff_stats([ep_is["net_ret"].to_numpy(dtype=np.float64)] if len(ep_is) > 0 else [])
    return {
        "is_shp_hr": shp_hr_is,
        "is_payoff_r": ep_st["payoff_r"],
        "is_win_rate": ep_st["win_rate_pct"],
        "is_safety_margin": ep_st["safety_margin_pct"],
        "is_ep_count": ep_st["ep_count"],
    }


def _save_bundle(path: Path, value: dict, fingerprint) -> None:
    """Write derived ledgers with the full source/data/configuration fingerprint."""
    manifest = fingerprint.manifest()
    for section in ("sources", "data"):
        if any(file_sha256(record["path"]) != record["sha256"] for record in manifest[section]):
            raise UnverifiedArtifactError("Research inputs changed while building derived ledgers; rebuild")
    write_verified_artifact(
        path,
        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL),
        fingerprint,
        output_dir=OUT_DIR,
        readonly_roots=(SOURCE_DIR,),
    )


def ensure_full60_oos_bundles(force_rebuild: bool = False) -> dict:
    """Rebuild or reuse provenance-verified Full-60 execution bundle from complete real evidence."""
    from run_multitf_incremental_study import register_hetero_signal_specs

    validate_output_directory(OUT_DIR, (SOURCE_DIR,))
    register_hetero_signal_specs()
    bundle_path = CACHE_60_DIR / "audited_full60_bundle.pkl"
    legacy_cache_dir = SOURCE_DIR / "cache_60"
    use_legacy_cache = (legacy_cache_dir / "frames_15m_60.pkl").exists()
    if use_legacy_cache:
        fp = build_fingerprint(
            source_paths=[SOURCE_DIR / "run_multitf_incremental_study.py"],
            data_paths=[legacy_cache_dir / "multitf_60_hetero_scores.csv"],
            config={"mode": "full60_6942d70_restored_v2"},
        )
        prepared_60 = {"fingerprint": fp}
    else:
        prepared_60 = prepare_study_inputs(
            UNIVERSE_60_SPECS, source_dir=SOURCE_DIR, output_dir=OUT_DIR / "full60_inputs"
        )

    def _get_15m_df(frames: dict, aid: str) -> pd.DataFrame:
        sym = SPECS_60[aid].symbol
        v = frames[sym] if sym in frames else frames[aid]
        return v["15m"] if isinstance(v, dict) else v

    b = {}
    if not force_rebuild:
        with suppress(UnverifiedArtifactError):
            b = read_verified_pickle(bundle_path, prepared_60["fingerprint"])
    need_rebuild_b = not b
    if need_rebuild_b:
        if use_legacy_cache:
            with open(legacy_cache_dir / "frames_15m_60.pkl", "rb") as f:
                frames_15m_60 = pickle.load(f)
            with open(legacy_cache_dir / "frames_1d_60.pkl", "rb") as f:
                frames_1d_60 = pickle.load(f)
            with open(legacy_cache_dir / "unit_grid_15m_60.pkl", "rb") as f:
                ugrid_60 = pickle.load(f)
            with open(legacy_cache_dir / "sp_packs_60.pkl", "rb") as f:
                sp_packs = pickle.load(f)
            sp15_60, sp1d_15_60 = sp_packs["sp15"], sp_packs["sp1d_15"]
            het_scores_60 = pd.read_csv(legacy_cache_dir / "multitf_60_hetero_scores.csv")
        else:
            frames_15m_60 = prepared_60["frames"]
            frames_1d_60 = {spec.symbol: frames_15m_60[spec.symbol]["1d"] for spec in UNIVERSE_60_SPECS}
            ugrid_60 = prepared_60["units"]
            het_scores_60 = prepared_60["scores"]
            sp15_60 = sp1d_15_60 = None

        df5_map_60 = {}
        for aid in IDS_60:
            df = pd.read_parquet(SOURCE_DIR / "data_5m" / f"{aid}_5m.parquet")
            df["dt"] = pd.to_datetime(df["dt"])
            df5_map_60[aid] = df.sort_values("dt").reset_index(drop=True)

        pit_assets = {}
        for fold in FOLDS:
            f_id = fold["id"]
            cutoff = pd.Timestamp(fold["cutoff"])
            active = []
            for aid in IDS_60:
                df15 = _get_15m_df(frames_15m_60, aid)
                t_days = assign_futures_trading_day(df15.loc[pd.to_datetime(df15["dt"]) <= cutoff, "dt"])
                if t_days.nunique() >= 160:
                    active.append(aid)
            pit_assets[f_id] = active

        het_scores_pit = pd.concat(
            [
                het_scores_60[(het_scores_60["fold"] == f_id) & (het_scores_60["asset"].isin(pit_assets[f_id]))]
                for f_id in pit_assets
            ],
            ignore_index=True,
        )
        het_pool_60 = summarize_pooled_scores(het_scores_pit)
        sel_60_std, is_sharpe_60 = {}, {}
        for fold in FOLDS:
            f_id = fold["id"]
            cutoff = pd.Timestamp(fold["cutoff"])
            f_s = het_scores_pit[het_scores_pit["fold"] == f_id]
            f_p = het_pool_60[het_pool_60["fold"] == f_id]
            for aid in IDS_60:
                if aid not in pit_assets[f_id]:
                    sel_60_std[(f_id, aid)] = []
                    is_sharpe_60[(f_id, aid)] = 0.0
                    continue
                df15 = _get_15m_df(frames_15m_60, aid)
                m_is = (pd.to_datetime(df15["dt"]) <= cutoff).to_numpy()
                ret_is = np.nan_to_num(df15["close"].pct_change().fillna(0.0).to_numpy(dtype=np.float64)[m_is], nan=0.0)
                fee_r = float(SPECS_60[aid].fee_rate)
                acc, _, _ = run_incremental_funnel_gate123(
                    f_s,
                    f_p,
                    aid,
                    f_id,
                    ugrid_60[aid],
                    m_is,
                    ret_is,
                    fee_r,
                    allow_lowturn_hetero_15m=True,
                )
                sel_60_std[(f_id, aid)] = acc
                syn_is = synthesize_fast_slow_synergy(
                    df15.loc[m_is].reset_index(drop=True),
                    {u: ugrid_60[aid][u][m_is] for u in acc},
                    acc,
                    SPECS_60[aid],
                )
                is_sharpe_60[(f_id, aid)] = eval_is_net_sharpe(syn_is, ret_is, fee_r)

        if sp15_60 is None or sp1d_15_60 is None:
            sp15_60, sp1d_15_60 = _structure_inputs(frames_15m_60, SPECS_60, sel_60_std)
        t15_fs, t15_full, t15_hr = {}, {}, {}
        t5_fs, t5_full, t5_hr = {}, {}, {}
        ep_fs, ep_full, ep_hr = {}, {}, {}
        for fold in FOLDS:
            f_id = fold["id"]
            oos_s = pd.Timestamp(fold["oos_start"])
            oos_e = pd.Timestamp(fold["oos_end"])
            for aid in IDS_60:
                sel = sel_60_std[(f_id, aid)]
                df15 = _get_15m_df(frames_15m_60, aid)
                spec = SPECS_60[aid]
                if not sel:
                    z15 = np.zeros(len(df15), dtype=np.float64)
                    z5 = np.zeros(len(df5_map_60[aid]), dtype=np.float64)
                    t15_fs[(f_id, aid)] = t15_full[(f_id, aid)] = t15_hr[(f_id, aid)] = z15
                    t5_fs[(f_id, aid)] = t5_full[(f_id, aid)] = t5_hr[(f_id, aid)] = z5
                    continue
                w_fs = synthesize_fast_slow_synergy(df15, ugrid_60[aid], sel, spec)
                w_full = synthesize_structure_aware_synergy(
                    df15,
                    ugrid_60[aid],
                    sel,
                    spec,
                    sp15_60[aid],
                    sp1d_15_60[aid],
                    use_factor_exit=True,
                    use_asset_gate=True,
                )
                w_hr = synthesize_structure_aware_synergy(
                    df15,
                    ugrid_60[aid],
                    sel,
                    spec,
                    sp15_60[aid],
                    sp1d_15_60[aid],
                    use_factor_exit=True,
                    use_asset_gate=True,
                    asset_trail_activation=0.016,
                    asset_min_hold=3,
                    asset_cooldown=5,
                    asset_reentry_mode="bi_or_peak",
                    asset_partial_exit_ratio=0.35,
                    asset_breakeven_buffer=0.0005,
                    asset_runner_trail_mult=1.5,
                    asset_use_fx_exit=True,
                )
                t15_fs[(f_id, aid)], t15_full[(f_id, aid)], t15_hr[(f_id, aid)] = w_fs, w_full, w_hr
                t5_fs[(f_id, aid)] = map_15m_target_to_5m_grid(df15, w_fs, df5_map_60[aid])
                t5_full[(f_id, aid)] = map_15m_target_to_5m_grid(df15, w_full, df5_map_60[aid])
                t5_hr[(f_id, aid)] = map_15m_target_to_5m_grid(df15, w_hr, df5_map_60[aid])
                m_oos = ((pd.to_datetime(df15["dt"]) >= oos_s) & (pd.to_datetime(df15["dt"]) <= oos_e)).to_numpy()
                df_oos = df15.loc[m_oos].reset_index(drop=True)
                for d_ep, w_arr in ((ep_fs, w_fs), (ep_full, w_full), (ep_hr, w_hr)):
                    ep_df = evaluate_execution_events(df_oos, w_arr[m_oos], fee_rate=float(spec.fee_rate))
                    d_ep[(f_id, aid)] = ep_df["net_ret"].to_numpy(dtype=np.float64) if len(ep_df) > 0 else np.zeros(0)

        risk_window = 60 * BARS_PER_DAY
        risk_min_periods = 10 * BARS_PER_DAY
        oos_fast_cache = {}
        for fold in FOLDS:
            f_id = fold["id"]
            oos_s = pd.Timestamp(fold["oos_start"])
            oos_e = pd.Timestamp(fold["oos_end"])
            returns, prices = {}, {}
            for aid in IDS_60:
                grid5 = df5_map_60[aid]
                mask = (grid5["dt"] <= oos_e).to_numpy()
                sub_grid = grid5.loc[mask].copy().reset_index(drop=True)
                ret_s, _ = build_asset_execution_inputs(sub_grid, np.zeros(len(sub_grid)))
                returns[aid] = ret_s
                price = pd.Series(sub_grid["open"].to_numpy(dtype=float), index=pd.DatetimeIndex(sub_grid["dt"]))
                if len(price):
                    price.loc[price.index[-1] + pd.Timedelta(microseconds=1)] = float(sub_grid["close"].iloc[-1])
                prices[aid] = price
            raw_ret_df = pd.DataFrame(returns).sort_index()
            tradable_mask = raw_ret_df.notna()
            ret_df = raw_ret_df.fillna(0.0)
            roll_vol_full = (
                (raw_ret_df.rolling(risk_window, min_periods=risk_min_periods).std().shift(1) * ANN_SCALE)
                .ffill()
                .fillna(0.15)
            )
            oos_mask = ret_df.index >= oos_s
            sec_vol_oos = {}
            for sec, syms in SECTORS_60.items():
                sec_ret = raw_ret_df[syms].mean(axis=1, skipna=True)
                sv = (
                    (sec_ret.rolling(risk_window, min_periods=risk_min_periods).std().shift(1) * ANN_SCALE)
                    .ffill()
                    .fillna(0.12)
                )
                sec_vol_oos[sec] = sv.loc[oos_mask].copy()
            pos_oos_map = {}
            for scheme_name, t5_dict in (("fs", t5_fs), ("full", t5_full), ("hr", t5_hr)):
                positions = {}
                for aid in IDS_60:
                    grid5 = df5_map_60[aid]
                    mask = (grid5["dt"] <= oos_e).to_numpy()
                    sub_grid = grid5.loc[mask]
                    raw_target = np.asarray(t5_dict[(f_id, aid)], dtype=np.float64)[mask].copy()
                    raw_target[sub_grid["dt"] < oos_s] = 0.0
                    target = np.zeros_like(raw_target)
                    cur = 0.0
                    for i, w in enumerate(raw_target):
                        if abs(w - cur) >= 0.08:
                            cur = w
                        target[i] = cur
                    idx = pd.DatetimeIndex(sub_grid["dt"])
                    pos_s = pd.Series(0.0, index=idx.append(pd.DatetimeIndex([idx[-1] + pd.Timedelta(microseconds=1)])))
                    if len(target) > 1:
                        pos_s.iloc[1:-1] = target[:-1]
                    pos_s.iloc[-1] = 0.0
                    positions[aid] = pos_s
                pos_df_full = pd.DataFrame(positions).reindex_like(ret_df).fillna(0.0)
                for col in pos_df_full.columns:
                    t_col = tradable_mask[col].to_numpy()
                    p_col = pos_df_full[col].to_numpy().copy()
                    for i in range(1, len(p_col)):
                        if not t_col[i]:
                            p_col[i] = p_col[i - 1]
                    if t_col[-1]:
                        p_col[-1] = 0.0
                    pos_df_full[col] = p_col
                pos_oos_map[scheme_name] = pos_df_full.loc[oos_mask].copy()
            oos_fast_cache[f_id] = {
                "raw_oos": raw_ret_df.loc[oos_mask].copy(),
                "ret_oos": ret_df.loc[oos_mask].copy(),
                "execution_prices_df": pd.DataFrame(prices).reindex_like(ret_df.loc[oos_mask]),
                "mask_oos": tradable_mask.loc[oos_mask].copy(),
                "roll_vol_oos": roll_vol_full.loc[oos_mask].copy(),
                "sec_vol_oos": sec_vol_oos,
                "pos_oos_map": pos_oos_map,
            }

        sleeve_daily_ret, is_meta_stats = {}, {}
        for fold in FOLDS:
            f_id = fold["id"]
            cutoff = pd.Timestamp(fold["cutoff"])
            for aid in IDS_60:
                df15 = _get_15m_df(frames_15m_60, aid)
                dt15 = pd.to_datetime(df15["dt"])
                ret15 = np.nan_to_num(df15["close"].pct_change().fillna(0.0).to_numpy(dtype=np.float64), nan=0.0)
                w = t15_hr[(f_id, aid)]
                pos = np.roll(w, 1)
                if len(pos):
                    pos[0] = 0.0
                turn = np.abs(np.diff(pos, prepend=0.0)) if len(pos) else np.zeros(0, dtype=np.float64)
                fee = float(SPECS_60[aid].fee_rate)
                net15 = pos * ret15 - turn * fee
                t_day_idx = assign_futures_trading_day(dt15)
                sleeve_daily_ret.setdefault(f_id, {})[aid] = pd.Series(net15, index=t_day_idx).groupby(level=0).sum()
                is_meta_stats[(f_id, aid)] = {
                    **_sleeve_training_metrics(df15, w, fee, cutoff, pd.Timestamp(fold["train_start"])),
                    "is_shp_fs": is_sharpe_60.get((f_id, aid), 0.0),
                    "n_factors": len(sel_60_std.get((f_id, aid), [])),
                }
        for f_id in list(sleeve_daily_ret.keys()):
            sleeve_daily_ret[f_id] = pd.DataFrame(sleeve_daily_ret[f_id]).sort_index().fillna(0.0)
        b = {
            "oos_fast_cache": oos_fast_cache,
            "sleeve_daily_ret": sleeve_daily_ret,
            "is_meta_stats": is_meta_stats,
            "ep_fs": ep_fs,
            "ep_full": ep_full,
            "ep_hr": ep_hr,
            "sel_60_std": sel_60_std,
            "is_sharpe_60": is_sharpe_60,
            "pit_assets": pit_assets,
            "trading_date_aligned": True,
            "stage2_unused_rust_integrated": True,
        }
        b["daily_frames"] = frames_1d_60
        b["fingerprint"] = prepared_60["fingerprint"]
        _save_bundle(bundle_path, b, prepared_60["fingerprint"])

    return b


def build_universe_60_summary_table() -> pd.DataFrame:
    """统计 60 个商品的 15m/5m 流动性、数据起止日期与所属板块。"""
    rows = []
    for aid, sym, sec, cn_name, has_night, fee in RAW_60_META:
        p15 = DATA_15M_60_DIR / f"{aid}_15m.parquet"
        df15 = pd.read_parquet(p15)
        dt = pd.to_datetime(df15["dt"])
        d_norm = assign_futures_trading_day(dt) if has_night else dt.dt.normalize()
        vol_col = "volume" if "volume" in df15.columns else "vol"
        amt_series = df15["amount"] if "amount" in df15.columns else pd.Series(0.0, index=df15.index)
        if float(amt_series.sum()) <= 0.0 and vol_col in df15.columns:
            # 历史本地 parquet 若未存成交额字段，使用 close * volume * 10 (标准合约乘数近似) 估算名义成交额
            amt_series = df15["close"].astype(np.float64) * df15[vol_col].astype(np.float64) * 10.0
        daily_amt_yi = amt_series.groupby(d_norm).sum() / 1e8
        daily_vol_wan = df15.groupby(d_norm)[vol_col].sum() / 1e4 if vol_col in df15.columns else pd.Series([0.0])
        rows.append(
            {
                "asset_id": aid,
                "cn_name": cn_name,
                "jq_symbol": sym,
                "sector": sec,
                "tier": "Full60_Unified",
                "has_night": has_night,
                "fee_rate_bp": round(fee * 1e4, 2),
                "start_date": str(d_norm.min().date()),
                "end_date": str(d_norm.max().date()),
                "trading_days": int(d_norm.nunique()),
                "bars_15m": int(len(df15)),
                "median_daily_turnover_yi": round(float(daily_amt_yi.median()), 2),
                "median_daily_vol_wan": round(float(daily_vol_wan.median()), 2),
            }
        )
    return pd.DataFrame(rows)


def _portfolio_execution_evidence(fc: dict) -> dict:
    """Retain observed execution prices and effective fees for independent replay."""
    fees = {aid: float(SPECS_60[aid].fee_rate) for aid in IDS_60}
    prices = fc["execution_prices_df"][IDS_60].copy()
    return {
        "execution_prices_df": prices,
        "returns_df": fc["ret_oos"][IDS_60],
        "tradable_mask": fc["mask_oos"][IDS_60],
        "fee_rates": fees,
        "execution_clock": "next_open",
        "accounting": "fixed_initial_capital",
    }


def attach_oos_roll_masks(oos_fast_cache: dict) -> dict[str, pd.DataFrame]:
    """为每个折次的 5m OOS 执行网格因果识别主连合约换月跳空掩码 `roll_mask_oos`。"""
    roll_masks_by_fold: dict[str, pd.DataFrame] = {}
    df5_loaded: dict[str, pd.DataFrame] = {}
    for fold in FOLDS:
        f_id = fold["id"]
        fc = oos_fast_cache[f_id]
        ret_oos = fc["ret_oos"][IDS_60]
        if "roll_mask_oos" in fc and isinstance(fc["roll_mask_oos"], pd.DataFrame):
            roll_masks_by_fold[f_id] = fc["roll_mask_oos"].reindex_like(ret_oos).fillna(False).astype(bool)
            continue
        oos_e = pd.Timestamp(fold.get("oos_end", ret_oos.index.max()))
        fold_rolls: dict[str, pd.Series] = {}
        for aid in IDS_60:
            p5 = SOURCE_DIR / "data_5m" / f"{aid}_5m.parquet"
            if not p5.exists():
                fold_rolls[aid] = pd.Series(False, index=ret_oos.index, dtype=bool)
                continue
            if aid not in df5_loaded:
                df5 = pd.read_parquet(p5)
                df5["dt"] = pd.to_datetime(df5["dt"])
                df5_loaded[aid] = df5.sort_values("dt").reset_index(drop=True)
            grid5 = df5_loaded[aid]
            sub_grid = grid5.loc[grid5["dt"] <= oos_e].reset_index(drop=True)
            spec = SPECS_60[aid]
            r_arr = detect_contract_roll_mask(
                sub_grid,
                has_night_session=spec.has_night_session,
                gap_atr_mult=spec.roll_gap_threshold_atr,
            )
            r_ser = pd.Series(r_arr, index=pd.DatetimeIndex(sub_grid["dt"]), dtype=bool)
            fold_rolls[aid] = r_ser.reindex(ret_oos.index, fill_value=False).fillna(False).astype(bool)
        roll_df = pd.DataFrame(fold_rolls, index=ret_oos.index, columns=IDS_60, dtype=bool)
        fc["roll_mask_oos"] = roll_df
        roll_masks_by_fold[f_id] = roll_df
    return roll_masks_by_fold


def _execute_sector_conserved_allocation(
    fc: dict,
    pos_df: pd.DataFrame,
    inv_vol: pd.DataFrame,
    active_mask: pd.DataFrame,
    *,
    sector_damp_power: float,
    symmetric_deadzone: float,
    max_single_asset_weight: float,
    min_effective_assets: int,
    target_portfolio_vol: float,
    adjust_roll_gaps: bool = False,
    charge_roll_fee: bool = True,
    fee_multiplier: float = 1.0,
) -> dict:
    """统一跨品种板块风险预算守恒、因果波动率预热缩放、死区防抖与期末闭合清算内核。"""
    ret_oos = fc["ret_oos"][IDS_60]
    mask_oos = fc["mask_oos"][IDS_60]
    roll_mask_oos: pd.DataFrame | None = None
    if adjust_roll_gaps and "roll_mask_oos" in fc:
        roll_mask_oos = fc["roll_mask_oos"][IDS_60].reindex_like(ret_oos).fillna(False).astype(bool)

    sector_active_counts = {}
    sector_inv_vol = {}
    for sec, syms in SECTORS_60.items():
        k_s_t = active_mask[syms].sum(axis=1)
        sector_active_counts[sec] = k_s_t
        sec_vol = fc["sec_vol_oos"][sec]
        sector_inv_vol[sec] = (1.0 / np.maximum(sec_vol, 0.05)) * (k_s_t > 0.0).astype(np.float64)

    sec_inv_vol_df = pd.DataFrame(sector_inv_vol, index=ret_oos.index)
    sec_inv_vol_sum = sec_inv_vol_df.sum(axis=1).replace(0.0, np.nan)
    sec_budget_share = sec_inv_vol_df.div(sec_inv_vol_sum, axis=0).fillna(0.0)

    sector_mult_df = pd.DataFrame(0.0, index=ret_oos.index, columns=IDS_60, dtype=np.float64)
    for sec, syms in SECTORS_60.items():
        k_s_t = sector_active_counts[sec]
        within_sec_damp = 1.0 / np.power(np.maximum(k_s_t, 1.0), sector_damp_power)
        sec_factor = sec_budget_share[sec] * within_sec_damp
        for sym in syms:
            sector_mult_df[sym] = sec_factor

    raw_alloc = inv_vol * sector_mult_df * active_mask
    n_rows, n_cols = raw_alloc.shape
    alloc_mat = np.zeros((n_rows, n_cols), dtype=np.float64)
    raw_mat = raw_alloc.to_numpy(dtype=np.float64)
    act_mat = active_mask.to_numpy(dtype=np.float64)
    for t_idx in range(n_rows):
        n_act = float(act_mat[t_idx].sum())
        if n_act <= 0.0:
            continue
        target_b = min(1.0, n_act / float(min_effective_assets))
        alloc_mat[t_idx] = _project_capped_simplex_row(
            raw_weights=raw_mat[t_idx],
            active_mask=act_mat[t_idx],
            target_budget=target_b,
            max_weight=max_single_asset_weight,
        )

    alloc_weights = pd.DataFrame(alloc_mat, index=ret_oos.index, columns=IDS_60)
    unscaled_positions = (alloc_weights * pos_df).where(mask_oos).ffill().fillna(0.0)
    unscaled_gross = (unscaled_positions.shift(1).fillna(0.0) * ret_oos).sum(axis=1)
    port_roll_vol = unscaled_gross.rolling(40 * BARS_PER_DAY, min_periods=15 * BARS_PER_DAY).std().shift(1)
    active_exp_bars = (
        (unscaled_positions.abs().sum(axis=1) > 1e-12)
        .astype(np.float64)
        .rolling(40 * BARS_PER_DAY, min_periods=1)
        .sum()
        .shift(1)
        .fillna(0.0)
    )
    port_roll_vol = port_roll_vol.where(active_exp_bars >= float(15 * BARS_PER_DAY))
    port_roll_vol = (port_roll_vol * ANN_SCALE).ffill().fillna(target_portfolio_vol)
    lev_scalar = np.clip(target_portfolio_vol / np.maximum(port_roll_vol, 0.05), 0.0, 1.65)
    budget_weights = alloc_weights.multiply(lev_scalar, axis=0).clip(lower=0.0, upper=max_single_asset_weight)
    final_weights_df = budget_weights * pos_df

    budget_fraction = active_mask.sum(axis=1).div(min_effective_assets).clip(upper=1.0)
    gross_budget_series = (budget_fraction * lev_scalar).to_numpy(dtype=np.float64)
    final_mat = final_weights_df.to_numpy().copy()
    observed = mask_oos.to_numpy()
    prev = np.zeros(n_cols)
    for t_idx in range(n_rows):
        target = final_mat[t_idx]
        if symmetric_deadzone > 0.0:
            same_dir = (target * prev) > 0.0
            keep = same_dir & (np.abs(target) > 1e-6) & (np.abs(target - prev) < symmetric_deadzone)
            target[keep] = prev[keep]
        target[~observed[t_idx]] = prev[~observed[t_idx]]
        frozen = float(np.abs(prev[~observed[t_idx]]).sum())
        avail = max(0.0, float(gross_budget_series[t_idx]) - frozen)
        prop = float(np.abs(target[observed[t_idx]]).sum())
        if prop > avail and prop > 0:
            target[observed[t_idx]] *= avail / prop
        final_mat[t_idx] = np.clip(target, -max_single_asset_weight, max_single_asset_weight)
        prev = final_mat[t_idx].copy()

    w_oos = pd.DataFrame(final_mat, index=ret_oos.index, columns=IDS_60)
    gross_exp = w_oos.abs().sum(axis=1)
    scale_cap = np.maximum(gross_exp, 1.0)
    w_oos = w_oos.div(scale_cap, axis=0)
    w_oos.iloc[0] = 0.0
    for col in IDS_60:
        t_col = mask_oos[col].to_numpy()
        w_col = w_oos[col].to_numpy().copy()
        for i in range(1, len(w_col)):
            if not t_col[i]:
                w_col[i] = w_col[i - 1]
        if t_col[-1]:
            w_col[-1] = 0.0
        w_oos[col] = w_col

    fees = {c: float(SPECS_60[c].fee_rate) for c in IDS_60}
    exec_ret_oos = ret_oos.mask(roll_mask_oos, 0.0) if roll_mask_oos is not None else ret_oos
    ledger = account_portfolio_execution(
        exec_ret_oos,
        w_oos,
        fees,
        tradable_mask=mask_oos,
        roll_mask=roll_mask_oos if (adjust_roll_gaps and charge_roll_fee) else None,
        fee_multiplier=fee_multiplier,
    )
    ev = _portfolio_execution_evidence(fc)
    if roll_mask_oos is not None:
        ev["returns_df"] = exec_ret_oos
    return {
        **ev,
        "daily_returns": ledger["daily_returns"],
        "bar_net_returns": ledger["bar_net_returns"],
        "gross_returns_df": ledger["gross_returns_df"],
        "cost_returns_df": ledger["cost_returns_df"],
        "turnover_df": ledger["turnover_df"],
        "weights_df": w_oos,
    }


def run_flat_60_alloc(
    name: str,
    oos_fast_cache: dict,
    pit_assets: dict,
    ep_map: dict,
    scheme: str = "hr",
    sector_damp_power: float = 0.50,
    symmetric_deadzone: float = 0.0,
    max_single_asset_weight: float = 0.25,
    min_effective_assets: int = 4,
    target_portfolio_vol: float = 0.14,
    adjust_roll_gaps: bool = False,
    charge_roll_fee: bool = True,
    fee_multiplier: float = 1.0,
) -> tuple[dict, list[dict]]:
    """运行全量 60 品种扁平化跨资产 HRP 配置（对比朴素等权 vs 板块预算守恒+对称死区）。"""
    ledgers = []
    ep_list = []
    for fold in FOLDS:
        f_id = fold["id"]
        fc = oos_fast_cache[f_id]
        mask_oos = fc["mask_oos"][IDS_60]
        roll_vol = fc["roll_vol_oos"][IDS_60]
        pos_df = fc["pos_oos_map"][scheme][IDS_60].copy()
        pit_set = set(pit_assets[f_id])
        for c in IDS_60:
            if c not in pit_set:
                pos_df[c] = 0.0

        listed_mask = mask_oos.astype(np.float64)
        active_mask = ((pos_df.abs() > 1e-3) & (listed_mask > 0.0)).astype(np.float64)
        inv_vol = (1.0 / np.maximum(roll_vol, 0.05)) * active_mask

        led = _execute_sector_conserved_allocation(
            fc,
            pos_df,
            inv_vol,
            active_mask,
            sector_damp_power=sector_damp_power,
            symmetric_deadzone=symmetric_deadzone,
            max_single_asset_weight=max_single_asset_weight,
            min_effective_assets=min_effective_assets,
            target_portfolio_vol=target_portfolio_vol,
            adjust_roll_gaps=adjust_roll_gaps,
            charge_roll_fee=charge_roll_fee,
            fee_multiplier=fee_multiplier,
        )
        ledgers.append(led)
        w_oos = led["weights_df"]
        for c in IDS_60:
            if w_oos[c].abs().max() > 1e-6 and len(ep_map.get((f_id, c), [])) > 0:
                ep_list.append(ep_map[(f_id, c)])

    s = summarize_stitched_oos(ledgers, name)
    ep_st = compute_episode_payoff_stats(ep_list)
    s.update(ep_st)
    return s, ledgers


def run_zero_hardcode_sop_stage3(
    name: str,
    oos_fast_cache: dict,
    sleeve_daily_ret: dict,
    is_meta_stats: dict,
    pit_assets: dict,
    ep_map: dict,
    scheme: str = "hr",
    min_nf: int = 5,
    min_shp_fs: float = 0.40,
    min_shp_hr: float = 0.50,
    min_r: float = 1.45,
    min_sm: float = 4.0,
    sector_damp_power: float = 0.55,
    symmetric_deadzone: float = 0.02,
    max_single_asset_weight: float = 0.25,
    min_effective_assets: int = 4,
    target_portfolio_vol: float = 0.14,
    adjust_roll_gaps: bool = False,
    charge_roll_fee: bool = True,
    fee_multiplier: float = 1.0,
) -> tuple[dict, list[dict], dict[str, list[str]]]:
    """
    运行 100% 零先验硬编码 (Zero Hardcoded Core-20) 的全市场 60 品种统一 SOP Stage-3 门禁配置（留存约 30 品种）。
    """
    ledgers = []
    ep_list = []
    selected_per_fold: dict[str, list[str]] = {}

    for fold in FOLDS:
        f_id = fold["id"]
        fc = oos_fast_cache[f_id]
        mask_oos = fc["mask_oos"][IDS_60]
        roll_vol = fc["roll_vol_oos"][IDS_60]
        pos_df = fc["pos_oos_map"][scheme][IDS_60].copy()

        sel_ids = []
        for aid in IDS_60:
            if aid not in pit_assets[f_id]:
                continue
            m = is_meta_stats[(f_id, aid)]
            eff_shp = max(float(m["is_shp_hr"]), 0.85 * float(m["is_shp_fs"]))
            if (
                m["n_factors"] >= min_nf
                and m["is_shp_fs"] >= min_shp_fs
                and eff_shp >= min_shp_hr
                and m["is_payoff_r"] >= min_r
                and m["is_safety_margin"] >= min_sm
                and m["is_ep_count"] >= 10
            ):
                sel_ids.append(aid)
        selected_per_fold[f_id] = sel_ids
        sel_set = set(sel_ids)
        for c in IDS_60:
            if c not in sel_set:
                pos_df[c] = 0.0

        listed_mask = mask_oos.astype(np.float64)
        active_mask = ((pos_df.abs() > 1e-3) & (listed_mask > 0.0)).astype(np.float64)
        inv_vol = (1.0 / np.maximum(roll_vol, 0.05)) * active_mask

        for c in sel_ids:
            m = is_meta_stats[(f_id, c)]
            eff_s = 0.55 * float(m["is_shp_hr"]) + 0.45 * float(m["is_shp_fs"])
            q_p = np.clip(eff_s, 0.5, 2.5) * np.clip(max(float(m["is_safety_margin"]), 5.0) / 15.0, 0.5, 2.0)
            inv_vol[c] *= q_p

        led = _execute_sector_conserved_allocation(
            fc,
            pos_df,
            inv_vol,
            active_mask,
            sector_damp_power=sector_damp_power,
            symmetric_deadzone=symmetric_deadzone,
            max_single_asset_weight=max_single_asset_weight,
            min_effective_assets=min_effective_assets,
            target_portfolio_vol=target_portfolio_vol,
            adjust_roll_gaps=adjust_roll_gaps,
            charge_roll_fee=charge_roll_fee,
            fee_multiplier=fee_multiplier,
        )
        ledgers.append(led)
        w_oos = led["weights_df"]
        for c in sel_ids:
            if w_oos[c].abs().max() > 1e-6 and len(ep_map.get((f_id, c), [])) > 0:
                ep_list.append(ep_map[(f_id, c)])

    s = summarize_stitched_oos(ledgers, name)
    ep_st = compute_episode_payoff_stats(ep_list)
    s.update(ep_st)
    return s, ledgers, selected_per_fold


def compute_is_liquidity_and_price_returns(
    frames_1d: dict,
) -> tuple[dict[tuple[str, str], dict[str, float]], pd.DataFrame]:
    """
    严格在每个折次的训练截止日 (<= cutoff) 前，按期货交易日归属计算全池 60 品种的样本内客观流动性深度
    （日均成交手数、名义成交额、有效上市天数）以及用于无监督产业链共线性风险分摊的日频价格收益率矩阵。
    全程 100% 零先验硬编码 (不引用 CORE_20_IDS)，杜绝任何未来信息泄露。
    """
    is_liq_stats: dict[tuple[str, str], dict[str, float]] = {}
    price_ret_dict: dict[str, pd.Series] = {}

    for aid in IDS_60:
        p15 = DATA_15M_60_DIR / f"{aid}_15m.parquet"
        df15 = pd.read_parquet(p15)
        dt = pd.to_datetime(df15["dt"])
        d_norm = assign_futures_trading_day(dt) if SPECS_60[aid].has_night_session else dt.dt.normalize()
        vol_col = "volume" if "volume" in df15.columns else "vol"
        amt_s = df15["amount"] if "amount" in df15.columns else pd.Series(0.0, index=df15.index)
        if float(amt_s.sum()) <= 0.0 and vol_col in df15.columns:
            amt_s = df15["close"].astype(np.float64) * df15[vol_col].astype(np.float64) * 10.0
        d_amt = amt_s.groupby(d_norm).sum() / 1e8
        d_vol = df15.groupby(d_norm)[vol_col].sum() / 1e4 if vol_col in df15.columns else pd.Series(0.0)

        for fold in FOLDS:
            f_id = fold["id"]
            cut = pd.Timestamp(fold["cutoff"]).normalize()
            sub_v = d_vol.loc[d_vol.index <= cut]
            sub_a = d_amt.loc[d_amt.index <= cut]
            is_liq_stats[(f_id, aid)] = {
                "is_days": float(len(sub_v)),
                "is_vol_wan": float(sub_v.median()) if len(sub_v) > 0 else 0.0,
                "is_amt_yi": float(sub_a.median()) if len(sub_a) > 0 else 0.0,
            }

        df1 = frames_1d[SPECS_60[aid].symbol]
        dt1 = pd.to_datetime(df1["dt"]).dt.normalize()
        c1 = pd.Series(df1["close"].to_numpy(dtype=np.float64), index=dt1)
        price_ret_dict[aid] = c1.pct_change().fillna(0.0)

    price_ret_df = pd.DataFrame(price_ret_dict).sort_index().fillna(0.0)
    return is_liq_stats, price_ret_df


def run_zero_hardcode_cluster_conserved(
    name: str,
    oos_fast_cache: dict,
    sleeve_daily_ret: dict,
    is_meta_stats: dict,
    pit_assets: dict,
    ep_map: dict,
    is_liq_stats: dict[tuple[str, str], dict[str, float]],
    price_ret_df: pd.DataFrame,
    scheme: str = "hr",
    min_nf: int = 5,
    min_ep: int = 10,
    min_shp_fs: float = 0.45,
    min_shp_hr: float = 0.52,
    min_r: float = 1.48,
    min_sm: float = 4.5,
    min_vol_wan: float = 2.0,
    max_per_corr_cluster: int = 2,
    corr_thresh: float = 0.80,
    cluster_damp_coeff: float = 0.15,
    sector_damp_power: float = 0.45,
    symmetric_deadzone: float = 0.02,
    max_single_asset_weight: float = 0.25,
    min_effective_assets: int = 4,
    target_portfolio_vol: float = 0.14,
    adjust_roll_gaps: bool = False,
    charge_roll_fee: bool = True,
    fee_multiplier: float = 1.0,
) -> tuple[dict, list[dict], dict[str, list[str]]]:
    """
    运行 100% 零先验硬编码 (Zero Hardcoded CORE_20_IDS) 的全市场 60 品种内生准入、
    同板块高共线性簇去重 + 连续相关性簇内风险分摊、板块风险预算守恒配置架构（约留存 30 个主力品种）。
    """
    ledgers = []
    ep_list = []
    selected_per_fold: dict[str, list[str]] = {}

    for fold in FOLDS:
        f_id = fold["id"]
        cut = pd.Timestamp(fold["cutoff"]).normalize()
        fc = oos_fast_cache[f_id]
        mask_oos = fc["mask_oos"][IDS_60]
        roll_vol = fc["roll_vol_oos"][IDS_60]
        pos_df = fc["pos_oos_map"][scheme][IDS_60].copy()

        d_all = sleeve_daily_ret[f_id]
        d_is = d_all.loc["2024-01-02":cut]
        sub_p = price_ret_df.loc["2024-01-02":cut]
        cp_mat = sub_p.corr().abs().fillna(0.0)
        cs_mat = d_is.corr().abs().fillna(0.0)

        # 1. 全池 60 品种统一客观准入与内生质量-流动性复合评分 (<= cutoff)
        cands: dict[str, float] = {}
        for aid in pit_assets[f_id]:
            m = is_meta_stats[(f_id, aid)]
            lq = is_liq_stats[(f_id, aid)]
            eff_shp = max(float(m["is_shp_hr"]), 0.85 * float(m["is_shp_fs"]))
            if (
                (lq["is_vol_wan"] >= min_vol_wan or lq.get("is_amt_yi", 0.0) >= 20.0)
                and m["n_factors"] >= min_nf
                and m["is_ep_count"] >= min_ep
                and m["is_shp_fs"] >= min_shp_fs
                and eff_shp >= min_shp_hr
                and m["is_payoff_r"] >= min_r
                and m["is_safety_margin"] >= min_sm
            ):
                cands[aid] = eff_shp * float(m["is_shp_fs"]) * float(m["is_payoff_r"])

        # 2. 同板块内高共线性簇限额去重 + 连续相关性簇内风险分摊 (<= cutoff)
        sel_ids: list[str] = []
        for _sec, syms in SECTORS_60.items():
            sec_cands = sorted([a for a in syms if a in cands], key=lambda x: cands[x], reverse=True)
            kept: list[str] = []
            for a in sec_cands:
                high_corr_cnt = sum(
                    1
                    for k in kept
                    if a in cp_mat.index and k in cp_mat.columns and float(cp_mat.loc[a, k]) >= corr_thresh
                )
                if high_corr_cnt < max_per_corr_cluster:
                    kept.append(a)
            sel_ids.extend(kept)

        cluster_share_w = compute_continuous_cluster_risk_shares(
            price_corr_df=cp_mat,
            strategy_corr_df=cs_mat,
            sectors=SECTORS_60,
            candidate_assets=sel_ids,
            cluster_damp_coeff=cluster_damp_coeff,
        )

        selected_per_fold[f_id] = sel_ids
        sel_set = set(sel_ids)
        for c in IDS_60:
            if c not in sel_set:
                pos_df[c] = 0.0

        listed_mask = mask_oos.astype(np.float64)
        active_mask = ((pos_df.abs() > 1e-3) & (listed_mask > 0.0)).astype(np.float64)
        inv_vol = (1.0 / np.maximum(roll_vol, 0.05)) * active_mask

        for c in sel_ids:
            m = is_meta_stats[(f_id, c)]
            eff_s = 0.55 * float(m["is_shp_hr"]) + 0.45 * float(m["is_shp_fs"])
            q_p = (
                np.clip(eff_s, 0.5, 2.5)
                * np.clip(max(float(m["is_safety_margin"]), 5.0) / 15.0, 0.5, 2.0)
                * cluster_share_w.get(c, 1.0)
            )
            inv_vol[c] *= q_p

        led = _execute_sector_conserved_allocation(
            fc,
            pos_df,
            inv_vol,
            active_mask,
            sector_damp_power=sector_damp_power,
            symmetric_deadzone=symmetric_deadzone,
            max_single_asset_weight=max_single_asset_weight,
            min_effective_assets=min_effective_assets,
            target_portfolio_vol=target_portfolio_vol,
            adjust_roll_gaps=adjust_roll_gaps,
            charge_roll_fee=charge_roll_fee,
            fee_multiplier=fee_multiplier,
        )
        ledgers.append(led)
        w_oos = led["weights_df"]
        for c in sel_ids:
            if w_oos[c].abs().max() > 1e-6 and len(ep_map.get((f_id, c), [])) > 0:
                ep_list.append(ep_map[(f_id, c)])

    s = summarize_stitched_oos(ledgers, name)
    ep_st = compute_episode_payoff_stats(ep_list)
    s.update(ep_st)
    return s, ledgers, selected_per_fold


def build_attribution_tables(
    flat_ledgers: list[dict],
    sop_ledgers: list[dict],
    sop_selected: dict[str, list[str]],
    zhc_ledgers: list[dict] | None = None,
    zhc_selected: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成 6大板块 与 60品种 在 Flat-60 HRP、ZeroHardcode-SOPStage3-60 与 ZeroHardcode-ClusterConserved-60 架构下的归因对比表。"""
    flat_gross = pd.concat([led["gross_returns_df"] for led in flat_ledgers]).sum() * 100.0
    flat_cost = pd.concat([led["cost_returns_df"] for led in flat_ledgers]).sum() * 100.0
    flat_net = flat_gross - flat_cost
    flat_turn = pd.concat([led["turnover_df"] for led in flat_ledgers]).sum() / (
        max(pd.concat([led["daily_returns"] for led in flat_ledgers]).index.nunique(), 1) / 252.0
    )

    sop_gross = pd.concat([led["gross_returns_df"] for led in sop_ledgers]).sum() * 100.0
    sop_cost = pd.concat([led["cost_returns_df"] for led in sop_ledgers]).sum() * 100.0
    sop_net = sop_gross - sop_cost
    sop_turn = pd.concat([led["turnover_df"] for led in sop_ledgers]).sum() / (
        max(pd.concat([led["daily_returns"] for led in sop_ledgers]).index.nunique(), 1) / 252.0
    )

    if zhc_ledgers is not None:
        zhc_gross = pd.concat([led["gross_returns_df"] for led in zhc_ledgers]).sum() * 100.0
        zhc_cost = pd.concat([led["cost_returns_df"] for led in zhc_ledgers]).sum() * 100.0
        zhc_net = zhc_gross - zhc_cost
        zhc_turn = pd.concat([led["turnover_df"] for led in zhc_ledgers]).sum() / (
            max(pd.concat([led["daily_returns"] for led in zhc_ledgers]).index.nunique(), 1) / 252.0
        )
    else:
        zhc_gross = sop_gross
        zhc_cost = sop_cost
        zhc_net = sop_net
        zhc_turn = sop_turn

    sop_union = set(sop_selected.get("Fold1_2025", [])) | set(sop_selected.get("Fold2_2026", []))
    zhc_union = (
        set(zhc_selected.get("Fold1_2025", [])) | set(zhc_selected.get("Fold2_2026", []))
        if zhc_selected is not None
        else set()
    )

    sym_rows = []
    for aid in IDS_60:
        role = "ZeroHC内生簇代表" if aid in zhc_union else ("SOP阶段三准入" if aid in sop_union else "SOP拦截影子/震荡")
        sym_rows.append(
            {
                "asset_id": aid,
                "cn_name": CN_NAME_MAP[aid],
                "sector": SPECS_60[aid].sector,
                "role": role,
                "flat60_gross_pct": round(float(flat_gross[aid]), 3),
                "flat60_cost_pct": round(float(flat_cost[aid]), 3),
                "flat60_net_pct": round(float(flat_net[aid]), 3),
                "flat60_ann_turn": round(float(flat_turn[aid]), 2),
                "sop60_gross_pct": round(float(sop_gross[aid]), 3),
                "sop60_cost_pct": round(float(sop_cost[aid]), 3),
                "sop60_net_pct": round(float(sop_net[aid]), 3),
                "sop60_ann_turn": round(float(sop_turn[aid]), 2),
                "zerohc60_gross_pct": round(float(zhc_gross[aid]), 3),
                "zerohc60_cost_pct": round(float(zhc_cost[aid]), 3),
                "zerohc60_net_pct": round(float(zhc_net[aid]), 3),
                "zerohc60_ann_turn": round(float(zhc_turn[aid]), 2),
                "net_improvement_pct": round(float(zhc_net[aid] - flat_net[aid]), 3),
            }
        )
    sym_df = pd.DataFrame(sym_rows)

    sec_rows = []
    for sec, syms in SECTORS_60.items():
        sub = sym_df[sym_df["sector"] == sec]
        sec_rows.append(
            {
                "sector": sec,
                "total_symbols": len(syms),
                "sop_admitted_count": sum(1 for c in syms if c in sop_union),
                "zerohc_cluster_count": sum(1 for c in syms if c in zhc_union),
                "flat60_gross_pct": round(float(sub["flat60_gross_pct"].sum()), 2),
                "flat60_cost_pct": round(float(sub["flat60_cost_pct"].sum()), 2),
                "flat60_net_pct": round(float(sub["flat60_net_pct"].sum()), 2),
                "flat60_ann_turn": round(float(sub["flat60_ann_turn"].sum()), 1),
                "sop60_gross_pct": round(float(sub["sop60_gross_pct"].sum()), 2),
                "sop60_cost_pct": round(float(sub["sop60_cost_pct"].sum()), 2),
                "sop60_net_pct": round(float(sub["sop60_net_pct"].sum()), 2),
                "sop60_ann_turn": round(float(sub["sop60_ann_turn"].sum()), 1),
                "zerohc60_gross_pct": round(float(sub["zerohc60_gross_pct"].sum()), 2),
                "zerohc60_cost_pct": round(float(sub["zerohc60_cost_pct"].sum()), 2),
                "zerohc60_net_pct": round(float(sub["zerohc60_net_pct"].sum()), 2),
                "zerohc60_ann_turn": round(float(sub["zerohc60_ann_turn"].sum()), 1),
                "net_improvement_pct": round(float(sub["zerohc60_net_pct"].sum() - sub["flat60_net_pct"].sum()), 2),
            }
        )
    sec_df = pd.DataFrame(sec_rows)
    return sec_df, sym_df


def build_satellite_sop_table(
    is_meta_stats: dict,
    sleeve_daily_ret: dict,
    sop_selected: dict[str, list[str]],
    zhc_selected: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """生成全池 60 个商品品种在两折样本内的 SOP Stage-3 门禁指标与样本外独立表现明细表。"""
    rows = []
    for aid in IDS_60:
        m1 = is_meta_stats[("Fold1_2025", aid)]
        m2 = is_meta_stats[("Fold2_2026", aid)]

        # 计算该品种在两折 OOS 的独立单品种袖套净 Sharpe
        oos_d1 = sleeve_daily_ret["Fold1_2025"][aid].loc["2025-01-01":"2025-12-31"]
        oos_d2 = sleeve_daily_ret["Fold2_2026"][aid].loc["2026-01-01":"2026-09-30"]
        oos_all = pd.concat([oos_d1, oos_d2])
        sd = float(oos_all.std(ddof=1))
        oos_shp = float((oos_all.mean() / sd) * np.sqrt(252.0)) if sd > 1e-12 else 0.0
        oos_ret_pct = float(oos_all.sum() * 100.0)

        rows.append(
            {
                "asset_id": aid,
                "cn_name": CN_NAME_MAP[aid],
                "sector": SPECS_60[aid].sector,
                "f1_n_factors": m1["n_factors"],
                "f1_is_shp_fs": round(float(m1["is_shp_fs"]), 2),
                "f1_is_shp_hr": round(float(m1["is_shp_hr"]), 2),
                "f1_is_payoff_r": round(float(m1["is_payoff_r"]), 2),
                "f1_is_safety_margin_pct": round(float(m1["is_safety_margin"]), 1),
                "admitted_fold1": aid in sop_selected.get("Fold1_2025", []),
                "zerohc_admitted_fold1": (aid in zhc_selected["Fold1_2025"]) if zhc_selected else False,
                "f2_n_factors": m2["n_factors"],
                "f2_is_shp_fs": round(float(m2["is_shp_fs"]), 2),
                "f2_is_shp_hr": round(float(m2["is_shp_hr"]), 2),
                "f2_is_payoff_r": round(float(m2["is_payoff_r"]), 2),
                "f2_is_safety_margin_pct": round(float(m2["is_safety_margin"]), 1),
                "admitted_fold2": aid in sop_selected.get("Fold2_2026", []),
                "zerohc_admitted_fold2": (aid in zhc_selected["Fold2_2026"]) if zhc_selected else False,
                "standalone_oos_ret_pct": round(oos_ret_pct, 2),
                "standalone_oos_sharpe": round(oos_shp, 2),
            }
        )
    return pd.DataFrame(rows)


def _count_roll_crossings(oos_fast_cache: dict, ledgers: list[dict]) -> tuple[int, int]:
    """统计两折样本外 60 品种换月跳空总事件数与持仓实际跨越换月次数。"""
    total_rolls = 0
    active_crossings = 0
    for fold, led in zip(FOLDS, ledgers, strict=True):
        fc = oos_fast_cache[fold["id"]]
        rm = fc.get("roll_mask_oos")
        if rm is None:
            continue
        rm_df = rm[IDS_60].reindex_like(led["weights_df"]).fillna(False).astype(bool)
        total_rolls += int(rm_df.to_numpy().sum())
        prev_w = led["weights_df"][IDS_60].shift(1).fillna(0.0).abs() > 1e-6
        active_crossings += int((rm_df & prev_w).to_numpy().sum())
    return total_rolls, active_crossings


def _run_variant_with_execution_knobs(
    variant_name: str,
    *,
    oos_fast_cache: dict,
    sleeve_daily_ret: dict,
    is_meta_stats: dict,
    pit_assets: dict,
    ep_hr: dict,
    is_liq_stats: dict,
    price_ret_df: dict,
    symmetric_deadzone: float | None = None,
    adjust_roll_gaps: bool = False,
    charge_roll_fee: bool = True,
    fee_multiplier: float = 1.0,
) -> tuple[dict, list[dict]]:
    """统一调度四大代表性组合架构，支持换月跳空剥离、移仓双边成本与多倍冲击费率注入。"""
    if variant_name == "Full60_FlatHRP_HysteresisRunner":
        dz = 0.0 if symmetric_deadzone is None else symmetric_deadzone
        return run_flat_60_alloc(
            variant_name,
            oos_fast_cache,
            pit_assets,
            ep_hr,
            scheme="hr",
            sector_damp_power=0.50,
            symmetric_deadzone=dz,
            adjust_roll_gaps=adjust_roll_gaps,
            charge_roll_fee=charge_roll_fee,
            fee_multiplier=fee_multiplier,
        )
    if variant_name == "Full60_SecConserved_SymDZ_HR":
        dz = 0.025 if symmetric_deadzone is None else symmetric_deadzone
        return run_flat_60_alloc(
            variant_name,
            oos_fast_cache,
            pit_assets,
            ep_hr,
            scheme="hr",
            sector_damp_power=0.90,
            symmetric_deadzone=dz,
            adjust_roll_gaps=adjust_roll_gaps,
            charge_roll_fee=charge_roll_fee,
            fee_multiplier=fee_multiplier,
        )
    if variant_name == "Full60_ZeroHardcode_SOPStage3":
        dz = 0.02 if symmetric_deadzone is None else symmetric_deadzone
        s, leds, _ = run_zero_hardcode_sop_stage3(
            variant_name,
            oos_fast_cache=oos_fast_cache,
            sleeve_daily_ret=sleeve_daily_ret,
            is_meta_stats=is_meta_stats,
            pit_assets=pit_assets,
            ep_map=ep_hr,
            scheme="hr",
            min_nf=5,
            min_shp_fs=0.40,
            min_shp_hr=0.50,
            min_r=1.45,
            min_sm=4.0,
            sector_damp_power=0.50,
            symmetric_deadzone=dz,
            adjust_roll_gaps=adjust_roll_gaps,
            charge_roll_fee=charge_roll_fee,
            fee_multiplier=fee_multiplier,
        )
        return s, leds
    dz = 0.02 if symmetric_deadzone is None else symmetric_deadzone
    s, leds, _ = run_zero_hardcode_cluster_conserved(
        variant_name,
        oos_fast_cache=oos_fast_cache,
        sleeve_daily_ret=sleeve_daily_ret,
        is_meta_stats=is_meta_stats,
        pit_assets=pit_assets,
        ep_map=ep_hr,
        is_liq_stats=is_liq_stats,
        price_ret_df=price_ret_df,
        scheme="hr",
        min_nf=5,
        min_ep=10,
        min_shp_fs=0.45,
        min_shp_hr=0.52,
        min_r=1.48,
        min_sm=4.5,
        min_vol_wan=2.0,
        max_per_corr_cluster=2,
        corr_thresh=0.80,
        cluster_damp_coeff=0.15,
        sector_damp_power=0.45,
        symmetric_deadzone=dz,
        target_portfolio_vol=0.14,
        adjust_roll_gaps=adjust_roll_gaps,
        charge_roll_fee=charge_roll_fee,
        fee_multiplier=fee_multiplier,
    )
    return s, leds


def build_roll_and_cost_stress_tables(
    *,
    oos_fast_cache: dict,
    sleeve_daily_ret: dict,
    is_meta_stats: dict,
    pit_assets: dict,
    ep_hr: dict,
    is_liq_stats: dict,
    price_ret_df: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """构建换月跳空剥离前后对比表、1x/1.5x/2x/3x 冲击成本矩阵与死区防抖保护表。"""
    attach_oos_roll_masks(oos_fast_cache)
    common_kw = {
        "oos_fast_cache": oos_fast_cache,
        "sleeve_daily_ret": sleeve_daily_ret,
        "is_meta_stats": is_meta_stats,
        "pit_assets": pit_assets,
        "ep_hr": ep_hr,
        "is_liq_stats": is_liq_stats,
        "price_ret_df": price_ret_df,
    }
    variants = (
        "Full60_FlatHRP_HysteresisRunner",
        "Full60_SecConserved_SymDZ_HR",
        "Full60_ZeroHardcode_SOPStage3",
        "Full60_ZeroHardcode_ClusterConserved_Optimal",
    )
    roll_rows = []
    cost_rows = []
    for var in variants:
        s_unadj, leds_unadj = _run_variant_with_execution_knobs(
            var, **common_kw, adjust_roll_gaps=False, charge_roll_fee=False, fee_multiplier=1.0
        )
        s_strip, _ = _run_variant_with_execution_knobs(
            var, **common_kw, adjust_roll_gaps=True, charge_roll_fee=False, fee_multiplier=1.0
        )
        s_roll, leds_roll = _run_variant_with_execution_knobs(
            var, **common_kw, adjust_roll_gaps=True, charge_roll_fee=True, fee_multiplier=1.0
        )
        tot_rolls, act_cross = _count_roll_crossings(oos_fast_cache, leds_roll)
        roll_rows.append(
            {
                "experiment": var,
                "total_roll_events": tot_rolls,
                "active_roll_crossings": act_cross,
                "unadj_gross_pct": round(float(s_unadj["gross_ret_pct"]), 3),
                "roll_adj_gross_pct": round(float(s_roll["gross_ret_pct"]), 3),
                "roll_gap_fake_pnl_pct": round(float(s_unadj["gross_ret_pct"] - s_roll["gross_ret_pct"]), 3),
                "unadj_cost_pct": round(float(s_unadj["cost_ret_pct"]), 3),
                "roll_adj_cost_pct": round(float(s_roll["cost_ret_pct"]), 3),
                "roll_friction_cost_pct": round(float(s_roll["cost_ret_pct"] - s_unadj["cost_ret_pct"]), 3),
                "unadj_net_pct": round(float(s_unadj["net_ret_pct"]), 3),
                "strip_only_net_pct": round(float(s_strip["net_ret_pct"]), 3),
                "roll_adj_net_pct": round(float(s_roll["net_ret_pct"]), 3),
                "unadj_sharpe": round(float(s_unadj["sharpe"]), 3),
                "strip_only_sharpe": round(float(s_strip["sharpe"]), 3),
                "roll_adj_sharpe": round(float(s_roll["sharpe"]), 3),
                "sharpe_delta": round(float(s_roll["sharpe"] - s_unadj["sharpe"]), 3),
                "roll_adj_max_dd_pct": round(float(s_roll["max_dd_pct"]), 3),
                "roll_adj_calmar": round(float(s_roll["calmar"]), 3),
            }
        )
        base_1x_shp = float(s_roll["sharpe"])
        base_1x_cost = float(s_roll["cost_ret_pct"])
        be_mult = round(float(s_roll["gross_ret_pct"]) / base_1x_cost, 2) if base_1x_cost > 1e-9 else 0.0
        for mult in (1.0, 1.5, 2.0, 3.0):
            s_m = (
                s_roll
                if mult == 1.0
                else _run_variant_with_execution_knobs(
                    var, **common_kw, adjust_roll_gaps=True, charge_roll_fee=True, fee_multiplier=mult
                )[0]
            )
            ret_pct = round(100.0 * float(s_m["sharpe"]) / base_1x_shp, 1) if abs(base_1x_shp) > 1e-9 else 0.0
            cost_rows.append(
                {
                    "experiment": var,
                    "cost_multiplier": f"{mult:.1f}x",
                    "gross_ret_pct": round(float(s_m["gross_ret_pct"]), 2),
                    "cost_ret_pct": round(float(s_m["cost_ret_pct"]), 2),
                    "net_ret_pct": round(float(s_m["net_ret_pct"]), 2),
                    "ann_ret_pct": round(float(s_m["ann_ret_pct"]), 2),
                    "max_dd_pct": round(float(s_m["max_dd_pct"]), 2),
                    "sharpe": round(float(s_m["sharpe"]), 2),
                    "calmar": round(float(s_m["calmar"]), 2),
                    "ann_turnover": round(float(s_m["ann_turnover"]), 2),
                    "sharpe_retention_pct": ret_pct,
                    "breakeven_fee_multiple": be_mult,
                }
            )

    dz_rows = []
    for var in ("Full60_ZeroHardcode_SOPStage3", "Full60_ZeroHardcode_ClusterConserved_Optimal"):
        for dz in (0.02, 0.03, 0.04):
            s_1x, _ = _run_variant_with_execution_knobs(
                var,
                **common_kw,
                symmetric_deadzone=dz,
                adjust_roll_gaps=True,
                charge_roll_fee=True,
                fee_multiplier=1.0,
            )
            c1 = float(s_1x["cost_ret_pct"])
            be_dz = round(float(s_1x["gross_ret_pct"]) / c1, 2) if c1 > 1e-9 else 0.0
            for mult in (1.0, 1.5, 2.0, 3.0):
                s_dz = (
                    s_1x
                    if mult == 1.0
                    else _run_variant_with_execution_knobs(
                        var,
                        **common_kw,
                        symmetric_deadzone=dz,
                        adjust_roll_gaps=True,
                        charge_roll_fee=True,
                        fee_multiplier=mult,
                    )[0]
                )
                dz_rows.append(
                    {
                        "architecture": var,
                        "symmetric_deadzone": f"{dz * 100.0:.1f}%",
                        "cost_multiplier": f"{mult:.1f}x",
                        "gross_ret_pct": round(float(s_dz["gross_ret_pct"]), 2),
                        "cost_ret_pct": round(float(s_dz["cost_ret_pct"]), 2),
                        "net_ret_pct": round(float(s_dz["net_ret_pct"]), 2),
                        "sharpe": round(float(s_dz["sharpe"]), 2),
                        "max_dd_pct": round(float(s_dz["max_dd_pct"]), 2),
                        "calmar": round(float(s_dz["calmar"]), 2),
                        "ann_turnover": round(float(s_dz["ann_turnover"]), 2),
                        "breakeven_fee_multiple": be_dz,
                    }
                )
    return pd.DataFrame(roll_rows), pd.DataFrame(cost_rows), pd.DataFrame(dz_rows)


def generate_html_report(
    ablation_df: pd.DataFrame,
    sec_df: pd.DataFrame,
    sym_df: pd.DataFrame,
    sat_df: pd.DataFrame,
    ledger_map: dict[str, list[dict]],
    output_path: Path,
    *,
    roll_comp_df: pd.DataFrame | None = None,
    cost_stress_df: pd.DataFrame | None = None,
    deadzone_df: pd.DataFrame | None = None,
) -> None:
    """生成自包含交互式 Plotly HTML 报告。"""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "1.0x 历史滚动验证累计净值曲线对比 (2025.01 ~ 2026.09, 按实际交易日统计)",
            "3.0x 机构目标敞口累计净收益对比",
            "6大板块净收益贡献对比：Flat-60 vs ZeroHC-SOPStage3-60 vs ZeroHC-ClusterConserved-60",
            "活跃品种净收益贡献排序（ZeroHC-ClusterConserved-60 vs Flat-60稀释）",
        ),
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )

    palette = {
        "Full60_FlatHRP_StaticExit": "#7f8c8d",
        "Full60_FlatHRP_FullStructExit": "#2980b9",
        "Full60_FlatHRP_HysteresisRunner": "#d35400",
        "Full60_SecConserved_SymDZ_HR": "#8e44ad",
        "Full60_ZeroHardcode_SOPStage3": "#16a085",
        "Full60_ZeroHardcode_ClusterConserved_Optimal": "#c0392b",
    }

    for k, color in palette.items():
        if k not in ledger_map:
            continue
        leds = ledger_map[k]
        d_ret = pd.concat([led["daily_returns"] for led in leds]).sort_index()
        cum_ret = d_ret.cumsum() * 100.0
        fig.add_trace(
            go.Scatter(
                x=cum_ret.index,
                y=cum_ret.values,
                mode="lines",
                name=k,
                line={"color": color, "width": 3.2 if "ClusterConserved" in k or "Optimal" in k else 2.0},
            ),
            row=1,
            col=1,
        )

    for k, color in (
        ("Full60_FlatHRP_StaticExit_3x", "#7f8c8d"),
        ("Full60_FlatHRP_FullStructExit_3x", "#2980b9"),
        ("Full60_FlatHRP_HysteresisRunner_3x", "#d35400"),
        ("Full60_SecConserved_SymDZ_HR_3x", "#8e44ad"),
        ("Full60_ZeroHardcode_SOPStage3_3x", "#16a085"),
        ("Full60_ZeroHardcode_ClusterConserved_3x", "#c0392b"),
    ):
        if k not in ledger_map:
            continue
        leds = ledger_map[k]
        d_ret = pd.concat([led["daily_returns"] for led in leds]).sort_index()
        cum_ret = d_ret.cumsum() * 100.0
        fig.add_trace(
            go.Scatter(
                x=cum_ret.index,
                y=cum_ret.values,
                mode="lines",
                name=k,
                line={"color": color, "width": 3.2 if "ClusterConserved" in k else 2.0, "dash": "solid"},
            ),
            row=1,
            col=2,
        )

    # Subplot 3: Sector Attribution Bar Chart
    fig.add_trace(
        go.Bar(
            x=sec_df["sector"],
            y=sec_df["flat60_net_pct"],
            name="Flat-60 等权HRP (净收益%)",
            marker_color="#e67e22",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=sec_df["sector"],
            y=sec_df["sop60_net_pct"],
            name="ZeroHC-SOPStage3-60 统一门禁配置 (净收益%)",
            marker_color="#16a085",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=sec_df["sector"],
            y=sec_df["zerohc60_net_pct"],
            name="ZeroHC-ClusterConserved-60 固定参数配置 (净收益%)",
            marker_color="#c0392b",
        ),
        row=2,
        col=1,
    )

    # Subplot 4: Top active symbols comparison
    active_sym = sym_df[
        (sym_df["zerohc60_ann_turn"] > 0.01)
        | (sym_df["sop60_ann_turn"] > 0.01)
        | (sym_df["flat60_net_pct"].abs() > 0.15)
    ].sort_values("zerohc60_net_pct", ascending=False)
    labels = [f"{r.asset_id}({r.cn_name})" for _, r in active_sym.iterrows()]
    fig.add_trace(
        go.Bar(
            x=labels,
            y=active_sym["zerohc60_net_pct"],
            name="ZeroHC-ClusterConserved-60 单品种净贡献(%)",
            marker_color="#c0392b",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=active_sym["flat60_net_pct"],
            name="Flat-60 单品种净贡献(%)",
            marker_color="#e67e22",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title="全商品池（方案B：60品种）零先验硬编码重构：F簇因果截面去噪 + 无监督产业链聚类与预算守恒实证报告",
        height=980,
        template="plotly_white",
        barmode="group",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
    )

    chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    ablation_show = ablation_df.drop(columns=["daily_series"], errors="ignore")
    extra_sections = ""
    if roll_comp_df is not None:
        extra_sections += (
            '<div class="card"><h2>四、主连合约换月跳空（Roll Gap）剥离与双边移仓成本对比表</h2>'
            f"{roll_comp_df.to_html(index=False)}</div>\n"
        )
    if cost_stress_df is not None:
        extra_sections += (
            '<div class="card"><h2>五、多倍滑点/冲击成本（1.0x / 1.5x / 2.0x / 3.0x）压力测试与盈亏平衡费率倍数表</h2>'
            f"{cost_stress_df.to_html(index=False)}</div>\n"
        )
    if deadzone_df is not None:
        extra_sections += (
            '<div class="card"><h2>六、对称调仓死区（2% vs 3% vs 4%）在高滑点环境下的换手抑制与净值保护效应表</h2>'
            f"{deadzone_df.to_html(index=False)}</div>\n"
        )
    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>60商品CTA修正后历史验证：完整执行证据与严格准入</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 24px; color: #2c3e50; background: #f8f9fa; }}
h1, h2, h3 {{ color: #1a252f; }}
.card {{ background: #fff; border-radius: 8px; padding: 20px; margin-bottom: 24px; box-shadow: 0 2px 6px rgba(0,0,0,0.06); }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin-top: 10px; }}
th, td {{ border: 1px solid #dfe6e9; padding: 8px 10px; text-align: right; }}
th {{ background: #f1f2f6; text-align: center; font-weight: 600; }}
td:first-child, td:nth-child(2), td:nth-child(3) {{ text-align: left; }}
.highlight {{ background: #e8f8f5; font-weight: 600; }}
</style>
</head>
<body>
<h1>60商品CTA修正后历史验证：完整执行证据与严格准入</h1>
<div class="card">
<p>本报告重新计算真实连续参考价格上的信号与完整执行事件，并专项纳入主连换月跳空（Roll Gap）剥离、双边移仓摩擦扣减与 1.0x~3.0x 多倍滑点压力测试。历史2025–2026区间属于回溯验证，2026-10-01起的数据封存。</p>
<h2>一、核心消融实验总表（2025.01 ~ 2026.09，历史滚动验证区间，原生5m网格计价）</h2>
{ablation_show.to_html(index=False)}
</div>
<div class="card">
{chart_html}
</div>
<div class="card">
<h2>二、6大商品板块样本外归因对比表（Flat-60等权HRP vs ZeroHC-SOPStage3-60 vs ZeroHC-ClusterConserved-60）</h2>
{sec_df.to_html(index=False)}
</div>
<div class="card">
<h2>三、全商品池 60 品种样本内 SOP Stage-3 准入门禁指标与样本外表现表</h2>
{sat_df.to_html(index=False)}
</div>
{extra_sections}</body>
</html>
"""
    output_path.write_text(html_doc, encoding="utf-8")


def save_execution_ledgers(ledger_map: dict[str, list[dict]], full_fingerprint) -> None:
    """Persist replayable fold evidence bound solely to the unified Full-60 inputs."""
    manifests = {"full60": full_fingerprint.manifest()}
    for manifest in manifests.values():
        for record in (*manifest["sources"], *manifest["data"]):
            if file_sha256(record["path"]) != record["sha256"]:
                raise UnverifiedArtifactError("Research inputs changed before publishing execution ledgers; rebuild")
    fingerprint = build_fingerprint(
        source_paths=[record["path"] for manifest in manifests.values() for record in manifest["sources"]],
        data_paths=[record["path"] for manifest in manifests.values() for record in manifest["data"]],
        config={"input_studies": manifests, "ledger_protocol": "next_open_fixed_initial_capital"},
    )
    required = {
        "execution_prices_df",
        "returns_df",
        "weights_df",
        "tradable_mask",
        "fee_rates",
        "gross_returns_df",
        "cost_returns_df",
        "turnover_df",
        "daily_returns",
        "bar_net_returns",
    }
    for variant, ledgers in ledger_map.items():
        if len(ledgers) != len(FOLDS):
            raise ValueError(f"Missing execution folds for {variant}")
        for fold, ledger in zip(FOLDS, ledgers, strict=True):
            missing = required - ledger.keys()
            if missing:
                raise ValueError(f"Incomplete execution evidence for {variant}/{fold['id']}: {sorted(missing)}")
            payload = {**ledger, "variant": variant, "fold": fold["id"], "fold_settings": dict(fold)}
            write_verified_artifact(
                OUT_DIR / "ledgers" / f"{variant}_{fold['id']}.pkl",
                pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL),
                fingerprint,
                output_dir=OUT_DIR,
                readonly_roots=(SOURCE_DIR,),
            )


def main() -> None:
    t0 = time.time()
    validate_output_directory(OUT_DIR, (SOURCE_DIR,))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 96)
    print("启动 Full-60（6大商品板块 60个流动性合格商品期货）全链路消融实证与归因分析...")
    print("=" * 96)

    univ_df = build_universe_60_summary_table()
    univ_csv = OUT_DIR / "full60_universe_summary.csv"
    univ_df.to_csv(univ_csv, index=False)
    print(f"[1/5] 已生成 60 品种流动性与数据覆盖表: {univ_csv.name} ({len(univ_df)} 个品种)")

    b = ensure_full60_oos_bundles(force_rebuild=False)

    oos_fast_cache = b["oos_fast_cache"]
    sleeve_daily_ret = b["sleeve_daily_ret"]
    is_meta_stats = b["is_meta_stats"]
    ep_fs, ep_full, ep_hr = b["ep_fs"], b["ep_full"], b["ep_hr"]
    pit_assets = b["pit_assets"]
    ep_60_map_all = {"fs": ep_fs, "full": ep_full, "hr": ep_hr}

    is_liq_stats, price_ret_df = compute_is_liquidity_and_price_returns(b["daily_frames"])

    results = []
    ledger_map: dict[str, list[dict]] = {}

    # 1. Full-60 朴素等权 HRP 系列 (展示板块风险预算膨胀 + 产业链影子品种稀释陷阱)
    for code, exp_name in (
        ("fs", "Full60_FlatHRP_StaticExit"),
        ("full", "Full60_FlatHRP_FullStructExit"),
        ("hr", "Full60_FlatHRP_HysteresisRunner"),
    ):
        s, leds = run_flat_60_alloc(
            exp_name,
            oos_fast_cache,
            pit_assets,
            ep_60_map_all[code],
            scheme=code,
            sector_damp_power=0.50,
            symmetric_deadzone=0.0,
        )
        s["active_universe"] = "Full-60 Flat (58/60)"
        results.append(s)
        ledger_map[exp_name] = leds

    # 2. Full-60 板块风险预算守恒 (1/K^0.9) + 对称调仓死区 (展示被动再平衡摩擦抑制)
    s_sec, leds_sec = run_flat_60_alloc(
        "Full60_SecConserved_SymDZ_HR",
        oos_fast_cache,
        pit_assets,
        ep_hr,
        scheme="hr",
        sector_damp_power=0.90,
        symmetric_deadzone=0.025,
    )
    s_sec["active_universe"] = "Full-60 SecConserved (58/60)"
    results.append(s_sec)
    ledger_map["Full60_SecConserved_SymDZ_HR"] = leds_sec

    # 3. Full-60 100% 零先验硬编码 全池统一 SOP 阶段三门禁基线
    s_zhc, leds_zhc, sel_zhc = run_zero_hardcode_sop_stage3(
        "Full60_ZeroHardcode_SOPStage3",
        oos_fast_cache=oos_fast_cache,
        sleeve_daily_ret=sleeve_daily_ret,
        is_meta_stats=is_meta_stats,
        pit_assets=pit_assets,
        ep_map=ep_hr,
        scheme="hr",
        min_nf=5,
        min_shp_fs=0.40,
        min_shp_hr=0.50,
        min_r=1.45,
        min_sm=4.0,
        sector_damp_power=0.50,
        symmetric_deadzone=0.02,
    )
    s_zhc["active_universe"] = f"ZeroHC-SOP (F1:{len(sel_zhc['Fold1_2025'])}, F2:{len(sel_zhc['Fold2_2026'])})"
    results.append(s_zhc)
    ledger_map["Full60_ZeroHardcode_SOPStage3"] = leds_zhc

    # 4. Full-60 100% 零先验硬编码 + F簇因果截面去噪 + 无监督产业链共线性聚类 + 板块风险预算守恒最优架构
    s_zhc_opt, leds_zhc_opt, sel_zhc_opt = run_zero_hardcode_cluster_conserved(
        "Full60_ZeroHardcode_ClusterConserved_Optimal",
        oos_fast_cache=oos_fast_cache,
        sleeve_daily_ret=sleeve_daily_ret,
        is_meta_stats=is_meta_stats,
        pit_assets=pit_assets,
        ep_map=ep_hr,
        is_liq_stats=is_liq_stats,
        price_ret_df=price_ret_df,
        scheme="hr",
        min_nf=5,
        min_ep=10,
        min_shp_fs=0.45,
        min_shp_hr=0.52,
        min_r=1.48,
        min_sm=4.5,
        min_vol_wan=2.0,
        max_per_corr_cluster=2,
        corr_thresh=0.80,
        cluster_damp_coeff=0.15,
        sector_damp_power=0.45,
        symmetric_deadzone=0.02,
        target_portfolio_vol=0.14,
    )
    s_zhc_opt["active_universe"] = (
        f"ZeroHC-Cluster (F1:{len(sel_zhc_opt['Fold1_2025'])}, F2:{len(sel_zhc_opt['Fold2_2026'])})"
    )
    results.append(s_zhc_opt)
    ledger_map["Full60_ZeroHardcode_ClusterConserved_Optimal"] = leds_zhc_opt

    # 5. 3.0x 机构目标敞口对比
    for base_key, name_3x in (
        ("Full60_FlatHRP_StaticExit", "Full60_FlatHRP_StaticExit_3x"),
        ("Full60_FlatHRP_FullStructExit", "Full60_FlatHRP_FullStructExit_3x"),
        ("Full60_FlatHRP_HysteresisRunner", "Full60_FlatHRP_HysteresisRunner_3x"),
        ("Full60_SecConserved_SymDZ_HR", "Full60_SecConserved_SymDZ_HR_3x"),
        ("Full60_ZeroHardcode_SOPStage3", "Full60_ZeroHardcode_SOPStage3_3x"),
        ("Full60_ZeroHardcode_ClusterConserved_Optimal", "Full60_ZeroHardcode_ClusterConserved_3x"),
    ):
        leds_3x = scale_fold_ledgers_exposure(ledger_map[base_key], 3.0)
        s3 = summarize_stitched_oos(leds_3x, name_3x)
        base_row = next(r for r in results if r["experiment"] == base_key)
        for k in ("ep_count", "win_rate_pct", "payoff_r", "safety_margin_pct", "active_universe"):
            if k in base_row:
                s3[k] = base_row[k]
        results.append(s3)
        ledger_map[name_3x] = leds_3x

    save_execution_ledgers(ledger_map, b["fingerprint"])
    ablation_df = pd.DataFrame(results)
    ablation_df["validation_scope"] = "retrospective_oos_validation"
    ablation_df["execution_evidence"] = np.where(
        ablation_df["ann_turnover"].abs() > 1e-12, "historical_reference_price_trades", "no_executed_positions"
    )
    ablation_csv = OUT_DIR / "full60_ablation_comparison.csv"
    ablation_df.to_csv(ablation_csv, index=False)
    print(f"[2/5] 已生成消融对比表: {ablation_csv.name}")
    print(
        ablation_df[
            [
                "experiment",
                "active_universe",
                "gross_ret_pct",
                "cost_ret_pct",
                "net_ret_pct",
                "sharpe",
                "max_dd_pct",
                "calmar",
                "ann_turnover",
                "payoff_r",
            ]
        ].to_string(index=False)
    )
    print(
        f"      ZeroHC-Cluster 100%零先验内生准入品种: Fold1_2025={sel_zhc_opt['Fold1_2025']} | Fold2_2026={sel_zhc_opt['Fold2_2026']}"
    )
    print(
        f"      ZeroHC-SOPStage3 全池统一门禁准入品种: Fold1_2025={sel_zhc['Fold1_2025']} | Fold2_2026={sel_zhc['Fold2_2026']}"
    )

    sec_df, sym_df = build_attribution_tables(
        ledger_map["Full60_FlatHRP_HysteresisRunner"],
        ledger_map["Full60_ZeroHardcode_SOPStage3"],
        sel_zhc,
        zhc_ledgers=ledger_map["Full60_ZeroHardcode_ClusterConserved_Optimal"],
        zhc_selected=sel_zhc_opt,
    )
    sec_csv = OUT_DIR / "full60_sector_attribution.csv"
    sym_csv = OUT_DIR / "full60_symbol_attribution.csv"
    sec_df.to_csv(sec_csv, index=False)
    sym_df.to_csv(sym_csv, index=False)
    print(f"[3/5] 已生成 6大板块归因表 ({sec_csv.name}) 与 60品种归因表 ({sym_csv.name})")

    sat_df = build_satellite_sop_table(is_meta_stats, sleeve_daily_ret, sel_zhc, zhc_selected=sel_zhc_opt)
    sat_csv = OUT_DIR / "full60_satellite_sop_admission.csv"
    sat_df.to_csv(sat_csv, index=False)
    print(f"[4/6] 已生成全商品池 60 品种 SOP Stage-3 准入明细表: {sat_csv.name}")

    roll_comp_df, cost_stress_df, deadzone_df = build_roll_and_cost_stress_tables(
        oos_fast_cache=oos_fast_cache,
        sleeve_daily_ret=sleeve_daily_ret,
        is_meta_stats=is_meta_stats,
        pit_assets=pit_assets,
        ep_hr=ep_hr,
        is_liq_stats=is_liq_stats,
        price_ret_df=price_ret_df,
    )
    roll_csv = OUT_DIR / "full60_roll_gap_comparison.csv"
    cost_csv = OUT_DIR / "full60_cost_stress_matrix.csv"
    dz_csv = OUT_DIR / "full60_deadzone_protection_matrix.csv"
    roll_comp_df.to_csv(roll_csv, index=False)
    cost_stress_df.to_csv(cost_csv, index=False)
    deadzone_df.to_csv(dz_csv, index=False)
    print(
        f"[5/6] 已生成换月跳空剥离对比表 ({roll_csv.name})、冲击成本压力矩阵 ({cost_csv.name}) 与死区防抖保护表 ({dz_csv.name})"
    )
    print(roll_comp_df.to_string(index=False))
    print(cost_stress_df.to_string(index=False))
    print(deadzone_df.to_string(index=False))

    html_path = OUT_DIR / "full60_cta_expansion_report.html"
    generate_html_report(
        ablation_df,
        sec_df,
        sym_df,
        sat_df,
        ledger_map,
        html_path,
        roll_comp_df=roll_comp_df,
        cost_stress_df=cost_stress_df,
        deadzone_df=deadzone_df,
    )
    print(f"[6/6] 已生成交互式 HTML 报告: {html_path} (总耗时 {time.time() - t0:.2f}s)")


if __name__ == "__main__":
    main()
