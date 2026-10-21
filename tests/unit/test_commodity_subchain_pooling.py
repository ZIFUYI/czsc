"""Fixed sub-chain pooling for oversized Full-60 commodity sectors."""

from examples.multi_factor_cta.forensic_validator import (
    COMMODITY_SUBCHAIN_FAMILIES,
    refine_commodity_pool_groups,
    run_stratified_pooled_gate1b_audit,
)

_SECTORS = {
    "commodity:energy_chem": "sc lu fu bu ta px eg pf pr ma ur l pp v eb pg sh ru br nr sp ec",
    "commodity:agriculture": "m y p oi rm a b sr cf ap cj pk lh jd cs c",
    "commodity:ferrous": "rb hc i j jm fg sa sm sf",
    "commodity:new_energy": "si lc ps",
    "commodity:base_metals": "cu al zn ni sn pb ao ss",
    "commodity:precious_metals": "au ag",
}


def _groups(upper: bool = False) -> dict[str, str]:
    return {a: (s.upper() if upper else s) for s, codes in _SECTORS.items() for a in codes.split()}


def test_subchains_are_disjoint_and_meet_floor():
    for chains in COMMODITY_SUBCHAIN_FAMILIES.values():
        seen = [a for codes in chains.values() for a in codes]
        assert len(seen) == len(set(seen))
        assert all(len(codes) >= 5 for codes in chains.values())


def test_full60_refinement_sizes_and_unmapped_assets():
    for upper in (False, True):
        refined = refine_commodity_pool_groups(_groups(upper))
        sizes: dict[str, int] = {}
        for label in refined.values():
            sizes[label.lower()] = sizes.get(label.lower(), 0) + 1
        assert sizes["commodity:chain_crude_aromatics_polyester"] == 9
        assert sizes["commodity:chain_coal_chlor_polyolefin_rubber"] == 12
        assert sizes["commodity:chain_oils_oilseeds"] == 7
        assert sizes["commodity:chain_softs_grains_livestock"] == 9
        assert sizes["commodity:chain_ferrous_burden"] == 5
        assert sizes["commodity:chain_buildmat_newenergy"] == 7
        assert refined["ec"].lower() == "commodity:energy_chem"  # no physical chain: never borrows a larger pool
        assert refined["cu"] == _groups(upper)["cu"] and refined["au"] == _groups(upper)["au"]  # metals untouched


def test_small_universe_keeps_legacy_sector_labels():
    groups = dict.fromkeys(["rb", "hc", "i", "j", "jm"], "commodity:ferrous")
    assert refine_commodity_pool_groups(groups) == groups
    groups.update(dict.fromkeys(["si", "lc"], "commodity:new_energy"))  # family <= 10 assets
    assert refine_commodity_pool_groups(groups) == groups


def test_full_exchange_symbol_normalization():
    raw_codes = ["rb", "hc", "i", "j", "jm", "fg", "sa", "sm", "sf"]
    new_energy_codes = ["si", "lc", "ps"]
    groups = {f"{c.upper()}9999.XSGE": "commodity:ferrous" for c in raw_codes}
    groups.update({f"KQ.m@GFEX.{c}": "commodity:new_energy" for c in new_energy_codes})
    refined = refine_commodity_pool_groups(groups)
    assert refined["RB9999.XSGE"] == "commodity:chain_ferrous_burden"
    assert refined["FG9999.XSGE"] == "commodity:chain_buildmat_newenergy"
    assert refined["KQ.m@GFEX.si"] == "commodity:chain_buildmat_newenergy"


def test_zero_event_companion_counted_as_non_positive_not_missing_audit():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    base_ts = pd.date_range("2022-01-02", periods=32, freq="4D")
    all_family_assets = ["rb", "hc", "i", "j", "jm", "fg", "sa", "sm", "sf", "si", "lc"]
    audits = {}
    for idx, asset in enumerate(all_family_assets):
        evs = [
            {
                "trigger_dt": ts + pd.Timedelta(hours=idx),
                "entry_dt": ts + pd.Timedelta(hours=idx + 1),
                "exit_dt": ts + pd.Timedelta(hours=idx + 12),
                "r_atr": float(0.45 + 0.08 * rng.standard_normal()),
                "net_ret": float(0.008 + 0.001 * rng.standard_normal()),
                "quality_w": 1.0,
                "regime": (k % 3) - 1,
            }
            for k, ts in enumerate(base_ts)
        ]
        audits[asset] = {"events": evs, "is_n_purged": len(evs)}
    # Newly-listed companion asset with 0 triggers (e.g., ps in Fold1_2025)
    audits["ps"] = {"events": [], "is_n_purged": 0}
    groups = _groups(False)
    res = run_stratified_pooled_gate1b_audit(
        asset_audits=audits,
        target_pool_group="commodity:new_energy",
        asset_groups=groups,
        target_asset="si",
    )
    assert set(res["evaluated_assets"]) == {"fg", "sa", "sm", "sf", "si", "lc", "ps"}
    assert res["pool_size"] == 7
    assert res["positive_assets"] == ["fg", "sa", "sm", "sf", "si", "lc"]
    assert abs(res["universal_ratio"] - round(6 / 7, 4)) < 1e-9
    assert res["passed"] is True

    # Fake summary-only audit (is_n_purged > 0 but events=[]) must still be rejected
    audits_fake = dict(audits)
    audits_fake["ps"] = {"events": [], "is_n_purged": 18}
    res_fake = run_stratified_pooled_gate1b_audit(
        asset_audits=audits_fake,
        target_pool_group="commodity:new_energy",
        asset_groups=groups,
        target_asset="si",
    )
    assert res_fake["passed"] is False
    assert "ps: 缺少真实events，不得用摘要生成样本" in res_fake["reason"]
