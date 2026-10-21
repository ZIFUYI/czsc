"""Fixed sub-chain pooling for oversized Full-60 commodity sectors."""

from examples.multi_factor_cta.forensic_validator import (
    COMMODITY_SUBCHAIN_FAMILIES,
    refine_commodity_pool_groups,
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
