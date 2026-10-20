#!/usr/bin/env python
"""Declared 32-commodity experiment with common admission and execution accounting.

The declared universe is retained. Exposure targets inferred from the whole
sample were removed; all ten shared causal allocation schemes are reported.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from examples.multi_factor_cta.data_engine import AssetClass, AssetSpec, ExecutionMode  # noqa: E402
from examples.multi_factor_cta.legacy_research import (  # noqa: E402
    ARTIFACT_DIR,
    WORKSPACE_DIR,
    load_experiment_inputs,
    save_experiment_report,
)
from examples.multi_factor_cta.test_all_weather_50commodities_portfolio import (  # noqa: E402
    evaluate_50_weighting_schemes as evaluate_32_weighting_schemes,
)

COMMODITY_DATA_DIR = ROOT_DIR / "examples" / "results" / "commodities_5m"
CTA_OUTPUT_DIR = ARTIFACT_DIR


def build_thirty_two_commodities_universe() -> list[tuple[str, AssetSpec]]:
    specs = [
        # 1. 贵金属
        ("ag", "AG9999.XSGE", "PRECIOUS_METALS", "ag_沪银主力连续", 0.00015),
        ("au", "AU9999.XSGE", "PRECIOUS_METALS", "au_沪金主力连续", 0.00010),
        # 2. 有色金属
        ("cu", "CU9999.XSGE", "BASE_METALS", "cu_沪铜主力连续", 0.00015),
        ("al", "AL9999.XSGE", "BASE_METALS", "al_沪铝主力连续", 0.00015),
        ("sn", "SN9999.XSGE", "BASE_METALS", "sn_沪锡主力连续", 0.00015),
        ("zn", "ZN9999.XSGE", "BASE_METALS", "zn_沪锌主力连续", 0.00015),
        ("ni", "NI9999.XSGE", "BASE_METALS", "ni_沪镍主力连续", 0.00015),
        # 3. 黑色建材
        ("rb", "RB9999.XSGE", "FERROUS", "rb_螺纹钢主力连续", 0.00015),
        ("i", "I9999.XDCE", "FERROUS", "i_铁矿石主力连续", 0.00015),
        ("jm", "JM9999.XDCE", "FERROUS", "jm_焦煤主力连续", 0.00015),
        ("j", "J9999.XDCE", "FERROUS", "j_焦炭主力连续", 0.00015),
        ("sa", "SA9999.XZCE", "FERROUS", "sa_纯碱主力连续", 0.00015),
        ("hc", "HC9999.XSGE", "FERROUS", "hc_热卷主力连续", 0.00015),
        ("fg", "FG9999.XZCE", "FERROUS", "fg_玻璃主力连续", 0.00015),
        # 4. 能源化工
        ("sc", "SC9999.XINE", "ENERGY_CHEM", "sc_原油主力连续", 0.00015),
        ("ta", "TA9999.XZCE", "ENERGY_CHEM", "ta_PTA主力连续", 0.00015),
        ("ma", "MA9999.XZCE", "ENERGY_CHEM", "ma_甲醇主力连续", 0.00015),
        ("ru", "RU9999.XSGE", "ENERGY_CHEM", "ru_橡胶主力连续", 0.00015),
        ("fu", "FU9999.XSGE", "ENERGY_CHEM", "fu_燃油主力连续", 0.00015),
        ("eb", "EB9999.XDCE", "ENERGY_CHEM", "eb_苯乙烯主力连续", 0.00015),
        ("eg", "EG9999.XDCE", "ENERGY_CHEM", "eg_乙二醇主力连续", 0.00015),
        ("pp", "PP9999.XDCE", "ENERGY_CHEM", "pp_聚丙烯主力连续", 0.00015),
        # 5. 农产品软商品
        ("m", "M9999.XDCE", "AGRICULTURE", "m_豆粕主力连续", 0.00015),
        ("p", "P9999.XDCE", "AGRICULTURE", "p_棕榈油主力连续", 0.00015),
        ("y", "Y9999.XDCE", "AGRICULTURE", "y_豆油主力连续", 0.00015),
        ("oi", "OI9999.XZCE", "AGRICULTURE", "oi_菜油主力连续", 0.00015),
        ("lh", "LH9999.XDCE", "AGRICULTURE", "lh_生猪主力连续", 0.00015),
        ("sr", "SR9999.XZCE", "AGRICULTURE", "sr_白糖主力连续", 0.00015),
        ("cf", "CF9999.XZCE", "AGRICULTURE", "cf_棉花主力连续", 0.00015),
        # 6. 新能源材料
        ("lc", "LC9999.GFEX", "NEW_ENERGY", "lc_碳酸锂主力连续", 0.00015),
        ("ps", "PS9999.GFEX", "NEW_ENERGY", "ps_多晶硅主力连续", 0.00015),
        ("si", "SI9999.GFEX", "NEW_ENERGY", "si_工业硅主力连续", 0.00015),
    ]

    out = []
    for short_id, symbol, sector, name, fee in specs:
        p15 = COMMODITY_DATA_DIR / f"{short_id}_15m_2020_2026.parquet"
        out.append(
            (
                short_id,
                AssetSpec(
                    symbol=symbol,
                    name=name,
                    asset_class=AssetClass.COMMODITY_FUTURES,
                    sector=sector,
                    base_tf="15m",
                    eligible_tfs=("1d", "60m", "30m", "15m"),
                    fee_rate=fee,
                    has_night_session=True,
                    execution_mode=ExecutionMode.LONG_SHORT,
                    data_path=p15,
                ),
            )
        )
    return out


def main() -> None:
    """Run admitted factors and save net curves and their admission record."""
    inputs = load_experiment_inputs(build_thirty_two_commodities_universe(), workspace_dir=WORKSPACE_DIR)
    results = evaluate_32_weighting_schemes(
        inputs["price_returns"],
        inputs["sectors"],
        asset_positions=inputs["positions"],
        asset_fee_rates=inputs["fees"],
    )
    save_experiment_report(
        results,
        inputs,
        ARTIFACT_DIR / "commodity_32assets_portfolio_chart.html",
        title="32 品种声明资产池：统一准入、最终成交扣费",
    )


if __name__ == "__main__":
    main()
