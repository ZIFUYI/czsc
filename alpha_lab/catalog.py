"""
Rust 原生信号资产目录与基于 Python AST 的因子依赖自动血缘分析器（Issue #20）。

核心能力：
1. `_FactorAstScanner(ast.NodeVisitor)`：对 `ATOMIC_FACTOR_DEFS` 中每个 `fdef.build` 函数进行静态 AST 解析，
   自动提取 `state`、`strength`、`oi`、`cs`、`htf` 五类底层资源引用，并计算规范化源码指纹 `code_sha256`；
2. 对接 `czsc._native.list_all_signals()` 与镜像普查元数据（`exact_mirror`、`active_rate`、`n_states`），
   提供 `get_factor_lineage_table()`、`get_signal_usage_summary()` 与 `list_unconsumed_mirror_signals()`。
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

import czsc._native as _native
from examples.alpha_mining_agent.hypothesis_bank import (
    ARCH_A,
    ARCH_B,
    ARCH_C,
    ARCH_D,
    ARCH_E,
    ARCH_F,
    STRENGTH_SIGNAL_KINDS,
)
from examples.multi_factor_cta.atomic_factors import (
    ATOMIC_FACTOR_DEFS,
    STATE_SIGNAL_MANIFEST,
    AtomicFactorDef,
)

# STATE_SIGNAL_MANIFEST 中 50 个 state_key 到 Rust 原生信号函数名的精确映射
NATIVE_CZSC_SIGNAL_MAP: dict[str, str] = {
    "ma_cls": "tas_ma_base_V221203",
    "td_v2": "coo_td_V221111",
    "clv": "clv_up_dw_line_V230605",
    "ntmdk": "ntmdk_V230824",
    "kcatr": "kcatr_up_dw_line_V230823",
    "fake_break": "bar_fake_break_V230204",
    "ma_joint": "tas_ma_joint_V260520",
    "double_ma": "tas_double_ma_V240208",
    "dif_layer": "tas_dif_layer_V241010",
    "dif_zero": "tas_dif_zero_V240612",
    "macd_bc_ubi": "tas_macd_bc_ubi_V230804",
    "macd_bc": "tas_macd_bc_V240307",
    "bar_acc": "bar_accelerate_V240428",
    "polyfit": "bar_polyfit_V240428",
    "plr_long": "bar_plr_V240427",
    "plr_short": "bar_plr_V240427",
    "tnr": "bar_tnr_V230630",
    "zfzd": "bar_zfzd_V241013",
    "overlap": "cxt_overlap_V240612",
    "bi_end_618": "cxt_bi_end_V230618",
    "zdy_dif": "zdy_dif_V230528",
    "macd_shrink": "zdy_macd_V230519",
    "xl_candle": "xl_bar_basis_V240412",
    "xl_trend": "xl_bar_trend_V240623",
    "xl_trend_v2": "xl_bar_trend_V240623",
    "sar": "coo_sar_V230325",
    "cvolp": "cvolp_up_dw_line_V230612",
    "pressure": "pressure_support_V240222",
    "vol_ts_v1": "vol_ti_suo_V221216",
    "vol_ts_v2": "vol_ti_suo_V221216",
    "vol_gd": "vol_gao_di_V221218",
    "boll_pwr_v1": "tas_boll_power_V221112",
    "boll_pwr_v2": "tas_boll_power_V221112",
    "ds_mid_v1": "cxt_ds_midline_break_V260521",
    "ds_mid_v2": "cxt_ds_midline_break_V260521",
    "sym_zs_v1": "byi_symmetry_zs_V221107",
    "sym_zs_v2": "byi_symmetry_zs_V221107",
    "bs2_sma": "cxt_second_bs_V230320",
    "bs2_macd": "byi_second_bs_V230324",
    "bs3_18": "cxt_third_bs_V230318",
    "bs3_19": "cxt_third_bs_V230319",
    "be_pause_v1": "byi_bi_end_V230106",
    "be_verif_v1": "byi_bi_end_V230107",
    "five_bi": "cxt_five_bi_V230619",
    "double_zs": "cxt_double_zs_V230311",
    "td9_v1": "bar_td9_V240616",
    "zs_sp_v1": "zdy_zs_space_V230421",
    "zs_fm_v1": "zdy_zs_V230423",
    "acc_w10": "bar_accelerate_V221110",
    "doji_rev": "xl_bar_trend_V240329",
}

STRENGTH_RUST_SIGNAL_MAP: dict[str, str] = {
    "fx": "strength_fx",
    "bi_div": "strength_bi_div",
    "zs_revert": "strength_zs_revert",
    "zs_break": "strength_zs_break",
    "flow": "strength_flow",
    "squeeze": "strength_squeeze",
}

# Rust 信号严格镜像对称普查元数据: signal_name -> (exact_mirror, active_rate, n_states)
_RUST_SIGNAL_CENSUS_META: dict[str, tuple[bool, float, int]] = {
    "adtm_up_dw_line_V230603": (True, 0.058, 3),
    "amv_up_dw_line_V230603": (True, 1.0, 4),
    "asi_up_dw_line_V230603": (True, 0.9147, 35),
    "bar_accelerate_V221110": (True, 0.1182, 4),
    "bar_accelerate_V221118": (False, 0.9916, 4),
    "bar_accelerate_V240428": (True, 0.2566, 4),
    "bar_amount_acc_V230214": (True, 0.3587, 4),
    "bar_big_solid_V230215": (True, 0.9552, 4),
    "bar_bpm_V230227": (True, 1.0, 4),
    "bar_break_V240428": (True, 0.0133, 2),
    "bar_channel_V230508": (True, 0.2958, 8),
    "bar_classify_V240606": (True, 1.0, 20),
    "bar_classify_V240607": (True, 1.0, 4),
    "bar_decision_V240608": (True, 0.0112, 2),
    "bar_decision_V240616": (True, 0.135, 2),
    "bar_dual_thrust_V230403": (True, 0.0133, 4),
    "bar_eight_V230702": (False, 0.0, 0),
    "bar_end_V221211": (False, 0.9972, 3),
    "bar_fake_break_V230204": (True, 0.9769, 4),
    "bar_fang_liang_break_V221216": (True, 0.3587, 14),
    "bar_intraday_direction_V260424": (True, 1.0, 1),
    "bar_intraday_exit_V260424": (False, 0.249, 4),
    "bar_limit_down_V230525": (False, 0.0, 0),
    "bar_mean_amount_V221112": (False, 1.0, 20),
    "bar_operate_span_V221111": (True, 0.3979, 16),
    "bar_plr_V240427": (True, 0.0161, 2),
    "bar_polyfit_V240428": (True, 1.0, 2),
    "bar_r_breaker_V230326": (True, 0.0706, 6),
    "bar_reversal_V230227": (False, 0.0378, 3),
    "bar_section_momentum_V221112": (True, 1.0, 3),
    "bar_shuang_fei_V230507": (True, 0.2587, 12),
    "bar_single_V230214": (False, 0.0161, 1),
    "bar_single_V230506": (True, 0.1706, 2),
    "bar_td9_V240616": (True, 0.1399, 2),
    "bar_time_V230327": (True, 0.1979, 1),
    "bar_tnr_V230629": (False, 0.0, 0),
    "bar_tnr_V230630": (True, 0.01, 2),
    "bar_trend_V240209": (True, 0.2007, 2),
    "bar_triple_V230506": (True, 1.0, 4),
    "bar_vol_bs1_V230224": (False, 0.0084, 2),
    "bar_vol_grow_V221112": (True, 0.0203, 2),
    "bar_volatility_V241013": (True, 0.0462, 2),
    "bar_weekday_V230328": (False, 0.6552, 2),
    "bar_window_ps_V230731": (True, 1.0, 2),
    "bar_window_ps_V230801": (True, 0.3385, 2),
    "bar_window_std_V230731": (True, 0.0748, 2),
    "bar_zdf_V221203": (True, 0.2028, 7),
    "bar_zdt_V230331": (False, 0.0, 0),
    "bar_zfzd_V241013": (True, 0.1042, 2),
    "bar_zfzd_V241014": (True, 0.142, 2),
    "bar_zt_count_V230504": (True, 0.1217, 2),
    "bias_up_dw_line_V230618": (True, 0.1112, 2),
    "byi_bi_end_V230106": (True, 0.0818, 2),
    "byi_bi_end_V230107": (True, 0.1322, 2),
    "byi_fx_num_V230628": (True, 0.1469, 2),
    "byi_second_bs_V230324": (True, 0.142, 5),
    "byi_symmetry_zs_V221107": (True, 0.979, 2),
    "cci_decision_V240620": (True, 0.1042, 2),
    "clv_up_dw_line_V230605": (True, 0.514, 2),
    "cmo_up_dw_line_V230605": (False, 0.1385, 4),
    "coo_cci_V230323": (True, 0.5545, 4),
    "coo_kdj_V230322": (False, 1.0, 4),
    "coo_sar_V230325": (True, 0.01, 2),
    "coo_td_V221110": (True, 0.0986, 25),
    "coo_td_V221111": (True, 1.0, 12),
    "cvolp_up_dw_line_V230612": (True, 1.0, 4),
    "cxt_bi_base_V230228": (False, 0.0035, 2),
    "cxt_bi_end_V230104": (False, 0.8503, 8),
    "cxt_bi_end_V230105": (True, 0.1958, 2),
    "cxt_bi_end_V230222": (False, 0.1217, 2),
    "cxt_bi_end_V230224": (False, 1.0, 4),
    "cxt_bi_end_V230312": (True, 0.0958, 2),
    "cxt_bi_end_V230320": (True, 0.0007, 1),
    "cxt_bi_end_V230322": (False, 0.0, 0),
    "cxt_bi_end_V230324": (False, 0.0364, 5),
    "cxt_bi_end_V230618": (True, 0.0874, 2),
    "cxt_bi_end_V230815": (True, 1.0, 10),
    "cxt_bi_status_V230101": (False, 0.0, 0),
    "cxt_bi_status_V230102": (True, 0.6713, 4),
    "cxt_bi_stop_V230815": (True, 0.1524, 2),
    "cxt_bi_trend_V230824": (True, 0.2734, 2),
    "cxt_bi_trend_V230913": (True, 0.9552, 3),
    "cxt_bi_zdf_V230601": (False, 0.0727, 4),
    "cxt_bs_V240526": (False, 0.014, 2),
    "cxt_bs_V240527": (True, 0.9685, 2),
    "cxt_decision_V240526": (True, 1.0, 4),
    "cxt_decision_V240612": (True, 0.9671, 10),
    "cxt_decision_V240613": (False, 0.0259, 5),
    "cxt_decision_V240614": (False, 1.0, 10),
    "cxt_double_zs_V230311": (True, 0.01, 2),
    "cxt_ds_midline_break_V260521": (True, 0.3112, 2),
    "cxt_eleven_bi_V230622": (False, 0.0448, 5),
    "cxt_first_buy_V221126": (True, 0.1727, 2),
    "cxt_first_sell_V221126": (False, 1.0, 2),
    "cxt_five_bi_V230619": (True, 1.0, 20),
    "cxt_fx_power_V221107": (True, 1.0, 10),
    "cxt_nine_bi_V230621": (True, 1.0, 4),
    "cxt_overlap_V240526": (True, 0.0161, 2),
    "cxt_overlap_V240612": (True, 0.6594, 2),
    "cxt_range_oscillation_V230620": (True, 1.0, 2),
    "cxt_second_bs_V230320": (True, 0.4224, 2),
    "cxt_second_bs_V240524": (True, 0.514, 4),
    "cxt_seven_bi_V230620": (True, 1.0, 1),
    "cxt_third_bs_V230318": (True, 0.3056, 14),
    "cxt_third_bs_V230319": (True, 1.0, 36),
    "cxt_third_buy_V230228": (True, 0.2287, 2),
    "cxt_three_bi_V230618": (True, 1.0, 7),
    "cxt_ubi_end_V230816": (True, 0.7245, 2),
    "dema_up_dw_line_V230605": (True, 0.1545, 2),
    "demakder_up_dw_line_V230605": (False, 0.0434, 2),
    "emv_up_dw_line_V230605": (True, 0.0133, 3),
    "er_up_dw_line_V230604": (True, 0.1343, 4),
    "jcc_ci_tou_V221101": (False, 0.1063, 2),
    "jcc_fan_ji_xian_V221121": (False, 0.0, 0),
    "jcc_fen_shou_xian_V20221113": (False, 0.2427, 2),
    "jcc_gap_yin_yang_V221121": (True, 0.028, 3),
    "jcc_ping_tou_V221113": (True, 0.986, 54),
    "jcc_san_fa_V20221115": (True, 0.0014, 1),
    "jcc_san_fa_V20221118": (True, 0.2664, 2),
    "jcc_san_szx_V221122": (False, 0.0, 0),
    "jcc_san_xing_xian_V221023": (False, 0.0, 0),
    "jcc_shan_chun_V221121": (True, 1.0, 2),
    "jcc_szx_V221111": (True, 0.3538, 2),
    "jcc_ta_xing_V221124": (False, 0.5294, 11),
    "jcc_ten_mo_V221028": (False, 0.1112, 2),
    "jcc_three_crow_V221108": (False, 0.2455, 1),
    "jcc_two_crow_V221108": (True, 0.9713, 11),
    "jcc_wu_yun_gai_ding_V221101": (True, 0.0867, 2),
    "jcc_xing_xian_V221118": (True, 0.1266, 2),
    "jcc_yun_xian_V221118": (True, 0.1524, 4),
    "jcc_zhu_huo_xian_V221027": (True, 0.3497, 6),
    "kcatr_up_dw_line_V230823": (True, 0.214, 2),
    "ntmdk_V230824": (True, 0.0476, 2),
    "obv_up_dw_line_V230719": (True, 0.0329, 2),
    "obvm_line_V230610": (False, 0.865, 6),
    "pressure_support_V240222": (True, 0.3371, 2),
    "pressure_support_V240402": (False, 1.0, 3),
    "pressure_support_V240406": (True, 0.0664, 4),
    "pressure_support_V240530": (False, 0.0, 0),
    "skdj_up_dw_line_V230611": (False, 0.0, 0),
    "strength_bi_div": (True, 1.0, 4),
    "strength_flow": (True, 1.0, 8),
    "strength_fx": (True, 0.9769, 35),
    "strength_squeeze": (True, 0.1629, 2),
    "strength_zs_break": (True, 1.0, 8),
    "strength_zs_revert": (True, 0.1189, 2),
    "tas_accelerate_V230531": (True, 0.228, 2),
    "tas_angle_V230802": (True, 1.0, 5),
    "tas_atr_V230630": (True, 0.4007, 6),
    "tas_atr_break_V230424": (False, 0.0, 0),
    "tas_boll_bc_V221118": (True, 0.0434, 2),
    "tas_boll_cc_V230312": (True, 0.0434, 2),
    "tas_boll_power_V221112": (True, 1.0, 3),
    "tas_boll_vt_V230212": (False, 0.0077, 1),
    "tas_cci_base_V230402": (True, 0.9972, 6),
    "tas_cross_status_V230619": (True, 0.0692, 4),
    "tas_cross_status_V230624": (True, 0.0476, 4),
    "tas_cross_status_V230625": (True, 0.6671, 12),
    "tas_dif_layer_V241010": (True, 0.0224, 2),
    "tas_dif_zero_V240612": (True, 0.01, 2),
    "tas_dif_zero_V240614": (True, 0.0434, 1),
    "tas_dma_bs_V240608": (True, 0.0853, 2),
    "tas_double_ma_V221203": (True, 0.0573, 2),
    "tas_double_ma_V230511": (False, 0.9336, 2),
    "tas_double_ma_V240208": (True, 0.3818, 2),
    "tas_ds_ma60_support_V260521": (True, 0.2301, 2),
    "tas_first_bs_V230217": (True, 0.0587, 2),
    "tas_hlma_V230301": (False, 0.672, 13),
    "tas_kdj_base_V221101": (False, 0.0, 0),
    "tas_kdj_evc_V221201": (False, 0.0, 0),
    "tas_kdj_evc_V230401": (False, 0.3252, 2),
    "tas_low_trend_V230627": (True, 0.8287, 8),
    "tas_ma_base_V221101": (False, 0.0, 0),
    "tas_ma_base_V221203": (True, 0.0986, 2),
    "tas_ma_base_V230313": (True, 0.8434, 3),
    "tas_ma_cohere_V230512": (True, 0.5965, 2),
    "tas_ma_joint_V260520": (True, 0.01, 2),
    "tas_ma_round_V221206": (True, 1.0, 2),
    "tas_ma_system_V230513": (False, 0.107, 7),
    "tas_macd_base_V221028": (True, 1.0, 2),
    "tas_macd_base_V230320": (True, 1.0, 1),
    "tas_macd_bc_V221201": (False, 0.0, 0),
    "tas_macd_bc_V230803": (False, 0.9895, 5),
    "tas_macd_bc_V230804": (True, 0.1629, 2),
    "tas_macd_bc_V240307": (True, 0.914, 2),
    "tas_macd_bc_ubi_V230804": (True, 1.0, 8),
    "tas_macd_bs1_V230312": (True, 1.0, 2),
    "tas_macd_bs1_V230313": (True, 1.0, 2),
    "tas_macd_bs1_V230411": (True, 1.0, 3),
    "tas_macd_bs1_V230412": (True, 0.0944, 26),
    "tas_macd_change_V221105": (True, 0.2937, 1),
    "tas_macd_direct_V221106": (False, 0.0, 0),
    "tas_macd_dist_V230408": (True, 0.1189, 2),
    "tas_macd_dist_V230409": (True, 0.9895, 2),
    "tas_macd_dist_V230410": (True, 0.1476, 11),
    "tas_macd_first_bs_V221201": (True, 0.1608, 22),
    "tas_macd_first_bs_V221216": (True, 0.651, 10),
    "tas_macd_power_V221108": (True, 0.0783, 2),
    "tas_macd_second_bs_V221201": (False, 1.0, 8),
    "tas_macd_xt_V221208": (False, 1.0, 8),
    "tas_rsi_base_V230227": (True, 0.0958, 2),
    "tas_rumi_V230704": (False, 0.0552, 6),
    "tas_sar_base_V230425": (True, 0.0133, 2),
    "tas_second_bs_V230228": (False, 0.9818, 4),
    "tas_second_bs_V230303": (True, 0.1741, 2),
    "tas_slope_V231019": (False, 0.0713, 2),
    "tas_slope_break_V261003": (False, 1.0, 2),
    "vol_double_ma_V230214": (True, 0.9902, 10),
    "vol_gao_di_V221218": (True, 0.0364, 2),
    "vol_single_ma_V230214": (True, 0.0909, 2),
    "vol_ti_suo_V221216": (True, 0.2566, 2),
    "vol_window_V230731": (True, 1.0, 3),
    "vol_window_V230801": (False, 1.0, 2),
    "xl_bar_basis_V240411": (True, 0.035, 2),
    "xl_bar_basis_V240412": (True, 0.01, 2),
    "xl_bar_position_V240328": (True, 1.0, 2),
    "xl_bar_trend_V240329": (True, 0.149, 4),
    "xl_bar_trend_V240330": (True, 0.9972, 8),
    "xl_bar_trend_V240331": (False, 0.0, 0),
    "xl_bar_trend_V240623": (True, 0.01, 2),
    "zdy_bi_end_V230406": (True, 0.0308, 2),
    "zdy_bi_end_V230407": (False, 1.0, 4),
    "zdy_dif_V230527": (True, 0.493, 2),
    "zdy_dif_V230528": (True, 0.9552, 4),
    "zdy_macd_V230518": (False, 0.9685, 2),
    "zdy_macd_V230519": (True, 0.1895, 4),
    "zdy_macd_V230527": (True, 0.2378, 2),
    "zdy_macd_bc_V230422": (False, 0.1245, 4),
    "zdy_macd_bs1_V230422": (False, 0.8364, 1),
    "zdy_macd_dif_V230516": (False, 1.0, 4),
    "zdy_macd_dif_V230517": (False, 0.7483, 4),
    "zdy_macd_dif_iqr_V230521": (True, 0.1364, 20),
    "zdy_zs_V230423": (True, 0.1357, 2),
    "zdy_zs_space_V230421": (True, 0.01, 2),
}

_CS_ATTR_NAMES: frozenset[str] = frozenset(
    {
        "xs",
        "cs_rank",
        "cs_strength_eff",
        "sector_leader_dir",
        "leader_lag_gap",
        "sector_spread_zscore",
        "has_external_cs",
        "cs_available",
    }
)
_HTF_ATTR_NAMES: frozenset[str] = frozenset({"htf_dir", "htf_q"})


@dataclass(frozen=True)
class FactorLineageInfo:
    """单个连续原子因子族的 AST 静态依赖与代码哈希快照。"""

    family_id: str
    archetype: str
    stage: str
    code_sha256: str
    states: tuple[str, ...]
    strengths: tuple[str, ...]
    oi_features: tuple[str, ...]
    cs_features: tuple[str, ...]
    htf_features: tuple[str, ...]
    rust_signals: tuple[str, ...]


class _FactorAstScanner(ast.NodeVisitor):
    """静态扫描 `fdef.build(ctx, side)` AST，自动提取 state / strength / oi / cs / htf 五类底层依赖。"""

    def __init__(self) -> None:
        self.states: set[str] = set()
        self.strengths: set[str] = set()
        self.oi_features: set[str] = set()
        self.cs_features: set[str] = set()
        self.htf_features: set[str] = set()
        self._local_str_bindings: dict[str, list[str]] = {}

    def visit_Assign(self, node: ast.Assign) -> None:
        str_vals = [
            c.value
            for c in ast.walk(node.value)
            if isinstance(c, ast.Constant) and isinstance(c.value, str) and c.value in STATE_SIGNAL_MANIFEST
        ]
        if str_vals:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self._local_str_bindings[target.id] = str_vals
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Attribute) and isinstance(node.value.value, ast.Name):
            if node.value.value.id == "ctx" and node.value.attr == "states":
                sl = node.slice
                if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                    self.states.add(sl.value)
                elif isinstance(sl, ast.Name) and sl.id in self._local_str_bindings:
                    self.states.update(self._local_str_bindings[sl.id])
            elif node.value.value.id == "ctx" and node.value.attr == "strength":
                sl = node.slice
                if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                    self.strengths.add(sl.value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "_dir_eq"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            self.strengths.add(node.args[1].value)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name) and node.value.id == "ctx":
            attr = node.attr
            if attr.startswith("oi_"):
                self.oi_features.add(attr)
            elif attr in _CS_ATTR_NAMES:
                self.cs_features.add(attr)
            elif attr in _HTF_ATTR_NAMES:
                self.htf_features.add(attr)
        self.generic_visit(node)


def compute_factor_code_sha256(build_fn: Callable) -> str:
    """计算因子构建函数规范化 AST 源码的 SHA-256 指纹。"""
    try:
        src = textwrap.dedent(inspect.getsource(build_fn))
        tree = ast.parse(src)
        canonical = ast.unparse(tree)
    except (OSError, TypeError, SyntaxError):
        code_obj = getattr(build_fn, "__code__", None)
        canonical = repr(code_obj.co_code) if code_obj is not None else repr(build_fn)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def resolve_state_to_rust_signal(state_key: str) -> str:
    """将 STATE_SIGNAL_MANIFEST 中的 state_key 解析为对应的 Rust 原生信号函数名。"""
    if state_key in NATIVE_CZSC_SIGNAL_MAP:
        return NATIVE_CZSC_SIGNAL_MAP[state_key]
    entry = STATE_SIGNAL_MANIFEST.get(state_key)
    if entry is not None:
        suffix = entry[0]
        for sig in _native.list_all_signals():
            if sig["param_template"].endswith(suffix) or suffix in sig["param_template"]:
                return sig["name"]
    return f"unknown_state:{state_key}"


def scan_factor_ast(family_or_fdef: str | AtomicFactorDef) -> FactorLineageInfo:
    """对单个连续原子因子族执行 AST 静态依赖扫描，返回结构化 FactorLineageInfo。"""
    fdef = ATOMIC_FACTOR_DEFS[family_or_fdef] if isinstance(family_or_fdef, str) else family_or_fdef
    scanner = _FactorAstScanner()
    try:
        src = textwrap.dedent(inspect.getsource(fdef.build))
        tree = ast.parse(src)
        scanner.visit(tree)
    except (OSError, TypeError, SyntaxError):
        pass

    states = tuple(sorted(scanner.states))
    strengths = tuple(sorted(scanner.strengths))
    oi_feats = tuple(sorted(scanner.oi_features))
    cs_feats = tuple(sorted(scanner.cs_features))
    htf_feats = tuple(sorted(scanner.htf_features))

    rust_sigs: list[str] = []
    for skey in states:
        rname = resolve_state_to_rust_signal(skey)
        if rname not in rust_sigs:
            rust_sigs.append(rname)
    for k in strengths:
        rname = STRENGTH_RUST_SIGNAL_MAP.get(k, f"strength_{k}")
        if rname not in rust_sigs:
            rust_sigs.append(rname)

    return FactorLineageInfo(
        family_id=fdef.family,
        archetype=fdef.archetype,
        stage=fdef.stage,
        code_sha256=compute_factor_code_sha256(fdef.build),
        states=states,
        strengths=strengths,
        oi_features=oi_feats,
        cs_features=cs_feats,
        htf_features=htf_feats,
        rust_signals=tuple(rust_sigs),
    )


def scan_all_atomic_factors() -> dict[str, FactorLineageInfo]:
    """扫描当前 ATOMIC_FACTOR_DEFS 中全部连续原子因子族。"""
    return {fam: scan_factor_ast(fdef) for fam, fdef in ATOMIC_FACTOR_DEFS.items()}


def get_factor_lineage_table() -> pd.DataFrame:
    """返回所有 (family_id, archetype, stage, resource_kind, resource_key, rust_signal_name, code_sha256) 血缘明细表。"""
    rows: list[dict[str, Any]] = []
    for fam, info in scan_all_atomic_factors().items():
        for skey in info.states:
            rows.append(
                {
                    "family_id": fam,
                    "archetype": info.archetype,
                    "stage": info.stage,
                    "resource_kind": "state",
                    "resource_key": skey,
                    "rust_signal_name": resolve_state_to_rust_signal(skey),
                    "code_sha256": info.code_sha256,
                }
            )
        for st_key in info.strengths:
            rows.append(
                {
                    "family_id": fam,
                    "archetype": info.archetype,
                    "stage": info.stage,
                    "resource_kind": "strength",
                    "resource_key": st_key,
                    "rust_signal_name": STRENGTH_RUST_SIGNAL_MAP.get(st_key, f"strength_{st_key}"),
                    "code_sha256": info.code_sha256,
                }
            )
        for oi_key in info.oi_features:
            rows.append(
                {
                    "family_id": fam,
                    "archetype": info.archetype,
                    "stage": info.stage,
                    "resource_kind": "oi",
                    "resource_key": oi_key,
                    "rust_signal_name": "attach_causal_oi_features",
                    "code_sha256": info.code_sha256,
                }
            )
        for cs_key in info.cs_features:
            rows.append(
                {
                    "family_id": fam,
                    "archetype": info.archetype,
                    "stage": info.stage,
                    "resource_kind": "cs",
                    "resource_key": cs_key,
                    "rust_signal_name": "compute_causal_cross_sectional_features",
                    "code_sha256": info.code_sha256,
                }
            )
        for htf_key in info.htf_features:
            rows.append(
                {
                    "family_id": fam,
                    "archetype": info.archetype,
                    "stage": info.stage,
                    "resource_kind": "htf",
                    "resource_key": htf_key,
                    "rust_signal_name": "cxt_bi_base_V230228+strength_zs_break",
                    "code_sha256": info.code_sha256,
                }
            )
    return pd.DataFrame(rows)


def get_signal_usage_summary() -> pd.DataFrame:
    """
    对接 `czsc._native.list_all_signals()` 与镜像普查元数据，统计全部 Rust 原生信号的挂载状态、
    镜像对称属性以及被各战略簇（A~F）原子因子引用的频次。
    """
    all_raw = _native.list_all_signals()
    kline_raw = [s for s in all_raw if s["category"] == "kline" and not s["name"].endswith("_V260601")]

    # 反查每个 Rust 信号对应的 state_key 列表
    sig_to_state_keys: dict[str, list[str]] = {}
    for skey in STATE_SIGNAL_MANIFEST:
        rname = resolve_state_to_rust_signal(skey)
        sig_to_state_keys.setdefault(rname, []).append(skey)
    for st_kind in STRENGTH_SIGNAL_KINDS:
        rname = STRENGTH_RUST_SIGNAL_MAP.get(st_kind, f"strength_{st_kind}")
        sig_to_state_keys.setdefault(rname, []).append(f"strength.{st_kind}")

    # 统计每个 Rust 信号被各簇原子因子引用的次数与因子列表
    cluster_keys = {
        ARCH_A: "cluster_A",
        ARCH_B: "cluster_B",
        ARCH_C: "cluster_C",
        ARCH_D: "cluster_D",
        ARCH_E: "cluster_E",
        ARCH_F: "cluster_F",
    }
    usage_fams: dict[str, list[str]] = {}
    usage_by_cluster: dict[str, dict[str, int]] = {}
    for fam, info in scan_all_atomic_factors().items():
        ckey = cluster_keys.get(info.archetype, "cluster_A")
        for rname in info.rust_signals:
            usage_fams.setdefault(rname, []).append(fam)
            cdict = usage_by_cluster.setdefault(rname, dict.fromkeys(cluster_keys.values(), 0))
            cdict[ckey] = cdict.get(ckey, 0) + 1

    rows: list[dict[str, Any]] = []
    for sig in kline_raw:
        sname = sig["name"]
        cat = _native.get_signal_category(sname) or sig["namespace"]
        exact_mirror, active_rate, n_states = _RUST_SIGNAL_CENSUS_META.get(sname, (False, 0.0, 0))
        skeys = sig_to_state_keys.get(sname, [])
        fams = usage_fams.get(sname, [])
        cdict = usage_by_cluster.get(sname, dict.fromkeys(cluster_keys.values(), 0))
        rows.append(
            {
                "signal_name": sname,
                "category": cat,
                "namespace": sig["namespace"],
                "param_template": sig["param_template"],
                "mounted_in_context": bool(skeys),
                "state_key": ",".join(skeys),
                "exact_mirror": bool(exact_mirror),
                "active_rate": float(active_rate),
                "n_states": int(n_states),
                "consumed_count": len(fams),
                "consumed_families": ",".join(fams),
                **cdict,
            }
        )
    return pd.DataFrame(rows)


def list_unconsumed_mirror_signals(category: str | None = None) -> pd.DataFrame:
    """筛选 exact_mirror=True 且 consumed_count == 0 的未开发高信噪比原生信号。"""
    df = get_signal_usage_summary()
    mask = df["exact_mirror"] & (df["consumed_count"] == 0) & (df["active_rate"] > 0.0)
    if category is not None:
        mask = mask & ((df["category"] == category) | (df["namespace"] == category))
    return df.loc[mask].sort_values(["active_rate", "signal_name"], ascending=[False, True]).reset_index(drop=True)
