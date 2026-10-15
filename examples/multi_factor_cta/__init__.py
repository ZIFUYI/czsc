"""
跨资产规则型多因子 CTA 策略框架 (Cross-Asset Multi-Factor Rule-Based CTA Framework)

涵盖三大资产类别：
1. 股指期货 / 指数衍生品 (Stock Index Futures: IF / IH / IC / IM)
2. 宽基/行业 ETF 日内 T+0 (Intraday Tradable ETFs: 159915 / 510500 / 510300)
3. 商品期货主力连续 (Commodity Futures with Night Sessions: SHFE / DCE / CZCE / INE / GFEX)
"""

from examples.multi_factor_cta.data_engine import (
    AssetClass,
    AssetSpec,
    CrossAssetDataEngine,
    ExecutionMode,
    map_tf_weight_to_base_grid,
)
from examples.multi_factor_cta.factor_engine import (
    ARCHETYPE_RULERS,
    RuleFactorRegistry,
    RuleFactorSpec,
)
from examples.multi_factor_cta.forensic_validator import (
    RuleFactorForensicValidator,
    compute_cross_sectional_concurrency,
    compute_signed_jaccard,
)
from examples.multi_factor_cta.synthesis_engine import (
    CrossAssetHRPSynthesizer,
    IntraAssetThreePillarSynthesizer,
    joint_is_coordinate_reweight,
)

__all__ = [
    "ARCHETYPE_RULERS",
    "AssetClass",
    "AssetSpec",
    "CrossAssetDataEngine",
    "CrossAssetHRPSynthesizer",
    "ExecutionMode",
    "IntraAssetThreePillarSynthesizer",
    "RuleFactorForensicValidator",
    "RuleFactorRegistry",
    "RuleFactorSpec",
    "compute_cross_sectional_concurrency",
    "compute_signed_jaccard",
    "joint_is_coordinate_reweight",
    "map_tf_weight_to_base_grid",
]
