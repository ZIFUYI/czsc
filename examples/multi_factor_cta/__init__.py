"""
跨资产规则型多因子 CTA 策略框架 (Cross-Asset Multi-Factor Rule-Based CTA Framework)

涵盖三大资产类别：
1. 股指期货 / 指数衍生品 (Stock Index Futures: IF / IH / IC / IM)
2. 宽基/行业 ETF 日内 T+0 (Intraday Tradable ETFs: 159915 / 510500 / 510300)
3. 商品期货主力连续 (Commodity Futures with Night Sessions: SHFE / DCE / CZCE / INE / GFEX)
"""

from examples.multi_factor_cta.atomic_factors import (
    ATOMIC_FACTOR_DEFS,
    ATOMIC_SIGNAL_IDS,
    CrossSectionProvider,
    DataFrameCrossSectionProvider,
    DecayConfig,
    FactorContext,
    NeutralCrossSection,
    SoftCrossSectionalGate,
    apply_dynamic_structure_decay,
    compute_atomic_signal,
    compute_family,
    two_stage_trigger,
)
from examples.multi_factor_cta.data_engine import (
    AssetClass,
    AssetSpec,
    CrossAssetDataEngine,
    CrossSectionalFeatureEngine,
    ExecutionMode,
    compute_causal_cross_sectional_features,
    map_tf_weight_to_base_grid,
)
from examples.multi_factor_cta.factor_engine import (
    ARCHETYPE_RULERS,
    RuleFactorRegistry,
    RuleFactorSpec,
)
from examples.multi_factor_cta.forensic_validator import (
    RollingEBWeightsResult,
    RollingEmpiricalBayesWeighter,
    RuleFactorForensicValidator,
    compute_cross_sectional_concurrency,
    compute_empirical_bayes_shrinkage_weight,
    compute_rolling_empirical_bayes_weights,
    compute_signed_jaccard,
)
from examples.multi_factor_cta.synthesis_engine import (
    ContinuousSignalStateMachine,
    CrossAssetHRPSynthesizer,
    DynamicActivePoolAllocator,
    IntraAssetSixClusterSynthesizer,
    IntraAssetThreePillarSynthesizer,
    apply_cross_sectional_budget_tilt,
    gram_schmidt_cluster_orthogonalize,
    joint_is_coordinate_reweight,
)

__all__ = [
    "ARCHETYPE_RULERS",
    "ATOMIC_FACTOR_DEFS",
    "ATOMIC_SIGNAL_IDS",
    "AssetClass",
    "AssetSpec",
    "ContinuousSignalStateMachine",
    "CrossAssetDataEngine",
    "CrossAssetHRPSynthesizer",
    "CrossSectionProvider",
    "CrossSectionalFeatureEngine",
    "DataFrameCrossSectionProvider",
    "DecayConfig",
    "DynamicActivePoolAllocator",
    "ExecutionMode",
    "FactorContext",
    "IntraAssetSixClusterSynthesizer",
    "IntraAssetThreePillarSynthesizer",
    "NeutralCrossSection",
    "RollingEBWeightsResult",
    "RollingEmpiricalBayesWeighter",
    "RuleFactorForensicValidator",
    "RuleFactorRegistry",
    "RuleFactorSpec",
    "SoftCrossSectionalGate",
    "apply_cross_sectional_budget_tilt",
    "apply_dynamic_structure_decay",
    "compute_atomic_signal",
    "compute_causal_cross_sectional_features",
    "compute_cross_sectional_concurrency",
    "compute_empirical_bayes_shrinkage_weight",
    "compute_family",
    "compute_rolling_empirical_bayes_weights",
    "compute_signed_jaccard",
    "gram_schmidt_cluster_orthogonalize",
    "joint_is_coordinate_reweight",
    "map_tf_weight_to_base_grid",
    "two_stage_trigger",
]
