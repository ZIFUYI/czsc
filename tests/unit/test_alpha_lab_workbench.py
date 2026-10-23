"""AlphaLab 工程化工作台全链路单元测试（Issues #20 ~ #23）。"""

from __future__ import annotations

from alpha_lab.catalog import (
    NATIVE_CZSC_SIGNAL_MAP,
    STRENGTH_RUST_SIGNAL_MAP,
    compute_factor_code_sha256,
    get_factor_lineage_table,
    get_signal_usage_summary,
    list_unconsumed_mirror_signals,
    resolve_state_to_rust_signal,
    scan_all_atomic_factors,
    scan_factor_ast,
)
from examples.multi_factor_cta.atomic_factors import (
    ATOMIC_FACTOR_DEFS,
    STATE_SIGNAL_MANIFEST,
)


def test_ast_scanner_extracts_all_five_dependency_classes_across_28_families():
    """验证 AST 扫描器对全部 28 个现有原子因子族准确提取 state / strength / oi / cs / htf 五类依赖，零遗漏。"""
    lineage_map = scan_all_atomic_factors()
    assert len(lineage_map) == 28
    assert set(lineage_map) == set(ATOMIC_FACTOR_DEFS)

    # 验证每个因子族均有 64 位 SHA-256 指纹
    for fam, info in lineage_map.items():
        assert len(info.code_sha256) == 64
        assert info.code_sha256 == compute_factor_code_sha256(ATOMIC_FACTOR_DEFS[fam].build)

    # 验证典型因子的五类依赖提取精确无误
    assert "zs_break" in lineage_map["AC01"].strengths
    assert "ma_cls" in lineage_map["AC02"].states
    assert {"ma_joint", "double_ma", "bar_acc", "xl_trend", "xl_trend_v2"} <= set(lineage_map["AC03"].states)
    assert {"oi_flow3_z", "oi_ratio"} <= set(lineage_map["AC05"].oi_features)
    assert {"plr_long", "plr_short", "zdy_dif", "clv"} <= set(lineage_map["DC03"].states)
    assert {"be_pause_v1", "be_verif_v1", "bs2_macd", "bs2_sma", "sym_zs_v1", "sym_zs_v2"} <= set(
        lineage_map["CC04"].states
    )
    assert {"htf_dir", "htf_q"} == set(lineage_map["FC01"].htf_features)
    assert {"xs", "has_external_cs"} <= set(lineage_map["FC01"].cs_features)
    assert {"sector_leader_dir", "leader_lag_gap"} <= set(lineage_map["FC02"].cs_features)
    assert "sector_spread_zscore" in lineage_map["FC03"].cs_features
    assert "cs_strength_eff" in lineage_map["FC04"].cs_features

    # 验证血缘明细表覆盖全部五类资源
    df_lineage = get_factor_lineage_table()
    assert not df_lineage.empty
    assert set(df_lineage["resource_kind"]) == {"state", "strength", "oi", "cs", "htf"}
    assert not df_lineage["rust_signal_name"].str.startswith("unknown_state:").any()


def test_state_signal_manifest_resolves_to_valid_rust_signals():
    """验证 STATE_SIGNAL_MANIFEST 全部状态键均可精确反查到 Rust 原生信号。"""
    summary_df = get_signal_usage_summary()
    known_signals = set(summary_df["signal_name"])
    for state_key in STATE_SIGNAL_MANIFEST:
        rname = resolve_state_to_rust_signal(state_key)
        assert rname in NATIVE_CZSC_SIGNAL_MAP.values()
        assert rname in known_signals, f"{state_key} -> {rname} 未在 Rust 信号目录中找到"
    for st_kind, rname in STRENGTH_RUST_SIGNAL_MAP.items():
        assert rname in known_signals, f"strength.{st_kind} -> {rname} 未在 Rust 信号目录中找到"


def test_signal_usage_summary_and_unconsumed_mirror_signals():
    """验证完整 Rust 原生信号目录、挂载状态、镜像对称属性、簇引用计数与未消费镜像信号筛选。"""
    df = get_signal_usage_summary()
    assert len(df) >= 222
    required_cols = {
        "signal_name",
        "category",
        "namespace",
        "param_template",
        "mounted_in_context",
        "state_key",
        "exact_mirror",
        "active_rate",
        "n_states",
        "consumed_count",
        "consumed_families",
        "cluster_A",
        "cluster_B",
        "cluster_C",
        "cluster_D",
        "cluster_E",
        "cluster_F",
    }
    assert required_cols <= set(df.columns)
    assert int(df["mounted_in_context"].sum()) >= 40
    assert int((df["consumed_count"] > 0).sum()) >= 35

    # 校验已被消费信号的簇计数之和等于 consumed_count
    cluster_sum = df[["cluster_A", "cluster_B", "cluster_C", "cluster_D", "cluster_E", "cluster_F"]].sum(axis=1)
    assert (cluster_sum == df["consumed_count"]).all()

    unconsumed = list_unconsumed_mirror_signals()
    assert not unconsumed.empty
    assert unconsumed["exact_mirror"].all()
    assert (unconsumed["consumed_count"] == 0).all()
    assert (unconsumed["active_rate"] > 0.0).all()

    # 按分类过滤
    cxt_unconsumed = list_unconsumed_mirror_signals(category="cxt")
    assert not cxt_unconsumed.empty
    assert (cxt_unconsumed["category"] == "cxt").all()

    # 验证 scan_factor_ast 支持直接传 family 字符串
    ac01_info = scan_factor_ast("AC01")
    assert ac01_info.family_id == "AC01" and "strength_zs_break" in ac01_info.rust_signals


def test_sqlite_memory_ledger_db_and_full_migration():
    """验证 SQLite 内存后端 (sqlite:///:memory:) 下 8 张核心表的建表、全量迁移与多维 SQL 聚合查询。"""
    from alpha_lab.ledger_db import (
        LEDGER_TABLES,
        AlphaLabLedgerDB,
        migrate_historical_data_to_db,
    )

    db = AlphaLabLedgerDB("sqlite:///:memory:")
    try:
        counts = migrate_historical_data_to_db(db=db)
        assert set(counts) == set(LEDGER_TABLES)
        assert counts["rust_signals"] >= 222
        assert counts["factor_defs"] >= 131
        assert counts["factor_signal_lineage"] >= 100
        assert counts["preflight_audits"] >= 28
        assert counts["factor_asset_scores"] >= 47000
        assert counts["factor_pool_scores"] >= 370
        assert counts["ablation_runs"] >= 18
        assert counts["factor_graveyard"] >= 80

        # 验证 6 簇 × 5 周期 = 30 个生态位饱和度矩阵
        niche_df = db.get_niche_saturation_matrix()
        assert len(niche_df) == 30
        assert set(niche_df["scarcity_status"]) <= {"SCARCE", "BALANCED", "SATURATED"}

        # 验证按 6 大商品板块 (sector) 的 SQL 聚合查询
        sec_df = db.get_sector_score_summary()
        assert len(sec_df) == 6
        assert int(sec_df["assets"].sum()) == 60

        # 验证按研发批次 (stage) 的 SQL 聚合查询
        stg_df = db.get_stage_score_summary()
        assert set(stg_df["stage"]) == {"Stage1_Base", "Stage2_Rust", "Stage3_OI", "Stage4_CE"}

        # 验证组合消融轨迹与各簇 Top 因子榜查询
        abl_df = db.get_ablation_evolution_table()
        assert {"Stage1_Base_Full60", "Stage3_OI_Incremental", "Stage4_CE_Incremental"} <= set(abl_df["run_id"])
        top_df = db.get_top_factors_by_cluster(top_k=3)
        assert len(top_df) >= 12

        # 验证证伪候选埋葬幂等写入
        db.bury_factor(
            "TEST_BAD01L@60m",
            failed_gate="Preflight_Mirror",
            reason="setup_diff=3 > 0 on affine mirror test",
            stage="Stage_Test",
        )
        g_df = db.query_df("SELECT * FROM alpha_lab.factor_graveyard WHERE uid = %s", ("TEST_BAD01L@60m",))
        assert len(g_df) == 1
        assert g_df["failed_gate"].iloc[0] == "Preflight_Mirror"
    finally:
        db.close()


def test_postgres_ledger_db_schema_and_counts():
    """验证本机 PostgreSQL structure_state_lab.alpha_lab Schema 下的 8 张表全量数据完整性（CI 无 PG 时自动回退验证）。"""
    from alpha_lab.ledger_db import DEFAULT_LEDGER_DB_URL, AlphaLabLedgerDB

    try:
        db = AlphaLabLedgerDB(DEFAULT_LEDGER_DB_URL, auto_init=False)
    except Exception:
        # 若在无外部 PG 的纯 CI 容器中运行，回退验证 sqlite:///:memory:
        db = AlphaLabLedgerDB("sqlite:///:memory:")
        from alpha_lab.ledger_db import migrate_historical_data_to_db

        migrate_historical_data_to_db(db=db)
    try:
        counts = db.get_table_counts()
        assert counts["rust_signals"] >= 222
        assert counts["factor_defs"] >= 131
        assert counts["factor_asset_scores"] >= 47000
        assert counts["factor_pool_scores"] >= 370
        assert counts["ablation_runs"] >= 18
    finally:
        db.close()
