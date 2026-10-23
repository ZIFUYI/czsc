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
    """验证 AST 扫描器对全部 36 个已注册原子因子族准确提取 state / strength / oi / cs / htf 五类依赖，零遗漏。"""
    lineage_map = scan_all_atomic_factors()
    assert len(lineage_map) == 36
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
    assert {"dema_v1", "bi_trend_v1", "bar_brk_w20"} <= set(lineage_map["AC06"].states)
    assert {"plr_long", "plr_short", "zdy_dif", "clv"} <= set(lineage_map["DC03"].states)
    assert {"be_pause_v1", "be_verif_v1", "bs2_macd", "bs2_sma", "sym_zs_v1", "sym_zs_v2"} <= set(
        lineage_map["CC04"].states
    )
    assert {"bs2_w9", "ubi_end_v1", "jcc_yun_v1", "jcc_xing_v2"} <= set(lineage_map["CC06"].states)
    assert {"cxt_dec_526", "macd_bs1_312", "fx_pwr_v1"} <= set(lineage_map["BC05"].states)
    assert {"xl_engulf", "xl_ma_flt", "dif_zero_614"} <= set(lineage_map["DC06"].states)
    assert {"vol_dma", "dif_dist_409", "bar_dec_608"} <= set(lineage_map["DC07"].states)
    assert {"zfzd_n5", "win_std_lo", "bar_triple_v1"} <= set(lineage_map["EC06"].states)
    assert {"bi_end_815", "vol_win_v1", "three_bi"} <= set(lineage_map["FC05"].states)
    assert {"sector_leader_dir", "leader_lag_gap", "cs_strength_eff"} <= set(lineage_map["FC05"].cs_features)
    assert {"sector_spread_zscore", "leader_lag_gap"} <= set(lineage_map["FC06"].cs_features)
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
        assert set(stg_df["stage"]) == {
            "Stage1_Base",
            "Stage2_Rust",
            "Stage3_OI",
            "Stage4_CE",
            "Stage5_FDB",
            "Stage6_ACE",
        }

        # 验证组合消融轨迹与各簇 Top 因子榜查询
        abl_df = db.get_ablation_evolution_table()
        assert {
            "Stage1_Base_Full60",
            "Stage3_OI_Incremental",
            "Stage4_CE_Incremental",
            "Stage5_FDB_Incremental",
            "Stage6_ACE_Incremental",
        } <= set(abl_df["run_id"])
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


def test_preflight_audit_passes_all_28_families_and_catches_bad_candidates():
    """验证 1 秒快速预检门禁对现有 36 个原子因子族 100% 通过，且 100% 拦截镜像不对称与前视偏差因子。"""
    import numpy as np

    from alpha_lab.ledger_db import AlphaLabLedgerDB
    from alpha_lab.preflight import run_preflight_audit
    from examples.alpha_mining_agent.hypothesis_bank import ARCH_A
    from examples.multi_factor_cta.atomic_factors import ATOMIC_FACTOR_DEFS, AtomicFactorDef

    db = AlphaLabLedgerDB("sqlite:///:memory:")
    try:
        df_ok = run_preflight_audit(db=db, persist=True)
        assert len(df_ok) == 36
        assert df_ok["passed"].all()
        assert (df_ok["setup_diff"] == 0).all()
        assert (df_ok["conf_diff"] == 0).all()
        assert (df_ok["fire_diff"] == 0).all()
        assert df_ok["zero_lookahead_pass"].all()
        assert df_ok["bounds_valid"].all()

        # 构造人为注入镜像不对称的坏因子 (多头常开 vs 空头按 AC01)
        def _build_asym(ctx, side):
            s, sq, c, cq, w = ATOMIC_FACTOR_DEFS["AC01"].build(ctx, side)
            if side > 0:
                s = s | (np.arange(ctx.n) % 17 == 0)
            return s, sq, c, cq, w

        # 构造人为注入未来前视偏差的坏因子 (引用未来第 t+5 根收盘价与全样本未来均值)
        def _build_lookahead(ctx, side):
            _, sq, _, cq, w = ATOMIC_FACTOR_DEFS["AC01"].build(ctx, side)
            future_ret = side * (np.roll(ctx.c, -5) - np.mean(ctx.c))
            s = future_ret > 0
            c = future_ret > 0
            return s, sq, c, cq, w

        bad_asym = AtomicFactorDef("BAD_ASYM", "镜像不对称测试因子", ARCH_A, _build_asym, ("30m",))
        bad_la = AtomicFactorDef("BAD_LA", "前视偏差测试因子", ARCH_A, _build_lookahead, ("30m",))

        df_bad = run_preflight_audit([bad_asym, bad_la], db=db, persist=True, auto_bury_failed=True)
        assert len(df_bad) == 2
        assert not df_bad["passed"].any()
        assert int(df_bad.loc[df_bad["family_id"] == "BAD_ASYM", "setup_diff"].iloc[0]) > 0
        assert not bool(df_bad.loc[df_bad["family_id"] == "BAD_LA", "zero_lookahead_pass"].iloc[0])

        # 验证失败候选已自动记入 factor_graveyard
        g_df = db.query_df("SELECT family_id, failed_gate FROM alpha_lab.factor_graveyard")
        assert {"BAD_ASYM", "BAD_LA"} <= set(g_df["family_id"])
    finally:
        db.close()


def test_full60_context_store_parity_hot_patch_and_speed():
    """验证 Full60ContextStore 数值结果与 FactorContext.from_df 完全一致、支持增量状态热补丁，且 60品种×3周期 评估耗时 <= 8s。"""
    import time

    import numpy as np

    from alpha_lab.context_store import Full60ContextStore
    from alpha_lab.ledger_db import DEFAULT_CACHE_60_DIR
    from alpha_lab.preflight import _mock_quantized_raw, _prepare_kdf
    from alpha_lab.runner import (
        map_tf_to_15m_causal_grid,
        run_full_sop_evaluation,
    )
    from examples.multi_factor_cta.atomic_factors import FactorContext, compute_family

    # 1. 验证单序列从头构建 FactorContext.from_df 与 Full60ContextStore 100% 逐元素一致 + 热补丁
    raw = _mock_quantized_raw(5).iloc[:600].reset_index(drop=True)
    sig_cache = DEFAULT_CACHE_60_DIR / "preflight_sig_cache"
    df_30m = _prepare_kdf(raw, "30m", sig_cache)
    ctx_direct = FactorContext.from_df(df_30m)
    store_custom = Full60ContextStore.from_dataframes(
        {("au", "30m"): df_30m},
        cache_dir=DEFAULT_CACHE_60_DIR,
        tag="unit_test_parity",
        persist=False,
    )
    res_store = store_custom.evaluate_families(["AC01", "BC01", "CC04", "EC05"], tfs=("30m",), assets=("au",))
    for fam in ("AC01", "BC01", "CC04", "EC05"):
        s_direct, trig_direct = compute_family(ctx_direct, fam)
        s_cached, trig_cached = res_store[("au", "30m", fam)]
        np.testing.assert_allclose(s_direct, s_cached, atol=1e-12, rtol=0.0)
        np.testing.assert_array_equal(trig_direct, trig_cached)

    # 验证增量状态热补丁 (patch_all_states)
    custom_manifest = {"custom_new_state": ("完全分类V241010", None, "tas", "Stage4_CE")}
    patched = store_custom.patch_all_states(manifest=custom_manifest)
    assert patched[("au", "30m")] == ["custom_new_state"]
    assert "custom_new_state" in store_custom.get("au", "30m").ctx.states

    # 2. 验证 Full-60 (180 序列) 命中缓存后评估 4 个因子族耗时 <= 8s
    t0 = time.perf_counter()
    store_60 = Full60ContextStore.load_full60(tfs=("1d", "60m", "30m"))
    res_60 = store_60.evaluate_families(["CC04", "CC05", "EC04", "EC05"], tfs=("1d", "60m", "30m"))
    elapsed = time.perf_counter() - t0
    assert len(store_60) == 180
    assert len(res_60) == 720
    assert elapsed <= 8.0, f"Full60ContextStore 命中缓存评估 4 族耗时 {elapsed:.2f}s > 8.0s"

    # 3. 验证 runner 因果网格对齐与一键 SOP 评估流水线
    w_aligned = map_tf_to_15m_causal_grid(df_30m["dt"], res_store[("au", "30m", "CC04")][0], df_30m["dt"])
    assert len(w_aligned) == len(df_30m)
    sop_out = run_full_sop_evaluation(
        ["CC04", "EC05"],
        stage="Stage4_CE",
        tfs=("30m",),
        assets=("au",),
        store=store_custom,
        persist_db=False,
    )
    assert sop_out["status"] == "OK"
    assert len(sop_out["ablation_runs"]) == 2


def test_cli_all_six_commands_and_leaderboard_generation(tmp_path):
    """验证 alpha_lab.cli 全部 6 个子命令 (migrate-db/status/preflight/eval/bury/report) 及 FACTORS_LEADERBOARD.md 生成。"""
    import json

    from typer.testing import CliRunner

    from alpha_lab.cli import DEFAULT_LEADERBOARD_PATH, app

    runner = CliRunner()
    sqlite_file = tmp_path / "cli_test.sqlite"
    db_url = f"sqlite:///{sqlite_file}"

    # 1. migrate-db --json
    r_mig = runner.invoke(app, ["migrate-db", "--db-url", db_url, "--json"])
    assert r_mig.exit_code == 0, r_mig.output
    mig_data = json.loads(r_mig.stdout)
    assert mig_data["status"] == "OK"
    assert mig_data["table_counts"]["rust_signals"] == 234
    assert mig_data["table_counts"]["factor_asset_scores"] >= 47000

    # 2. status --json
    r_st = runner.invoke(app, ["status", "--db-url", db_url, "--json"])
    assert r_st.exit_code == 0, r_st.output
    st_data = json.loads(r_st.stdout)
    assert st_data["signal_coverage"]["total_kline_signals"] == 234
    assert st_data["unconsumed_mirror_signals_count"] >= 80
    assert set(st_data["niche_saturation"]["clusters"]) == {"A", "B", "C", "D", "E", "F"}
    assert set(st_data["niche_saturation"]["timeframes"]) == {"1d", "60m", "30m", "15m", "5m"}

    # 3. preflight --families AC01,CC04,EC05 --json
    r_pf = runner.invoke(app, ["preflight", "--families", "AC01,CC04,EC05", "--db-url", db_url, "--json"])
    assert r_pf.exit_code == 0, r_pf.output
    pf_data = json.loads(r_pf.stdout)
    assert pf_data["status"] == "OK"
    assert pf_data["passed"] == 3

    # 4. eval --families CC04,EC05 --tfs 30m --assets au,rb --json
    r_ev = runner.invoke(
        app,
        [
            "eval",
            "--families",
            "CC04,EC05",
            "--stage",
            "Stage4_CE",
            "--tfs",
            "30m",
            "--assets",
            "au,rb",
            "--db-url",
            db_url,
            "--json",
        ],
    )
    assert r_ev.exit_code == 0, r_ev.output
    ev_data = json.loads(r_ev.stdout)
    assert ev_data["status"] == "OK"
    assert ev_data["asset_score_rows"] == 16
    assert ev_data["pool_score_rows"] == 8

    # 5. bury --json
    r_bu = runner.invoke(
        app,
        [
            "bury",
            "--family-id",
            "TEST_EXP01",
            "--candidate-key",
            "TEST_EXP01@30m",
            "--stage",
            "Stage4_CE",
            "--failed-gate",
            "Gate2_WeakPoolEdge",
            "--reason",
            "单元测试模拟淘汰",
            "--db-url",
            db_url,
            "--json",
        ],
    )
    assert r_bu.exit_code == 0, r_bu.output
    bu_data = json.loads(r_bu.stdout)
    assert bu_data["status"] == "OK"
    assert bu_data["candidate_key"] == "TEST_EXP01@30m"

    # 6. report --output ... --json
    md_out = tmp_path / "FACTORS_LEADERBOARD.md"
    r_rp = runner.invoke(app, ["report", "--output", str(md_out), "--db-url", db_url, "--json"])
    assert r_rp.exit_code == 0, r_rp.output
    rp_data = json.loads(r_rp.stdout)
    assert rp_data["status"] == "OK"
    assert md_out.exists()
    md_text = md_out.read_text(encoding="utf-8")
    assert "6×5 生态位饱和度矩阵" in md_text
    assert "Stage 1 ~ Stage 6 全周期组合消融演进对比" in md_text
    assert "TEST_EXP01@30m" in md_text
    assert DEFAULT_LEADERBOARD_PATH.exists()


def test_dashboard_service_http_api_and_static_html_export(tmp_path):
    """验证 AlphaLabDashboardService、HTTP REST API、只读 SQL 安全拦截与静态 HTML 导出。"""
    import json
    import threading
    import urllib.error
    import urllib.request

    import pytest
    from typer.testing import CliRunner

    from alpha_lab.cli import app
    from alpha_lab.dashboard import (
        AlphaLabDashboardService,
        create_dashboard_server,
        validate_readonly_sql,
    )
    from alpha_lab.ledger_db import AlphaLabLedgerDB, migrate_historical_data_to_db

    # 1. 只读 SQL 校验器测试
    assert validate_readonly_sql("SELECT COUNT(*) FROM alpha_lab.ablation_runs;") == (
        "SELECT COUNT(*) FROM alpha_lab.ablation_runs"
    )
    for bad_sql in (
        "DROP TABLE alpha_lab.ablation_runs",
        "DELETE FROM alpha_lab.factor_graveyard",
        "SELECT 1; SELECT 2",
        "UPDATE alpha_lab.rust_signals SET consumed_count = 0",
    ):
        with pytest.raises(ValueError):
            validate_readonly_sql(bad_sql)

    # 2. 基于临时 SQLite 台账验证 DashboardService 与 CLI 导出
    sqlite_file = tmp_path / "dash_test.sqlite"
    db_url = f"sqlite:///{sqlite_file}"
    db = AlphaLabLedgerDB(db_url)
    migrate_historical_data_to_db(db=db)
    db.close()

    svc = AlphaLabDashboardService(db_url)
    ov = svc.get_overview()
    assert ov["table_counts"]["rust_signals"] == 234
    assert len(svc.get_ablation_runs()) >= 10
    assert len(svc.get_pool_scores(gate1b_only=True)) > 0
    asset_res = svc.get_asset_scores(sector="PRECIOUS_METALS", limit=10, offset=0)
    assert asset_res["summary"]["total_rows"] > 0
    assert len(asset_res["rows"]) <= 10

    # 3. 启动后台 HTTP 服务验证 REST 端点与 POST /api/sql
    server = create_dashboard_server(host="127.0.0.1", port=0, db_url=db_url)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{port}"
        with urllib.request.urlopen(f"{base_url}/") as resp:
            html_index = resp.read().decode("utf-8")
            assert "CZSC AlphaLab" in html_index
            assert "window.__LIVE_MODE__ = true;" in html_index

        with urllib.request.urlopen(f"{base_url}/api/asset-scores?sector=FERROUS&limit=5") as resp:
            as_json = json.loads(resp.read().decode("utf-8"))
            assert len(as_json["rows"]) == 5

        req_ok = urllib.request.Request(
            f"{base_url}/api/sql",
            data=json.dumps(
                {"sql": "SELECT stage, COUNT(*) AS cnt FROM alpha_lab.ablation_runs GROUP BY stage"}
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_ok) as resp:
            sql_json = json.loads(resp.read().decode("utf-8"))
            assert sql_json["status"] == "OK"
            assert sql_json["total_rows"] >= 4

        req_bad = urllib.request.Request(
            f"{base_url}/api/sql",
            data=json.dumps({"sql": "DELETE FROM alpha_lab.ablation_runs"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req_bad)
        assert exc_info.value.code == 400
    finally:
        server.shutdown()
        server.server_close()

    # 4. 验证 CLI dashboard --export
    runner = CliRunner()
    html_out = tmp_path / "DASHBOARD.html"
    r_exp = runner.invoke(app, ["dashboard", "--export", str(html_out), "--db-url", db_url, "--json"])
    assert r_exp.exit_code == 0, r_exp.output
    assert html_out.exists()
    assert "window.__LIVE_MODE__ = false;" in html_out.read_text(encoding="utf-8")
