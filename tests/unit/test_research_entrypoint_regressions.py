"""正式研究入口的准入契约；不以模拟收益证明经济有效性。"""

from __future__ import annotations

import json
import sys

import numpy as np

from czsc.mock import generate_symbol_kines
from examples.alpha_mining_agent.run_evolution_loop import evaluate_candidate_pooled_gate1b, main
from examples.alpha_mining_agent.state_ledger import AgentStateLedger


def test_research_entrypoint_rejects_summary_only_admission(tmp_path):
    """漂亮的摘要不能代替真实逐事件收益，即使所有摘要都是正值。"""
    summaries = {
        symbol: {"is_n_purged": 100, "is_mean_pct": 1.0, "is_net_sharpe": 2.0}
        for symbol in ("159915", "510300", "510500", "510050", "588000")
    }
    ledger = AgentStateLedger(tmp_path)
    result = evaluate_candidate_pooled_gate1b(
        sid="BC01L",
        tf="30m",
        primary_symbol="159915",
        primary_g1=summaries["159915"],
        pool_tf_dfs={},
        rng=np.random.default_rng(42),
        per_symbol_gate1_overrides=summaries,
        ledger=ledger,
    )
    assert result["passed"] is False
    assert ledger.load_signal_zoo() == {}
    assert len(ledger.load_graveyard_records()) == 1


def test_formal_entrypoint_loads_real_pipeline_and_preserves_ledger(tmp_path, monkeypatch):
    """通过真实CLI路径加载mock行情，接入截面；OOS-only输入不成为IS准入。"""
    paths = {}
    for index, symbol in enumerate(("159915", "510500")):
        frame = generate_symbol_kines(symbol, "5分钟", "20240102", "20240119", seed=42 + index)
        path = tmp_path / f"{symbol}.csv"
        frame.to_csv(path, index=False)
        paths[symbol] = path
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    old_ledger = workspace / "signal_zoo.json"
    old_ledger.write_text('{"existing": {"user_data": true}}', encoding="utf-8")
    original = old_ledger.read_bytes()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_evolution_loop",
            "--symbol",
            "159915",
            "--data-path",
            str(paths["159915"]),
            "--pool-symbols",
            f"510500={paths['510500']}",
            "--workspace-dir",
            str(workspace),
            "--validate-only",
            "--audit-unit",
            "FC02L@30m",
        ],
    )
    main()
    result = json.loads((workspace / "entrypoint_validation.json").read_text(encoding="utf-8"))
    assert result["cross_section_attached"] is True
    assert result["pool_symbols"] == ["159915", "510500"]
    assert result["passed"] is False
    assert result["gate1"]["is_n_purged"] == 0
    assert result["gate1b"]["passed"] is False
    assert old_ledger.read_bytes() == original


def test_evolution_entrypoint_does_not_seed_unqualified_factors(tmp_path, monkeypatch):
    """即使没有旧库，完整研究主循环也不能绕过多资产门禁写入种子。"""
    frame = generate_symbol_kines("159915", "5分钟", "20220104", "20220114", seed=97)
    data_path = tmp_path / "159915.csv"
    frame.to_csv(data_path, index=False)
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_evolution_loop",
            "--symbol",
            "159915",
            "--data-path",
            str(data_path),
            "--workspace-dir",
            str(workspace),
            "--max-candidates",
            "0",
        ],
    )
    main()
    ledger = AgentStateLedger(workspace)
    assert ledger.load_signal_zoo() == {}
    assert ledger.load_graveyard_records()
    assert (workspace / "autonomous_evolution_dashboard.html").exists()


def test_cross_asset_demo_runs_complete_report_with_rejected_pool(tmp_path, monkeypatch):
    """真正执行多资产demo，包括报表，检验门禁拒绝后空仓流程也能完成。"""
    from examples.multi_factor_cta import run_framework_demo as entry
    from examples.multi_factor_cta.data_engine import AssetClass, AssetSpec, ExecutionMode

    specs = []
    for seed, symbol in enumerate(("159915", "510300", "510500"), 70):
        frame = generate_symbol_kines(symbol, "5分钟", "20240102", "20240105", seed=seed)
        path = tmp_path / f"{symbol}.csv"
        frame.to_csv(path, index=False)
        specs.append(
            AssetSpec(
                symbol=symbol,
                name=f"fixture_{symbol}",
                asset_class=AssetClass.INDEX_FUTURES,
                sector="test_equity",
                base_tf="5m",
                eligible_tfs=("1d", "60m", "30m", "15m", "5m"),
                fee_rate=0.001,
                execution_mode=ExecutionMode.LONG_SHORT,
                data_path=path,
            )
        )
    workspace, output = tmp_path / "workspace", tmp_path / "output"
    workspace.mkdir()
    champion = workspace / "champion_portfolio.json"
    champion.write_text(json.dumps({"alloc_map": {"FC02L@30m": 1.0, "S1L@30m": 1.0}}), encoding="utf-8")
    monkeypatch.setattr(entry, "WORKSPACE_DIR", workspace)
    monkeypatch.setattr(entry, "CTA_OUTPUT_DIR", output)
    monkeypatch.setattr(entry, "build_default_cross_asset_universe", lambda: specs)
    entry.main()
    assert list(output.glob("*.html"))
    assert len(list(output.glob("*_admission.json"))) == 3
    for path in output.glob("*_admission.json"):
        rejects = json.loads(path.read_text(encoding="utf-8"))
        assert rejects and all(not result["passed"] for result in rejects.values())
