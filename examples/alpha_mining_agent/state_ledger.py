#!/usr/bin/env python
"""
持久态记忆账本与 4×4 生态位矩阵管理 (Persistent State Ledger & MAP-Elites Niche Matrix)

管理五大核心磁盘资产 (位于 examples/results/alpha_agent_workspace/)：
1. signal_zoo.json: 已入库原子信号基因库 (以 sig_id@tf 为原子单元，跨越 4大战略方向 × 4个周期)
2. niche_matrix.json: 4×4 生态位配额矩阵分布状态
3. graveyard.jsonl: 被淘汰候选信号的证伪墓地（记录死因，防重复探索）
4. evolution_history.jsonl: 每一轮进化迭代的组合净值与指标演进轨迹
5. champion_portfolio.json: 当前最优四周期 (1d+60m+30m+15m) 复合组合配置与水位线
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from examples.alpha_mining_agent.hypothesis_bank import migrate_archetype

WORKSPACE_DIR = Path("examples/results/alpha_agent_workspace")


@dataclass
class AdmittedSignalRecord:
    """已入库真金原子信号档案 (sig_id@timeframe)"""

    unit_id: str  # 例如 "C01L@30m"
    sig_id: str  # 例如 "C01L"
    timeframe: str  # "1d" | "60m" | "30m" | "15m"
    name: str
    archetype: str  # 6 簇之一 (见 hypothesis_bank.ALL_ARCHETYPES)；旧四簇名在加载时由 migrate_archetype 归一
    ruler_desc: str  # 所用专属法医标尺描述
    direction: int  # 1 = 多, -1 = 空
    weight_alloc: float
    is_base_seed: bool
    # Gate 1: 样本内专属法医标尺指标 (2020-2023)
    is_n_purged: int
    is_mean_pct: float
    is_excess_alpha_pct: float
    is_win_rate_pct: float
    is_pl_ratio: float
    is_safety_margin_pct: float
    is_boot_pval: float
    # Gate 2: 三维互补性指标 (时序正交 + 空仓补盲 + 回撤期雪中送炭)
    max_corr_with_zoo: float
    max_corr_peer: str
    gap_fill_ratio_pct: float
    dd_rescue_alpha_bp: float
    # Gate 3: 边际组合贡献与样本外盲测表现 (2024-2026)
    marginal_sharpe_delta: float
    marginal_calmar_delta: float
    oos_mean_pct: float
    oos_excess_alpha_pct: float
    oos_safety_margin_pct: float
    admitted_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


class AgentStateLedger:
    """支持断点续跑、防重复探索与 4×4 生态位配额管理的持久态账本"""

    def __init__(self, workspace_dir: Path = WORKSPACE_DIR) -> None:
        self.workspace_dir = workspace_dir
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.zoo_path = self.workspace_dir / "signal_zoo.json"
        self.niche_path = self.workspace_dir / "niche_matrix.json"
        self.graveyard_path = self.workspace_dir / "graveyard.jsonl"
        self.history_path = self.workspace_dir / "evolution_history.jsonl"
        self.champion_path = self.workspace_dir / "champion_portfolio.json"
        self.leaderboard_path = self.workspace_dir / "leaderboard.md"

    def reset_workspace(self) -> None:
        for p in (
            self.zoo_path,
            self.niche_path,
            self.graveyard_path,
            self.history_path,
            self.champion_path,
            self.leaderboard_path,
        ):
            if p.exists():
                p.unlink()

    @staticmethod
    def _migrate_record_archetype(rec: dict[str, Any]) -> dict[str, Any]:
        """旧四簇账本 -> 六簇：按 sig_id 归一 archetype (内存中迁移，不改写磁盘文件)。"""
        arch = rec.get("archetype")
        if isinstance(arch, str):
            sig_id = rec.get("sig_id") or str(rec.get("unit_id", "")).split("@")[0] or None
            # 无法判定的历史记录原样保留，避免加载失败
            with contextlib.suppress(ValueError):
                rec["archetype"] = migrate_archetype(arch, sig_id)
        return rec

    def load_signal_zoo(self) -> dict[str, dict[str, Any]]:
        if not self.zoo_path.exists():
            return {}
        with open(self.zoo_path, encoding="utf-8") as f:
            zoo = json.load(f)
        return {uid: self._migrate_record_archetype(rec) for uid, rec in zoo.items()}

    def save_signal_to_zoo(self, record: AdmittedSignalRecord) -> None:
        zoo = self.load_signal_zoo()
        zoo[record.unit_id] = asdict(record)
        with open(self.zoo_path, "w", encoding="utf-8") as f:
            json.dump(zoo, f, ensure_ascii=False, indent=2)

    def remove_signal_from_zoo(self, unit_id: str) -> dict[str, Any] | None:
        zoo = self.load_signal_zoo()
        removed = zoo.pop(unit_id, None)
        if removed is not None:
            with open(self.zoo_path, "w", encoding="utf-8") as f:
                json.dump(zoo, f, ensure_ascii=False, indent=2)
        return removed

    def save_niche_matrix(self, matrix_data: dict[str, Any]) -> None:
        with open(self.niche_path, "w", encoding="utf-8") as f:
            json.dump(matrix_data, f, ensure_ascii=False, indent=2)

    def load_graveyard_ids(self) -> set[str]:
        if not self.graveyard_path.exists():
            return set()
        tested = set()
        with open(self.graveyard_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                tested.add(obj.get("unit_id", obj.get("sig_id", "")))
        return tested

    def load_graveyard_records(self) -> list[dict[str, Any]]:
        if not self.graveyard_path.exists():
            return []
        records = []
        with open(self.graveyard_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(self._migrate_record_archetype(json.loads(line)))
        return records

    def bury_rejected_signal(
        self,
        unit_id: str,
        sig_id: str,
        timeframe: str,
        name: str,
        archetype: str,
        direction: int,
        failed_gate: str,
        reason: str,
        metrics: dict[str, Any],
    ) -> None:
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "unit_id": unit_id,
            "sig_id": sig_id,
            "timeframe": timeframe,
            "name": name,
            "archetype": archetype,
            "direction": "多" if direction == 1 else "空",
            "failed_gate": failed_gate,
            "reason": reason,
            "metrics": metrics,
        }
        with open(self.graveyard_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def log_evolution_step(self, step_info: dict[str, Any]) -> None:
        step_info["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(step_info, ensure_ascii=False) + "\n")

    def load_evolution_history(self) -> list[dict[str, Any]]:
        if not self.history_path.exists():
            return []
        rows = []
        with open(self.history_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    def save_champion_portfolio(self, champion_data: dict[str, Any]) -> None:
        champion_data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.champion_path, "w", encoding="utf-8") as f:
            json.dump(champion_data, f, ensure_ascii=False, indent=2)
