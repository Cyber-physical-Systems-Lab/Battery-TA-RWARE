"""
Role-specialization metrics for AGV mission histories.

The tracker records mission starts by AGV, including picking, delivering,
returning, charging, and optional rack-section assignments. The metrics object
then reports dominant roles, role switches, mission entropy, section entropy,
team specialization, role stability, and charger-escort prevalence for
heuristic or learned policies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# ────────────────────────────────────────────────────────────────────────────
# Per-agent role profile
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentRoleProfile:
    """Accumulates mission history for one agent over an episode."""

    agent_id: int

    # Mission-type counts (productive vs charging)
    mission_counts: Dict[str, int] = field(default_factory=lambda: {
        "picking": 0, "delivering": 0, "returning": 0, "charging": 0,
    })

    # Section-level specialization (section_idx → count of missions)
    section_counts: Dict[int, int] = field(default_factory=dict)

    # Role-switch tracking: sequence of dominant roles at each mission start
    _role_history: List[str] = field(default_factory=list, repr=False)

    def record_mission(self, mission_type: str, section_idx: Optional[int] = None) -> None:
        self.mission_counts[mission_type] = self.mission_counts.get(mission_type, 0) + 1
        if section_idx is not None:
            self.section_counts[section_idx] = self.section_counts.get(section_idx, 0) + 1
        self._role_history.append(mission_type)

    def dominant_role(self) -> str:
        """Mission type with highest count ('idle' if none recorded)."""
        counts = {k: v for k, v in self.mission_counts.items() if v > 0}
        if not counts:
            return "idle"
        return max(counts, key=counts.get)

    def specialization_entropy(self) -> float:
        """
        Shannon entropy over mission-type distribution.
        0.0  = perfectly specialized (all missions of one type).
        log4 ≈ 1.386 = uniform over 4 types (generalist).
        """
        counts = np.array(list(self.mission_counts.values()), dtype=float)
        total = counts.sum()
        if total == 0:
            return 0.0
        p = counts / total
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    def section_entropy(self) -> float:
        """
        Shannon entropy over section distribution.
        0.0  = section specialist (always serves same rack group).
        High = generalist across all sections.
        """
        counts = np.array(list(self.section_counts.values()), dtype=float)
        total = counts.sum()
        if total == 0:
            return 0.0
        p = counts / total
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    def role_switches(self) -> int:
        """Count transitions between different mission types in history."""
        if len(self._role_history) < 2:
            return 0
        return sum(
            1 for a, b in zip(self._role_history, self._role_history[1:]) if a != b
        )

    def is_charger_escort(self, threshold: float = 0.5) -> bool:
        """True if charging fraction exceeds `threshold`."""
        total = sum(self.mission_counts.values())
        if total == 0:
            return False
        return self.mission_counts.get("charging", 0) / total >= threshold


# ────────────────────────────────────────────────────────────────────────────
# Episode-level tracker
# ────────────────────────────────────────────────────────────────────────────

class RoleEmergenceTracker:
    """
    In-episode tracker called from heuristic_episode at each mission event.
    Mirrors the design of ReplanningController (RQ3) — same lifecycle pattern.
    """

    def __init__(self, agv_ids: List[int], num_pickers: int) -> None:
        self.num_pickers = num_pickers
        self._profiles: Dict[int, AgentRoleProfile] = {
            aid: AgentRoleProfile(agent_id=aid) for aid in agv_ids
        }
        self._total_missions: int = 0

    # ── Event recording ──────────────────────────────────────────────────

    def on_task_assigned(
        self,
        agent_id: int,
        mission_type: str,
        section_idx: Optional[int] = None,
    ) -> None:
        """Call every time an AGV is assigned a new mission."""
        if agent_id not in self._profiles:
            self._profiles[agent_id] = AgentRoleProfile(agent_id=agent_id)
        self._profiles[agent_id].record_mission(mission_type, section_idx)
        self._total_missions += 1

    # ── Metrics extraction ───────────────────────────────────────────────

    def get_metrics(self) -> "RoleEmergenceMetrics":
        profiles = dict(self._profiles)
        return RoleEmergenceMetrics(
            num_agvs=len(profiles),
            num_pickers=self.num_pickers,
            total_missions=self._total_missions,
            profiles=profiles,
        )


# ────────────────────────────────────────────────────────────────────────────
# Episode-level metrics dataclass
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class RoleEmergenceMetrics:
    """
    Immutable snapshot of role emergence statistics for one episode.
    Returned as the 6th element of heuristic_episode().
    """

    num_agvs: int
    num_pickers: int
    total_missions: int
    profiles: Dict[int, AgentRoleProfile]

    # ── Team-level derived metrics ────────────────────────────────────────

    def team_specialization_index(self) -> float:
        """
        Team Specialization Index (TSI).

        Measures how differentiated agent roles are vs. a fully uniform
        mission distribution.  Range [0, 1]:
          0 = all agents have identical, uniform mission mixes (no specialization)
          1 = each agent is perfectly specialized in a single mission type

        Computed as 1 − mean_normalized_entropy, where entropy is normalized
        by log(4) (max entropy over 4 mission types).
        """
        if not self.profiles:
            return 0.0
        max_h = np.log(4)
        entropies = [p.specialization_entropy() / max_h for p in self.profiles.values()]
        return float(1.0 - np.mean(entropies))

    def role_stability_index(self) -> float:
        """
        Role Stability Index (RSI).

        Fraction of mission transitions that keep the same type.
        Range [0, 1]:
          0 = every consecutive mission switches type (maximum thrashing)
          1 = no role switches (perfect stability)
        """
        total_switches = sum(p.role_switches() for p in self.profiles.values())
        total_transitions = max(self.total_missions - len(self.profiles), 1)
        return float(1.0 - total_switches / total_transitions)

    def charger_escort_fraction(self) -> float:
        """Fraction of all AGV missions that are charging missions."""
        total = self.total_missions
        if total == 0:
            return 0.0
        charge = sum(
            p.mission_counts.get("charging", 0) for p in self.profiles.values()
        )
        return float(charge / total)

    def charger_escort_count(self, threshold: float = 0.5) -> int:
        """Number of AGVs that qualify as de facto charger escorts."""
        return sum(1 for p in self.profiles.values() if p.is_charger_escort(threshold))

    def section_specialization_mean(self) -> float:
        """
        Mean section specialization across AGVs.
        Low section entropy → high section specialization → agents stick to
        their assigned rack groups (similar to prescribed section assignment).
        """
        if not self.profiles:
            return 0.0
        entropies = [p.section_entropy() for p in self.profiles.values()]
        # Invert: lower entropy = higher specialization
        max_e = np.log(max(len(p.section_counts) for p in self.profiles.values()) + 1)
        if max_e == 0:
            return 0.0
        normalized = [e / max_e for e in entropies]
        return float(1.0 - np.mean(normalized))

    def dominant_role_distribution(self) -> Dict[str, int]:
        """Count of agents per dominant role type across the team."""
        dist: Dict[str, int] = {}
        for p in self.profiles.values():
            role = p.dominant_role()
            dist[role] = dist.get(role, 0) + 1
        return dist

    # ── Per-agent summary ─────────────────────────────────────────────────

    def per_agent_summary(self) -> Dict[int, Dict]:
        return {
            aid: {
                "mission_counts": dict(p.mission_counts),
                "dominant_role": p.dominant_role(),
                "specialization_entropy": round(p.specialization_entropy(), 4),
                "section_entropy": round(p.section_entropy(), 4),
                "role_switches": p.role_switches(),
                "is_charger_escort": p.is_charger_escort(),
                "total_missions": sum(p.mission_counts.values()),
            }
            for aid, p in self.profiles.items()
        }

    # ── Full summary dict (for JSON serialization) ────────────────────────

    def summary(self) -> Dict:
        return {
            "num_agvs": self.num_agvs,
            "num_pickers": self.num_pickers,
            "total_missions": self.total_missions,
            "team_specialization_index": round(self.team_specialization_index(), 4),
            "role_stability_index": round(self.role_stability_index(), 4),
            "charger_escort_fraction": round(self.charger_escort_fraction(), 4),
            "charger_escort_count": self.charger_escort_count(),
            "section_specialization_mean": round(self.section_specialization_mean(), 4),
            "dominant_role_distribution": self.dominant_role_distribution(),
            "per_agent": self.per_agent_summary(),
        }
