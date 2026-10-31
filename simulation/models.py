"""数据模型定义"""
from __future__ import annotations
from typing import Any, TypedDict
from dataclasses import dataclass, field

Coord = tuple[int, int]
MOORE_OFFSETS: tuple[tuple[int, int], ...] = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if not (dr == 0 and dc == 0))
PROBE_OUTLIER_SIGMA: float = 1.75

@dataclass(slots=True, frozen=True)
class CellObservation:
    coord: Coord
    state: str
    action_step: int
    revealed: bool
    source: str

@dataclass(slots=True, frozen=True)
class ActionFeedback:
    action_type: str
    success: bool
    comparison: str
    theta: float
    expected_g: float

@dataclass(slots=True)
class Probe:
    center: Coord
    remaining_steps: int

@dataclass(slots=True)
class CellProbeStats:
    total_observations: int = 0
    mine_observations: int = 0
    change_intervals: list[int] = field(default_factory=list)
    last_state: str | None = None
    last_change_step: int | None = None

@dataclass(slots=True, frozen=True)
class IntervalEvent:
    due_step: int
    priority: int | None
    event_type: str
    target: Coord | None = None

@dataclass(slots=True)
class MemoryEntry:
    coord: Coord
    action_step: int
    observed_state: str
    marker: str = "none"
    revealed: bool = False
    visible_mutation_ratio: float = 0.0
    confidence: float = 0.0
    source: str = "unknown"
    note: str = ""

@dataclass(slots=True, frozen=True)
class PlannedAction:
    action_type: str
    target: Coord | None

class SimulationResult(TypedDict):
    history: list[dict[str, Any]]
    snapshots: list[dict[str, Any]]
    summary: dict[str, Any]
    parameter_profile: dict[str, Any]
 
