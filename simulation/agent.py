"""约束代理模型"""
from __future__ import annotations
import math
from collections import deque
import numpy as np

from simulation.config import AgentConfig
from simulation.models import (
    MOORE_OFFSETS, CellObservation, CellProbeStats, Coord, 
    IntervalEvent, MemoryEntry, PlannedAction, Probe, PROBE_OUTLIER_SIGMA
)

class ConstrainedAgent:
    def __init__(self, config: AgentConfig, shape: tuple[int, int], rng: np.random.Generator, *, initial_mine_ratio_a1: float) -> None:
        self.config = config
        self.rows, self.cols = shape
        self.rng = rng
        self.initial_mine_ratio_a1 = initial_mine_ratio_a1
        self.estimated_mutation_b4 = initial_mine_ratio_a1
        self.b4_dataset: deque[float] = deque(maxlen=config.probe_estimate_window)

        self.opened = np.zeros(shape, dtype=bool)
        self.flags = np.zeros(shape, dtype=bool)
        self.questions = np.zeros(shape, dtype=bool)
        self.observation_log: list[MemoryEntry] = []
        self.memory: dict[Coord, MemoryEntry] = {}
        self.probes: list[Probe] = []
        self.cell_probe_stats: dict[Coord, CellProbeStats] = {}

        self.action_step = 0
        self.decision_index = 0
        self.probe_grant_progress = 0
        self.next_probe_grant_progress = max(1, config.probe_interval_d2)
        self.pending_interval_events: list[IntervalEvent] = []
        self.probe_opportunity = False
        self.adjudication_opportunity = False
        self.adjudication_granted = False
        self.defuse_enabled = True
        self.current_anchor: Coord | None = None
        self.first_number_coord: Coord | None = None
        self.defuse_history: deque[bool] = deque(maxlen=10)
        
        self.N = self.rows * self.cols 
        self.true_b3 = initial_mine_ratio_a1 
        self.defused_mines = 0
        self.failed_defuses = 0
        self.overload_events = 0
        self.placed_probes = 0
        self.probe_update_events = 0
        self.probe_expiration_events = 0
        self.memory_decay_events = 0
        self.probe_grant_events = 0
        self.adjudication_grant_events = 0
        self.adjudications_used = 0
@property
def estimated_mutation(self) -> float:
    return self.estimated_mutation_b4

@property
def step_count(self) -> int:
    return self.action_step

@property
def decision_step_count(self) -> int:
    return self.decision_index

@property
def in_stabilization(self) -> bool:
    return not self.defuse_enabled

@property
def disabled(self) -> bool:
    return not self.defuse_enabled

def in_bounds(self, row: int, col: int) -> bool:
    return 0 <= row < self.rows and 0 <= col < self.cols

def neighbors(self, coord: Coord) -> list[Coord]:
    row, col = coord
    return [(row + dr, col + dc) for dr, dc in MOORE_OFFSETS if self.in_bounds(row + dr, col + dc)]

def register_decision(self) -> None:
    self.decision_index += 1

def _remember(self, coord: Coord, *, observed_state: str, marker: str = "none", revealed: bool = False, visible_mutation_ratio: float = 0.0, confidence: float = 0.0, source: str, note: str = "") -> None:
    entry = MemoryEntry(coord=coord, action_step=self.action_step, observed_state=observed_state, marker=marker, revealed=revealed, visible_mutation_ratio=visible_mutation_ratio, confidence=confidence, source=source, note=note)
    self.observation_log.append(entry)
    self.memory[coord] = entry

def observe_cells(self, observations: list[CellObservation]) -> None:
    for observation in observations:
        coord = observation.coord
        if observation.revealed:
            self.opened[coord] = True
            self.flags[coord] = False
            self.questions[coord] = False
        if observation.state.startswith("number:") and self.first_number_coord is None:
            self.first_number_coord = coord
        current = self.memory.get(coord)
        self._remember(coord, observed_state=observation.state, marker="none" if observation.revealed else (current.marker if current else "none"), revealed=bool(self.opened[coord]), visible_mutation_ratio=current.visible_mutation_ratio if current else 0.0, confidence=1.0 if observation.revealed else 0.8, source=observation.source)
    if observations and self.current_anchor is None:
        self.current_anchor = observations[0].coord
def apply_marker(self, action_type: str, coord: Coord) -> None:
    current = self.memory.get(coord)
    marker = "flag" if action_type == "mark_flag" else "question"
    self.flags[coord] = marker == "flag"
    self.questions[coord] = marker == "question"
    self._remember(coord, observed_state=current.observed_state if current else "unknown", marker=marker, revealed=bool(self.opened[coord]), visible_mutation_ratio=current.visible_mutation_ratio if current else 0.0, confidence=0.85 if marker == "flag" else 0.35, source="marker", note="constraint marker")

def apply_marker(self, action_type: str, coord: Coord) -> None:
    current = self.memory.get(coord)
    marker = "flag" if action_type == "mark_flag" else "question"
    self.flags[coord] = marker == "flag"
    self.questions[coord] = marker == "question"
    self._remember(coord, observed_state=current.observed_state if current else "unknown", marker=marker, revealed=bool(self.opened[coord]), visible_mutation_ratio=current.visible_mutation_ratio if current else 0.0, confidence=0.85 if marker == "flag" else 0.35, source="marker", note="constraint marker")

def revealed_mine_ratio(self) -> float:
    entries = [entry for entry in self.memory.values() if entry.revealed]
    if not entries: return 0.0
    mines = sum(entry.observed_state == "mine" for entry in entries)
    return mines / len(entries)

def update_environment_state(self, b3: float) -> None:
    self.true_b3 = b3

def defuse_cost(self) -> float:
    if self.estimated_mutation_b4 <= 0: return float('inf')
    return max(1.0, (self.true_b3 ** 2) / (self.estimated_mutation_b4 * 0.5 * self.N))

def defuse_cost_steps(self) -> int:
    return max(1, int(math.ceil(self.defuse_cost())))

def check_lose_defuse_ability(self) -> None:
    # 双向晃动失能判定
    if self.defuse_cost() > self.N * 0.02:
        if self.defuse_enabled: 
            self.overload_events += 1
        self.defuse_enabled = False
    else:
        self.defuse_enabled = True

def record_defuse_result(self, success: bool) -> None:
    self.defuse_history.append(success)

def defuse_success_rate(self) -> float:
    if not self.defuse_history: return 1.0
    return sum(self.defuse_history) / len(self.defuse_history)

def consume_adjudication_opportunity(self) -> None:
    if self.adjudication_opportunity:
        self.adjudication_opportunity = False
        self.adjudications_used += 1

def _calculate_g_and_f(self) -> tuple[float, float]:
    estimated_a3 = sum(1 for e in self.memory.values() if e.observed_state.startswith("number:")) / max(1, len(self.memory))
    estimated_a2 = self.initial_mine_ratio_a1
    f_est = 0.6 * estimated_a3 + 0.4 * self.estimated_mutation_b4
    g_est = 0.5 * estimated_a2 + 0.5 * self.estimated_mutation_b4 + 0.02
    return f_est, g_est

def in_predictable_mode(self) -> bool:
    f_est, g_est = self._calculate_g_and_f()
    return g_est < f_est

def adjudication_region(self, center: Coord) -> set[Coord]:
    row, col = center
    cells: set[Coord] = set()
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            if abs(dr) == 2 and abs(dc) == 2: continue
            nr, nc = row + dr, col + dc
            if self.in_bounds(nr, nc): cells.add((nr, nc))
    return cells

def _estimated_local_mutation(self, center: Coord) -> float:
    row, col = center
    samples: list[float] = []
    radius = self.config.compare_radius
    for nr in range(max(0, row - radius), min(self.rows, row + radius + 1)):
        for nc in range(max(0, col - radius), min(self.cols, col + radius + 1)):
            entry = self.memory.get((nr, nc))
            if entry is not None and entry.visible_mutation_ratio > 0:
                samples.append(entry.visible_mutation_ratio)
    if not samples: return self.estimated_mutation_b4
    return float(np.clip(np.mean(samples), 0.0, 1.0))

defconsume_probe_opportunity(self) -> None:
    self.probe_opportunity = False

def place_probe(self, center: Coord, initial_scan: dict[Coord, str], duration: int) -> None:
    observed_by_active_probe = {coord for probe in self.probes if probe.remaining_steps > 0 for coord in self.neighbors(probe.center)}
    self.probes.append(Probe(center=center, remaining_steps=duration))
    self.placed_probes += 1
    for coord, state in initial_scan.items():
        stats = self.cell_probe_stats.get(coord)
        if stats is None:
            stats = CellProbeStats()
            self.cell_probe_stats[coord] = stats
        if coord not in observed_by_active_probe:
            stats.last_state = state
            stats.last_change_step = self.action_step
        current = self.memory.get(coord)
        self._remember(coord, observed_state=state, marker=current.marker if current else "none", revealed=bool(self.opened[coord]), visible_mutation_ratio=current.visible_mutation_ratio if current else 0.0, confidence=0.8, source="probe", note="probe initial scan")

def _probe_cell_estimate(self, intervals: list[int]) -> float | None:
    if len(intervals) < 2: return None
    values = np.asarray(intervals[-self.config.probe_estimate_window:], dtype=float)
    mean_interval = float(values.mean())
    if mean_interval <= 0: return None
    std_interval = float(values.std(ddof=0))
    if std_interval > 0:
        deviations = np.abs(values - mean_interval)
        values = values[deviations <= PROBE_OUTLIER_SIGMA * std_interval]
        if values.size == 0: return None
    return float(np.clip(1.0 / float(values.mean()), 0.0, 1.0))

def update_probes(self, readings: dict[Coord, dict[Coord, str]]) -> None:
    cell_estimates: dict[Coord, list[float]] = {}
    observed_cells: set[Coord] = set()
    for probe in self.probes:
        if probe.remaining_steps <= 0: continue
        scan = readings.get(probe.center, {})
        for coord, state in scan.items():
            if coord in observed_cells: continue
            observed_cells.add(coord)
            is_mine = state == "mine"
            stats = self.cell_probe_stats.setdefault(coord, CellProbeStats())
            stats.total_observations += 1
            if is_mine: stats.mine_observations += 1
            if stats.last_state is not None and (stats.last_state == "mine") != is_mine:
                previous_change = stats.last_change_step
                interval = (self.action_step - previous_change) if previous_change is not None else 0
                if interval > 0:
                    stats.change_intervals.append(interval)
                    del stats.change_intervals[:-self.config.probe_estimate_window]
                stats.last_change_step = self.action_step
            stats.last_state = state
            cell_estimate = self._probe_cell_estimate(stats.change_intervals)
            if cell_estimate is not None: cell_estimates.setdefault(coord, []).append(cell_estimate)
            current = self.memory.get(coord)
            self._remember(coord, observed_state=state, marker=current.marker if current else "none", revealed=bool(self.opened[coord]), visible_mutation_ratio=(cell_estimate if cell_estimate is not None else (current.visible_mutation_ratio if current else 0.0)), confidence=0.85 if cell_estimate is not None else 0.65, source="probe", note="visible state interval update")
        probe.remaining_steps -= 1
        if probe.remaining_steps == 0:
            self.pending_interval_events.append(IntervalEvent(due_step=self.action_step, priority=1, event_type="probe_expired", target=probe.center))
    if cell_estimates:
        self.b4_dataset.append(float(np.mean([np.mean(estimates) for estimates in cell_estimates.values()])))
        self.estimated_mutation_b4 = float(np.mean(self.b4_dataset))
        self.probe_update_events += 1

def decay_memory(self) -> None:
    if not self.observation_log: return
    drop_count = int(math.ceil(len(self.observation_log) * self.config.memory_decay_ratio_c3))
    removed = self.observation_log[:drop_count]
    self.observation_log = self.observation_log[drop_count:]
    self.memory = {}
    for entry in self.observation_log: self.memory[entry.coord] = entry
    removed_coords = {entry.coord for entry in removed}
    remaining_coords = {entry.coord for entry in self.observation_log}
    for coord in removed_coords - remaining_coords:
        self.flags[coord] = False
        self.questions[coord] = False
        self.opened[coord] = False # 统子修改：连带遗忘物理踩点

def advance_action_tick(self, readings: dict[Coord, dict[Coord, str]], *, counts_for_probe_grant: bool = True) -> None:
    self.action_step += 1
    self.update_probes(readings)
    if not counts_for_probe_grant: return
    self.probe_grant_progress += 1
    while self.probe_grant_progress >= self.next_probe_grant_progress:
        self.pending_interval_events.append(IntervalEvent(due_step=self.action_step, priority=3, event_type="probe_grant"))
        self.next_probe_grant_progress += max(1, self.config.probe_interval_d2)

def take_pending_interval_events(self) -> list[IntervalEvent]:
    events = sorted(self.pending_interval_events, key=lambda event: (event.due_step, event.priority))
    self.pending_interval_events = []
    return events

def collect_clearing_events(self) -> list[IntervalEvent]:
    events = self.take_pending_interval_events()
    if self.config.memory_decay_ratio_c3 > 0 and self.memory_size() > self.config.memory_record_limit:
        events.append(IntervalEvent(due_step=self.action_step, priority=2, event_type="memory_decay"))
    events.sort(key=lambda event: (event.due_step, event.priority))
    return events

def apply_interval_event(self, event: IntervalEvent) -> None:
    if event.event_type == "probe_expired":
        self.probes = [probe for probe in self.probes if probe.center != event.target or probe.remaining_steps > 0]
        self.probe_expiration_events += 1
    elif event.event_type == "memory_decay":
        self.decay_memory()
        self.memory_decay_events += 1
    elif event.event_type == "probe_grant":
        self.probe_opportunity = True
        self.probe_grant_events += 1
    elif event.event_type == "adjudication_grant":
        if not self.defuse_enabled and not self.adjudication_granted:
            self.adjudication_opportunity = True
            self.adjudication_granted = True
        self.adjudication_grant_events += 1

def advance_action_step(self, readings: dict[Coord, dict[Coord, str]]) -> None:
    self.advance_action_tick(readings)
    for event in self.collect_clearing_events(): self.apply_interval_event(event)

def _constraint_hints(self) -> tuple[set[Coord], set[Coord], set[Coord]]:
    safe: set[Coord] = set()
    mines: set[Coord] = set()
    uncertain: set[Coord] = set()
    for coord, entry in self.memory.items():
        if not entry.observed_state.startswith("number:"): continue
        number = int(entry.observed_state.split(":", maxsplit=1)[1])
        neighbors = self.neighbors(coord)
        hidden = [cell for cell in neighbors if not self.opened[cell]]
        flagged = [cell for cell in neighbors if self.flags[cell]]
        if not hidden: continue
        remaining_mines = max(0, number - len(flagged))
        unflagged_hidden = [cell for cell in hidden if not self.flags[cell]]
        if remaining_mines == 0: safe.update(unflagged_hidden)
        elif remaining_mines == len(unflagged_hidden): mines.update(unflagged_hidden)
        else: uncertain.update(unflagged_hidden)
    conflicted = safe & mines
    safe -= conflicted
    mines -= conflicted
    uncertain |= conflicted
    return safe, mines, uncertain

def _layer_count(self, coord: Coord) -> int:
    count = 0
    for neighbor in self.neighbors(coord):
        entry = self.memory.get(neighbor)
        if entry is not None and entry.observed_state.startswith("number:"): count += 1
    return count

def _hazard_score(self, coord: Coord) -> float:
    entry = self.memory.get(coord)
    score = self.estimated_mutation_b4
    if entry is not None:
        score = 0.6 * score + 0.4 * entry.visible_mutation_ratio
        if entry.observed_state == "mine" or entry.marker == "flag": score += 0.5
        elif entry.marker == "question": score += 0.15
    score += self.config.layer_penalty_e * self._layer_count(coord)
    if self.current_anchor is not None:
        score += 0.03 * (abs(coord[0] - self.current_anchor[0]) + abs(coord[1] - self.current_anchor[1]))
    return score

def _choose_hidden_target(self, preferred: set[Coord] | None = None) -> Coord | None:
    if not np.any(~self.opened & ~self.flags): return None
    candidates = [coord for coord in (preferred or set()) if not self.opened[coord] and not self.flags[coord]]
    if not candidates:
        frontier: set[Coord] = set()
        for row, col in np.argwhere(self.opened):
            for neighbor in self.neighbors((int(row), int(col))):
                if not self.opened[neighbor] and not self.flags[neighbor]: frontier.add(neighbor)
        candidates = list(frontier)
    if not candidates:
        candidates = [(int(row), int(col)) for row, col in np.argwhere(~self.opened) if not self.flags[(int(row), int(col))]]
    if not candidates: return None
    return min(candidates, key=self._hazard_score)

def _choose_defuse_target(self) -> Coord | None:
    targets = {(int(row), int(col)) for row, col in np.argwhere(self.flags)}
    targets.update(coord for coord, entry in self.memory.items() if entry.observed_state == "mine")
    if not targets: return None
    return max(targets, key=self._hazard_score)

def _choose_probe_center(self) -> Coord | None:
    active = {probe.center for probe in self.probes}
    candidates = [(int(row), int(col)) for row, col in np.argwhere(self.opened) if (int(row), int(col)) not in active]
    if not candidates: return None
    return max(candidates, key=lambda coord: sum(not self.opened[cell] for cell in self.neighbors(coord)))

def _choose_flag_probe_center(self) -> Coord | None:
    active = {probe.center for probe in self.probes}
    candidates: set[Coord] = set()
    for row, col in np.argwhere(self.flags):
        for neighbor in self.neighbors((int(row), int(col))):
            if self.opened[neighbor] and neighbor not in active: candidates.add(neighbor)
    if not candidates: return None
    return max(candidates, key=lambda coord: self._estimated_local_mutation(coord))

def _choose_probe_center_with_flag_prior(self) -> Coord | None:
    if self.rng.random() < self.config.flag_probe_probability_i:
        center = self._choose_flag_probe_center()
        if center is not None: return center
    return self._choose_probe_center()

def _choose_adjudicate_center(self) -> Coord | None:
    candidates = [(int(row), int(col)) for row, col in np.argwhere(~self.opened)]
    if not candidates: return None
    def score(coord: Coord) -> int: return sum(1 for cell in self.adjudication_region(coord) if not self.opened[cell])
    return max(candidates, key=score)

def _number_followup(self) -> Coord | None:
    # 常态化扫描所有数字周围
    if self.initial_mine_ratio_a1 > self.config.number_followup_threshold: return None
    if self.rng.random() >= self.config.number_followup_probability: return None
    all_candidates = []
    for coord, entry in self.memory.items():
        if entry.observed_state.startswith("number:"):
            for cell in self.neighbors(coord):
                if not self.opened[cell] and not self.flags[cell]: all_candidates.append(cell)
    if not all_candidates: return None
    return all_candidates[int(self.rng.integers(0, len(all_candidates)))]

def _sync_markers(self) -> None:
    safe, mines, uncertain = self._constraint_hints()
    for coord in mines:
        if not self.opened[coord] and not self.flags[coord]: self.apply_marker("mark_flag", coord)
    for coord in uncertain:
        if not self.opened[coord] and not self.flags[coord] and not self.questions[coord]: self.apply_marker("mark_question", coord)

def choose_action(self) -> PlannedAction:
    self._sync_markers()
    if not self.opened.any(): return PlannedAction("reveal", (int(self.rng.integers(0, self.rows)), int(self.rng.integers(0, self.cols))))
    
    if not self.defuse_enabled:
        # 失能后不能再放探测器
        if self.adjudication_opportunity:
            center = self._choose_adjudicate_center()
            if center is not None: return PlannedAction("adjudicate", center)
        safe, _, _ = self._constraint_hints()
        safe_target = self._choose_hidden_target(safe)
        if safe_target is not None and safe_target in safe: return PlannedAction("reveal", safe_target)
        hidden = self._choose_hidden_target()
        if hidden is not None: return PlannedAction("reveal", hidden)
        return PlannedAction("wait", None)
        
    predictable = self.in_predictable_mode()
    safe, _, _ = self._constraint_hints()
    safe_target = self._choose_hidden_target(safe)
    if safe_target is not None and safe_target in safe: return PlannedAction("reveal", safe_target)
    
    if self.probe_opportunity and not predictable:
        center = self._choose_probe_center_with_flag_prior()
        if center is not None: return PlannedAction("probe", center)
        
    defuse_target = self._choose_defuse_target()
    if defuse_target is not None:
        self.check_lose_defuse_ability()
        if self.defuse_enabled: return PlannedAction("defuse", defuse_target)
        
    if self.probe_opportunity:
        center = self._choose_probe_center_with_flag_prior()
    if center is not None: return PlannedAction("probe", center)
    
followup = self._number_followup() 
if followup is not None: return PlannedAction("reveal", followup)

hidden = self._choose_hidden_target()
if hidden is not None: return PlannedAction("reveal", hidden)
return PlannedAction("wait", None)

def memory_size(self) -> int:
    return len(self.observation_log)

def memory_heatmap(self) -> np.ndarray:
    heatmap = np.zeros((self.rows, self.cols), dtype=float)
    for coord, entry in self.memory.items(): 
        heatmap[coord] = max(entry.visible_mutation_ratio, entry.confidence * 0.5)
    return heatmap

def estimated_volatility_grid(self) -> np.ndarray:
    grid = np.zeros((self.rows, self.cols), dtype=float)
    for coord, entry in self.memory.items(): 
        grid[coord] = entry.visible_mutation_ratio
    return grid

def known_state_grid(self) -> np.ndarray:
    grid = np.full((self.rows, self.cols), -1, dtype=int)
    for coord, entry in self.memory.items():
        if entry.observed_state == "mine": grid[coord] = 9
        elif entry.observed_state == "empty": grid[coord] = 0
        elif entry.observed_state.startswith("number:"): 
            grid[coord] = int(entry.observed_state.split(":", maxsplit=1)[1])
    return grid
