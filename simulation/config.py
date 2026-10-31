"""仿真配置数据类"""
from __future__ import annotations
import math
from dataclasses import dataclass

@dataclass(slots=True)
class EnvironmentConfig:
    rows: int = 20
    cols: int = 20
    initial_state_rate_b1: float = 0.15
    infection_trigger_c1: float = 1.2
    infection_center_c2: float = 0.3
    infection_step_h1: float = 0.05
    empty_reveal_min_h2: int = 4
    empty_reveal_max_h2: int = 12
    mutation_feedback_rate: float = 0.2
    def __post_init__(self) -> None: pass

@dataclass(slots=True)
class CouplingConfig:
    expected_alpha: float = 0.6
    expected_beta: float = 0.4
    actual_gamma: float = 0.5
    actual_delta: float = 0.5
    theta: float = 0.02
    def __post_init__(self) -> None: pass

@dataclass(slots=True)
class AgentConfig:
    probe_interval_d2: int = 10
    probe_duration_d6: int = 15
    flag_probe_probability_i: float = 0.3
    memory_record_limit: int = 1000
    memory_decay_ratio_c3: float = 0.22
    mine_reveal_cost_d3: int = 5
    adjudicate_cost_d4: int = 6
    probe_placement_cost_d5: int = 2
    compare_radius: int = 2
    layer_penalty_e: float = 0.08
    number_followup_threshold: float = 0.22
    number_followup_probability: float = 0.60
    probe_estimate_window: int = 32
    @property
    def N(self) -> int: return 400
    @property
    def disable_threshold(self) -> float: return self.N * 0.02
    def __post_init__(self) -> None: pass

@dataclass(slots=True)
class SimulationConfig:
    seed: int = 7
    max_action_steps: int = 10000
    snapshot_interval: int = 100

@dataclass(slots=True)
class ScanConfig:
    b1_start: float = 0.02
    b1_end: float = 0.30
    b1_points: int = 5
    theta_start: float = -0.10
    theta_end: float = 0.10
    theta_points: int = 5
    seed_start: int = 7
    seed_count: int = 3
    max_runs: int = 500
