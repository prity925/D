"""仿真引擎"""
from __future__ import annotations
from dataclasses import replace
from itertools import product
from typing import Any
import numpy as np

from simulation.agent import ConstrainedAgentFinal as ConstrainedAgent
from simulation.config import AgentConfig, CouplingConfig, EnvironmentConfig, ScanConfig, SimulationConfig
from simulation.environment import DynamicEnvironment
from simulation.models import CellObservation, Coord, IntervalEvent, PlannedAction, SimulationResult

def _expected_f(environment: DynamicEnvironment, coupling: CouplingConfig) -> float:
    value = (coupling.expected_alpha * environment.initial_mine_ratio_a1 + coupling.expected_beta * environment.global_average_mutation())
    return float(np.clip(value, 0.0, 1.0))

def _actual_g(environment: DynamicEnvironment, agent: ConstrainedAgent, coupling: CouplingConfig) -> float:
    value = (coupling.actual_gamma * environment.initial_state_rate_b1 + coupling.actual_delta * agent.estimated_mutation_b4 + coupling.theta)
    return float(np.clip(value, 0.0, 1.0))

def _comparison(environment: DynamicEnvironment, agent: ConstrainedAgent, coupling: CouplingConfig) -> str:
    f_value = _expected_f(environment, coupling)
    g_value = _actual_g(environment, agent, coupling)
    return "g_lt_f" if g_value < f_value else "g_ge_f"

def _capture_snapshot(*, action: str, environment: DynamicEnvironment, agent: ConstrainedAgent, coupling: CouplingConfig) -> dict[str, Any]:
    f_value = _expected_f(environment, coupling)
    g_value = _actual_g(environment, agent, coupling)
    opened = agent.opened.copy()
    frozen = environment.frozen.copy()
    return {
        "step": agent.action_step, "action_step": agent.action_step, "environment_tick": environment.elapsed_ticks,
        "decision_index": agent.decision_index, "decision_steps": agent.decision_index, "action": action,
        "minefield": environment.minefield.copy(), "revealed": opened, "opened": opened.copy(),
        "neighbor_counts": environment.neighbor_counts.copy(), "mutation_rate": environment.b2.copy(),
        "mutation_count": environment.state_change_count.copy(), "state_change_count": environment.state_change_count.copy(),
        "resample_count": environment.resample_count.copy(), "stabilization_mask": frozen, "frozen_cells": int(frozen.sum()),
        "flags": agent.flags.copy(), "questions": agent.questions.copy(), "known_state_grid": agent.known_state_grid(),
        "memory_heatmap": agent.memory_heatmap(), "b4_grid": agent.estimated_volatility_grid(),
        "estimated_mutation": agent.estimated_mutation_b4, "actual_mutation": environment.global_average_mutation(),
        "f": f_value, "expected_line": f_value, "g": g_value, "actual_pressure": g_value,
        "a3": environment.mine_ratio(), "number_ratio": environment.current_number_ratio(),
        "initial_number_ratio": environment.initial_number_ratio_a2, "a1": environment.initial_mine_ratio_a1,
        "anchor": agent.current_anchor, "disabled": agent.disabled, "stabilizing": agent.in_stabilization,
        "defuse_enabled": agent.defuse_enabled,
    }

def _history_row(*, action: str, action_target: Coord | None, action_phase: str, comparison: str, environment: DynamicEnvironment, agent: ConstrainedAgent, coupling: CouplingConfig, interval_due_step: int | None = None, interval_priority: int | None = None) -> dict[str, Any]:
    target_row = action_target[0] if action_target is not None else None
    target_col = action_target[1] if action_target is not None else None
    b3, b4 = environment.global_average_mutation(), agent.estimated_mutation_b4
    f_value, g_value = _expected_f(environment, coupling), _actual_g(environment, agent, coupling)
    a3 = environment.mine_ratio()
    return {
        "action_step": agent.action_step, "environment_tick": environment.elapsed_ticks, "decision_index": agent.decision_index,
        "action": action, "action_phase": action_phase, "interval_due_step": interval_due_step, "interval_priority": interval_priority,
        "target_row": target_row, "target_col": target_col, "a1": environment.initial_mine_ratio_a1,
        "a2": environment.initial_number_ratio_a2, "a3": a3, "b1": environment.initial_state_rate_b1,
        "b3": b3, "b4": b4, "f": f_value, "g": g_value, "theta": coupling.theta, "mine_ratio": a3,
        "number_ratio": environment.current_number_ratio(), "revealed_mine_ratio": agent.revealed_mine_ratio(),
        "defuse_cost": agent.defuse_cost(), "mutation_gap": b4 - b3, "mutation_error": abs(b4 - b3),
        "comparison": comparison, "actual_comparison": "g_lt_f" if g_value < f_value else "g_ge_f",
        "active_probes": len(agent.probes), "probe_opportunity": agent.probe_opportunity,
        "adjudication_opportunity": agent.adjudication_opportunity, "memory_size": agent.memory_size(),
        "opened_cells": int(agent.opened.sum()), "flags": int(agent.flags.sum()), "questions": int(agent.questions.sum()),
        "defuse_enabled": agent.defuse_enabled, "frozen_cells": int(environment.frozen.sum()),
        "disable_events": agent.overload_events, "defused_mines": agent.defused_mines,
    }

def _probe_readings(environment: DynamicEnvironment, agent: ConstrainedAgent) -> dict[Coord, dict[Coord, str]]:
    return {probe.center: environment.probe_scan(probe.center) for probe in agent.probes if probe.remaining_steps > 0}

def _advance_action_ticks(*, count: int, action: PlannedAction, action_phase: str, comparison: str, environment: DynamicEnvironment, agent: ConstrainedAgent, coupling: CouplingConfig, simulation: SimulationConfig, history: list[dict[str, Any]], snapshots: list[dict[str, Any]], capture_snapshots: bool, counts_for_probe_grant: bool = True) -> int:
    if agent.action_step + count > simulation.max_action_steps: return 0
    for _ in range(count):
        environment.step()
        agent.advance_action_tick(_probe_readings(environment, agent), counts_for_probe_grant=counts_for_probe_grant)
        history.append(_history_row(action=action.action_type, action_target=action.target, action_phase=action_phase, comparison=comparison, environment=environment, agent=agent, coupling=coupling))
        if capture_snapshots and simulation.snapshot_interval > 0:
            if (agent.action_step - 1) // simulation.snapshot_interval < agent.action_step // simulation.snapshot_interval:
                snapshots.append(_capture_snapshot(action=action.action_type, environment=environment, agent=agent, coupling=coupling))
    return count

def _complete_action(*, action: PlannedAction, action_start_step: int, comparison: str, environment: DynamicEnvironment, agent: ConstrainedAgent, agent_config: AgentConfig, coupling: CouplingConfig, simulation: SimulationConfig, history: list[dict[str, Any]], snapshots: list[dict[str, Any]], capture_snapshots: bool) -> None:
    for event in agent.collect_clearing_events():
        agent.apply_interval_event(event)
        history.append(_history_row(action=action.action_type, action_target=action.target, action_phase=f"interval_{event.event_type}", comparison=comparison, environment=environment, agent=agent, coupling=coupling, interval_due_step=event.due_step, interval_priority=event.priority))

def _cell_observations(raw: list[tuple[Coord, str]], *, action_step: int, revealed: bool, source: str) -> list[CellObservation]:
    return [CellObservation(coord=coord, state=state, action_step=action_step, revealed=revealed, source=source) for coord, state in raw]

def _handle_reveal(*, action: PlannedAction, environment: DynamicEnvironment, agent: ConstrainedAgent, agent_config: AgentConfig, coupling: CouplingConfig, simulation_config: SimulationConfig, comparison: str, history: list[dict[str, Any]], snapshots: list[dict[str, Any]], capture_snapshots: bool, **_kwargs: Any) -> bool:
    hit_mine = bool(environment.minefield[action.target])
    extra_steps = agent_config.mine_reveal_cost_d3 if hit_mine else 0
    cost = 1 + extra_steps
    if agent.action_step + cost > simulation_config.max_action_steps: return False
    action_start_step = agent.action_step
    _advance_action_ticks(count=1, action=action, action_phase="reveal_mine" if hit_mine else "reveal", comparison=comparison, environment=environment, agent=agent, coupling=coupling, simulation=simulation_config, history=history, snapshots=snapshots, capture_snapshots=capture_snapshots, counts_for_probe_grant=True)
    if extra_steps:
        _advance_action_ticks(count=extra_steps, action=action, action_phase="reveal_mine_penalty", comparison=comparison, environment=environment, agent=agent, coupling=coupling, simulation=simulation_config, history=history, snapshots=snapshots, capture_snapshots=capture_snapshots, counts_for_probe_grant=False)
    reveal = environment.reveal(action.target, agent.opened)
    agent.observe_cells(_cell_observations(reveal["observations"], action_step=agent.action_step, revealed=True, source="reveal"))
    history.append(_history_row(action=action.action_type, action_target=action.target, action_phase="reveal_mine_result" if hit_mine else "reveal_done", comparison=comparison, environment=environment, agent=agent, coupling=coupling))
    _complete_action(action=action, action_start_step=action_start_step, comparison=comparison, environment=environment, agent=agent, agent_config=agent_config, coupling=coupling, simulation=simulation_config, history=history, snapshots=snapshots, capture_snapshots=capture_snapshots)
    return True

def _handle_probe(*, action: PlannedAction, environment: DynamicEnvironment, agent: ConstrainedAgent, agent_config: AgentConfig, coupling: CouplingConfig, simulation_config: SimulationConfig, comparison: str, history: list[dict[str, Any]], snapshots: list[dict[str, Any]], capture_snapshots: bool, **_kwargs: Any) -> bool:
    cost = agent_config.probe_placement_cost_d5
    if agent.action_step + cost > simulation_config.max_action_steps: return False
    action_start_step = agent.action_step
    agent.consume_probe_opportunity()
    _advance_action_ticks(count=cost, action=action, action_phase="probe_deployment", comparison=comparison, environment=environment, agent=agent, coupling=coupling, simulation=simulation_config, history=history, snapshots=snapshots, capture_snapshots=capture_snapshots, counts_for_probe_grant=True)
    agent.place_probe(action.target, environment.probe_scan(action.target), agent_config.probe_duration_d6)
    history.append(_history_row(action=action.action_type, action_target=action.target, action_phase="probe_deployed", comparison=comparison, environment=environment, agent=agent, coupling=coupling))
    _complete_action(action=action, action_start_step=action_start_step, comparison=comparison, environment=environment, agent=agent, agent_config=agent_config, coupling=coupling, simulation=simulation_config, history=history, snapshots=snapshots, capture_snapshots=capture_snapshots)
    return True

def _handle_defuse(*, action: PlannedAction, environment: DynamicEnvironment, agent: ConstrainedAgent, agent_config: AgentConfig, coupling: CouplingConfig, simulation_config: SimulationConfig, comparison: str, history: list[dict[str, Any]], snapshots: list[dict[str, Any]], capture_snapshots: bool, **_kwargs: Any) -> bool:
    agent.check_lose_defuse_ability()
    if not agent.defuse_enabled:
        action_start_step = agent.action_step
        agent.pending_interval_events.append(IntervalEvent(due_step=agent.action_step, priority=4, event_type="adjudication_grant"))
        history.append(_history_row(action=action.action_type, action_target=action.target, action_phase="defuse_disabled", comparison=comparison, environment=environment, agent=agent, coupling=coupling))
        _complete_action(action=action, action_start_step=action_start_step, comparison=comparison, environment=environment, agent=agent, agent_config=agent_config, coupling=coupling, simulation=simulation_config, history=history, snapshots=snapshots, capture_snapshots=capture_snapshots)
        return True
    cost = agent.defuse_cost_steps()
    if agent.action_step + cost > simulation_config.max_action_steps: return False
    action_start_step = agent.action_step
    _advance_action_ticks(count=cost, action=action, action_phase="defuse_work", comparison=comparison, environment=environment, agent=agent, coupling=coupling, simulation=simulation_config, history=history, snapshots=snapshots, capture_snapshots=capture_snapshots, counts_for_probe_grant=True)
    result = environment.defuse(action.target)
    success = result == "mine"
    if success: agent.defused_mines += 1
    else: agent.failed_defuses += 1
    agent.record_defuse_result(success)
    agent.observe_cells([CellObservation(coord=action.target, state=environment.observable_label(action.target), action_step=agent.action_step, revealed=True, source="defuse")])
    history.append(_history_row(action=action.action_type, action_target=action.target, action_phase=f"defuse_result_{result}", comparison=comparison, environment=environment, agent=agent, coupling=coupling))
    _complete_action(action=action, action_start_step=action_start_step, comparison=comparison, environment=environment, agent=agent, agent_config=agent_config, coupling=coupling, simulation=simulation_config, history=history, snapshots=snapshots, capture_snapshots=capture_snapshots)
    return True

def _handle_adjudicate(*, action: PlannedAction, environment: DynamicEnvironment, agent: ConstrainedAgent, agent_config: AgentConfig, coupling: CouplingConfig, simulation_config: SimulationConfig, comparison: str, history: list[dict[str, Any]], snapshots: list[dict[str, Any]], capture_snapshots: bool, **_kwargs: Any) -> bool:
    cost = agent_config.adjudicate_cost_d4
    if agent.action_step + cost > simulation_config.max_action_steps: return False
    action_start_step = agent.action_step
    agent.consume_adjudication_opportunity()
    center = action.target
    frozen_neighbors = set(agent.neighbors(center))
    environment.freeze_cells(frozen_neighbors)
    _advance_action_ticks(count=cost, action=action, action_phase="adjudicate_work", comparison=comparison, environment=environment, agent=agent, coupling=coupling, simulation=simulation_config, history=history, snapshots=snapshots, capture_snapshots=capture_snapshots, counts_for_probe_grant=True)
    environment.unfreeze_cells(frozen_neighbors)
    cells = agent.adjudication_region(center)
    agent.observe_cells(_cell_observations([(coord, environment.observable_label(coord)) for coord in cells], action_step=agent.action_step, revealed=True, source="adjudicate"))
    history.append(_history_row(action=action.action_type, action_target=action.target, action_phase="adjudicate_done", comparison=comparison, environment=environment, agent=agent, coupling=coupling))
    _complete_action(action=action, action_start_step=action_start_step, comparison=comparison, environment=environment, agent=agent, agent_config=agent_config, coupling=coupling, simulation=simulation_config, history=history, snapshots=snapshots, capture_snapshots=capture_snapshots)
    return True

def _handle_wait(*, action: PlannedAction, environment: DynamicEnvironment, agent: ConstrainedAgent, agent_config: AgentConfig, coupling: CouplingConfig, simulation_config: SimulationConfig, comparison: str, history: list[dict[str, Any]], snapshots: list[dict[str, Any]], capture_snapshots: bool, **_kwargs: Any) -> bool:
    if agent.action_step + 1 > simulation_config.max_action_steps: return False
    action_start_step = agent.action_step
    _advance_action_ticks(count=1, action=action, action_phase="wait", comparison=comparison, environment=environment, agent=agent, coupling=coupling, simulation=simulation_config, history=history, snapshots=snapshots, capture_snapshots=capture_snapshots, counts_for_probe_grant=True)
    _complete_action(action=action, action_start_step=action_start_step, comparison=comparison, environment=environment, agent=agent, agent_config=agent_config, coupling=coupling, simulation=simulation_config, history=history, snapshots=snapshots, capture_snapshots=capture_snapshots)
    return True

def run_simulation(*, environment_config: EnvironmentConfig, agent_config: AgentConfig, coupling_config: CouplingConfig | None = None, simulation_config: SimulationConfig, capture_snapshots: bool = True) -> SimulationResult:
    coupling = coupling_config or CouplingConfig()
    seed_sequence = np.random.SeedSequence(simulation_config.seed)
    environment_seed, agent_seed = seed_sequence.spawn(2)
    environment = DynamicEnvironment(environment_config, np.random.default_rng(environment_seed))
    agent = ConstrainedAgent(agent_config, (environment.rows, environment.cols), np.random.default_rng(agent_seed), initial_mine_ratio_a1=environment.initial_mine_ratio_a1)

    history: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    if capture_snapshots:
        snapshots.append(_capture_snapshot(action="initial", environment=environment, agent=agent, coupling=coupling))

    history.append(_history_row(action="initial", action_target=None, action_phase="initial", comparison=_comparison(environment, agent, coupling), environment=environment, agent=agent, coupling=coupling))

    error_message: str | None = None
    termination_reason = "max_action_steps"
    try:
        while agent.action_step < simulation_config.max_action_steps:
            agent.update_environment_state(environment.global_average_mutation())
            action = agent.choose_action()
            agent.register_decision()
            comparison = _comparison(environment, agent, coupling)
            action_completed: bool
            
            if action.action_type == "reveal" and action.target is not None:
                action_completed = _handle_reveal(action=action, environment=environment, agent=agent, agent_config=agent_config, coupling=coupling, simulation_config=simulation_config, comparison=comparison, history=history, snapshots=snapshots, capture_snapshots=capture_snapshots)
            elif action.action_type == "probe" and action.target is not None:
                action_completed = _handle_probe(action=action, environment=environment, agent=agent, agent_config=agent_config, coupling=coupling, simulation_config=simulation_config, comparison=comparison, history=history, snapshots=snapshots, capture_snapshots=capture_snapshots)
            elif action.action_type == "defuse" and action.target is not None:
                action_completed = _handle_defuse(action=action, environment=environment, agent=agent, agent_config=agent_config, coupling=coupling, simulation_config=simulation_config, comparison=comparison, history=history, snapshots=snapshots, capture_snapshots=capture_snapshots)
            elif action.action_type == "adjudicate" and action.target is not None:
                action_completed = _handle_adjudicate(action=action, environment=environment, agent=agent, agent_config=agent_config, coupling=coupling, simulation_config=simulation_config, comparison=comparison, history=history, snapshots=snapshots, capture_snapshots=capture_snapshots)
            else:
                action_completed = _handle_wait(action=action, environment=environment, agent=agent, agent_config=agent_config, coupling=coupling, simulation_config=simulation_config, comparison=comparison, history=history, snapshots=snapshots, capture_snapshots=capture_snapshots)
                
            if not action_completed:
                termination_reason = "insufficient_budget_for_atomic_action"
                history.append(_history_row(action=action.action_type, action_target=action.target, action_phase="action_not_started_insufficient_budget", comparison=comparison, environment=environment, agent=agent, coupling=coupling))
                break
    except Exception as exc:
        error_message = str(exc)
        termination_reason = "error"

    if capture_snapshots:
        snapshots.append(_capture_snapshot(action="final", environment=environment, agent=agent, coupling=coupling))

    summary: dict[str, Any] = {
        "executed_steps": agent.action_step, "action_steps": agent.action_step, "environment_ticks": environment.elapsed_ticks,
        "decision_steps": agent.decision_index, "remaining_budget": max(0, simulation_config.max_action_steps - agent.action_step),
        "termination_reason": termination_reason, "a1": environment.initial_mine_ratio_a1, "a2": environment.initial_number_ratio_a2,
        "a3": environment.mine_ratio(), "initial_number_ratio": environment.initial_number_ratio_a2,
        "final_mine_ratio": environment.mine_ratio(), "final_number_ratio": environment.current_number_ratio(),
        "final_actual_mutation": environment.global_average_mutation(), "final_estimated_mutation": agent.estimated_mutation_b4,
        "final_f": _expected_f(environment, coupling), "final_actual_pressure": _actual_g(environment, agent, coupling),
        "final_g": _actual_g(environment, agent, coupling), "theta": coupling.theta,
        "mutation_gap": agent.estimated_mutation_b4 - environment.global_average_mutation(),
        "defused_mines": agent.defused_mines, "failed_defuses": agent.failed_defuses, "disable_events": agent.overload_events,
        "overload_events": agent.overload_events, "placed_probes": agent.placed_probes, "probe_update_events": agent.probe_update_events,
        "probe_expiration_events": agent.probe_expiration_events, "memory_decay_events": agent.memory_decay_events,
        "probe_grant_events": agent.probe_grant_events, "adjudication_grant_events": agent.adjudication_grant_events,
        "adjudications_used": agent.adjudications_used, "global_mutation_count": int(environment.state_change_count.sum()),
        "active_probes": len(agent.probes), "frozen_cells": int(environment.frozen.sum()),
        "revealed_cells": int(agent.opened.sum()), "remaining_mines": int(environment.minefield.sum()),
    }
    if error_message is not None: summary["error"] = error_message

    return {
        "history": history, "snapshots": snapshots, "summary": summary,
        "parameter_profile": {
            "b1": environment_config.initial_state_rate_b1, "a1": environment.initial_mine_ratio_a1,
            "c1": environment_config.infection_trigger_c1, "c2": environment_config.infection_center_c2,
            "h1": environment_config.infection_step_h1, "mutation_feedback_rate": environment_config.mutation_feedback_rate,
            "probe_estimate_window": agent_config.probe_estimate_window, "theta": coupling.theta,
            "alpha": coupling.expected_alpha, "beta": coupling.expected_beta,
            "gamma": coupling.actual_gamma, "delta": coupling.actual_delta,
            "i": agent_config.flag_probe_probability_i, "c3": agent_config.memory_decay_ratio_c3,
            "max_steps": simulation_config.max_action_steps,
        },
    }

def run_parameter_scan(*, environment_config: EnvironmentConfig, agent_config: AgentConfig, coupling_config: CouplingConfig, simulation_config: SimulationConfig, scan_config: ScanConfig) -> list[dict[str, Any]]:
    b1_values = np.linspace(scan_config.b1_start, scan_config.b1_end, scan_config.b1_points)
    theta_values = np.linspace(scan_config.theta_start, scan_config.theta_end, scan_config.theta_points)
    seeds = range(scan_config.seed_start, scan_config.seed_start + scan_config.seed_count)
    combinations = list(product(b1_values, theta_values, seeds))
    if len(combinations) > scan_config.max_runs:
        raise ValueError(f"scan requests {len(combinations)} runs; maximum is {scan_config.max_runs}")
    rows: list[dict[str, Any]] = []
    for run_index, (b1, theta, seed) in enumerate(combinations, start=1):
        environment = replace(environment_config, initial_state_rate_b1=float(b1))
        coupling = replace(coupling_config, theta=float(theta))
        simulation = replace(simulation_config, seed=int(seed))
        result = run_simulation(environment_config=environment, agent_config=agent_config, coupling_config=coupling, simulation_config=simulation, capture_snapshots=False)
        for history_row in result["history"]:
            rows.append({"run_id": run_index, "scan_b1": float(b1), "scan_theta": float(theta), "seed": int(seed), **history_row})
    return rows
