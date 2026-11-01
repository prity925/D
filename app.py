from __future__ import annotations
from dataclasses import replace
import pandas as pd
import streamlit as st

from simulation.config import AgentConfig, CouplingConfig, EnvironmentConfig, ScanConfig, SimulationConfig
from simulation.engine import run_parameter_scan, run_simulation
from simulation.formulas import FormulaError, evaluate_expression
from simulation.visualization import (
    ACTION_LABELS, build_cell_stats_frame, build_history_frame, build_mutation_rate_grid_frame,
    build_state_grid_frame, critical_point_steps, plot_grid_panels, plot_minesweeper_board,
    plot_mutation_gap, plot_summary_bars, plot_volatility_heatmap, style_mutation_rate_grid, style_state_grid,
)

st.set_page_config(page_title="双系统场域耦合模型 DSFCM", page_icon="▦", layout="wide")

def expression_input(label: str, default: str, *, key: str, minimum: float | None = None, maximum: float | None = None) -> float:
    raw = st.sidebar.text_input(label, value=default, key=key)
    try:
        return evaluate_expression(raw, default=float(default), minimum=minimum, maximum=maximum)
    except FormulaError as exc:
        st.sidebar.warning(f"{label} 使用默认值 {default}：{exc}")
        return evaluate_expression(default, default=float(default), minimum=minimum, maximum=maximum)

def build_environment_config() -> EnvironmentConfig:
    st.sidebar.subheader("系统 A")
    b1 = expression_input("b1 初始全局场波动率", "0.15", key="b1", minimum=0.0, maximum=1.0)
    c1 = expression_input("c1 波动传导密度触发阈值", "1.2", key="c1", minimum=1.000001, maximum=10.0)
    c2 = expression_input("c2 传导激活动作选择概率", "0.3", key="c2", minimum=0.0, maximum=1.0)
    h1 = expression_input("h1 单次波动传导步长", "0.05", key="h1", minimum=0.0, maximum=1.0)
    h2_min = int(st.sidebar.number_input("h2 波及下限", 0, 16, 4, 1))
    h2_max = int(st.sidebar.number_input("h2 波及上限", 0, 16, 12, 1))
    return EnvironmentConfig(initial_state_rate_b1=b1, infection_trigger_c1=c1, infection_center_c2=c2, infection_step_h1=h1, empty_reveal_min_h2=h2_min, empty_reveal_max_h2=h2_max)

def build_coupling_config() -> CouplingConfig:
    st.sidebar.subheader("AB 耦合")
    alpha = expression_input("α：f 中 a3 系数", "0.6", key="alpha", minimum=0.0, maximum=1.0)
    beta = 1.0 - alpha
    st.sidebar.caption(f"β 自动取 1−α = {beta:.3f}")
    gamma = expression_input("γ：g 中 a2 系数", "0.5", key="gamma", minimum=0.0, maximum=1.0)
    delta = expression_input("δ：g 中 b4 系数", "0.5", key="delta", minimum=0.0, maximum=1.0)
    theta = expression_input("θ 残差项", "0.02", key="theta", minimum=-1.0, maximum=1.0)
    return CouplingConfig(expected_alpha=alpha, expected_beta=beta, actual_gamma=gamma, actual_delta=delta, theta=theta)

def build_agent_config() -> AgentConfig:
    st.sidebar.subheader("系统 B")
    d2 = int(st.sidebar.number_input("d2 探测冷却间隔", 1, 120, 10, 1))
    d3 = int(st.sidebar.number_input("d3 踩雷追加惩罚步", 0, 60, 5, 1))
    d4 = int(st.sidebar.number_input("d4 判定动作步数", 1, 60, 6, 1))
    d5 = int(st.sidebar.number_input("d5 部署探测器步数", 1, 60, 2, 1))
    d6 = int(st.sidebar.number_input("d6 探测器持续步数", 1, 120, 15, 1))
    i = expression_input("i 标记雷旁主动探测先验", "0.3", key="probe_i", minimum=0.0, maximum=1.0)
    memory_limit = int(st.sidebar.number_input("记忆数据库记录上限", 1, 100_000, 1000, 100))
    c3 = expression_input("c3 记忆衰减单次删除比例", "0.22", key="c3", minimum=0.0, maximum=1.0)
    estimate_window = int(st.sidebar.number_input("探测估计窗口", 2, 1000, 32, 1))
    return AgentConfig(probe_interval_d2=d2, probe_duration_d6=d6, flag_probe_probability_i=i, memory_record_limit=memory_limit, memory_decay_ratio_c3=c3, mine_reveal_cost_d3=d3, adjudicate_cost_d4=d4, probe_placement_cost_d5=d5, probe_estimate_window=estimate_window)

def build_simulation_config() -> SimulationConfig:
    st.sidebar.subheader("仿真")
    seed = int(st.sidebar.number_input("随机种子", 0, 1_000_000, 7, 1))
    max_steps = int(st.sidebar.slider("最大总步数 T", 100, 20000, 10000, 100))
    snapshot = int(st.sidebar.slider("快照间隔", 10, 500, 100, 10))
    return SimulationConfig(seed=seed, max_action_steps=max_steps, snapshot_interval=snapshot)
def scan_controls(simulation: SimulationConfig) -> tuple[ScanConfig, int]:
    col1, col2, col3 = st.columns(3)
    with col1:
        b1_start = st.number_input("b1 起点", 0.0, 1.0, 0.02, 0.01, key="scan_b1_start")
        b1_end = st.number_input("b1 终点", 0.0, 1.0, 0.30, 0.01, key="scan_b1_end")
        b1_points = int(st.number_input("b1 点数", 1, 30, 5, 1, key="scan_b1_points"))
    with col2:
        theta_start = st.number_input("θ 起点", -1.0, 1.0, -0.10, 0.01, key="scan_theta_start")
        theta_end = st.number_input("θ 终点", -1.0, 1.0, 0.10, 0.01, key="scan_theta_end")
        theta_points = int(st.number_input("θ 点数", 1, 30, 5, 1, key="scan_theta_points"))
    with col3:
        seed_start = int(st.number_input("种子起点", 0, 1_000_000, simulation.seed, 1, key="scan_seed_start"))
        seed_count = int(st.number_input("种子数量", 1, 100, 3, 1, key="scan_seed_count"))
        st.metric("扫描轮数", b1_points * theta_points * seed_count)
    scan_steps = int(st.number_input("扫描单轮最大步数", 20, 20_000, min(1000, simulation.max_action_steps), 100, key="scan_steps"))
    return ScanConfig(b1_start=float(b1_start), b1_end=float(b1_end), b1_points=b1_points, theta_start=float(theta_start), theta_end=float(theta_end), theta_points=theta_points, seed_start=seed_start, seed_count=seed_count), scan_steps

st.title("双系统场域耦合模型 DSFCM")
environment_config = build_environment_config()
coupling_config = build_coupling_config()
agent_config = build_agent_config()
simulation_config = build_simulation_config()

run_clicked = st.sidebar.button("运行单次仿真", type="primary", width="stretch")
current_configs = ("dynamic-feedback-v1", environment_config, coupling_config, agent_config, simulation_config)

if run_clicked:
    with st.spinner("正在运行单次仿真…"):
        st.session_state.simulation_result = run_simulation(environment_config=environment_config, agent_config=agent_config, coupling_config=coupling_config, simulation_config=simulation_config)
    st.session_state.simulation_configs = current_configs
elif ("simulation_result" in st.session_state and st.session_state.get("simulation_configs") != current_configs):
    st.warning("侧边栏参数已修改，下方结果仍对应上次运行的参数，请点击「运行单次仿真」刷新。")

if "simulation_result" not in st.session_state:
    st.info("请先在左侧设置参数，然后点击「运行单次仿真」按钮。")
    st.stop()

result = st.session_state.simulation_result
history = build_history_frame(result["history"])
summary = result["summary"]
snapshots = result["snapshots"]

metric_row_one = st.columns(3)
metric_row_one[0].metric("行动 / 环境步", f"{summary['action_steps']} / {summary['environment_ticks']}")
metric_row_one[1].metric("a1 / a3", f"{summary['a1']:.3f} / {summary['a3']:.3f}")
metric_row_one[2].metric("b3 / b4", f"{summary['final_actual_mutation']:.3f} / {summary['final_estimated_mutation']:.3f}")
metric_row_two = st.columns(3)
metric_row_two[0].metric("f 预期承受线", f"{summary['final_f']:.3f}")
metric_row_two[1].metric("g 真实承受线", f"{summary['final_g']:.3f}")
metric_row_two[2].metric("雷比例 / 数字格比例", f"{summary['final_mine_ratio']:.3f} / {summary['final_number_ratio']:.3f}")
metric_row_three = st.columns(3)
metric_row_three[0].metric("全局变异次数(翻转)", f"{summary.get('global_mutation_count', 0)}")
metric_row_three[1].metric("失能次数", f"{summary.get('disable_events', 0)}")
metric_row_three[2].metric("判定次数", f"{summary.get('adjudications_used', 0)}")

tab_trace, tab_board, tab_raw, tab_snapshot, tab_scan = st.tabs(["单次轨迹", "雷场回放", "原始数据", "快照", "参数扫描"])

with tab_trace:
    trace_controls = st.columns(2)
    with trace_controls[0]: trace_view = st.radio("曲线视图", ["分行波形", "叠加比较"], horizontal=True)
    with trace_controls[1]: trace_range = st.selectbox("显示范围", ["最近 500 步", "最近 2000 步", "全程"])
    trace_history = history if trace_range == "全程" else history[history["action_step"] >= history["action_step"].max() - (500 if trace_range == "最近 500 步" else 2000)]
    enable_criterion = st.checkbox("启用建议判据（默认关闭；红色竖虚线标记候选临界点）", value=False, key="chaos_criterion_enabled")
    critical_steps = critical_point_steps(history) if enable_criterion else []
    st.pyplot(plot_mutation_gap(trace_history, critical_steps=critical_steps, separate_axes=trace_view == "分行波形"), clear_figure=True, width="stretch")
    left, right = st.columns([2.2, 1])
    with left: st.dataframe(history[["action_step", "environment_tick", "action_label", "b3", "b4", "f", "g", "a3"]].tail(12), width="stretch", hide_index=True)
    with right: st.pyplot(plot_summary_bars(summary), clear_figure=True, width="stretch")

with tab_board:
    if not snapshots: st.info("本次运行未采集快照。")
    else:
        board_index = st.slider("回放步", 0, len(snapshots) - 1, len(snapshots) - 1, key="board_snapshot_index") if len(snapshots) > 1 else 0
        board_snapshot = snapshots[board_index]
        board_metrics = st.columns(4)
        board_metrics[0].metric("行动步数", board_snapshot["action_step"])
        board_metrics[1].metric("环境步数", board_snapshot["environment_tick"])
        board_metrics[2].metric("决策次数", board_snapshot["decision_index"])
        board_metrics[3].metric("当前动作", ACTION_LABELS.get(board_snapshot["action"], board_snapshot["action"]))
        heatmap_view = st.radio("场域热力图视图", ["A 真实雷分布", "B 认知波动率估计"], horizontal=True, key="board_heatmap_view")
        st.pyplot((plot_minesweeper_board(board_snapshot) if heatmap_view == "A 真实雷分布" else plot_volatility_heatmap(board_snapshot)), clear_figure=True, width="stretch")

with tab_raw:
    st.dataframe(history, width="stretch", hide_index=True, height=620)
    interval_rows = history[history["interval_due_step"].notna()].copy()
    if not interval_rows.empty:
        interval_rows["event_type"] = interval_rows["action_phase"].str.replace("interval_", "", regex=False)
        st.subheader("事件日志（清算窗口）")
        st.dataframe(interval_rows[["interval_due_step", "interval_priority", "event_type", "action_step", "action", "memory_size"]], width="stretch", hide_index=True)
    st.download_button("下载单次原始 CSV", data=history.to_csv(index=False).encode("utf-8-sig"), file_name="single_run_raw.csv", mime="text/csv")

with tab_snapshot:
    if not snapshots: st.info("本次运行未采集快照。")
    else:
        snapshot_index = st.slider("快照", 0, len(snapshots) - 1, len(snapshots) - 1) if len(snapshots) > 1 else 0
        selected = snapshots[snapshot_index]
        st.caption(f"行动步 {selected['action_step']} · 环境步 {selected['environment_tick']} · 决策 {selected['decision_index']} · {ACTION_LABELS.get(selected['action'], selected['action'])}")
        st.pyplot(plot_grid_panels(selected), clear_figure=True, width="stretch")
        state_col, b2_col = st.columns(2)
        with state_col: st.dataframe(style_state_grid(build_state_grid_frame(selected)), width="stretch")
        with b2_col: st.dataframe(style_mutation_rate_grid(build_mutation_rate_grid_frame(selected)), width="stretch")
        st.dataframe(build_cell_stats_frame(selected), width="stretch", hide_index=True, height=420)

  with tab_scan:
    scan_config, scan_steps = scan_controls(simulation_config)
    requested_runs = scan_config.b1_points* scan_config.theta_points * scan_config.seed_count
    scan_clicked = st.button("运行参数扫描", type="primary", disabled=requested_runs > scan_config.max_runs)
    if requested_runs > scan_config.max_runs:
       st.error(f"当前为 {requested_runs} 轮，最多允许 {scan_config.max_runs} 轮。")
    if scan_clicked:
       with st.spinner(f"正在运行 {requested_runs} 轮仿真"):
           rows = run_parameter_scan(environment_config=environment_config, agent_config=agent_config, coupling_config=coupling_config, simulation_config=replace(simulation_config, max_action_steps=scan_steps), scan_config=scan_config)
       st.session_state.scan_frame = pd.DataFrame(rows)
       st.session_state.scan_configs = (current_configs, scan_config)
    scan_frame = st.session_state.get("scan_frame")
    if isinstance(scan_frame, pd.DataFrame) and not scan_frame.empty:
      if st.session_state.get("scan_configs") != (current_configs, scan_config):
        st.warning("侧边栏或扫描参数已修改，下方扫描结果仍对应上次运行的参数，请重新点击「运行参数扫描」。")
    run_ids = sorted(scan_frame["run_id"].unique().tolist())
    selected_runs = st.multiselect("运行编号", run_ids, default=run_ids[: min(5, len(run_ids))])
    filtered = scan_frame[scan_frame["run_id"].isin(selected_runs)] if selected_runs else scan_frame
    st.dataframe(filtered, width="stretch", hide_index=True, height=620)
    st.download_button("下载扫描原始 CSV", data=scan_frame.to_csv(index=False).encode("utf-8-sig"), file_name="b1_theta_scan_raw.csv", mime="text/csv") 
