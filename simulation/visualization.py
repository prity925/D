"""可视化工具：将仿真历史和网格状态转换为图表和数据框。"""

from __future__ import annotations

import matplotlib

# Streamlit 在工作线程中绘图，使用无窗口后端避免 Tk 主循环冲突。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Micro Hei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


ACTION_LABELS = {
    "initial": "初始",
    "reveal": "翻开",
    "probe": "探测器",
    "defuse": "拔雷",
    "adjudicate": "判定",
    "wait": "等待",
    "final": "最终",
}


def build_history_frame(history: list[dict[str, object]]) -> pd.DataFrame:
    """将历史记录列表转换为 DataFrame，并添加中文动作标签列。"""
    frame = pd.DataFrame(history)
    if frame.empty:
        return frame
    frame["action_label"] = frame["action"].map(ACTION_LABELS).fillna(frame["action"])
    return frame


def critical_point_steps(
    frame: pd.DataFrame,
    *,
    trailing: int = 50,
    baseline: int = 100,
    multiplier: float = 3.0,
    variance_threshold: float = 0.01,
) -> list[int]:
    """按文档第 5 节建议判据返回候选临界点 T 值列表（默认关闭，仅供辅助参考）。"""
    if frame.empty:
        return []
    frame = (
        frame.drop_duplicates(subset="action_step", keep="last")
        .sort_values("action_step")
        .reset_index(drop=True)
    )
    if len(frame) < trailing + baseline:
        return []
    error = frame["mutation_error"].to_numpy(dtype=float)
    b3 = frame["b3"].to_numpy(dtype=float)
    steps = frame["action_step"].to_numpy(dtype=int)
    points: list[int] = []
    for index in range(trailing + baseline - 1, len(frame)):
        current = error[index - trailing + 1 : index + 1]
        prior = error[index - trailing + 1 - baseline : index - trailing + 1]
        if (
            float(np.std(current))
            > multiplier * float(np.mean(prior))
            and float(np.var(b3[index - trailing + 1 : index + 1]))
            > variance_threshold
        ):
            points.append(int(steps[index]))
    return points


def plot_mutation_gap(
    frame: pd.DataFrame,
    *,
    critical_steps: list[int] | None = None,
    separate_axes: bool = True,
) -> Figure:
    """绘制双系统轨迹图：b3/b4/f/g/a3（当前雷比例）随总步数 T 的变化曲线。"""
    figure, axes = plt.subplots(
        5 if separate_axes else 1,
        1,
        figsize=(12, 9 if separate_axes else 5.6),
        sharex=True,
        squeeze=False,
    )
    axis = axes[0, 0]
    if frame.empty:
        axis.set_title("双系统场域混沌模型轨迹")
        return figure

    # 动作结果与间隔事件行共享同一 action_step，只保留每步末行避免垂直锯齿。
    frame = frame.drop_duplicates(subset="action_step", keep="last")
    x = frame["action_step"].to_numpy(dtype=float)
    series = [
        ("b3", "b3 全局平均波动率", "#b42318"),
        ("b4", "b4 B 估计波动率", "#1565c0"),
        ("f", "f 预期承受线（动态）", "#087f5b"),
        ("g", "g 真实承受线（动态）", "#6a1b9a"),
        ("a3", "a3 当前雷比例", "#d97706"),
    ]
    for index, (key, label, color) in enumerate(series):
        axis = axes[index if separate_axes else 0, 0]
        axis.plot(x, frame[key], label=label, linewidth=1.0, color=color)
        if separate_axes:
            low, high = float(frame[key].min()), float(frame[key].max())
            margin = max(0.002, (high - low) * 0.15)
            axis.set_ylim(max(0, low - margin), min(1, high + margin))
            axis.set_ylabel(key)
            axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    for axis in axes.flat:
        axis.set_xlim(left=float(x.min()), right=max(float(x.max()), float(x.min()) + 1))
        if not separate_axes:
            axis.set_ylim(0, 1)
            axis.set_ylabel("比例 / 压力")
        for step in critical_steps or []:
            if x.min() <= step <= x.max():
                axis.axvline(step, color="#c53030", linestyle="--", linewidth=1, alpha=0.7)
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right", ncol=1 if separate_axes else 2, fontsize=8)
    axes[-1, 0].set_xlabel("总步数 T")
    axes[0, 0].set_title(
        "双系统场域混沌模型轨迹" + ("（各行独立纵轴，原始数值）" if separate_axes else "")
    )
    figure.tight_layout()
    return figure


def plot_history(frame: pd.DataFrame) -> Figure:
    """绘制仿真历史图（plot_mutation_gap 的别名）。"""
    return plot_mutation_gap(frame)


def plot_summary_bars(summary: dict[str, object]) -> Figure:
    """绘制行动结果汇总柱状图：拔雷/误拔/探测/失效/衰减/失能/判定次数。"""
    labels = ["已拔雷", "误拔", "探测器", "探测失效", "记忆衰减", "失能", "判定"]
    values = [
        int(summary.get("defused_mines", 0)),
        int(summary.get("failed_defuses", 0)),
        int(summary.get("placed_probes", 0)),
        int(summary.get("probe_expiration_events", 0)),
        int(summary.get("memory_decay_events", 0)),
        int(summary.get("overload_events", 0)),
        int(summary.get("adjudications_used", 0)),
    ]
    colors = ["#2f855a", "#c05621", "#2b6cb0", "#805ad5", "#d97706", "#c53030", "#3182ce"]
    figure, axis = plt.subplots(figsize=(6.5, 3.2))
    axis.bar(labels, values, color=colors)
    axis.set_ylabel("次数")
    axis.set_title("行动结果")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    return figure


def _grid_columns(cols: int) -> list[str]:
    return [f"C{col + 1}" for col in range(cols)]


def _grid_index(rows: int) -> list[str]:
    return [f"R{row + 1}" for row in range(rows)]


def build_state_grid_frame(snapshot: dict[str, object]) -> pd.DataFrame:
    """从快照构建状态格网 DataFrame，显示雷/空/数字。"""
    minefield = np.asarray(snapshot["minefield"], dtype=bool)
    counts = np.asarray(snapshot["neighbor_counts"], dtype=int)
    grid = counts.astype(str)
    grid[np.logical_and(~minefield, counts == 0)] = "空"
    grid[minefield] = "雷"
    rows, cols = minefield.shape
    return pd.DataFrame(grid, index=_grid_index(rows), columns=_grid_columns(cols))


def style_state_grid(frame: pd.DataFrame) -> "Styler":
    """为状态格网 DataFrame 应用颜色样式。"""
    def cell_style(value: object) -> str:
        if value == "雷":
            return "background-color:#c53030;color:white;font-weight:700;text-align:center"
        if value == "空":
            return "background-color:#e6fffa;color:#22543d;text-align:center"
        return "background-color:#ebf8ff;color:#1a365d;font-weight:600;text-align:center"

    return frame.style.map(cell_style)


def build_mutation_rate_grid_frame(snapshot: dict[str, object]) -> pd.DataFrame:
    """从快照构建逐格 b2 变异率 DataFrame。"""
    b2 = np.asarray(snapshot["mutation_rate"], dtype=float)
    rows, cols = b2.shape
    return pd.DataFrame(b2, index=_grid_index(rows), columns=_grid_columns(cols))


def style_mutation_rate_grid(frame: pd.DataFrame) -> "Styler":
    """为变异率格网应用 magma 渐变色样式。"""
    return frame.style.background_gradient(cmap="magma", vmin=0.0, vmax=1.0).format("{:.3f}")


def build_cell_stats_frame(snapshot: dict[str, object]) -> pd.DataFrame:
    """从快照构建逐格统计表，包含状态、b2、重采样和状态改变次数。"""
    b2 = np.asarray(snapshot["mutation_rate"], dtype=float)
    changes = np.asarray(snapshot["state_change_count"], dtype=int)
    resamples = np.asarray(snapshot["resample_count"], dtype=int)
    minefield = np.asarray(snapshot["minefield"], dtype=bool)
    counts = np.asarray(snapshot["neighbor_counts"], dtype=int)
    stabilized = np.asarray(
        snapshot.get("stabilization_mask", np.zeros_like(b2, dtype=bool)),
        dtype=bool,
    )
    rows: list[dict[str, object]] = []
    for row in range(b2.shape[0]):
        for col in range(b2.shape[1]):
            state = "雷" if minefield[row, col] else ("空" if counts[row, col] == 0 else str(counts[row, col]))
            rows.append(
                {
                    "坐标": f"R{row + 1}C{col + 1}",
                    "状态": state,
                    "b2": float(b2[row, col]),
                    "重采样次数": int(resamples[row, col]),
                    "变异次数(翻转)": int(changes[row, col]),
                    "判定冻结区": "是" if stabilized[row, col] else "否",
                }
            )
    return pd.DataFrame(rows)


def _truth_grid(snapshot: dict[str, object]) -> np.ndarray:
    minefield = np.asarray(snapshot["minefield"], dtype=bool)
    counts = np.asarray(snapshot["neighbor_counts"], dtype=int)
    grid = counts.copy()
    grid[minefield] = 9
    return grid


def _agent_grid(snapshot: dict[str, object]) -> np.ndarray:
    known = np.asarray(snapshot["known_state_grid"], dtype=int).copy()
    flags = np.asarray(snapshot["flags"], dtype=bool)
    questions = np.asarray(snapshot["questions"], dtype=bool)
    known[questions] = 10
    known[flags] = 11
    return known


def plot_minesweeper_board(snapshot: dict[str, object]) -> Figure:
    """绘制带行动步标题、数字文本和明确雷格的扫雷式真实状态图。"""
    minefield = np.asarray(snapshot["minefield"], dtype=bool)
    counts = np.asarray(snapshot["neighbor_counts"], dtype=int)
    changes = np.asarray(
        snapshot.get("state_change_count", np.zeros_like(minefield, dtype=int)),
        dtype=int,
    )
    b2 = np.asarray(
        snapshot.get("mutation_rate", np.full(minefield.shape, np.nan)), dtype=float,
    )
    stabilized = np.asarray(
        snapshot.get("stabilization_mask", np.zeros_like(minefield, dtype=bool)),
        dtype=bool,
    )
    rows, cols = minefield.shape
    background = np.zeros((rows, cols), dtype=int)
    background[np.logical_and(~minefield, counts > 0)] = 1
    background[minefield] = 2

    figure, axis = plt.subplots(figsize=(13, 13), dpi=160)
    axis.imshow(
        background,
        cmap=ListedColormap(["#f7f7f5", "#e8f1f7", "#c93d35"]),
        vmin=0,
        vmax=2,
    )
    number_colors = {
        1: "#1769aa",
        2: "#287a3d",
        3: "#b42318",
        4: "#5b3f99",
        5: "#8b4513",
        6: "#007c83",
        7: "#202124",
        8: "#5f6368",
    }
    for row in range(rows):
        for col in range(cols):
            if minefield[row, col]:
                label = "雷"
                color = "#ffffff"
            elif counts[row, col] > 0:
                value = int(counts[row, col])
                label = str(value)
                color = number_colors.get(value, "#202124")
            else:
                label = ""
                color = "#202124"
            if label:
                axis.text(
                    col,
                    row - 0.28,
                    label,
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                    fontweight="bold",
                )
            axis.text(
                col,
                row + 0.04,
                f"b2={b2[row, col]:.3f}" if np.isfinite(b2[row, col]) else "b2=未知",
                ha="center",
                va="center",
                fontsize=5.5,
                color="#ffffff" if minefield[row, col] else "#334155",
            )
            axis.text(
                col,
                row + 0.33,
                f"{int(changes[row, col])}次",
                ha="center",
                va="center",
                fontsize=5.5,
                color="#ffffff" if minefield[row, col] else "#334155",
            )
            if stabilized[row, col]:
                axis.add_patch(
                    Rectangle(
                        (col - 0.48, row - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        edgecolor="#008f95",
                        linewidth=1.4,
                    )
                )

    axis.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    axis.grid(which="minor", color="#9aa0a6", linewidth=0.45)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.set_xticks(np.arange(cols))
    axis.set_yticks(np.arange(rows))
    axis.set_xticklabels([f"C{col + 1}" for col in range(cols)], fontsize=5)
    axis.set_yticklabels([f"R{row + 1}" for row in range(rows)], fontsize=5)
    axis.xaxis.tick_top()
    axis.set_title(
        f"行动步 {snapshot['action_step']}  |  环境步 {snapshot['environment_tick']}  |  "
        f"决策 {snapshot['decision_index']}\n每格：状态 / b2 翻转概率 / 累计变异次数",
        fontsize=13,
        pad=18,
    )
    axis.set_xlim(-0.5, cols - 0.5)
    axis.set_ylim(rows - 0.5, -0.5)
    axis.set_aspect("equal")
    figure.tight_layout()
    return figure


def plot_volatility_heatmap(snapshot: dict[str, object]) -> Figure:
    """绘制 B 认知波动率估计热力图（探测器反推的逐格 b2 估计）。"""
    b2_shape = np.asarray(snapshot["mutation_rate"], dtype=float).shape
    grid = np.asarray(
        snapshot.get("b4_grid", np.zeros(b2_shape, dtype=float)),
        dtype=float,
    )
    figure, axis = plt.subplots(figsize=(9.2, 9.2))
    image = axis.imshow(grid, cmap="magma", vmin=0.0, vmax=1.0)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_title(
        f"B 认知波动率估计 | 行动步 {snapshot['action_step']} | "
        f"环境步 {snapshot['environment_tick']} | 决策 {snapshot['decision_index']}",
        fontsize=13,
        pad=14,
    )
    axis.set_xticks([])
    axis.set_yticks([])
    figure.tight_layout()
    return figure


def plot_grid_panels(snapshot: dict[str, object]) -> Figure:
    """绘制 2x3 面板图：真实状态、代理视角、状态改变、b2、记忆强度、重采样。"""
    truth = _truth_grid(snapshot)
    agent = _agent_grid(snapshot)
    changes = np.asarray(snapshot["state_change_count"], dtype=float)
    b2 = np.asarray(snapshot["mutation_rate"], dtype=float)
    stabilized = np.asarray(
        snapshot.get("stabilization_mask", np.zeros_like(b2, dtype=bool)),
        dtype=bool,
    )
    memory = np.asarray(
        snapshot.get(
            "b4_grid",
            snapshot.get("memory_heatmap", np.zeros_like(b2)),
        ),
        dtype=float,
    )
    resamples = np.asarray(snapshot["resample_count"], dtype=float)

    figure, axes = plt.subplots(2, 3, figsize=(13, 8.2))
    state_colors = [
        "#f7fafc",
        "#bee3f8",
        "#90cdf4",
        "#63b3ed",
        "#4299e1",
        "#3182ce",
        "#2b6cb0",
        "#2c5282",
        "#2a4365",
        "#c53030",
    ]
    truth_cmap = ListedColormap(state_colors)
    truth_norm = BoundaryNorm(np.arange(-0.5, 10.5, 1), truth_cmap.N)
    axes[0, 0].imshow(truth, cmap=truth_cmap, norm=truth_norm)
    axes[0, 0].set_title("A 真实状态")

    agent_cmap = ListedColormap(["#edf2f7", *state_colors, "#d69e2e", "#9b2c2c"])
    axes[0, 1].imshow(agent + 1, cmap=agent_cmap, vmin=0, vmax=12)
    axes[0, 1].set_title("B 数据库视角")

    plots = [
        (axes[0, 2], changes, "cividis", "累计状态改变"),
        (axes[1, 0], b2, "magma", "逐格 b2"),
        (axes[1, 1], memory, "viridis", "B 认知波动率估计"),
        (axes[1, 2], resamples, "Blues", "累计重采样"),
    ]
    for axis, values, cmap, title in plots:
        image = axis.imshow(values, cmap=cmap)
        axis.set_title(title)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    if stabilized.any():
        axes[1, 0].contour(
            stabilized.astype(float),
            levels=[0.5],
            colors=["#00a6a6"],
            linewidths=1.2,
        )

    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(
        f"行动步 {snapshot['action_step']} | 环境步 {snapshot['environment_tick']} | "
        f"决策 {snapshot['decision_index']}",
        fontsize=13,
    )
    figure.tight_layout()
    return figure
