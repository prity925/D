from __future__ import annotations
import numpy as np
from simulation.config import EnvironmentConfig
from simulation.models import Coord, MOORE_OFFSETS

CARDINAL_OFFSETS = ((1, 0), (-1, 0), (0, 1), (0, -1))

class DynamicEnvironment:
    def __init__(self, config: EnvironmentConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng
        self.rows = config.rows
        self.cols = config.cols
        self.elapsed_ticks = 0
        self.minefield = (self.rng.random((self.rows, self.cols)) < config.initial_state_rate_b1)
        self.initial_mine_ratio_a1 = float(self.minefield.mean())
        self.region_weight = self._build_region_weights()
        self.b2 = np.clip(self.initial_mine_ratio_a1 * self.region_weight, 0.0, 1.0)
        self.initial_average_mutation_b3 = float(self.b2.mean())
        self.frozen = np.zeros((self.rows, self.cols), dtype=bool)
        self.resample_count = np.zeros((self.rows, self.cols), dtype=int)
        self.state_change_count = np.zeros((self.rows, self.cols), dtype=int)
        self.last_state_change = np.zeros((self.rows, self.cols), dtype=bool)
        self.neighbor_counts = np.zeros((self.rows, self.cols), dtype=int)
        self.observable_grid = np.zeros((self.rows, self.cols), dtype=int)
        self.scan_global_state()
        self.initial_number_ratio_a2 = self.current_number_ratio()

    @property
    def initial_state_rate_b1(self) -> float:
        return self.config.initial_state_rate_b1

    @property
    def current_mutation(self) -> np.ndarray:
        return self.b2

    @property
    def mutation_count(self) -> np.ndarray:
        return self.state_change_count

    @property
    def initial_number_ratio(self) -> float:
        return self.initial_number_ratio_a2

    def _build_region_weights(self) -> np.ndarray:
        weights = np.ones((self.rows, self.cols), dtype=float)
        row_edges = np.linspace(0, self.rows, 4, dtype=int)
        col_edges = np.linspace(0, self.cols, 4, dtype=int)
        region_id = np.zeros((self.rows, self.cols), dtype=int)
        for row_band in range(3):
            for col_band in range(3):
                region_id[row_edges[row_band]:row_edges[row_band + 1], col_edges[col_band]:col_edges[col_band + 1]] = row_band * 3 + col_band
        region_id[region_id == 1] = 0
        region_id[region_id == 7] = 8
        for region in range(9):
            if region in (1, 7):
                continue
            mask = region_id == region
            if mask.any():
                weights[mask] = self.rng.uniform(0.5, 1.5)
        return weights

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    def neighbors(self, coord: Coord, moore: bool = True) -> list[Coord]:
        offsets = MOORE_OFFSETS if moore else CARDINAL_OFFSETS
        row, col = coord
        return [(row + dr, col + dc) for dr, dc in offsets if self.in_bounds(row + dr, col + dc)]

    def _scan_neighbor_mines(self) -> np.ndarray:
        padded = np.pad(self.minefield.astype(int), 1, mode='constant')
        counts = np.zeros((self.rows, self.cols), dtype=int)
        for dr, dc in MOORE_OFFSETS:
            counts += padded[1 + dr : 1 + dr + self.rows, 1 + dc : 1 + dc + self.cols]
        return counts

    def scan_global_state(self) -> None:
        self.neighbor_counts = self._scan_neighbor_mines()
        self.observable_grid = self.neighbor_counts.copy()
        self.observable_grid[self.minefield] = 9

    def observable_value(self, coord: Coord) -> int:
        return int(self.observable_grid[coord])

    def observable_label(self, coord: Coord) -> str:
        value = self.observable_value(coord)
        if value == 9:
            return "mine"
        if value == 0:
            return "empty"
        return f"number:{value}"

    def mine_ratio(self) -> float:
        return float(self.minefield.mean())

    def current_number_ratio(self) -> float:
        mask = np.logical_and(~self.minefield, self.neighbor_counts > 0)
        return float(mask.mean())

    def empty_ratio(self) -> float:
        mask = np.logical_and(~self.minefield, self.neighbor_counts == 0)
        return float(mask.mean())

    def global_average_mutation(self) -> float:
        return float(self.b2.mean())

    def reveal(self, coord: Coord, opened: np.ndarray) -> dict[str, object]:
        if opened[coord]:
            return {"hit_mine": False, "observations": []}
        revealed = [coord]
        hit_mine = bool(self.minefield[coord])
        if not hit_mine and self.neighbor_counts[coord] == 0:
            candidates = [n for n in self.neighbors(coord) if not opened[n] and not self.minefield[n]]
            if candidates:
                lower = min(self.config.empty_reveal_min_h2, self.config.empty_reveal_max_h2)
                upper = max(self.config.empty_reveal_min_h2, self.config.empty_reveal_max_h2)
                count = min(len(candidates), int(self.rng.integers(lower, upper + 1)))
                if count > 0:
                    indices = self.rng.choice(len(candidates), size=count, replace=False)
                    revealed.extend(candidates[int(i)] for i in np.atleast_1d(indices))
        observations = [(cell, self.observable_label(cell)) for cell in revealed]
        return {"hit_mine": hit_mine, "observations": observations}

    def probe_scan(self, center: Coord) -> dict[Coord, str]:
        return {coord: self.observable_label(coord) for coord in self.neighbors(center)}

    def freeze_cells(self, cells: set[Coord]) -> None:
        for row, col in cells:
            self.frozen[row, col] = True

    def unfreeze_cells(self, cells: set[Coord]) -> None:
        for row, col in cells:
            self.frozen[row, col] = False

    def boost_b2_around(self, center: Coord, delta: float) -> None:
        row, col = center
        top = max(0, row - 1)
        bottom = min(self.rows, row + 2)
        left = max(0, col - 1)
        right = min(self.cols, col + 2)
        self.b2[top:bottom, left:right] = np.clip(self.b2[top:bottom, left:right] + delta, 0.0, 1.0)

    def defuse(self, coord: Coord) -> str:
        if self.minefield[coord]:
            self.minefield[coord] = False
            self.state_change_count[coord] += 1
            self.last_state_change[coord] = True
            result = "mine"
        else:
            self.boost_b2_around(coord, self.config.infection_step_h1)
            result = "non_mine"
        self.scan_global_state()
        return result

    def _conduction_active(self) -> bool:
        return self.mine_ratio() > (self.initial_number_ratio_a2 * self.config.infection_trigger_c1)

    def _apply_conduction(self) -> None:
        if not self._conduction_active(): return
        selection_probability = float(np.clip(self.initial_number_ratio_a2 * self.config.infection_center_c2, 0.0, 1.0))
        for row, col in np.argwhere(self.minefield):
            center = (int(row), int(col))
            if self.rng.random() >= selection_probability: continue
        neighbors = self.neighbors(center)
        non_empty_neighbors = [coord for coord in neighbors if self.observable_label(coord) != "empty"]
        n = len(non_empty_neighbors)
        if n == 0: continue
        count = n // 2 if self.rng.random() < 0.5 else (n + 1) // 2
        candidates = [center, *neighbors]
        if count <= 0: continue
        indices = self.rng.choice(len(candidates), size=min(count, len(candidates)), replace=False)
        for index in np.atleast_1d(indices):
            coord = candidates[int(index)]
            if self.rng.random() < 0.7:
                delta = self.config.infection_step_h1
            else:
                delta = -self.config.infection_step_h1
            self.b2[coord] = np.clip(self.b2[coord] + delta, 0.0, 1.0)

def step(self) -> None:
    self.elapsed_ticks += 1
    mutable = ~self.frozen
    previous = self.minefield.copy()
    flip = self.rng.random((self.rows, self.cols)) < self.b2
    self.minefield[mutable] ^= flip[mutable]
    self.resample_count[mutable] += 1
    self.last_state_change = np.logical_and(previous != self.minefield, mutable)
    self.state_change_count[self.last_state_change] += 1
    rate = self.config.mutation_feedback_rate
    self.b2[mutable] += rate * (self.last_state_change[mutable].astype(float) - 0.5)
    self.scan_global_state()
    self._apply_conduction()
    self.b2 = np.clip(self.b2, 0.0, 1.0)
