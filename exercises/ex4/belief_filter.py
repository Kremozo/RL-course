"""Particle-filter belief tracker over the agent(s)' unknown position(s).

Follows Silver & Veness (2010): the belief update is unweighted rejection
sampling driven by the SAME generative model the planner uses -- sample a
particle, simulate the real action on it, keep the outcome only if the
simulated observation matches the real one. Box positions are always fully
known and passed in externally at each update.
"""
from __future__ import annotations

import random
from typing import FrozenSet, Iterable, List, Optional, Tuple

from observation import Window, window_at
from world_model import Dynamics, Pos, Settings, State


class BeliefFilter:
    """Unweighted particle representation of the belief over the agent(s)'
    position(s). Box positions are never part of a particle -- they are
    fully observed and supplied externally at every update."""

    def __init__(self, dynamics: Dynamics, settings: Settings,
                 rng: random.Random):
        self.dynamics = dynamics
        self.settings = settings
        self.rng = rng
        self.particles: List[Tuple[Pos, ...]] = []

    # ------------------------------------------------------------------
    def seed_uniform(self, small: FrozenSet[Pos], heavy: Optional[Pos]) -> None:
        """Fill the belief with `belief_size` particles, drawing each
        agent's coordinate independently and uniformly over every box-free
        floor cell (bulk sampling-with-replacement, one call per agent)."""
        cells = self.dynamics.board.free_agent_cells(small, heavy)
        n = self.settings.belief_size
        per_agent_draws = [
            self.rng.choices(cells, k=n) for _ in range(self.dynamics.n_agents)
        ]
        self.particles = list(zip(*per_agent_draws))

    def cells_matching(self, window: Window, small: FrozenSet[Pos],
                        heavy: Optional[Pos]) -> List[Pos]:
        """Every cell whose egocentric window equals `window` exactly."""
        board = self.dynamics.board
        matches: List[Pos] = []
        for cell in board.free_agent_cells(small, heavy):
            if window_at(cell, board, small, heavy) == window:
                matches.append(cell)
        return matches

    # ------------------------------------------------------------------
    def _draw_and_test(self, action: int, real_obs: Tuple[Window, ...],
                        small_before: FrozenSet[Pos], heavy_before: Optional[Pos],
                        small_after: FrozenSet[Pos], heavy_after: Optional[Pos]
                        ) -> Optional[Tuple[Pos, ...]]:
        idx = self.rng.randrange(len(self.particles))
        state: State = (self.particles[idx], small_before, heavy_before)
        next_state, sim_obs, _reward, _done = self.dynamics.step(
            state, action, self.rng
        )
        if next_state[1] != small_after or next_state[2] != heavy_after:
            return None
        if sim_obs != real_obs:
            return None
        return next_state[0]

    def _reinvigorate(self, n_needed: int, real_obs: Tuple[Window, ...],
                       small: FrozenSet[Pos], heavy: Optional[Pos]
                       ) -> Iterable[Tuple[Pos, ...]]:
        pools = [self.cells_matching(w, small, heavy) for w in real_obs]
        pools = [
            pool if pool else self.dynamics.board.free_agent_cells(small, heavy)
            for pool in pools
        ]
        per_agent_draws = [self.rng.choices(pool, k=n_needed) for pool in pools]
        return zip(*per_agent_draws)

    def update(self, action: int, real_obs: Tuple[Window, ...],
               small_before: FrozenSet[Pos], heavy_before: Optional[Pos],
               small_after: FrozenSet[Pos], heavy_after: Optional[Pos]) -> None:
        target = self.settings.belief_size
        cap = self.settings.resample_attempts_cap

        refreshed: List[Tuple[Pos, ...]] = []
        trials = 0
        while len(refreshed) < target and trials < cap:
            trials += 1
            hit = self._draw_and_test(
                action, real_obs, small_before, heavy_before,
                small_after, heavy_after,
            )
            if hit is not None:
                refreshed.append(hit)

        shortfall = target - len(refreshed)
        if shortfall > 0:
            refreshed.extend(
                self._reinvigorate(shortfall, real_obs, small_after, heavy_after)
            )

        self.particles = refreshed