"""POMCP planner: Monte-Carlo tree search over a particle belief."""
from __future__ import annotations

import math
import random
import time
from typing import Dict, FrozenSet, List, Optional, Tuple

from world_model import STEP, Dynamics, Pos, Settings, State


class _HistoryStats:
    __slots__ = ("visits", "action_visits", "action_values")

    def __init__(self, n_actions: int):
        self.visits = 0
        self.action_visits = [0] * n_actions
        self.action_values = [0.0] * n_actions


class Planner:
    def __init__(self, dynamics: Dynamics, settings: Settings, rng: random.Random):
        self.dynamics = dynamics
        self.settings = settings
        self.rng = rng
        self.tree: Dict[Tuple, _HistoryStats] = {}
        self.last_sim_count = 0

        gamma, eps = settings.discount, settings.value_epsilon
        eps_depth = (math.ceil(math.log(eps) / math.log(gamma))
                     if 0 < gamma < 1 else 10 ** 9)
        self.max_depth = min(settings.horizon, eps_depth)

    def decide(self, particles: List[Tuple[Pos, ...]], small: FrozenSet[Pos],
               heavy: Optional[Pos], time_budget: float) -> int:
        self.tree = {(): _HistoryStats(self.dynamics.n_actions)}
        deadline = time.perf_counter() + time_budget
        n_sims = 0
        while time.perf_counter() < deadline:
            state: State = (self.rng.choice(particles), small, heavy)
            if not self.dynamics.is_terminal(state):
                self._run_simulation(state, ())
            n_sims += 1
        self.last_sim_count = n_sims

        root = self.tree[()]
        candidates = [
            (root.action_values[a], a) for a in range(self.dynamics.n_actions)
            if root.action_visits[a] > 0
        ]
        return max(candidates)[1] if candidates else self.rng.randrange(self.dynamics.n_actions)

    def _select_ucb(self, stats: _HistoryStats) -> int:
        untried = [a for a in range(self.dynamics.n_actions) if stats.action_visits[a] == 0]
        if untried:
            return self.rng.choice(untried)
        log_total = math.log(stats.visits)
        c = self.settings.exploration_const

        def score(a: int) -> float:
            return stats.action_values[a] + c * math.sqrt(log_total / stats.action_visits[a])

        return max(range(self.dynamics.n_actions), key=score)

    def _run_simulation(self, state: State, history: Tuple) -> float:
        trail: List[Tuple[_HistoryStats, int, float]] = []
        depth = 0
        leaf_value = 0.0

        while True:
            if depth >= self.max_depth or self.dynamics.is_terminal(state):
                break
            stats = self.tree.get(history)
            if stats is None:
                self.tree[history] = _HistoryStats(self.dynamics.n_actions)
                leaf_value = self._rollout(state, depth)
                break
            action = self._select_ucb(stats)
            next_state, obs, reward, done = self.dynamics.step(state, action, self.rng)
            shaped = (reward
                      + self.settings.discount * self.dynamics.potential(next_state)
                      - self.dynamics.potential(state))
            trail.append((stats, action, shaped))
            if done:
                break
            history = history + ((action, obs),)
            state = next_state
            depth += 1

        total = leaf_value
        for stats, action, shaped in reversed(trail):
            total = shaped + self.settings.discount * total
            stats.visits += 1
            stats.action_visits[action] += 1
            stats.action_values[action] += (
                (total - stats.action_values[action]) / stats.action_visits[action]
            )
        return total

    def _rollout(self, state: State, depth: int) -> float:
        total, discount = 0.0, 1.0
        phi = self.dynamics.potential(state)
        while depth < self.max_depth and not self.dynamics.is_terminal(state):
            action = self._rollout_policy(state)
            state, _, reward, done = self.dynamics.step(state, action, self.rng)
            next_phi = self.dynamics.potential(state)
            total += discount * (reward + self.settings.discount * next_phi - phi)
            discount *= self.settings.discount
            phi = next_phi
            depth += 1
            if done:
                break
        return total

    def _rollout_policy(self, state: State) -> int:
        if self.rng.random() < self.settings.rollout_explore_p:
            return self.rng.randrange(self.dynamics.n_actions)

        agents, small, heavy = state
        board = self.dynamics.board
        all_boxes = set(small) | ({heavy} if heavy is not None else set())
        open_goals = [g for g in board.goals if g not in all_boxes]
        loose_small = [b for b in small if b not in board.goals]
        heavy_plan = None
        if heavy is not None and heavy not in board.goals and open_goals:
            heavy_plan = self._plan_push(heavy, open_goals, small, heavy)

        directions: List[int] = []
        for pos in agents:
            plan = None
            if loose_small and open_goals:
                target = min(loose_small, key=lambda b: abs(b[0] - pos[0]) + abs(b[1] - pos[1]))
                plan = self._plan_push(target, open_goals, small, heavy)
            plan = plan or heavy_plan
            if plan is None:
                directions.append(self.rng.randrange(4))
                continue
            push_dir, push_from = plan
            directions.append(push_dir if pos == push_from
                               else self._step_toward(pos, push_from, small, heavy))

        if self.dynamics.n_agents == 1:
            return directions[0]
        return directions[0] * 4 + directions[1]

    def _plan_push(self, box: Pos, open_goals: List[Pos], small: FrozenSet[Pos],
                    heavy: Optional[Pos]) -> Optional[Tuple[int, Pos]]:
        goal = min(open_goals, key=lambda g: abs(g[0] - box[0]) + abs(g[1] - box[1]))
        axis_moves = []
        if goal[0] > box[0]:
            axis_moves.append(2)
        if goal[0] < box[0]:
            axis_moves.append(3)
        if goal[1] > box[1]:
            axis_moves.append(1)
        if goal[1] < box[1]:
            axis_moves.append(0)
        self.rng.shuffle(axis_moves)

        board = self.dynamics.board
        for d in axis_moves:
            dx, dy = STEP[d]
            dest = (box[0] + dx, box[1] + dy)
            if self.dynamics.occupied(dest, small, heavy) or dest in board.dead_cells:
                continue
            push_from = (box[0] - dx, box[1] - dy)
            if not board.passable(push_from) or push_from in small or push_from == heavy:
                continue
            return d, push_from
        return None

    def _step_toward(self, pos: Pos, target: Pos, small: FrozenSet[Pos],
                      heavy: Optional[Pos]) -> int:
        best, best_dist = [], float("inf")
        for d in range(4):
            dx, dy = STEP[d]
            cand = (pos[0] + dx, pos[1] + dy)
            if self.dynamics.occupied(cand, small, heavy):
                continue
            dist = abs(cand[0] - target[0]) + abs(cand[1] - target[1])
            if dist < best_dist:
                best, best_dist = [d], dist
            elif dist == best_dist:
                best.append(d)
        return self.rng.choice(best) if best else self.rng.randrange(4)