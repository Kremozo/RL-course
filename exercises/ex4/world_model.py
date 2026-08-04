"""Map layouts, settings, board geometry, and the stochastic generative
model shared by planning and belief update."""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, NamedTuple, Optional, Tuple

import numpy as np

from observation import Window, window_at

Pos = Tuple[int, int]


class State(NamedTuple):
    agents: Tuple[Pos, ...]
    small: FrozenSet[Pos]
    heavy: Optional[Pos]


TWO_ROBOT_LAYOUT: List[str] = [
    "WWWWWWWW",
    "WA  A  W",
    "W   C  W",
    "W B    W",
    "W      W",
    "W G G  W",
    "WWWWWWWW",
]

ONE_ROBOT_LAYOUT: List[str] = [
    "WWWWWWWW",
    "W    A W",
    "W      W",
    "W  B   W",
    "W      W",
    "W G    W",
    "WWWWWWWW",
]

LAYOUTS = {"single": ONE_ROBOT_LAYOUT, "two_agent": TWO_ROBOT_LAYOUT}


class Move(NamedTuple):
    delta: Pos
    left: int
    right: int


DIRECTIONS: Dict[int, Move] = {
    0: Move((0, -1), 2, 3),
    1: Move((0, 1), 2, 3),
    2: Move((1, 0), 0, 1),
    3: Move((-1, 0), 0, 1),
}
STEP: Dict[int, Pos] = {d: m.delta for d, m in DIRECTIONS.items()}
SIDEWAYS: Dict[int, Tuple[int, int]] = {d: (m.left, m.right) for d, m in DIRECTIONS.items()}
_PUSH_DELTAS = tuple(m.delta for m in DIRECTIONS.values())
_CHAR_ROLE = {"W": "wall", "A": "agent", "B": "small", "C": "heavy", "G": "goal"}


@dataclass
class Settings:
    budget_fast: float = 1.0
    budget_slow: float = 20.0
    episodes_per_setting: int = 30
    max_steps_per_episode: int = 200
    belief_size: int = 500
    resample_attempts_cap: int = 20_000
    exploration_const: float = 3.0
    horizon: int = 50
    value_epsilon: float = 0.01
    discount: float = 0.95
    move_cost: float = -1.0
    terminal_bonus: float = 0.0
    shaping_coef: float = 0.5
    rollout_explore_p: float = 0.2
    scenario: str = "single"
    random_start: bool = True
    seed: int = 0

    @property
    def layout(self) -> List[str]:
        return LAYOUTS[self.scenario]


def _parse_layout(layout: List[str]) -> Tuple[np.ndarray, Dict[str, List[Pos]]]:
    height, width = len(layout), len(layout[0])
    walls = np.zeros((height, width), dtype=bool)
    buckets: Dict[str, List[Pos]] = defaultdict(list)
    for y, row in enumerate(layout):
        for x, ch in enumerate(row):
            role = _CHAR_ROLE.get(ch)
            if role == "wall":
                walls[y, x] = True
            elif role is not None:
                buckets[role].append((x, y))
    return walls, buckets


def _compute_dead_cells(walls: np.ndarray, floor_cells: List[Pos],
                         goals: FrozenSet[Pos]) -> FrozenSet[Pos]:
    height, width = walls.shape

    def passable(p: Pos) -> bool:
        x, y = p
        return 0 <= x < width and 0 <= y < height and not walls[y, x]

    reachable = set(goals)
    changed = True
    while changed:
        changed = False
        for cell in floor_cells:
            if cell in reachable:
                continue
            for dx, dy in _PUSH_DELTAS:
                dest = (cell[0] + dx, cell[1] + dy)
                pusher = (cell[0] - dx, cell[1] - dy)
                if dest in reachable and passable(pusher):
                    reachable.add(cell)
                    changed = True
                    break
    return frozenset(c for c in floor_cells if c not in reachable)


class Board:
    def __init__(self, layout: List[str]):
        self.walls, buckets = _parse_layout(layout)
        self.height, self.width = self.walls.shape
        self.agent_starts: List[Pos] = buckets.get("agent", [])
        self.small_start: FrozenSet[Pos] = frozenset(buckets.get("small", []))
        heavy_list = buckets.get("heavy", [])
        self.heavy_start: Optional[Pos] = heavy_list[0] if heavy_list else None
        self.goals: FrozenSet[Pos] = frozenset(buckets.get("goal", []))
        self.floor_cells: List[Pos] = [
            (x, y) for y in range(self.height) for x in range(self.width)
            if not self.walls[y, x]
        ]
        self.n_agents = len(self.agent_starts)
        self.dead_cells: FrozenSet[Pos] = _compute_dead_cells(
            self.walls, self.floor_cells, self.goals
        )

    def passable(self, p: Pos) -> bool:
        x, y = p
        return 0 <= x < self.width and 0 <= y < self.height and not self.walls[y, x]

    def free_agent_cells(self, small: FrozenSet[Pos],
                         heavy: Optional[Pos]) -> List[Pos]:
        occupied = set(small)
        if heavy is not None:
            occupied.add(heavy)
        return [c for c in self.floor_cells if c not in occupied]


def _blocked(board: Board, pos: Pos, small: FrozenSet[Pos],
             heavy: Optional[Pos]) -> bool:
    if not board.passable(pos):
        return True
    return pos in small or pos == heavy


def _drift(direction: int, rng: random.Random) -> int:
    options = (direction, *SIDEWAYS[direction])
    return rng.choices(options, weights=(0.8, 0.1, 0.1))[0]


def _resolve_agent_step(board: Board, direction: int, pos: Pos,
                         small: FrozenSet[Pos], heavy: Optional[Pos],
                         rng: random.Random) -> Tuple[Pos, FrozenSet[Pos]]:
    dx, dy = STEP[direction]
    target = (pos[0] + dx, pos[1] + dy)
    if not board.passable(target):
        return pos, small

    if target in small:
        beyond = (target[0] + dx, target[1] + dy)
        if _blocked(board, beyond, small, heavy):
            return pos, small
        if rng.random() < 0.8:
            return target, frozenset((small - {target}) | {beyond})
        return pos, small

    if heavy is not None and target == heavy:
        return pos, small

    ax, ay = STEP[_drift(direction, rng)]
    dest = (pos[0] + ax, pos[1] + ay)
    if _blocked(board, dest, small, heavy):
        return pos, small
    return dest, small


def _try_joint_heavy_push(board: Board, p0: Pos, direction: int,
                           small: FrozenSet[Pos], heavy: Pos,
                           rng: random.Random) -> Optional[Tuple[Pos, Pos]]:
    """None means "not a heavy-push scenario, resolve normally". Otherwise
    returns the (new agent position, new heavy position) outcome -- which
    may equal the inputs if the push attempt failed or was blocked."""
    dx, dy = STEP[direction]
    target = (p0[0] + dx, p0[1] + dy)
    if target != heavy:
        return None
    beyond = (target[0] + dx, target[1] + dy)
    if _blocked(board, beyond, small, None):
        return p0, heavy
    if rng.random() < 0.8:
        return target, beyond
    return p0, heavy


class Dynamics:
    def __init__(self, board: Board, settings: Settings):
        self.board = board
        self.settings = settings
        self.n_agents = board.n_agents
        self.n_actions = 4 ** self.n_agents
        self._obs_memo: Dict[Tuple, Tuple[Window, ...]] = {}

    def split_action(self, joint_action: int) -> Tuple[int, ...]:
        if self.n_agents == 1:
            return (joint_action,)
        return (joint_action // 4, joint_action % 4)

    def occupied(self, pos: Pos, small: FrozenSet[Pos],
                 heavy: Optional[Pos]) -> bool:
        return _blocked(self.board, pos, small, heavy)

    def is_terminal(self, state: State) -> bool:
        _, small, heavy = state
        boxes = set(small) | ({heavy} if heavy is not None else set())
        return self.board.goals <= boxes

    def is_deadlocked(self, state: State) -> bool:
        _, small, heavy = state
        dead = self.board.dead_cells
        return heavy in dead or any(b in dead for b in small)

    def potential(self, state: State) -> float:
        w = self.settings.shaping_coef
        if w == 0.0:
            return 0.0
        _, small, heavy = state
        goals = self.board.goals
        total = sum(min(abs(b[0] - g[0]) + abs(b[1] - g[1]) for g in goals) for b in small)
        if heavy is not None:
            total += min(abs(heavy[0] - g[0]) + abs(heavy[1] - g[1]) for g in goals)
        return -w * total

    def observe(self, state: State) -> Tuple[Window, ...]:
        agents, small, heavy = state
        key = (agents, small, heavy)
        if key not in self._obs_memo:
            self._obs_memo[key] = tuple(
                window_at(p, self.board, small, heavy) for p in agents
            )
        return self._obs_memo[key]

    def step(self, state: State, joint_action: int, rng: random.Random
             ) -> Tuple[State, Tuple[Window, ...], float, bool]:
        agents, small, heavy = state
        directions = self.split_action(joint_action)

        if self.n_agents == 2:
            p0, p1 = agents
            d0, d1 = directions
            if heavy is not None and p0 == p1 and d0 == d1:
                outcome = _try_joint_heavy_push(self.board, p0, d0, small, heavy, rng)
                if outcome is not None:
                    new_pos, heavy = outcome
                    next_state = State((new_pos, new_pos), small, heavy)
                    obs = self.observe(next_state)
                    return (next_state, obs, self.settings.move_cost,
                            self.is_terminal(next_state))

            p0, small = _resolve_agent_step(self.board, d0, p0, small, heavy, rng)
            p1, small = _resolve_agent_step(self.board, d1, p1, small, heavy, rng)
            next_state = State((p0, p1), small, heavy)
        else:
            (p0,) = agents
            p0, small = _resolve_agent_step(self.board, directions[0], p0, small, heavy, rng)
            next_state = State((p0,), small, heavy)

        obs = self.observe(next_state)
        done = self.is_terminal(next_state)
        reward = self.settings.move_cost + (self.settings.terminal_bonus if done else 0.0)
        return next_state, obs, reward, done


class Simulator:
    def __init__(self, dynamics: Dynamics, settings: Settings, rng: random.Random):
        self.dynamics = dynamics
        self.settings = settings
        self.rng = rng
        self._state: Optional[State] = None
        self.elapsed_steps = 0

    def reset(self):
        board = self.dynamics.board
        small, heavy = board.small_start, board.heavy_start
        if self.settings.random_start:
            cells = board.free_agent_cells(small, heavy)
            agents = tuple(self.rng.choice(cells) for _ in range(board.n_agents))
        else:
            agents = tuple(board.agent_starts)
        self._state = State(agents, small, heavy)
        self.elapsed_steps = 0
        return self.dynamics.observe(self._state), {"small": small, "heavy": heavy}

    def step(self, joint_action: int):
        next_state, obs, reward, done = self.dynamics.step(
            self._state, joint_action, self.rng
        )
        self._state = next_state
        self.elapsed_steps += 1
        truncated = self.elapsed_steps >= self.settings.max_steps_per_episode and not done
        info = {"small": next_state[1], "heavy": next_state[2]}
        return obs, reward, done, truncated, info

    @property
    def true_state(self) -> State:
        return self._state