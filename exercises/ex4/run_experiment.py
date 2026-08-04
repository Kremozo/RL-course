"""Online plan/act/observe/update loop, benchmark harness, smoke tests, CLI."""
from __future__ import annotations

import argparse
import itertools
import random
import time
from typing import Iterator, List, Optional, Tuple

import numpy as np

from belief_filter import BeliefFilter
from pomcp import Planner
from world_model import Board, Dynamics, Settings, Simulator, State


def _align_belief_with_first_observation(belief: BeliefFilter, board: Board,
                                          obs, small, heavy, settings: Settings,
                                          rng: random.Random) -> None:
    pools = [belief.cells_matching(w, small, heavy) or board.free_agent_cells(small, heavy)
             for w in obs]
    belief.particles = list(zip(*(rng.choices(pool, k=settings.belief_size) for pool in pools)))


def run_single_episode(settings: Settings, time_budget: float, seed: int,
                        verbose: bool = False) -> int:
    rng = random.Random(seed)
    board = Board(settings.layout)
    dynamics = Dynamics(board, settings)
    sim = Simulator(dynamics, settings, rng)
    planner = Planner(dynamics, settings, rng)
    belief = BeliefFilter(dynamics, settings, rng)

    obs, info = sim.reset()
    small, heavy = info["small"], info["heavy"]
    belief.seed_uniform(small, heavy)
    _align_belief_with_first_observation(belief, board, obs, small, heavy, settings, rng)

    steps = 0
    done = False
    while not done and steps < settings.max_steps_per_episode:
        action = planner.decide(belief.particles, small, heavy, time_budget)
        obs, reward, terminated, truncated, info = sim.step(action)
        belief.update(action, obs, small, heavy, info["small"], info["heavy"])
        small, heavy = info["small"], info["heavy"]
        steps += 1
        done = terminated or truncated
        if verbose:
            print(f"    step {steps:>3}  action={action:<2} "
                  f"sims={planner.last_sim_count:<6} "
                  f"belief_support={len(set(belief.particles))}")
    return steps


def _episodes_for_setting(settings: Settings, budget: float, verbose: bool
                           ) -> Iterator[Tuple[int, int, float]]:
    for run in range(settings.episodes_per_setting):
        t0 = time.perf_counter()
        steps = run_single_episode(settings, budget, seed=settings.seed + run,
                                    verbose=(verbose and run == 0))
        yield run, steps, time.perf_counter() - t0


def run_benchmark(base_settings: Settings, scenarios: Optional[List[str]] = None,
                   budgets: Optional[List[float]] = None,
                   verbose_first: bool = False, results_path: Optional[str] = None):
    scenarios = scenarios or ["single", "two_agent"]
    budgets = budgets or [base_settings.budget_fast, base_settings.budget_slow]
    results = {}
    log: List[str] = []

    def emit(line: str = "") -> None:
        print(line)
        log.append(line)

    for scenario, budget in itertools.product(scenarios, budgets):
        settings = Settings(**{**base_settings.__dict__, "scenario": scenario})
        emit(f"\n=== scenario={scenario}  budget={budget}s  "
             f"({settings.episodes_per_setting} runs) ===")
        step_counts = []
        for run, steps, dt in _episodes_for_setting(settings, budget, verbose_first):
            step_counts.append(steps)
            emit(f"  run {run + 1:>2}/{settings.episodes_per_setting}: "
                 f"steps={steps:<4} ({dt:.1f}s)")
        arr = np.array(step_counts, dtype=float)
        results[(scenario, budget)] = (arr.mean(), arr.std())
        emit(f"  --> mean={arr.mean():.2f}  std={arr.std():.2f}")

    emit("\n" + "=" * 62)
    emit(f"{'scenario':<12} {'budget':>8} {'mean steps':>12} {'std':>10}")
    emit("-" * 62)
    for (scenario, budget), (mean, std) in results.items():
        emit(f"{scenario:<12} {budget:>7}s {mean:>12.2f} {std:>10.2f}")
    emit("=" * 62)

    if results_path:
        with open(results_path, "w") as f:
            f.write("\n".join(log) + "\n")
    return results


def _check_dynamics(scenario: str) -> Tuple[Board, Dynamics, int]:
    rng = random.Random(0)
    settings = Settings(scenario=scenario, belief_size=200)
    board = Board(settings.layout)
    dynamics = Dynamics(board, settings)

    n_boxes = len(board.small_start) + (1 if board.heavy_start else 0)
    assert n_boxes == len(board.goals)

    s0 = State(tuple(board.agent_starts), board.small_start, board.heavy_start)
    assert dynamics.observe(s0) == dynamics.observe(s0)
    assert isinstance(hash(dynamics.observe(s0)), int)

    state = s0
    for _ in range(300):
        state, _, _, done = dynamics.step(state, rng.randrange(dynamics.n_actions), rng)
        agents, small, heavy = state
        assert len(small) + (1 if heavy else 0) == n_boxes
        assert all(not board.walls[y, x] for x, y in agents)
        if done:
            break
    print(f"  [{scenario}] board/observation/dynamics OK")
    return board, dynamics, n_boxes


def _check_belief_localizes(scenario: str) -> None:
    settings = Settings(scenario=scenario, belief_size=300)
    rng = random.Random(1)
    dynamics = Dynamics(Board(settings.layout), settings)
    sim = Simulator(dynamics, settings, rng)
    belief = BeliefFilter(dynamics, settings, rng)

    obs, info = sim.reset()
    small, heavy = info["small"], info["heavy"]
    belief.seed_uniform(small, heavy)
    for _ in range(12):
        a = rng.randrange(dynamics.n_actions)
        obs, _, term, _, info = sim.step(a)
        belief.update(a, obs, small, heavy, info["small"], info["heavy"])
        small, heavy = info["small"], info["heavy"]
        if term:
            break

    true_agents = sim.true_state[0]
    frac = sum(p == true_agents for p in belief.particles) / len(belief.particles)
    print(f"  [{scenario}] belief: true state holds {frac:.0%} of the "
          f"particles after 12 random steps")
    assert any(p == true_agents for p in belief.particles)


def run_smoke_tests() -> None:
    print("== smoke tests ==")
    for scenario in ("single", "two_agent"):
        _check_dynamics(scenario)
        _check_belief_localizes(scenario)
    print("== all smoke tests passed ==\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="POMDP/POMCP box pushing")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--verbose", action="store_true",
                        help="print per-step diagnostics for the first run of each setting")
    parser.add_argument("--scenario", choices=["single", "two_agent", "both"], default="both")
    parser.add_argument("--budget", type=float, default=None,
                        help="single time budget in seconds (default: budget_fast and budget_slow)")
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--results-file", type=str, default="results.txt")
    args = parser.parse_args()

    if args.selftest:
        run_smoke_tests()
        return

    settings = Settings()
    if args.runs:
        settings.episodes_per_setting = args.runs

    scenarios = ["single", "two_agent"] if args.scenario == "both" else [args.scenario]
    budgets = [args.budget] if args.budget is not None else [settings.budget_fast, settings.budget_slow]

    run_benchmark(settings, scenarios=scenarios, budgets=budgets,
                  verbose_first=args.verbose, results_path=args.results_file)


if __name__ == "__main__":
    main()