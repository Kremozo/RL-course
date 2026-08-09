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
                        verbose: bool = False) -> Tuple[int, bool]:
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
    terminated = False
    truncated = False
    deadlocked = False
    while not (terminated or truncated or deadlocked) and steps < settings.max_steps_per_episode:
        action = planner.decide(belief.particles, small, heavy, time_budget)
        obs, reward, terminated, truncated, info = sim.step(action)
        belief.update(action, obs, small, heavy, info["small"], info["heavy"])
        small, heavy = info["small"], info["heavy"]
        steps += 1
        deadlocked = (not terminated) and dynamics.is_deadlocked(sim.true_state)
        if verbose:
            print(f"    step {steps:>3}  action={action:<2} "
                  f"sims={planner.last_sim_count:<6} "
                  f"belief_support={len(set(belief.particles))}")
    success = terminated and not deadlocked
    return steps, success


def _episodes_for_setting(settings: Settings, budget: float, verbose: bool
                           ) -> Iterator[Tuple[int, int, bool, float]]:
    for run in range(settings.episodes_per_setting):
        t0 = time.perf_counter()
        steps, success = run_single_episode(settings, budget, seed=settings.seed + run,
                                             verbose=(verbose and run == 0))
        yield run, steps, success, time.perf_counter() - t0


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
        successes = []
        for run, steps, success, dt in _episodes_for_setting(settings, budget, verbose_first):
            step_counts.append(steps)
            successes.append(success)
            emit(f"  run {run + 1:>2}/{settings.episodes_per_setting}: "
                 f"steps={steps:<4} success={'yes' if success else 'no ':<3} ({dt:.1f}s)")
        arr = np.array(step_counts, dtype=float)
        succ_arr = np.array(successes, dtype=bool)
        success_rate = succ_arr.mean() if len(succ_arr) else 0.0
        if succ_arr.any():
            solved_steps = arr[succ_arr]
            mean_solved, std_solved = solved_steps.mean(), solved_steps.std()
        else:
            mean_solved, std_solved = float("nan"), float("nan")
        results[(scenario, budget)] = (arr.mean(), arr.std(), success_rate,
                                        mean_solved, std_solved)
        emit(f"  --> mean_steps(all)={arr.mean():.2f}  std(all)={arr.std():.2f}  "
             f"success_rate={success_rate:.0%}  "
             f"mean_steps(solved only)={mean_solved:.2f}  std(solved only)={std_solved:.2f}")

    emit("\n" + "=" * 90)
    emit(f"{'scenario':<12} {'budget':>8} {'mean(all)':>10} {'std(all)':>9} "
         f"{'success%':>9} {'mean(solved)':>13} {'std(solved)':>12}")
    emit("-" * 90)
    for (scenario, budget), (mean, std, succ, mean_s, std_s) in results.items():
        emit(f"{scenario:<12} {budget:>7}s {mean:>10.2f} {std:>9.2f} "
             f"{succ:>8.0%} {mean_s:>13.2f} {std_s:>12.2f}")
    emit("=" * 90)

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