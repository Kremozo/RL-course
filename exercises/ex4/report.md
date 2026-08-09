# Programming Exercise 4 — POMDP and POMCP in Box Pushing

## 1. Results Table
 
| Scenario | Budget | Mean steps | Std | Success rate |
|---|---|---|---|---|
| single | 1s | 7.43 | 3.18 | 100% |
| single | 20s | 7.80 | 3.09 | 100% |
| two_agent | 1s | 5.57 | 2.45 | 100% |
| two_agent | 20s | 5.80 | 2.12 | 100% |

## 2. Hyperparameters
 
- **Particles:** `belief_size = 500`, `resample_attempts_cap = 20,000`
- **POMCP:** `exploration_const (UCB c) = 1.0`, `horizon = 50`, `value_epsilon = 0.01` → effective max search depth = `min(horizon, ceil(log(eps)/log(gamma))) = min(50, 90)`
- **Rollout policy:** heuristic goal-directed push planner with `rollout_explore_p = 0.2` random-action mixing
- **MDP:** `discount (gamma) = 0.95`, `move_cost = -1.0`, `terminal_bonus = 0.0`
- **Reward shaping:** potential-based, `shaping_coef = 0.5` (Manhattan distance of boxes to nearest goal)
- **Episodes:** `episodes_per_setting = 30`, `max_steps_per_episode = 200`, `random_start = True`
- **Time budgets:** `budget_fast = 1.0s`, `budget_slow = 20.0s`, enforced via a wall-clock deadline inside the POMCP simulation loop
All values are the assignment's recommended defaults; the only ones we set explicitly rather than leaving unstated are `belief_size=500` and `exploration_const=1.0`, both matching the assignment's own suggested starting points.

## 3. Observation Alternative Choice
 
We implemented **Option B (egocentric window)**: a 3×3 window always centered on the agent's true position, with the agent occupying the center cell, and out-of-bounds cells treated as walls.
 
**Justification:** actions in this environment are direct cardinal moves (up/down/left/right) with no notion of a "facing direction." So there's nothing for a fixed-north window (Option A) to be oriented relative to that an egocentric window doesn't already capture more usefully. An egocentric window gives the agent symmetric local information in every direction it could plausibly move or push next, which directly supports the push-planning logic (checking `dest`/`push_from` cells around a box) and the belief update (matching windows to candidate cells). A fixed-north window would frequently return the same content regardless of the agent's actual surroundings to the south/east/west, discarding exactly the information needed to decide the next action.

## 4. Discussion of Results
 
### Effect of compute budget
Increasing the per-decision budget from 1s to 20s did **not** meaningfully change performance: mean steps went from 7.43→7.80 (single) and 5.57→5.80 (two_agent) — i.e. slightly *worse* nominally, not better. Given the standard errors here (std/√30 ≈ 0.58 and 0.39 for single and two_agent respectively), a difference of 0.37 and 0.23 steps is well within noise — not a statistically meaningful effect in either direction.

The most likely explanation is that the board is small enough (a 6×4 interior with a single loose box and one goal) that even 30 simulations' worth of search in one second is already sufficient to find a near-optimal push plan; the extra 19 seconds of budget buys many more simulations per decision, but there's no remaining decision quality to extract on a problem this size. We'd expect the budget effect to become visible on a larger map, a longer horizon to the goal, or a scenario with a heavy box requiring coordinated joint pushes, where a shallow one-second search is more likely to miss the correct multi-step plan.

### Effect of the two-agent scenario
 
Two agents solved the task in consistently fewer steps than one agent, at both budgets (5.57 vs 7.43 at 1s; 5.80 vs 7.80 at 20s) — roughly a 25% reduction. Two effects plausibly combine here:
 
1. **Faster belief convergence** — with two agents, two simultaneous 3×3 windows arrive per step instead of one, which narrows the joint position hypothesis space more aggressively with each rejection-sampling update.
2. **Reduced time to reach the box** — whichever of the two agents happens to be closer to the single box can push it, effectively halving the expected time to first contact. Since this layout has no heavy box requiring joint action, the second agent mainly adds redundancy in search and localization rather than coordination overhead.
