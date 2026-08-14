"""Cost of one exact dual evaluation: blockwise knapsacks vs joint subproblem.

Block-angular instances with strongly correlated data: p-1 copied single-row
knapsacks over a common support of q variables, weights uniform in
{1,...,100}, each profit equal to the mean weight of its column plus small
uniform noise, capacities at half the row sums. On such data the blockwise
part of one aggregated dual evaluation (p-1 single-row 0/1 knapsacks, solved
exactly by dynamic programming) costs milliseconds, while the joint
subproblem of the merged decomposition (one knapsack with p-1 constraints
over the same q variables) admits no pseudo-polynomial recursion of
practical size and is submitted to an exact MILP solver (HiGHS) under a
time limit.

Separation criterion, fixed before running: a cell separates when the MILP
does not close within the limit while the blockwise total stays below 0.5 s.
Profits used in each evaluation are the objective shifted by small random
multipliers, one independent draw per evaluation, mimicking a generic
iteration of a dual method; the per-cell RNG is seeded independently of the
execution order, so any subset of cells can be reproduced in isolation.

Usage:
    python3 jointly_hard.py             # full grid, q in {50,100,200},
                                        # p-1 in {10,20,40}, 2 draws per cell
    python3 jointly_hard.py Q M         # a single cell (q=Q, p-1=M)

Output: one line per cell; exit code 0 when every completed cell separates.
Wall times are machine dependent; the qualitative outcome (closed or not
closed within the limit) is the reported result.
"""
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subgradient_support import knapsack01  # noqa: E402

import highspy  # noqa: E402

SEED = 4051
CELLS = [(q, m) for q in (50, 100, 200) for m in (10, 20, 40)]
DRAWS = 2
MILP_LIMIT = 15.0   # seconds per joint evaluation
BLOCK_EASY = 0.5    # seconds: ceiling for "blockwise trivial"


def gen_cell(q, m, rng):
    """Strongly correlated block-angular cell: weights, capacities, profits."""
    W = rng.integers(1, 101, (m, q)).astype(float)
    caps = np.floor(0.5 * W.sum(1))
    base = W.mean(0) + rng.uniform(0, 10, q)
    return W, caps, base


def solve_joint_milp(profit, W, caps, limit):
    """max profit.x s.t. W x <= caps, x binary; HiGHS with a time limit."""
    m, q = W.shape
    h = highspy.Highs()
    h.silent()
    h.setOptionValue('time_limit', limit)
    inf = highspy.kHighsInf
    h.addVars(q, np.zeros(q), np.ones(q))
    h.changeColsIntegrality(q, np.arange(q, dtype=np.int32),
                            np.array([highspy.HighsVarType.kInteger] * q))
    h.changeColsCost(q, np.arange(q, dtype=np.int32), -profit)
    for i in range(m):
        h.addRow(-inf, caps[i], q, np.arange(q, dtype=np.int32), W[i])
    t0 = time.time()
    h.run()
    tt = time.time() - t0
    optimal = (h.getModelStatus() == highspy.HighsModelStatus.kOptimal)
    val = -h.getInfo().objective_function_value if optimal else None
    return optimal, val, tt


def run_cell(q, m):
    """One cell; per-cell RNG independent of execution order."""
    rng = np.random.default_rng(SEED + 1000 * q + m)
    W, caps, base = gen_cell(q, m, rng)
    tb_list, tj_list, closed = [], [], 0
    for _ in range(DRAWS):
        profit = base + rng.uniform(-2, 2, q)
        t0 = time.time()
        for i in range(m):
            knapsack01(profit, W[i], caps[i])
        tb_list.append(time.time() - t0)
        optimal, _, tj = solve_joint_milp(profit, W, caps, MILP_LIMIT)
        closed += int(optimal)
        tj_list.append(tj)
    sep = (closed < DRAWS) and (max(tb_list) < BLOCK_EASY)
    print(f"q={q:3d} p-1={m:2d} | blockwise (DP) max {max(tb_list):.3f}s | "
          f"joint (MILP, limit {MILP_LIMIT:.0f}s) closed {closed}/{DRAWS} | "
          f"separation: {'YES' if sep else 'no'}")
    return sep


if __name__ == '__main__':
    if len(sys.argv) == 3:
        ok = run_cell(int(sys.argv[1]), int(sys.argv[2]))
    else:
        ok = all([run_cell(q, m) for (q, m) in CELLS])
    sys.exit(0 if ok else 1)
