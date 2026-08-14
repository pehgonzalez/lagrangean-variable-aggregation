"""Merged dual on strongly correlated data: visited multipliers and
truncated evaluations at equal time.

Complements jointly_hard.py in two directions on the same strongly
correlated generator (weights uniform in {1..100}, profits equal to the mean
weight of the column plus small uniform noise, capacities at half the row
sums), now embedded in a full instance: ambient dimension equal to the
support size q, one fresh retained row of the same kind, m copied rows.

Part 1 (visited, not drawn). The merged dual is run under the projected
subgradient method with TRUNCATED joint evaluations: HiGHS receives the
joint m-row 0-1 knapsack with a per-call time limit and returns its best
dual bound, which is a valid upper bound on the subproblem maximum, so the
truncated evaluation is a valid dual bound with a quantified inflation. At a
fixed sample of iterations the joint subproblem is additionally solved with
a 15-second cap at the multipliers actually visited, reporting closed or not
and the time, so the hardness claim refers to the trajectory rather than to
random multiplier draws.

Part 2 (equal time). The aggregated dual (one copy per row, mean-form
coupling, q multipliers, blockwise dynamic-programming knapsacks) is run for
ITERS iterations; the merged dual with truncated evaluations then runs for
the same wall-clock budget, and the two best bounds are compared. Both duals
carry q multipliers here, so the comparison isolates the subproblem cost.

Deterministic given the seed except for wall-clock effects on the iteration
counts of the equal-time run. Usage: python3 q3_trajectory.py
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subgradient_support import knapsack01
import highspy

SEED = 4051
CELLS = [(50, 10), (100, 20)]
ITERS = 400
TRUNC_LIMIT = 1.0      # seconds per truncated joint evaluation
EXACT_LIMIT = 15.0     # seconds for the exact probes at visited multipliers
PROBE_ITERS = (1, 5, 10, 20, 40)  # fixed iterations for the exact probes


def gen_full(q, m, rng):
    """Strongly correlated cell plus retained row and objective."""
    W = rng.integers(1, 101, (m, q)).astype(float)
    caps = np.floor(0.5 * W.sum(1))
    c = W.mean(0) + rng.uniform(0, 10, q)
    a_ret = rng.integers(1, 101, q).astype(float)
    b_ret = np.floor(0.5 * a_ret.sum())
    return c, W, caps, a_ret, b_ret


def joint_bound(profit, W, caps, limit):
    """Valid upper bound on max profit.y s.t. W y <= caps, y binary.

    HiGHS minimizes -profit.y; its dual bound lower-bounds that minimum, so
    its negation upper-bounds the maximum. Returns (upper bound, incumbent
    or None, closed flag, time)."""
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
    closed = (h.getModelStatus() == highspy.HighsModelStatus.kOptimal)
    ub = -h.getInfo().mip_dual_bound
    sol = np.array(h.getSolution().col_value)
    inc = np.round(sol) if sol.size == q else None
    return ub, inc, closed, tt


def run_aggregated(c, W, caps, a_ret, b_ret, iters):
    """Blockwise aggregated dual: m copies, mean-form coupling, q mults."""
    q = len(c)
    m = W.shape[0]
    beta = np.zeros(q)
    best = np.inf
    delta = 0.05 * abs(c).sum()
    stall = 0
    t0 = time.time()
    for _ in range(iters):
        vx, x = knapsack01(c - beta, a_ret, b_ret)
        theta = vx
        ysum = np.zeros(q)
        for k in range(m):
            vy, y = knapsack01(beta / m, W[k], caps[k])
            theta += vy
            ysum += y
        g = ysum / m - x
        if theta < best - 1e-9:
            best = theta
            stall = 0
        else:
            stall += 1
            if stall >= 30:
                delta *= 0.5
                stall = 0
        gn2 = (g * g).sum()
        if gn2 < 1e-12:
            break
        step = (theta - (best - delta)) / gn2
        beta -= step * g
    return best, time.time() - t0


def run_merged_truncated(c, W, caps, a_ret, b_ret, budget, probe_iters):
    """Merged dual, q multipliers, truncated joint evals, wall budget (s)."""
    q = len(c)
    lam = np.zeros(q)
    best = np.inf
    delta = 0.05 * abs(c).sum()
    stall = 0
    evals = []
    probes = []
    it = 0
    t0 = time.time()
    while time.time() - t0 < budget:
        it += 1
        vx, x = knapsack01(c - lam, a_ret, b_ret)
        ub, y, closed, tt = joint_bound(lam, W, caps, TRUNC_LIMIT)
        evals.append(tt)
        theta = vx + ub
        if theta < best - 1e-9:
            best = theta
            stall = 0
        else:
            stall += 1
            if stall >= 30:
                delta *= 0.5
                stall = 0
        if it in probe_iters:
            pub, _, pclosed, ptt = joint_bound(lam, W, caps, EXACT_LIMIT)
            probes.append((it, pclosed, ptt))
        if y is None:
            continue
        g = y - x
        gn2 = (g * g).sum()
        if gn2 < 1e-12:
            break
        step = (theta - (best - delta)) / gn2
        lam -= step * g
    return best, it, evals, probes


def main():
    for (q, m) in CELLS:
        rng = np.random.default_rng(SEED + 1000 * q + m)
        c, W, caps, a_ret, b_ret = gen_full(q, m, rng)
        bA, tA = run_aggregated(c, W, caps, a_ret, b_ret, ITERS)
        bM, itM, evals, probes = run_merged_truncated(
            c, W, caps, a_ret, b_ret, tA, set(PROBE_ITERS))
        ev = np.array(evals)
        print(f"cell q={q} m={m}: aggregated {ITERS} iters, {tA:.1f}s, "
              f"best bound {bA:.2f}")
        print(f"  merged truncated at equal time: {itM} iters, best bound "
              f"{bM:.2f} ({'tighter' if bM < bA else 'weaker'} by "
              f"{abs(bM - bA):.2f})")
        print(f"  truncated joint evals: median {np.median(ev):.2f}s, "
              f"max {ev.max():.2f}s, at the {TRUNC_LIMIT:.1f}s cap on "
              f"{(ev > 0.95 * TRUNC_LIMIT).sum()} of {len(ev)}")
        for (it, closed, tt) in probes:
            print(f"  exact probe at visited iter {it}: "
                  f"{'closed' if closed else 'NOT closed'} in {tt:.1f}s "
                  f"(cap {EXACT_LIMIT:.0f}s)")
    print("done")


if __name__ == '__main__':
    main()
