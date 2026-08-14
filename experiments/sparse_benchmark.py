"""Three-dual timing comparison: dense, aggregated and sparse coupling.

Runs the four block-angular configurations of the subgradient demonstration
(n, q, p) with five seeds each, under the same projected subgradient method
and step rule as subgradient_support.py, for three duals over identical
instances:

  G  dense Guignard-Kim coupling, one copy per block, (p-1)*n multipliers;
  A  aggregated coupling with S* = J, (p-1)*q + (n-q) multipliers;
  S  sparse coupling on the shared support, y^k_J = x_J per copy,
     (p-1)*q multipliers and no coupling outside J.

In the sparse dual the off-support coordinates of the copies appear neither
in the coupling nor in the Lagrangean objective, so each copy solves the
same q-dimensional knapsack as in the other duals and the retained problem
is the same n-dimensional knapsack: all three duals solve p knapsacks per
iteration, and any time difference comes from the multiplier space alone.

Bounds are deterministic given the seeds; wall times vary with the machine.
For each configuration the script prints per-seed bounds and times, then a
summary line with mean [min, max] statistics, including the per-seed
wall-time ratios S/G and S/A and the iterations needed to come within 0.1%
of the best bound found by any of the three duals.

Usage:  python3 sparse_benchmark.py [iters]
"""
import sys
import time
import numpy as np

from subgradient_support import knapsack01, run

CONFIGS = [(100, 20, 11), (100, 20, 21), (150, 30, 11), (150, 30, 21)]
SEEDS = [7, 13, 29, 41, 57]


def instance(n, q, p, seed):
    """Same generator as the __main__ block of subgradient_support.py."""
    rng = np.random.default_rng(seed)
    J = np.arange(q)
    AJ = rng.integers(1, 10, (p - 1, q)).astype(float)
    bJ = np.floor(0.5 * AJ.sum(1))
    a_ret = rng.integers(1, 10, n).astype(float)
    b_ret = np.floor(0.5 * a_ret.sum())
    c = rng.integers(1, 10, n).astype(float)
    return c, AJ, bJ, a_ret, b_ret, J


def run_sparse(c, AJ, bJ, a_ret, b_ret, J, iters):
    """Subgradient minimization of the sparse dual: mu (p-1) x q, nothing off J.

    L(mu) = max_x [(c - sum_k mu^k on J) x] + sum_k max_{yJ} mu^k yJ,
    the retained maximization over the n-dimensional knapsack and each copy
    over its q-dimensional knapsack. Same step rule as subgradient_support.run.
    """
    q = len(J)
    p1 = AJ.shape[0]
    mu = np.zeros((p1, q))
    best = np.inf
    traj = []
    solves = 0
    delta = 0.05 * abs(c).sum()
    stall = 0
    t0 = time.time()
    for _ in range(iters):
        cx = c.copy()
        cx[J] -= mu.sum(0)
        vx, x = knapsack01(cx, a_ret, b_ret)
        solves += 1
        theta = vx
        gmu = np.empty((p1, q))
        for k in range(p1):
            vy, yJ = knapsack01(mu[k], AJ[k], bJ[k])
            solves += 1
            theta += vy
            gmu[k] = yJ - x[J]
        if theta < best - 1e-9:
            best = theta
            stall = 0
        else:
            stall += 1
            if stall >= 30:
                delta *= 0.5
                stall = 0
        traj.append(best)
        gn2 = (gmu * gmu).sum()
        if gn2 < 1e-12:
            break
        step = (theta - (best - delta)) / gn2
        mu -= step * gmu
    return np.array(traj), time.time() - t0, solves


def hits(traj, ref):
    """First iteration whose bound is within 0.1% of the reference."""
    for i, v in enumerate(traj):
        if v <= 1.001 * ref:
            return i + 1
    return None


def main(iters=400):
    fmt = lambda v: f"{np.mean(v):.2f} [{min(v):.2f}, {max(v):.2f}]"
    fmth = lambda v: '-' if any(x is None for x in v) else f"{np.mean(v):.0f}"
    for (n, q, p) in CONFIGS:
        dims = ((p - 1) * n, (p - 1) * q + (n - q), (p - 1) * q)
        tG, tA, tS, rSG, rSA, reldiff = [], [], [], [], [], []
        hG, hA, hS = [], [], []
        print(f"n={n} q={q} p={p}  dual dims G/A/S = {dims[0]}/{dims[1]}/{dims[2]}")
        for seed in SEEDS:
            c, AJ, bJ, a_ret, b_ret, J = instance(n, q, p, seed)
            trajG, timeG, _ = run('guignard', c, AJ, bJ, a_ret, b_ret, J, iters)
            trajA, timeA, _ = run('support', c, AJ, bJ, a_ret, b_ret, J, iters)
            trajS, timeS, sS = run_sparse(c, AJ, bJ, a_ret, b_ret, J, iters)
            assert sS == p * len(trajS)
            bG, bA, bS = float(trajG[-1]), float(trajA[-1]), float(trajS[-1])
            ref = min(bG, bA, bS)
            reldiff.append(max(bG, bA, bS) / ref - 1)
            tG.append(timeG); tA.append(timeA); tS.append(timeS)
            rSG.append(timeS / timeG); rSA.append(timeS / timeA)
            hG.append(hits(trajG, ref)); hA.append(hits(trajA, ref)); hS.append(hits(trajS, ref))
            print(f"  seed {seed:2d}: bounds G/A/S {bG:10.3f} {bA:10.3f} {bS:10.3f} "
                  f"| times {timeG:5.2f} {timeA:5.2f} {timeS:5.2f}")
        print(f"  summary: max rel. diff bounds {100*max(reldiff):.4f}% "
              f"| tG(s) {fmt(tG)} | tA(s) {fmt(tA)} | tS(s) {fmt(tS)} "
              f"| S/G {fmt(rSG)} | S/A {fmt(rSA)} "
              f"| iters to 0.1%: G {fmth(hG)} A {fmth(hA)} S {fmth(hS)}")
    print("done")


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 400)
