"""Support-rule demonstration under a projected subgradient method.

Block-angular 0-1 instances in which no deduplication is possible: the p-1
copied single-row knapsack blocks share a small support J (|J| = q << n) but
carry independently drawn weights and capacities; the retained row and the
objective touch all n variables. The support rule guarantees
V(D'(S*)) = V(D) for S* = J.

Both duals are minimized by the same projected subgradient method under the
same step rule. The aggregated coupling is dualized in MEAN form,
(1/(p-1)) sum_k y^k_off = x_off, an equivalent scaling that keeps the
aggregated multipliers on the same footing as the copied ones under a
uniform step. Per iteration both duals solve one retained knapsack and p-1
distinct copied knapsacks; the aggregated dual resolves the coordinates
outside J, common to all copies, once instead of p-1 times.

All knapsacks are solved exactly by 0/1 dynamic programming with solution
recovery. Metrics: dual dimension, best bound, wall time, knapsack solves,
iterations to reach 0.1% of the best bound found by either method.
Deterministic given the seed.
Usage:  python3 subgradient_support.py [n] [q] [p] [iters] [seed]
"""
import sys, time
import numpy as np

def knapsack01(profit, a, b):
    """Exact 0/1 knapsack: max profit.x s.t. a.x <= b, x binary.
    Integer weights a >= 0, integer capacity b >= 0, real profits
    (items with profit <= 0 are simply skipped: x_j = 0 is optimal for them).
    Returns (value, x) via DP over capacity with parent traceback."""
    n = len(profit); b = int(b)
    dp = np.zeros(b + 1)
    take = np.zeros((n, b + 1), dtype=bool)
    for j in range(n):
        if profit[j] <= 0:
            continue
        w = int(a[j])
        if w == 0:
            dp += profit[j]; take[j, :] = True; continue
        cand = dp[:-w] + profit[j] if w <= b else np.array([])
        if w <= b:
            better = cand > dp[w:]
            dp[w:] = np.where(better, cand, dp[w:])
            take[j, w:] = better
    # traceback
    x = np.zeros(n); wpos = int(np.argmax(dp))
    val = dp[wpos]
    for j in range(n - 1, -1, -1):
        if take[j, wpos]:
            x[j] = 1
            wpos -= int(a[j]) if a[j] > 0 else 0
    return val, x



def run(mode, c, AJ, bJ, a_ret, b_ret, J, iters):
    """Subgradient minimization of the chosen dual.
    mode='guignard': lam (p-1) x n.  mode='support': mu (p-1) x q + beta (n-q).
    AJ: (p-1) x q weights of the copied rows on J; bJ: capacities."""
    n = len(c); q = len(J); p1 = AJ.shape[0]
    off = np.setdiff1d(np.arange(n), J)
    if mode == 'guignard':
        lam = np.zeros((p1, n))
    else:
        mu = np.zeros((p1, q)); beta = np.zeros(n - q)
    best = np.inf; traj = []; solves = 0
    delta = 0.05 * abs(c).sum(); stall = 0
    t0 = time.time()
    for t in range(iters):
        if mode == 'guignard':
            vx, x = knapsack01(c - lam.sum(0), a_ret, b_ret); solves += 1
            theta = vx; g = np.empty((p1, n))
            for k in range(p1):
                prof = lam[k]
                # knapsack on J (weighted) ; coordinates off J have weight 0
                wfull = np.zeros(n); wfull[J] = AJ[k]
                vy, y = knapsack01(prof, wfull, bJ[k]); solves += 1
                theta += vy; g[k] = y - x
            gvec = g
        else:
            cx = c.copy(); cx[J] -= mu.sum(0); cx[off] -= beta
            vx, x = knapsack01(cx, a_ret, b_ret); solves += 1
            # shared free part: off-J coordinates of every copy take j iff beta_j>0
            yoff = (beta > 0).astype(float); voff = beta[beta > 0].sum() / p1
            theta = vx + p1 * voff
            gmu = np.empty((p1, q))
            for k in range(p1):
                vy, yJ = knapsack01(mu[k], AJ[k], bJ[k]); solves += 1
                theta += vy; gmu[k] = yJ[:] - x[J]
            gbeta = yoff - x[off]
            gvec = (gmu, gbeta)
        if theta < best - 1e-9:
            best = theta; stall = 0
        else:
            stall += 1
            if stall >= 30: delta *= 0.5; stall = 0
        traj.append(best)
        if mode == 'guignard':
            gn2 = (gvec * gvec).sum()
            if gn2 < 1e-12: break
            step = (theta - (best - delta)) / gn2
            lam -= step * gvec
        else:
            gn2 = (gvec[0]**2).sum() + (gvec[1]**2).sum()
            if gn2 < 1e-12: break
            step = (theta - (best - delta)) / gn2
            mu -= step * gvec[0]; beta -= step * gvec[1]
    return np.array(traj), time.time() - t0, solves

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    q = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    p = int(sys.argv[3]) if len(sys.argv) > 3 else 11
    iters = int(sys.argv[4]) if len(sys.argv) > 4 else 400
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 7
    rng = np.random.default_rng(seed)
    J = np.arange(q)
    AJ = rng.integers(1, 10, (p - 1, q)).astype(float)
    bJ = np.floor(0.5 * AJ.sum(1))
    a_ret = rng.integers(1, 10, n).astype(float); b_ret = np.floor(0.5 * a_ret.sum())
    c = rng.integers(1, 10, n).astype(float)
    dims = {'guignard': (p - 1) * n, 'support': (p - 1) * q + (n - q)}
    out = {}
    for mode in ('guignard', 'support'):
        out[mode] = run(mode, c, AJ, bJ, a_ret, b_ret, J, iters)
    ref = min(out['guignard'][0][-1], out['support'][0][-1])
    print(f"n={n} q={q} p={p} iters={iters} seed={seed}  (dual dims: {dims['guignard']} vs {dims['support']})")
    for mode in ('guignard', 'support'):
        traj, tt, solves = out[mode]
        hit = next((i + 1 for i, v in enumerate(traj) if v <= 1.001 * ref), None)
        print(f"  {mode:9s}: best bound {traj[-1]:10.3f} | time {tt:6.2f}s | knapsacks {solves:6d} | iters to 0.1% of {ref:.3f}: {hit}")
