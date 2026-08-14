"""Four-dual timing benchmark: dense, aggregated, sparse and merged.

Extends sparse_benchmark.py in four directions, on the same block-angular
configurations (n, q, p) and under the same projected subgradient method and
step rule:

  1. a fourth dual, M, the merged decomposition: all copied rows are imposed
     jointly on one copy y and the coupling y = x is dualized at n free
     multipliers. Its bound dominates V(D) and its dual evaluation needs one
     joint multi-row 0-1 knapsack, solved exactly here by HiGHS as a MILP,
     against the dynamic-programming knapsacks of the other three duals. The
     implementations are therefore heterogeneous (compiled MILP solver vs
     interpreted dynamic program) and wall times across that divide measure
     the protocol, not the abstract schemes;

  2. twenty-five seeds per configuration instead of five, with medians and
     interquartile ranges, so that ratios carry a stated resolution;

  3. instrumentation: the time spent inside knapsack solves (and, for M,
     inside the MILP) is accumulated separately from the total, per dual;

  4. a sum-form variant of the aggregated dual, dualizing
     sum_k y^k_off = (p-1) x_off without the mean rescaling, since a fixed
     step rule is not invariant to the scaling of a dualized equation.

Bounds of G, A, S converge to V(D) (support rule); M converges to the merged
value, which is at least as tight. Bounds are deterministic given the seeds;
wall times vary with the machine. Output: per-seed lines, a summary per
configuration, and one JSON line per run in four_dual_benchmark.jsonl.

Usage:  python3 four_dual_benchmark.py [iters]
"""
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import subgradient_support as ss
import highspy

CONFIGS = [(100, 20, 11), (100, 20, 21), (150, 30, 11), (150, 30, 21)]
SEEDS = [7, 13, 29, 41, 57, 71, 89, 101, 113, 131, 149, 163, 179, 193,
         211, 227, 241, 257, 271, 283, 307, 313, 331, 349, 367]

KNAP = {'t': 0.0}
_orig_knap = ss.knapsack01


def knapsack01(profit, a, b):
    """Timed wrapper around the public exact 0/1 knapsack."""
    t0 = time.perf_counter()
    r = _orig_knap(profit, a, b)
    KNAP['t'] += time.perf_counter() - t0
    return r


ss.knapsack01 = knapsack01  # instruments run('guignard') and run('support')


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
    """Sparse coupling y^k_J = x_J per copy: mu (p-1) x q, nothing off J."""
    q = len(J)
    p1 = AJ.shape[0]
    mu = np.zeros((p1, q))
    best = np.inf
    traj = []
    delta = 0.05 * abs(c).sum()
    stall = 0
    t0 = time.time()
    for _ in range(iters):
        cx = c.copy()
        cx[J] -= mu.sum(0)
        vx, x = knapsack01(cx, a_ret, b_ret)
        theta = vx
        gmu = np.empty((p1, q))
        for k in range(p1):
            vy, yJ = knapsack01(mu[k], AJ[k], bJ[k])
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
    return np.array(traj), time.time() - t0


def run_support_sum(c, AJ, bJ, a_ret, b_ret, J, iters):
    """Aggregated dual in SUM form: dualizes sum_k y^k_off = (p-1) x_off.

    Same step rule as subgradient_support.run('support'), which uses the
    mean form; the two are the same optimization problem under a rescaling
    of beta, but a fixed step rule follows different trajectories.
    """
    n = len(c)
    q = len(J)
    p1 = AJ.shape[0]
    off = np.setdiff1d(np.arange(n), J)
    mu = np.zeros((p1, q))
    beta = np.zeros(n - q)
    best = np.inf
    traj = []
    delta = 0.05 * abs(c).sum()
    stall = 0
    t0 = time.time()
    for _ in range(iters):
        cx = c.copy()
        cx[J] -= mu.sum(0)
        cx[off] -= p1 * beta
        vx, x = knapsack01(cx, a_ret, b_ret)
        yoff = (beta > 0).astype(float)
        theta = vx + p1 * beta[beta > 0].sum()
        gmu = np.empty((p1, q))
        for k in range(p1):
            vy, yJ = knapsack01(mu[k], AJ[k], bJ[k])
            theta += vy
            gmu[k] = yJ - x[J]
        gbeta = p1 * (yoff - x[off])
        if theta < best - 1e-9:
            best = theta
            stall = 0
        else:
            stall += 1
            if stall >= 30:
                delta *= 0.5
                stall = 0
        traj.append(best)
        gn2 = (gmu * gmu).sum() + (gbeta * gbeta).sum()
        if gn2 < 1e-12:
            break
        step = (theta - (best - delta)) / gn2
        mu -= step * gmu
        beta -= step * gbeta
    return np.array(traj), time.time() - t0


def joint_milp(profit, AJ, bJ):
    """max profit.y over AJ y <= bJ, y binary (q variables); exact HiGHS."""
    m, q = AJ.shape
    h = highspy.Highs()
    h.silent()
    inf = highspy.kHighsInf
    h.addVars(q, np.zeros(q), np.ones(q))
    h.changeColsIntegrality(q, np.arange(q, dtype=np.int32),
                            np.array([highspy.HighsVarType.kInteger] * q))
    h.changeColsCost(q, np.arange(q, dtype=np.int32), -profit)
    for i in range(m):
        h.addRow(-inf, bJ[i], q, np.arange(q, dtype=np.int32), AJ[i])
    h.run()
    val = -h.getInfo().objective_function_value
    sol = np.array(h.getSolution().col_value)
    return val, np.round(sol)


def run_merged(c, AJ, bJ, a_ret, b_ret, J, iters):
    """Merged decomposition: one copy under ALL copied rows, coupling y = x
    at n free multipliers. Joint J-part solved exactly by HiGHS per
    iteration; off-J coordinates of the copy are free binaries resolved by
    sign. Returns (trajectory, total time, time inside the MILP)."""
    n = len(c)
    q = len(J)
    off = np.setdiff1d(np.arange(n), J)
    lam = np.zeros(n)
    best = np.inf
    traj = []
    delta = 0.05 * abs(c).sum()
    stall = 0
    tmilp = 0.0
    t0 = time.time()
    for _ in range(iters):
        vx, x = knapsack01(c - lam, a_ret, b_ret)
        y = np.zeros(n)
        y[off] = (lam[off] > 0).astype(float)
        t1 = time.perf_counter()
        vj, yJ = joint_milp(lam[J], AJ, bJ)
        tmilp += time.perf_counter() - t1
        y[J] = yJ
        theta = vx + vj + lam[off][lam[off] > 0].sum()
        if theta < best - 1e-9:
            best = theta
            stall = 0
        else:
            stall += 1
            if stall >= 30:
                delta *= 0.5
                stall = 0
        traj.append(best)
        g = y - x
        gn2 = (g * g).sum()
        if gn2 < 1e-12:
            break
        step = (theta - (best - delta)) / gn2
        lam -= step * g
    return np.array(traj), time.time() - t0, tmilp


def med_iqr(v):
    a = np.percentile(v, [25, 50, 75])
    return f"{a[1]:.2f} [{a[0]:.2f}, {a[2]:.2f}]"


def main(iters=400):
    raw = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'four_dual_benchmark.jsonl'), 'w')
    for (n, q, p) in CONFIGS:
        dims = {'G': (p - 1) * n, 'A': (p - 1) * q + (n - q),
                'S': (p - 1) * q, 'M': n}
        res = {k: [] for k in ('tG', 'tA', 'tS', 'tM', 'tAsum', 'kfG', 'kfA',
                               'kfS', 'mfM', 'bG', 'bA', 'bS', 'bM', 'bAsum',
                               'AG', 'SG', 'SA', 'MG', 'MA')}
        print(f"n={n} q={q} p={p}  dims G/A/S/M = "
              f"{dims['G']}/{dims['A']}/{dims['S']}/{dims['M']}")
        for seed in SEEDS:
            c, AJ, bJ, a_ret, b_ret, J = instance(n, q, p, seed)
            out = {}
            for tag, fn in (('G', lambda: ss.run('guignard', c, AJ, bJ, a_ret, b_ret, J, iters)),
                            ('A', lambda: ss.run('support', c, AJ, bJ, a_ret, b_ret, J, iters)),
                            ('S', lambda: run_sparse(c, AJ, bJ, a_ret, b_ret, J, iters)),
                            ('Asum', lambda: run_support_sum(c, AJ, bJ, a_ret, b_ret, J, iters)),
                            ('M', lambda: run_merged(c, AJ, bJ, a_ret, b_ret, J, iters))):
                KNAP['t'] = 0.0
                r = fn()
                traj, tt = r[0], r[1]
                out[tag] = (float(traj[-1]), float(tt), KNAP['t'],
                            r[2] if tag == 'M' else None)
            bG, tG, kG, _ = out['G']
            bA, tA, kA, _ = out['A']
            bS, tS, kS, _ = out['S']
            bAs, tAs, _, _ = out['Asum']
            bM, tM, kM, mM = out['M']
            res['tG'].append(tG); res['tA'].append(tA); res['tS'].append(tS)
            res['tM'].append(tM); res['tAsum'].append(tAs)
            res['kfG'].append(kG / tG); res['kfA'].append(kA / tA)
            res['kfS'].append(kS / tS); res['mfM'].append(mM / tM)
            res['bG'].append(bG); res['bA'].append(bA); res['bS'].append(bS)
            res['bM'].append(bM); res['bAsum'].append(bAs)
            res['AG'].append(tA / tG); res['SG'].append(tS / tG)
            res['SA'].append(tS / tA); res['MG'].append(tM / tG)
            res['MA'].append(tM / tA)
            raw.write(json.dumps({'n': n, 'q': q, 'p': p, 'seed': seed,
                                  'bounds': [bG, bA, bS, bAs, bM],
                                  'times': [tG, tA, tS, tAs, tM],
                                  'knap_frac': [kG / tG, kA / tA, kS / tS],
                                  'milp_frac_M': mM / tM}) + '\n')
        gas = np.minimum.reduce([np.array(res['bG']), np.array(res['bA']),
                                 np.array(res['bS'])])
        agree = max(max(res['bG'][i], res['bA'][i], res['bS'][i],
                        res['bAsum'][i]) / gas[i] - 1 for i in range(len(SEEDS)))
        mtight = np.array(res['bM']) - gas
        print(f"  bounds: G/A/S/Asum agree to {100*agree:.4f}% | "
              f"M minus best of G,A,S: median {np.median(mtight):+.3f} "
              f"(<=0 means tighter)")
        print(f"  time medians [IQR] (s): G {med_iqr(res['tG'])} | "
              f"A {med_iqr(res['tA'])} | S {med_iqr(res['tS'])} | "
              f"Asum {med_iqr(res['tAsum'])} | M {med_iqr(res['tM'])}")
        print(f"  ratio medians [IQR]: A/G {med_iqr(res['AG'])} | "
              f"S/G {med_iqr(res['SG'])} | S/A {med_iqr(res['SA'])} | "
              f"M/G {med_iqr(res['MG'])} | M/A {med_iqr(res['MA'])}")
        print(f"  time inside knapsack solves: G {100*np.median(res['kfG']):.0f}% "
              f"A {100*np.median(res['kfA']):.0f}% S {100*np.median(res['kfS']):.0f}% "
              f"| inside the MILP for M {100*np.median(res['mfM']):.0f}%")
    raw.close()
    print("done")


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 400)
