"""Paired control of the two dual-value implementations.

Recomputes V(D'(empty)) and V(R) on a control sample of 2,500 instances,
500 from each of the five structural families of the battery, through two
independently coded linear programs:

  1. the primal hull form of Proposition 3.2: convex weights on the
     enumerated 0-1 points of each factor, couplings as equality rows;
  2. the direct minimization of the piecewise linear dual function through
     its epigraph: multipliers plus one epigraph variable per factor, one
     row per enumerated point, as in the proof of the Geoffrion
     characterization.

The two programs are linear programming duals of one another, so their
agreement validates the encodings and the solver interfaces, not the
characterization itself, which is proved. The script prints the largest
absolute difference over the five thousand paired values; that figure is
the one quoted in the paper.

Deterministic (seed 4242). Usage: python3 paired_control.py [n_per_family]
"""
import itertools as it
import sys
import numpy as np
from scipy.optimize import linprog

SEED = 4242


def points(A, b, n):
    """All 0-1 points of {x : A x <= b}."""
    return np.array([x for x in it.product([0, 1], repeat=n)
                     if np.all(A @ np.array(x, float) <= b)], float)


# ---------------------------------------------------------------- generator
def gen_instance(kind, rng):
    """One instance of the requested family; returns (c, blocks) with
    blocks = [(A_ret, b_ret), (A_1, b_1), ..., (A_{p-1}, b_{p-1})],
    the retained block first. Mirrors the generator documented in the
    paper's computational section."""
    def row(n, lo=0, hi=9):
        while True:
            a = rng.integers(lo, hi + 1, n).astype(float)
            if a.sum() > 0:
                return a

    def rhs(a, atleast=0):
        return float(rng.integers(atleast, max(int(a.sum()) - 1, atleast) + 1))

    if kind == 'generic':
        n = int(rng.integers(3, 6))
        p = int(rng.integers(3, 5))
        blocks = []
        for _ in range(p):
            rows = int(rng.integers(1, 3))
            A = np.array([row(n) for _ in range(rows)])
            b = np.array([rhs(A[i]) for i in range(rows)])
            blocks.append((A, b))
        c = rng.integers(1, 10, n).astype(float)
        return c, blocks
    if kind == 'identical':
        n = int(rng.integers(3, 6))
        A = np.array([row(n)])
        b = np.array([rhs(A[0], 1)])
        Ar = np.array([row(n)])
        br = np.array([rhs(Ar[0], 1)])
        c = rng.integers(1, 10, n).astype(float)
        return c, [(Ar, br), (A, b), (A.copy(), b.copy())]
    if kind == 'blockangular':
        n = 5
        A1 = np.zeros((1, n)); A1[0, :3] = rng.integers(1, 10, 3)
        A2 = np.zeros((1, n)); A2[0, 2:] = rng.integers(1, 10, 3)
        b1 = np.array([rhs(A1[0], 1)]); b2 = np.array([rhs(A2[0], 1)])
        Ar = np.array([row(n, 1)]); br = np.array([rhs(Ar[0], 1)])
        c = rng.integers(1, 10, n).astype(float)
        return c, [(Ar, br), (A1, b1), (A2, b2)]
    if kind == 'sharedmatrix':
        n = int(rng.integers(3, 6))
        A = np.array([row(n, 1)])
        b1 = np.array([rhs(A[0], 1)]); b2 = np.array([rhs(A[0], 1)])
        Ar = np.array([row(n, 1)]); br = np.array([rhs(Ar[0], 1)])
        c = rng.integers(1, 10, n).astype(float)
        return c, [(Ar, br), (A, b1), (A.copy(), b2)]
    if kind == 'injective':
        n = 2
        def inj():
            while True:
                A = rng.integers(0, 10, (2, 2)).astype(float)
                if abs(np.linalg.det(A)) > 1e-9 and A.sum(1).min() > 0:
                    return A
        A1, A2 = inj(), inj()
        b1 = np.array([rhs(A1[i], 1) for i in range(2)])
        b2 = np.array([rhs(A2[i], 1) for i in range(2)])
        Ar = np.array([row(n, 1)]); br = np.array([rhs(Ar[0], 1)])
        c = rng.integers(1, 10, n).astype(float)
        return c, [(Ar, br), (A1, b1), (A2, b2)]
    raise ValueError(kind)


# ------------------------------------------------------------ hull form LPs
def hull_Dp(c, Xs, Ys, n):
    """V(D'(empty)) by the primal hull form: weights per factor, coupling
    sum_k y^k = (p-1) x as equality rows."""
    parts = [Xs] + Ys
    tam = [len(P) for P in parts]
    off = np.cumsum([0] + tam)
    N = off[-1]
    p1 = len(Ys)
    obj = np.zeros(N)
    obj[:tam[0]] = -(Xs @ c)
    Aeq, beq = [], []
    for j in range(n):
        r = np.zeros(N)
        r[:tam[0]] = -p1 * Xs[:, j]
        for k in range(p1):
            r[off[k + 1]:off[k + 2]] = Ys[k][:, j]
        Aeq.append(r); beq.append(0.0)
    for f in range(len(parts)):
        r = np.zeros(N)
        r[off[f]:off[f + 1]] = 1.0
        Aeq.append(r); beq.append(1.0)
    r = linprog(obj, A_eq=np.array(Aeq), b_eq=np.array(beq),
                bounds=[(0, None)] * N, method='highs')
    return -r.fun if r.status == 0 else None


def hull_R(c, Xs, copied):
    """V(R) by the primal hull form: weights on X_s, dualized rows kept."""
    m = len(Xs)
    obj = -(Xs @ c)
    Aub, bub = [], []
    for (A, b) in copied:
        for i in range(len(b)):
            Aub.append(Xs @ A[i]); bub.append(b[i])
    r = linprog(obj, A_ub=np.array(Aub), b_ub=np.array(bub),
                A_eq=np.ones((1, m)), b_eq=np.array([1.0]),
                bounds=[(0, None)] * m, method='highs')
    return -r.fun if r.status == 0 else None


# -------------------------------------------------------- epigraph form LPs
def epi_Dp(c, Xs, Ys, n):
    """V(D'(empty)) by minimizing the piecewise linear dual function.

    Variables (beta in R^n free, t_0, t_1..t_{p-1}), objective t_0 + sum t_k,
    one row per enumerated point:
      t_0 >= (c - p1*beta) x   for every x in X_s,
      t_k >= beta y            for every y in Y_k."""
    p1 = len(Ys)
    nv = n + 1 + p1
    obj = np.zeros(nv)
    obj[n:] = 1.0
    Aub, bub = [], []
    for x in Xs:          # c.x - p1*(beta.x) - t0 <= 0
        r = np.zeros(nv)
        r[:n] = -p1 * x
        r[n] = -1.0
        Aub.append(r); bub.append(-(c @ x))
    for k in range(p1):   # beta.y - t_k <= 0
        for y in Ys[k]:
            r = np.zeros(nv)
            r[:n] = y
            r[n + 1 + k] = -1.0
            Aub.append(r); bub.append(0.0)
    r = linprog(obj, A_ub=np.array(Aub), b_ub=np.array(bub),
                bounds=[(None, None)] * nv, method='highs')
    return r.fun if r.status == 0 else None


def epi_R(c, Xs, copied):
    """V(R) by minimizing the piecewise linear dual function.

    Variables (nu >= 0 stacked over the dualized rows, t), objective t,
    one row per point of X_s:  t >= c x + sum_r nu_r (b_r - a_r x)."""
    rows = [(A[i], b[i]) for (A, b) in copied for i in range(len(b))]
    m = len(rows)
    nv = m + 1
    obj = np.zeros(nv)
    obj[m] = 1.0
    Aub, bub = [], []
    for x in Xs:          # c.x + sum_r nu_r (b_r - a_r.x) - t <= 0
        r = np.zeros(nv)
        for j, (a, bb) in enumerate(rows):
            r[j] = bb - a @ x
        r[m] = -1.0
        Aub.append(r); bub.append(-(c @ x))
    r = linprog(obj, A_ub=np.array(Aub), b_ub=np.array(bub),
                bounds=[(0, None)] * m + [(None, None)], method='highs')
    return r.fun if r.status == 0 else None


def main(nper=500):
    rng = np.random.default_rng(SEED)
    fams = ['generic', 'identical', 'blockangular', 'sharedmatrix', 'injective']
    worst = 0.0
    pairs = 0
    for fam in fams:
        done = 0
        while done < nper:
            c, blocks = gen_instance(fam, rng)
            n = len(c)
            Xs = points(*blocks[0], n)
            Ys = [points(A, b, n) for (A, b) in blocks[1:]]
            if len(Xs) == 0 or any(len(Y) == 0 for Y in Ys):
                continue
            for v1, v2 in ((hull_Dp(c, Xs, Ys, n), epi_Dp(c, Xs, Ys, n)),
                           (hull_R(c, Xs, blocks[1:]), epi_R(c, Xs, blocks[1:]))):
                if v1 is None or v2 is None:
                    continue
                worst = max(worst, abs(v1 - v2))
                pairs += 1
            done += 1
        print(f"{fam}: done ({nper} instances)")
    print(f"paired values: {pairs} | largest absolute difference: {worst:.3e}")


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
