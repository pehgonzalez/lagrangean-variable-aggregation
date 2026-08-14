"""Exhaustive verification over a complete finite universe of instances.

Unlike sampling batteries, this script enumerates ALL instances of a finite
universe (n=2, p=3, one retained and two copied single-row blocks, matrix
coefficients in {0,1,2} with nonzero rows, right-hand sides in {0,1},
objectives in {1,2}^2 - 16,384 instances) and checks, on every one of them,
seven statements of the paper, including monotonicity over the COMPLETE
lattice of copied subsets S. A violation of any theorem inside this universe
would be found with certainty rather than with sampling probability.

In addition, on 30 random instances a third independent implementation
(subgradient method with exact-target Polyak steps, guaranteed convergent)
is checked against the hull-LP value, closing the triangle
primal LP / dual LP / iterative method (relative tolerance 1e-3).

Deterministic. Usage: python3 exhaustive_small_universe.py
"""
import itertools as it
import numpy as np
import highspy

INF = highspy.kHighsInf
H = highspy.Highs()

def hlp_max(obj, rows, rlb, rub, ub):
    H.clear(); H.setOptionValue('output_flag', False)
    N = len(obj)
    H.addVars(N, np.zeros(N), np.array(ub, float))
    H.changeColsCost(N, np.arange(N, dtype=np.int32), -np.array(obj, float))
    for r, a, b in zip(rows, rlb, rub):
        nz = np.nonzero(r)[0].astype(np.int32)
        H.addRow(float(a), float(b), len(nz), nz, np.array(r, float)[nz])
    H.run()
    if H.getModelStatus() != highspy.HighsModelStatus.kOptimal: return None
    return -H.getInfo().objective_function_value

PTS2 = [np.array(x, float) for x in it.product([0, 1], repeat=2)]

def hull(a, b):
    return np.array([x for x in PTS2 if a @ x <= b], float)

def V_Dp(c, Xk, Ys, S):
    partes = [Xk] + Ys; tam = [len(P) for P in partes]
    off = np.cumsum([0] + tam); N = off[-1]; q = len(Ys)
    obj = np.zeros(N); obj[:tam[0]] = c @ Xk.T
    rows, lb, ub = [], [], []
    for j in range(2):
        if j in S:
            for i in range(1, len(partes)):
                l = np.zeros(N); l[:off[1]] = Xk[:, j]
                l[off[i]:off[i+1]] = -partes[i][:, j]; rows.append(l); lb.append(0); ub.append(0)
        else:
            l = np.zeros(N); l[:off[1]] = q * Xk[:, j]
            for i in range(1, len(partes)): l[off[i]:off[i+1]] = -partes[i][:, j]
            rows.append(l); lb.append(0); ub.append(0)
    for i in range(len(partes)):
        l = np.zeros(N); l[off[i]:off[i+1]] = 1; rows.append(l); lb.append(1); ub.append(1)
    return hlp_max(obj, rows, lb, ub, [INF] * N)

def V_DS(c, Xk, Ys, avec):
    partes = [Xk] + Ys; tam = [len(P) for P in partes]
    off = np.cumsum([0] + tam); N = off[-1]
    obj = np.zeros(N); obj[:tam[0]] = c @ Xk.T
    rows, lb, ub = [], [], []
    for i, a in enumerate(avec):
        l = np.zeros(N); l[:off[1]] = Xk @ a
        l[off[i+1]:off[i+2]] = -(partes[i+1] @ a); rows.append(l); lb.append(0); ub.append(0)
    for i in range(len(partes)):
        l = np.zeros(N); l[off[i]:off[i+1]] = 1; rows.append(l); lb.append(1); ub.append(1)
    return hlp_max(obj, rows, lb, ub, [INF] * N)

def V_R(c, Xk, Ar, br):
    rows = [Xk @ a for a in Ar] + [np.ones(len(Xk))]
    return hlp_max(c @ Xk.T, rows, [-INF] * len(br) + [1], list(br) + [1], [INF] * len(Xk))

E = 1e-9
linhas = [(np.array(a, float), float(b))
          for a in it.product([0, 1, 2], repeat=2) if any(a)
          for b in (0.0, 1.0)]
SUBS = [set(), {0}, {1}, {0, 1}]
PARES = [(s1, s2) for s1 in SUBS for s2 in SUBS if s1 < s2]

cont = {f"C{i}": 0 for i in range(1, 8)}
aplic = {"C6": 0, "C7": 0}
tot = 0
for (a1, b1), (a2, b2), (ak, bk) in it.product(linhas, linhas, linhas):
    Y1, Y2, Xk = hull(a1, b1), hull(a2, b2), hull(ak, bk)
    if min(len(Y1), len(Y2), len(Xk)) == 0: continue
    for c in [np.array(v, float) for v in it.product([1, 2], repeat=2)]:
        tot += 1
        feas = [x for x in PTS2 if a1@x <= b1 and a2@x <= b2 and ak@x <= bk]
        VP = max((c @ x for x in feas), default=None)
        Dv = {frozenset(S): V_Dp(c, Xk, [Y1, Y2], S) for S in SUBS}
        D = Dv[frozenset({0, 1})]
        DS = V_DS(c, Xk, [Y1, Y2], [a1, a2])
        R = V_R(c, Xk, [a1, a2], [b1, b2])
        Ycap = np.array([x for x in PTS2 if a1@x <= b1 and a2@x <= b2], float)
        Dcap = V_Dp(c, Xk, [Ycap], {0, 1}) if len(Ycap) else None
        if None in list(Dv.values()) + [DS, R]: continue
        if VP is not None and VP > D + E: cont["C1"] += 1
        if any(D > Dv[frozenset(S)] + E for S in SUBS): cont["C2"] += 1
        if any(Dv[frozenset(s2)] > Dv[frozenset(s1)] + E for s1, s2 in PARES): cont["C3"] += 1
        if D > DS + E or DS > R + E: cont["C4"] += 1
        if Dcap is not None and ((VP is not None and VP > Dcap + E) or Dcap > D + E): cont["C5"] += 1
        cruz = all(a1@y <= b1 + E and a2@y <= b2 + E for y in np.vstack([Y1, Y2]))
        if cruz:
            aplic["C6"] += 1
            if any(Dv[frozenset(S)] > R + E for S in SUBS): cont["C6"] += 1
        if np.array_equal(a1, a2) and b1 == b2:
            aplic["C7"] += 1
            if any(abs(Dv[frozenset(S)] - D) > 1e-7 for S in SUBS): cont["C7"] += 1
print(f"instancias enumeradas na caixa: {tot}")
for k in sorted(cont):
    extra = f" (aplicaveis: {aplic[k]})" if k in aplic else ""
    print(f"  {k}: {cont[k]} violacoes{extra}")
ok_caixa = all(v == 0 for v in cont.values())

# ---- terceiro metodo: subgradiente converge ao valor do LP-envoltorio ----
rng = np.random.default_rng(909)
piores = 0.0; nsub = 0
for t in range(30):
    n = int(rng.integers(2, 4))
    pts = [np.array(x, float) for x in it.product([0, 1], repeat=n)]
    def h(a, b): return np.array([x for x in pts if a@x <= b], float)
    a1 = rng.integers(0, 5, n).astype(float); b1 = float(rng.integers(1, max(2, int(a1.sum()))))
    a2 = rng.integers(0, 5, n).astype(float); b2 = float(rng.integers(1, max(2, int(a2.sum()))))
    ak = rng.integers(0, 5, n).astype(float); bk = float(rng.integers(1, max(2, int(ak.sum()))))
    c = rng.integers(1, 6, n).astype(float)
    Y1, Y2, Xk = h(a1, b1), h(a2, b2), h(ak, bk)
    if min(len(Y1), len(Y2), len(Xk)) == 0: continue
    ref = V_Dp(c, Xk, [Y1, Y2], set())
    if ref is None: continue
    beta = np.zeros(n); best = np.inf
    for _ in range(4000):
        ix = int(np.argmax((c - 2*beta) @ Xk.T)); x = Xk[ix]
        i1 = int(np.argmax(beta @ Y1.T)); y1 = Y1[i1]
        i2 = int(np.argmax(beta @ Y2.T)); y2 = Y2[i2]
        theta = (c - 2*beta) @ x + beta @ y1 + beta @ y2
        best = min(best, theta)
        if best - ref < 1e-6 * max(1.0, abs(ref)): break
        g = y1 + y2 - 2*x; gn = g @ g
        if gn < 1e-15: break
        beta -= (theta - ref) / gn * g          # Polyak com alvo exato
    nsub += 1
    piores = max(piores, abs(best - ref) / max(1.0, abs(ref)))
print(f"terceiro metodo: {nsub} instancias, pior desvio relativo subgradiente vs LP-envoltorio: {piores:.2e}")
ok_sub = piores < 1e-3
print("RESULTADO:", "EXHAUSTIVE UNIVERSE AND METHOD TRIANGLE: ALL PASSED" if ok_caixa and ok_sub else "VIOLATIONS FOUND")
