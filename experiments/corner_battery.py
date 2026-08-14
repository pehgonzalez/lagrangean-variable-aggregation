"""Targeted verification battery for hypothesis variants of each theorem.

Each verifier exercises a theorem in configurations complementary to the main
randomized battery:

  V1  merged decomposition, V(P) <= V(Dcap) <= V(D), on random instances;
  V2  cross-feasibility theorem with a random NONEMPTY copied set S;
  V3  block-symmetry exactness with identical blocks of up to 3 rows;
  V4  support rule with MULTI-ROW copied blocks on random local supports;
  V5  greedy disaggregation paths end to end (monotonicity, at most n steps,
      certified termination at the decomposition value);
  V6  substitution = decomposition for RECTANGULAR injective copied matrices.

Exact dual values via primal characterizations (scipy/HiGHS); tolerance 1e-6;
deterministic seed. Usage: python3 corner_battery.py
"""
import itertools as it
import numpy as np
from scipy.optimize import linprog

rng = np.random.default_rng(4242)
E = 1e-6

def pontos(A, b, n):
    return np.array([x for x in it.product([0, 1], repeat=n)
                     if np.all(A @ np.array(x, float) <= b)], float)

def _max(obj, Aeq, beq, Aub=None, bub=None, N=None, caixa=None):
    r = linprog(obj, A_ub=Aub, b_ub=bub, A_eq=np.array(Aeq), b_eq=beq,
                bounds=caixa or [(0, None)] * N, method='highs')
    return -r.fun if r.status == 0 else None

def V_Dp(c, Xk, Ys, S, n, retorna_sol=False):
    partes = [Xk] + Ys; tam = [len(P) for P in partes]
    off = np.cumsum([0] + tam); N = off[-1]; q = len(Ys)
    obj = np.zeros(N); obj[:tam[0]] = -(c @ Xk.T)
    Aeq, beq = [], []
    for j in range(n):
        if j in S:
            for i in range(1, len(partes)):
                l = np.zeros(N); l[:off[1]] = Xk[:, j]
                l[off[i]:off[i+1]] = -partes[i][:, j]; Aeq.append(l); beq.append(0)
        else:
            l = np.zeros(N); l[:off[1]] = q * Xk[:, j]
            for i in range(1, len(partes)):
                l[off[i]:off[i+1]] = -partes[i][:, j]
            Aeq.append(l); beq.append(0)
    for i in range(len(partes)):
        l = np.zeros(N); l[off[i]:off[i+1]] = 1; Aeq.append(l); beq.append(1)
    r = linprog(obj, A_eq=np.array(Aeq), b_eq=beq, bounds=[(0, None)] * N, method='highs')
    if r.status != 0: return (None, None, None) if retorna_sol else None
    if not retorna_sol: return -r.fun
    w = r.x
    xh = Xk.T @ w[:off[1]]
    yh = [partes[i].T @ w[off[i]:off[i+1]] for i in range(1, len(partes))]
    return -r.fun, xh, yh

def V_R(c, Xk, Ar, br):
    return _max(-(c @ Xk.T), [np.ones(len(Xk))], [1],
                Aub=Ar @ Xk.T, bub=br, N=len(Xk))

def V_DS(c, Xk, Ys, As, n):
    partes = [Xk] + Ys; tam = [len(P) for P in partes]
    off = np.cumsum([0] + tam); N = off[-1]
    obj = np.zeros(N); obj[:tam[0]] = -(c @ Xk.T)
    Aeq, beq = [], []
    for i, Ak in enumerate(As):
        for r_ in range(Ak.shape[0]):
            l = np.zeros(N); l[:off[1]] = Ak[r_] @ Xk.T
            l[off[i+1]:off[i+2]] = -(Ak[r_] @ partes[i+1].T); Aeq.append(l); beq.append(0)
    for i in range(len(partes)):
        l = np.zeros(N); l[off[i]:off[i+1]] = 1; Aeq.append(l); beq.append(1)
    return _max(obj, Aeq, beq, N=N)

def gera_bloco(n, m, hi=10):
    A = rng.integers(0, hi, size=(m, n)).astype(float)
    b = np.array([max(1, int(rng.integers(1, max(2, int(A[r].sum()))))) for r in range(m)], float)
    return A, b

res = {}

# ---- V1: prop:merge em instancias aleatorias ----
viol = tot = 0
for t in range(300):
    n = int(rng.integers(3, 6)); p = int(rng.integers(3, 5))
    blocos = [gera_bloco(n, int(rng.integers(1, 3))) for _ in range(p)]
    c = rng.integers(1, 10, size=n).astype(float)
    keep = int(rng.integers(0, p)); oth = [k for k in range(p) if k != keep]
    Xk = pontos(*blocos[keep], n)
    Ys = [pontos(*blocos[k], n) for k in oth]
    Acap = np.vstack([blocos[k][0] for k in oth]); bcap = np.concatenate([blocos[k][1] for k in oth])
    Ycap = pontos(Acap, bcap, n)
    if len(Xk) == 0 or len(Ycap) == 0 or any(len(Y) == 0 for Y in Ys): continue
    feas = [x for x in it.product([0, 1], repeat=n)
            if all(np.all(A @ np.array(x, float) <= b) for A, b in blocos)]
    VP = max((c @ np.array(x, float) for x in feas), default=None)
    Dcap = V_Dp(c, Xk, [Ycap], set(range(n)), n)
    D = V_Dp(c, Xk, Ys, set(range(n)), n)
    if None in (Dcap, D): continue
    tot += 1
    if (VP is not None and VP > Dcap + E) or Dcap > D + E: viol += 1
res['V1 merge'] = (tot, viol)

# ---- V2: cruzada com S aleatorio nao vazio ----
viol = tot = 0
for t in range(200):
    n = int(rng.integers(3, 6))
    A1, b1 = gera_bloco(n, 1)
    A2 = rng.integers(0, 10, size=(1, n)).astype(float)
    Y1 = pontos(A1, b1, n)
    if len(Y1) == 0: continue
    b2 = np.array([max(float(rng.integers(1, max(2, int(A2.sum())))),
                       float(max(float((A2 @ y)[0]) for y in Y1)))])
    Y2 = pontos(A2, b2, n)
    if not all(np.all(A1 @ y <= b1) for y in Y2): continue   # exigir cruzada nas 2 direcoes
    Ak, bk = gera_bloco(n, 1)
    Xk = pontos(Ak, bk, n)
    if len(Xk) == 0 or len(Y2) == 0: continue
    c = rng.integers(1, 10, size=n).astype(float)
    S = set(int(j) for j in rng.choice(n, size=int(rng.integers(1, n)), replace=False))
    Dp = V_Dp(c, Xk, [Y1, Y2], S, n)
    R = V_R(c, Xk, np.vstack([A1, A2]), np.concatenate([b1, b2]))
    if None in (Dp, R): continue
    tot += 1
    if Dp > R + E: viol += 1
res['V2 cruzada S!=vazio'] = (tot, viol)

# ---- V3: simetria com blocos de ate 3 linhas e S aleatorio ----
viol = tot = 0
for t in range(200):
    n = int(rng.integers(3, 6)); p = int(rng.integers(3, 5))
    Ac, bc = gera_bloco(n, int(rng.integers(1, 4)))
    Ak, bk = gera_bloco(n, 1)
    c = rng.integers(1, 10, size=n).astype(float)
    Xk = pontos(Ak, bk, n); Yc = pontos(Ac, bc, n)
    if len(Xk) == 0 or len(Yc) == 0: continue
    Ys = [Yc] * (p - 1)
    S = set(int(j) for j in rng.choice(n, size=int(rng.integers(0, n)), replace=False))
    Dp = V_Dp(c, Xk, Ys, S, n)
    D = V_Dp(c, Xk, Ys, set(range(n)), n)
    if None in (Dp, D): continue
    tot += 1
    if abs(Dp - D) > E: viol += 1
res['V3 simetria 3 linhas'] = (tot, viol)

# ---- V4: suporte multi-linha, suportes aleatorios ----
viol = tot = 0
for t in range(200):
    n = 6
    Js = [sorted(rng.choice(n, size=int(rng.integers(2, 4)), replace=False)) for _ in range(2)]
    As, bs = [], []
    for J in Js:
        m = int(rng.integers(1, 3))
        A = np.zeros((m, n)); A[:, J] = rng.integers(1, 10, size=(m, len(J)))
        b = np.array([max(1, int(rng.integers(1, max(2, int(A[r].sum()))))) for r in range(m)], float)
        As.append(A); bs.append(b)
    Ak, bk = gera_bloco(n, 1)
    c = rng.integers(1, 10, size=n).astype(float)
    Xk = pontos(Ak, bk, n); Ys = [pontos(As[i], bs[i], n) for i in range(2)]
    if len(Xk) == 0 or any(len(Y) == 0 for Y in Ys): continue
    Sstar = set(int(j) for J in Js for j in J)
    Dp = V_Dp(c, Xk, Ys, Sstar, n)
    D = V_Dp(c, Xk, Ys, set(range(n)), n)
    if None in (Dp, D): continue
    tot += 1
    if abs(Dp - D) > E: viol += 1
res['V4 suporte multi-linha'] = (tot, viol)

# ---- V5: caminhos de desagregacao em instancias aleatorias ----
viol = tot = 0
for t in range(80):
    n = int(rng.integers(3, 5)); p = 3
    blocos = [gera_bloco(n, 1) for _ in range(p)]
    c = rng.integers(1, 10, size=n).astype(float)
    Xk = pontos(*blocos[2], n); Ys = [pontos(*blocos[0], n), pontos(*blocos[1], n)]
    if len(Xk) == 0 or any(len(Y) == 0 for Y in Ys): continue
    D = V_Dp(c, Xk, Ys, set(range(n)), n)
    if D is None: continue
    S = set(); anterior = np.inf; ok = True; passos = 0
    while True:
        val, xh, yh = V_Dp(c, Xk, Ys, S, n, retorna_sol=True)
        if val is None: ok = False; break
        if val > anterior + E: ok = False; break        # monotonia
        anterior = val
        disp = np.zeros(n)
        for j in range(n):
            if j not in S: disp[j] = sum(abs(y[j] - xh[j]) for y in yh)
        if disp.max() < 1e-7:
            ok &= abs(val - D) < 1e-5                   # parada certificada => = V(D)
            break
        S.add(int(np.argmax(disp))); passos += 1
        if passos > n: ok = False; break                # <= n passos
    tot += 1
    if not ok: viol += 1
res['V5 caminhos'] = (tot, viol)

# ---- V6: injetivos retangulares 3x2 ----
viol = tot = 0
for t in range(150):
    n = 2
    As, bs = [], []
    for k in range(2):
        while True:
            A = rng.integers(0, 6, size=(3, n)).astype(float)
            if np.linalg.matrix_rank(A) == 2: break
        b = np.array([max(1, int(rng.integers(1, max(2, int(A[r].sum()))))) for r in range(3)], float)
        As.append(A); bs.append(b)
    Ak, bk = gera_bloco(n, 1)
    c = rng.integers(1, 9, size=n).astype(float)
    Xk = pontos(Ak, bk, n); Ys = [pontos(As[k], bs[k], n) for k in range(2)]
    if len(Xk) == 0 or any(len(Y) == 0 for Y in Ys): continue
    DS = V_DS(c, Xk, Ys, As, n)
    D = V_Dp(c, Xk, Ys, set(range(n)), n)
    if None in (DS, D): continue
    tot += 1
    if abs(DS - D) > E: viol += 1
res['V6 injetivos 3x2'] = (tot, viol)

print("=" * 60)
tudo_ok = True
for k, (tot, viol) in res.items():
    print(f"{k:28s}: {tot:4d} instancias, {viol} violacoes")
    tudo_ok &= (viol == 0 and tot > 0)
print("=" * 60)
print("RESULTADO:", "ALL CHECKS PASSED" if tudo_ok else "VIOLATIONS FOUND")
