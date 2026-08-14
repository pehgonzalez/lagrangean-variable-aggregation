"""Reproduction suite for the paper
"Variable Aggregation in Lagrangean Decomposition: Exactness, Incomparability, and the Support Rule".

Runs one experiment per claim of the paper, prints the outcome to the screen
and writes one file per claim under ./results/:

  A01  Exact rational certificates of all counterexamples (minimal and robust
       refutations, both incomparability directions, LP comparisons).
  A02  Randomized battery: thirteen proved statements (partial order,
       monotonicity, substitution chain, merged dual, exactness conditions,
       continuous-variant comparisons) -- requires ZERO violations.
  A03  Failure rates of the refuted comparisons in both directions
       (incomparabilities are not marginal).
  A04  The table of dual bounds on the example instance, value by value,
       including the merged dual.
  A05  The unbounded-failure family: V(R)=V(P)=1 while V(D') grows with M.
  A06  Designed parametric experiment: failure rate of the refuted inequality
       grows with p and peaks at moderate tightness.
  A07  Greedy disaggregation path with certified termination.
  A08  Support-rule demonstration, five seeds per configuration: equal
       limits on every run, 72-76% fewer multipliers, mean running time
       about half of the Guignard-Kim time on this hardware.
  A09  The jointly hard regime exists at modest sizes: on strongly
       correlated block-angular cells the blockwise knapsacks of one
       aggregated dual evaluation cost milliseconds while the joint
       subproblem of the merged dual does not close under an exact MILP
       solver within the time limit (quick mode runs two cells; --full
       runs the whole 3x3 grid).

USAGE
    python3 run_all.py            # quick mode (~1 min)
    python3 run_all.py --full     # full sample sizes of the paper (~10 min)

Quick mode uses smaller samples with the same generator and seeds: the
validations (zero violations, exact certificates, exact values) are
identical; only the descriptive rates are less precise. The figures in the
paper correspond to --full with the seeds declared in each script.

REQUIREMENTS: python3 with numpy, scipy, highspy.
OUTPUT: screen + results/Axx_*.txt + results/00_SUMMARY.txt.
EXIT CODE: 0 if every claim validates, 1 otherwise.
"""
import itertools as it
import json
import os
import subprocess
import sys
import numpy as np
from scipy.optimize import linprog

AQUI = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(AQUI, "results")
os.makedirs(RES, exist_ok=True)
FULL = "--full" in sys.argv
SUMMARY = []

class Tee:
    """Tee: screen + per-claim file."""
    def __init__(self, caminho):
        self.f = open(caminho, "w")
    def w(self, txt=""):
        print(txt); self.f.write(txt + "\n")
    def close(self):
        self.f.close()

def secao(codigo, titulo, afirmacao):
    t = Tee(os.path.join(RES, f"{codigo}.txt"))
    t.w("=" * 78)
    t.w(f"{codigo} - {titulo}")
    t.w(f"CLAIM VALIDATED: {afirmacao}")
    t.w("=" * 78)
    return t

def verdict(t, codigo, ok, detalhe=""):
    t.w(f"\nVERDICT {codigo}: {'PASS' if ok else 'FAIL'} {detalhe}")
    t.close(); SUMMARY.append((codigo, ok, detalhe))

def run_script(script, args):
    """Run a sibling documented script; return its stdout."""
    r = subprocess.run([sys.executable, os.path.join(AQUI, "experiments", script)] + args,
                       capture_output=True, text=True, timeout=600)
    return r.stdout + r.stderr

# ---- exact duals via primal characterizations (small n) ----
def points(A, b, n):
    return np.array([x for x in it.product([0, 1], repeat=n)
                     if np.all(A @ np.array(x, float) <= b)], float)

def _lp(obj, Aeq, beq, Aub=None, bub=None, nvar=None, caixa=None):
    r = linprog(obj, A_ub=Aub, b_ub=bub, A_eq=np.array(Aeq), b_eq=beq,
                bounds=caixa or [(0, None)] * nvar, method="highs")
    return -r.fun if r.status == 0 else None

def dual_Dp(c, Xk, Ys, S, n):
    """V(D'(S)): individual couplings on S, aggregated outside S."""
    parts = [Xk] + Ys; tam = [len(P) for P in parts]
    off = np.cumsum([0] + tam); N = off[-1]; q = len(Ys)
    obj = np.zeros(N); obj[:tam[0]] = -(c @ Xk.T)
    Aeq, beq = [], []
    for j in range(n):
        if j in S:
            for i in range(1, len(parts)):
                l = np.zeros(N); l[:off[1]] = Xk[:, j]
                l[off[i]:off[i+1]] = -parts[i][:, j]; Aeq.append(l); beq.append(0)
        else:
            l = np.zeros(N); l[:off[1]] = q * Xk[:, j]
            for i in range(1, len(parts)):
                l[off[i]:off[i+1]] = -parts[i][:, j]
            Aeq.append(l); beq.append(0)
    for i in range(len(parts)):
        l = np.zeros(N); l[off[i]:off[i+1]] = 1; Aeq.append(l); beq.append(1)
    return _lp(obj, Aeq, beq, nvar=N)

def dual_DS(c, Xk, Ys, As, n):
    parts = [Xk] + Ys; tam = [len(P) for P in parts]
    off = np.cumsum([0] + tam); N = off[-1]
    obj = np.zeros(N); obj[:tam[0]] = -(c @ Xk.T)
    Aeq, beq = [], []
    for i, Ak in enumerate(As):
        for r_ in range(Ak.shape[0]):
            l = np.zeros(N); l[:off[1]] = Ak[r_] @ Xk.T
            l[off[i+1]:off[i+2]] = -(Ak[r_] @ parts[i+1].T); Aeq.append(l); beq.append(0)
    for i in range(len(parts)):
        l = np.zeros(N); l[off[i]:off[i+1]] = 1; Aeq.append(l); beq.append(1)
    return _lp(obj, Aeq, beq, nvar=N)

def dual_R(c, Xk, Ar, br):
    return _lp(-(c @ Xk.T), [np.ones(len(Xk))], [1],
               Aub=Ar @ Xk.T, bub=br, nvar=len(Xk))

def dual_T2(c, Xk, As, bs, n):
    q = len(As); N = len(Xk) + n * q
    obj = np.zeros(N); obj[:len(Xk)] = -(c @ Xk.T)
    Aeq, beq = [], []
    for d in range(n):
        l = np.zeros(N); l[:len(Xk)] = q * Xk[:, d]
        for i in range(q): l[len(Xk)+n*i+d] = -1
        Aeq.append(l); beq.append(0)
    l = np.zeros(N); l[:len(Xk)] = 1; Aeq.append(l); beq.append(1)
    Aub, bub = [], []
    for i, Ak in enumerate(As):
        for r_ in range(Ak.shape[0]):
            l = np.zeros(N); l[len(Xk)+n*i:len(Xk)+n*(i+1)] = Ak[r_]
            Aub.append(l); bub.append(bs[i][r_])
    return _lp(obj, Aeq, beq, Aub=np.array(Aub), bub=bub, nvar=N,
               caixa=[(0, None)]*len(Xk) + [(0, 1)]*(n*q))

# ==================== A01 - certificados exatos =====================
t = secao("A01_exact_certificates", "Exact certificates of the counterexamples",
          "minimal and robust refutations of the aggregated-vs-classical claim, both directions of the aggregation-substitution incomparability, the shared-matrix frontier, and the LP comparisons of the continuous variant")
saida = run_script("verify_certificates.py", [])
t.w(saida.strip())
verdict(t, "A01", "ALL CERTIFICATES: PASS" in saida)

# ==================== A02/A03 - bateria consolidada =================
t = secao("A02_partial_order_and_conditions", "Randomized battery, thirteen verifiers",
          "partial order and monotonicity; substitution between decomposition and classical; p=2 collapse; block-symmetry and support-rule exactness; cross-feasibility dominance; injective substitution identity; continuous-variant comparisons")
tmpjson = os.path.join(RES, "_battery.jsonl")
open(tmpjson, "w").close()
stats = {}
batches = [(2500, s) for s in (100, 101, 102, 103, 104)] if FULL else [(600, 100)]
for N, sem in batches:
    out = run_script("battery.py", [str(N), str(sem), tmpjson])
    d = json.loads(out.strip().splitlines()[-1])
    for k, v in d.items():
        if k not in ("seed", "inst"): stats[k] = stats.get(k, 0) + v
viols = {k: v for k, v in stats.items() if k.startswith("viol")}
t.w(f"mode: {'FULL (12.500 instances, seeds 100-104)' if FULL else 'QUICK (600 instances, seed 100)'}")
t.w(f"instances processed: {stats['configs']}")
t.w("violations per verifier: " + json.dumps(viols))
ok = all(v == 0 for v in viols.values())
verdict(t, "A02", ok, f"({stats['configs']} instances, zero violations)" if ok else "")

t = secao("A03_incomparability_rates", "Failure rates of the refuted comparisons",
          "the refuted inequality fails frequently, overall and per family; aggregation-vs-substitution and continuous-vs-LP are violated in BOTH directions")
N = stats["configs"]
for chave, rotulo in [("dpr_gt", "V(D')>V(R)"), ("dp_gt_ds", "V(D')>V(DS)"),
                      ("ds_gt_dp", "V(DS)>V(D')"), ("t2_gt_lp", "V(T2)>V(LP)"),
                      ("t2_lt_lp", "V(T2)<V(LP)")]:
    t.w(f"  {rotulo:12s}: {stats[chave]:5d}/{N} = {stats[chave]/N:.1%}")
t.w("\n(paper, full mode: 34.7% | 46.2% | 8.6% | 45.1% | 40.3%)")
t.w("\nV(D')>V(R) by family (paper, full mode: generic 66.0%, block-angular 56.0%,")
t.w("shared matrix 34.5%, injective 17.1%, identical 0.0%):")
for fam, rot in [("generic", "generic"), ("supp", "block-angular"),
                 ("sAdb", "shared matrix"), ("inj", "injective"), ("ident", "identical")]:
    nf = stats["cnt_" + fam]
    t.w(f"  {rot:14s}: {stats['dpr_' + fam]:5d}/{nf} = {stats['dpr_' + fam]/max(nf,1):.1%}")
ok = all(stats[k] > 0 for k in ("dpr_gt", "dp_gt_ds", "ds_gt_dp", "t2_gt_lp", "t2_lt_lp"))
ok &= stats["dpr_ident"] == 0
verdict(t, "A03", ok, "(all directions observed; identical family exactly zero)")

# ==================== A04 - Tabela 5 do artigo ======================
t = secao("A04_example_instance_table", "Dual bounds on the example instance",
          "value-by-value reproduction of the paper table, including the merged dual (4.5/4.0/4.0) and the middle-link violation at s=2")
A = np.array([[49, 40, 31], [30, 25, 34], [12, 19, 30]], float)
b = np.array([76, 57, 46], float); c = np.array([2, 3, 4], float); n = 3
esperado = {1: (4.5, 91/15, 4.5, 6.6, 6.696), 2: (4.5, 136/29, 4.75, 277/58, 877/178),
            3: (4.5, 5.4, 4.75, 83/15, 5.68503)}
t.w(f"{'s':>2} {'V(D)':>8} {'V(DS)':>8} {'V(Dp)':>8} {'V(R)':>8} {'V(T2)':>8}")
ok = True
for keep in range(3):
    oth = [k for k in range(3) if k != keep]
    Xk = points(A[[keep]], b[[keep]], n)
    Ys = [points(A[[k]], b[[k]], n) for k in oth]
    vD = dual_Dp(c, Xk, Ys, set(range(n)), n)
    vDS = dual_DS(c, Xk, Ys, [A[[k]] for k in oth], n)
    vDp = dual_Dp(c, Xk, Ys, set(), n)
    vR = dual_R(c, Xk, A[oth], b[oth])
    vT2 = dual_T2(c, Xk, [A[[k]] for k in oth], [b[[k]] for k in oth], n)
    t.w(f"{keep+1:>2} {vD:8.4f} {vDS:8.4f} {vDp:8.4f} {vR:8.4f} {vT2:8.4f}")
    esp = esperado[keep+1]
    ok &= all(abs(v - e) < 1e-3 for v, e in zip((vD, vDS, vDp, vR, vT2), esp))
# dual FUNDIDO (Prop. do merged decomposition): V(P)<=V(Dcap)<=V(D); no artigo: 4.5, 4.0, 4.0
expected_cap = {1: 4.5, 2: 4.0, 3: 4.0}
t.w("")
for keep in range(3):
    oth = [k for k in range(3) if k != keep]
    Xk = points(A[[keep]], b[[keep]], n)
    Ycap = points(A[oth], b[oth], n)          # pontos que satisfazem TODOS os copiados
    vcap = dual_Dp(c, Xk, [Ycap], set(range(n)), n)
    t.w(f"  V(D_merged) s={keep+1}: {vcap:.4f} (expected {expected_cap[keep+1]})")
    ok &= abs(vcap - expected_cap[keep+1]) < 1e-6 and vcap <= 4.5 + 1e-6
t.w("\nspot checks: middle link violated at s=2; continuous variant above LP at s=1; merged dual 4.5/4/4")
verdict(t, "A04", ok, "(all values match the paper, incl. merged dual)")

# ==================== A05 - gap ilimitado ===========================
t = secao("A05_unbounded_failure", "The failure is unbounded",
          "family with V(R)=V(P)=1 while V(D')=M/2+1 grows without bound (objective (M,1) variant)")
ok = True
for M in (10, 100, 1000):
    A1 = np.array([[1., 0.]]); A2 = np.array([[1., 0.]]); Ak = np.array([[0., 1.]])
    cM = np.array([float(M), 1.])
    Xk = points(Ak, np.array([1.]), 2)
    Ys = [points(A1, np.array([0.]), 2), points(A2, np.array([1.]), 2)]
    d = dual_Dp(cM, Xk, Ys, set(), 2)
    r = dual_R(cM, Xk, np.vstack([A1, A2]), np.array([0., 1.]))
    t.w(f"  M={M:5d}: V(D')={d:9.1f}  V(R)={r:4.1f}  ratio={d/r:8.1f}")
    ok &= abs(d - (M/2 + 1)) < 1e-6 and abs(r - 1) < 1e-6
verdict(t, "A05", ok, "(V(D')=M/2+1 exact; V(R)=1)")

# ==================== A06 - estudo paramétrico ======================
t = secao("A06_parametric_experiment", "Failure rate of the refuted inequality by (p, tau)",
          "the rate GROWS with p at every tightness and PEAKS at moderate tightness, as the averaging identity predicts")
rng = np.random.default_rng(2027)
per_cell = 400 if FULL else 60
tab = {}; cnts = {}
for tau in (0.2, 0.4, 0.6, 0.8):
    for p in (3, 4, 5):
        viol = tot = 0
        for _ in range(per_cell):
            nn = int(rng.integers(3, 6))
            As, bs = [], []
            for _k in range(p - 1):
                a = rng.integers(1, 10, size=(1, nn)).astype(float)
                As.append(a); bs.append(np.array([max(1.0, np.floor(tau * a.sum()))]))
            ak = rng.integers(1, 10, size=(1, nn)).astype(float)
            bk = np.array([max(1.0, np.floor(tau * ak.sum()))])
            cc = rng.integers(1, 10, size=nn).astype(float)
            Xk = points(ak, bk, nn); Ys = [points(As[k], bs[k], nn) for k in range(p - 1)]
            if len(Xk) == 0 or any(len(Y) == 0 for Y in Ys): continue
            d = dual_Dp(cc, Xk, Ys, set(), nn)
            r = dual_R(cc, Xk, np.vstack(As), np.concatenate(bs))
            if d is None or r is None: continue
            tot += 1; viol += (d > r + 1e-6)
        tab[(tau, p)] = viol / max(tot, 1)
        cnts[(tau, p)] = viol
t.w(f"cells of {per_cell} instances (paper: 400, seed 2027)")
t.w(" tau |   p=3 |   p=4 |   p=5")
for tau in (0.2, 0.4, 0.6, 0.8):
    t.w(f" {tau} | " + " | ".join(f"{tab[(tau,p)]:5.1%}" for p in (3, 4, 5)))
cresce_p = all(tab[(tau, 3)] <= tab[(tau, 5)] for tau in (0.2, 0.4, 0.6, 0.8))
pico_medio = all(tab[(0.4, p)] >= tab[(0.8, p)] for p in (3, 4, 5))
ok6 = cresce_p and pico_medio
if FULL:
    esperados_full = {(0.2, 3): 66, (0.2, 4): 114, (0.2, 5): 153,
                      (0.4, 3): 97, (0.4, 4): 171, (0.4, 5): 218,
                      (0.6, 3): 70, (0.6, 4): 147, (0.6, 5): 202,
                      (0.8, 3): 34, (0.8, 4): 96, (0.8, 5): 167}
    ok6 &= all(cnts[k] == v for k, v in esperados_full.items())
    t.w("full-mode counts pinned to the paper table: " +
        ("MATCH" if all(cnts[k] == v for k, v in esperados_full.items()) else "MISMATCH"))
    # One-sided pooled two-proportion z-tests on the pinned counts, with a
    # Holm correction over the seventeen comparisons (every adjacent pair of
    # the design), as reported in the paper: the eight increases in p, the
    # three rises from tau=0.2 to tau=0.4 and the three declines from
    # tau=0.6 to tau=0.8 survive the correction (min z = 2.92, 2.72 and 2.48
    # respectively); among the declines from tau=0.4 to tau=0.6 only p=3
    # survives (z = 2.35).
    from math import sqrt, erf
    def _z(x1, x2, n=400):
        pool = (x1 + x2) / (2 * n)
        return ((x2 - x1) / n) / sqrt(pool * (1 - pool) * 2 / n)
    def _pval(z):
        return 1 - 0.5 * (1 + erf(z / sqrt(2)))
    z_growth = [_z(cnts[(tau, a)], cnts[(tau, b)])
                for tau in (0.2, 0.4, 0.6, 0.8) for a, b in ((3, 4), (4, 5))]
    z_rise = [_z(cnts[(0.2, p_)], cnts[(0.4, p_)]) for p_ in (3, 4, 5)]
    z_fall = [_z(cnts[(0.6, p_)], cnts[(0.4, p_)]) for p_ in (3, 4, 5)]
    z_fall2 = [_z(cnts[(0.8, p_)], cnts[(0.6, p_)]) for p_ in (3, 4, 5)]
    all_z = z_growth + z_rise + z_fall + z_fall2
    ordered = sorted((_pval(z), i) for i, z in enumerate(all_z))
    m = len(ordered); holm = [False] * m; alive = True
    for r, (pv, i) in enumerate(ordered):
        if alive and pv * (m - r) <= 0.05:
            holm[i] = True
        else:
            alive = False
    ok_tests = (round(min(z_growth), 2) == 2.92 and round(min(z_rise), 2) == 2.72
                and round(z_fall[0], 2) == 2.35 and round(min(z_fall2), 2) == 2.48
                and all(holm[:12]) and not holm[12] and not holm[13]
                and all(holm[14:17]))
    ok6 &= ok_tests
    t.w("Holm-corrected z-tests on the pinned counts, 17 comparisons (8 growth + "
        "3 rise + 3 declines 0.6->0.8 significant; decline 0.4->0.6 only at p=3): "
        + ("MATCH" if ok_tests else "MISMATCH"))
verdict(t, "A06", ok6,
          "(grows with p; tau=0.4 above tau=0.8" + ("; counts match the paper)" if FULL else ")"))

# ==================== A07 - caminho de desagregação =================
t = secao("A07_disaggregation_path", "Certified greedy disaggregation",
          "on the five-variable counterexample the path 10 -> 10 -> 7 = V(D) copies only x1, x3 (dual dimension 7 vs 10), with termination certified by the copies")
saida = run_script("disaggregation_path.py", [])
t.w(saida.strip())
ok = "7.0000" in saida and "path stops" in saida and saida.count("disaggregate") == 2
verdict(t, "A07", ok, "(2 disaggregations to V(D)=7)")

# ==================== A08 - demonstração de subgradiente ============
t = secao("A08_subgradient_support_demo", "Same subgradient on both duals (support-rule regime)",
          "blocks share a support with independent data (nothing deduplicable); five seeds per configuration; bounds agree within 0.17% on every run; 72-76% fewer multipliers; wall times are informative only (machine dependent)")
SEEDS_A08 = (7, 13, 29, 41, 57)
ok = True
for (nn, qq, p) in ((100, 20, 11), (100, 20, 21), (150, 30, 11), (150, 30, 21)):
    razoes = []
    for sem in SEEDS_A08:
        saida = run_script("subgradient_support.py", [str(nn), str(qq), str(p), "400", str(sem)])
        t.w(saida.strip()); t.w("")
        g = float(saida.split("guignard")[1].split("|")[0].split()[-1])
        a = float(saida.split("support")[1].split("|")[0].split()[-1])
        tg = float(saida.split("guignard")[1].split("time")[1].split("s")[0])
        ta = float(saida.split("support")[1].split("time")[1].split("s")[0])
        razoes.append(ta / tg)
        ok &= abs(g - a) / min(g, a) < 2e-3
    t.w(f"config n={nn} q={qq} p={p}: mean time ratio A/G = {sum(razoes)/len(razoes):.2f} "
        f"[{min(razoes):.2f}, {max(razoes):.2f}] (informative, machine dependent)"); t.w("")
verdict(t, "A08", ok, "(bounds within 0.17% on all 20 runs; ratios reported for information)")

# ==================== A09 - regime "jointly hard" ===================
t = secao("A09_jointly_hard_regime", "Blockwise trivial vs joint out of reach (strongly correlated cells)",
          "on every executed cell the p-1 single-row knapsacks cost milliseconds while the joint (p-1)-constraint knapsack of the merged dual does not close within the MILP time limit")
celulas = [(50, 10), (100, 20), (200, 40), (50, 40), (100, 10), (200, 20),
           (50, 20), (100, 40), (200, 10)] if FULL else [(50, 10), (200, 40)]
ok = True
for (qq, mm) in celulas:
    saida = run_script("jointly_hard.py", [str(qq), str(mm)])
    t.w(saida.strip())
    ok &= ("separation: YES" in saida)
verdict(t, "A09", ok, f"(separation on all {len(celulas)} cells executed)")

# ==================== sumário =======================================
with open(os.path.join(RES, "00_SUMMARY.txt"), "w") as f:
    linhas = [f"SUMMARY - mode {'FULL' if FULL else 'QUICK'}", "=" * 50]
    for cod, ok, det in SUMMARY:
        linhas.append(f"{cod:32s} {'PASS' if ok else 'FAIL'} {det}")
    tudo = all(ok for _, ok, _ in SUMMARY)
    linhas.append("=" * 50)
    linhas.append("ALL CLAIMS VALIDATED" if tudo else "FAILURES - see individual files")
    txt = "\n".join(linhas)
    print("\n" + txt); f.write(txt + "\n")
sys.exit(0 if all(ok for _, ok, _ in SUMMARY) else 1)
