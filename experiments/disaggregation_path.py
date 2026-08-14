"""
disaggregation_path.py - Greedy disaggregation path (corollary on disaggregation paths).

Purpose
-------
Illustrates the disaggregation-path corollary of the paper on the
nondegenerate counterexample of Remark 4.3 (n=5, p=3):
starting from total aggregation (S = empty set), repeatedly disaggregate the
variable whose copies disagree the most with the retained solution, measured
by the dispersion sum_k |y^k_j - x_j| at the current optimum (the aggregate
residual sum_k y^k_j - (p-1) x_j is zero by construction, so it carries no
information), recomputing V(D'(S)) at every step, until the copies coincide
with the retained solution, which certifies that the Guignard-Kim value V(D)
has been reached.

Method
------
Each dual value is computed EXACTLY through its Geoffrion primal
characterization (Proposition 2.2 of the manuscript): a linear program over
convex multipliers of the enumerated 0-1 points of each block, solved with
HiGHS. This is exact for the small instance used here (n = 5).

Instance (Remark 4.3 of the manuscript)
---------------------------------------
copied block 1: 4x1+4x2+2x3+     5x5 <= 9 ;  8x1+     8x3+8x4+2x5 <= 4
copied block 3:      5x2+ x3+7x4+6x5 <= 17
retained block: 7x1+3x2+     9x4+4x5 <= 8 ;  8x1+6x2+7x3+7x4+ x5 <= 13
objective     : c = (7,4,9,4,3), maximize.

Output
------
One line per step: current S, V(D'(S)), and the variable chosen next.
Last line reports V(D) for reference. No randomness involved.
"""
import itertools as it
import numpy as np
import highspy

INF = highspy.kHighsInf
H = highspy.Highs()

def lp_max(obj, rows, rlb, rub, ub):
    """Solve max obj'w s.t. rlb <= rows.w <= rub, 0 <= w <= ub via HiGHS.
    Returns (optimal value, weight vector) or (None, None) if not optimal."""
    H.clear(); H.setOptionValue('output_flag', False)
    N = len(obj)
    H.addVars(N, np.zeros(N), np.array(ub, float))
    H.changeColsCost(N, np.arange(N, dtype=np.int32), -np.array(obj, float))
    for r, a, b in zip(rows, rlb, rub):
        nz = np.nonzero(r)[0].astype(np.int32)
        H.addRow(float(a), float(b), len(nz), nz, np.array(r, float)[nz])
    H.run()
    if H.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        return None, None
    sol = np.array(H.getSolution().col_value)
    return -H.getInfo().objective_function_value, sol

def binary_points(A, b, n):
    """All points of {0,1}^n satisfying A x <= b (componentwise)."""
    return np.array([x for x in it.product([0, 1], repeat=n)
                     if np.all(A @ np.array(x, float) <= b)], float)

def V_Dprime(c, Xk, Ys, S, n):
    """V(D'(S)) by the primal characterization; also returns the optimal
    (x_hat, list of y_hat^k) reconstructed from the convex weights."""
    parts = [Xk] + Ys
    sizes = [len(P) for P in parts]
    off = np.cumsum([0] + sizes); N = off[-1]; q = len(Ys)
    obj = np.zeros(N); obj[:sizes[0]] = c @ Xk.T
    rows, lb, ub = [], [], []
    for j in range(n):
        if j in S:                      # individual couplings x_j = y^k_j
            for i in range(1, len(parts)):
                row = np.zeros(N)
                row[:off[1]] = Xk[:, j]
                row[off[i]:off[i+1]] = -parts[i][:, j]
                rows.append(row); lb.append(0); ub.append(0)
        else:                           # aggregated coupling
            row = np.zeros(N); row[:off[1]] = q * Xk[:, j]
            for i in range(1, len(parts)):
                row[off[i]:off[i+1]] = -parts[i][:, j]
            rows.append(row); lb.append(0); ub.append(0)
    for i in range(len(parts)):        # convexity rows
        row = np.zeros(N); row[off[i]:off[i+1]] = 1
        rows.append(row); lb.append(1); ub.append(1)
    val, w = lp_max(obj, rows, lb, ub, [INF] * N)
    xh = Xk.T @ w[:off[1]]
    yh = [parts[i].T @ w[off[i]:off[i+1]] for i in range(1, len(parts))]
    return val, xh, yh

# ---- instance of Remark 4.3 ----
A1 = np.array([[4,4,2,0,5],[8,0,8,8,2]], float); b1 = np.array([9,4], float)
A3 = np.array([[0,5,1,7,6]], float);             b3 = np.array([17], float)
Ak = np.array([[7,3,0,9,4],[8,6,7,7,1]], float); bk = np.array([8,13], float)
c  = np.array([7,4,9,4,3], float); n = 5

Xk = binary_points(Ak, bk, n)
Ys = [binary_points(A1, b1, n), binary_points(A3, b3, n)]

S = set()
while True:
    val, xh, yh = V_Dprime(c, Xk, Ys, S, n)
    viol = np.zeros(n)
    for j in range(n):
        if j not in S:
            viol[j] = sum(abs(y[j] - xh[j]) for y in yh)
    if viol.max() < 1e-9:
        print(f"S={sorted(S) if S else '{}'}: V(D'(S)) = {val:.4f}  (copies agree with x; path stops)")
        break
    jstar = int(np.argmax(viol))
    print(f"S={sorted(S) if S else '{}'}: V(D'(S)) = {val:.4f}  -> disaggregate x{jstar+1} (dispersion {viol[jstar]:.3f})")
    S.add(jstar)
vD, _, _ = V_Dprime(c, Xk, Ys, set(range(n)), n)
print(f"reference: V(D) = {vD:.4f}")
