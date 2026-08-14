"""Randomized verification battery for every theorem proved in the paper.

Five structural families are generated in rotation (generic, identical copied
blocks, block-angular, shared matrix with distinct right-hand sides, injective
copied matrices). On each instance the thirteen proved statements are checked
against exact dual values computed through the Geoffrion primal
characterizations (linear programs over enumerated 0-1 points, HiGHS).
Violation counters must all be zero. Descriptive failure rates of the refuted
comparisons are also tallied, overall and per family (the per-family rates of
V(D') > V(R) are the ones reported in the paper's family table).

For copied blocks with more than one row the continuous-copy comparisons use
one continuous copy per BLOCK, constrained by all rows of that block; the
three statements checked for it (R <= T2, D'(empty) <= T2, and T2 = R on the
identical family) hold for this blockwise variant by the same proofs given in
the paper, which states the single-row case.

Usage: battery.py N_INSTANCES SEED OUTFILE   (appends one JSON line to OUTFILE)
"""
import itertools as it, numpy as np, sys, json, highspy
N_INST=int(sys.argv[1]); SEED=int(sys.argv[2]); OUT=sys.argv[3]
rng=np.random.default_rng(SEED); INF=highspy.kHighsInf
H=highspy.Highs()
def hlp_max(obj,rows,rlb,rub,ub):
    # max obj'w s.t. rlb <= rows@w <= rub, 0<=w<=ub
    H.clear(); H.setOptionValue('output_flag',False)
    N=len(obj)
    H.addVars(N,np.zeros(N),np.array(ub,float))
    H.changeColsCost(N,np.arange(N,dtype=np.int32),-np.array(obj,float))
    for r,lb_,ub_ in zip(rows,rlb,rub):
        nz=np.nonzero(r)[0].astype(np.int32)
        H.addRow(float(lb_),float(ub_),len(nz),nz,np.array(r,float)[nz])
    H.run()
    if H.getModelStatus()!=highspy.HighsModelStatus.kOptimal: return None
    return -H.getInfo().objective_function_value
def pts(A,b,n): return np.array([x for x in it.product([0,1],repeat=n) if np.all(A@np.array(x,float)<=b)],float)
def solveD(c,Xk,Ys,S,n):
    parts=[Xk]+Ys; sizes=[len(P) for P in parts]; off=np.cumsum([0]+sizes); N=off[-1]; q=len(Ys)
    obj=np.zeros(N);obj[:sizes[0]]=c@Xk.T
    rows=[];lb=[];ub=[]
    for j in range(n):
        if j in S:
            for i in range(1,len(parts)):
                row=np.zeros(N);row[:off[1]]=Xk[:,j];row[off[i]:off[i+1]]=-parts[i][:,j];rows.append(row);lb.append(0);ub.append(0)
        else:
            row=np.zeros(N);row[:off[1]]=q*Xk[:,j]
            for i in range(1,len(parts)): row[off[i]:off[i+1]]=-parts[i][:,j]
            rows.append(row);lb.append(0);ub.append(0)
    for i in range(len(parts)):
        row=np.zeros(N);row[off[i]:off[i+1]]=1;rows.append(row);lb.append(1);ub.append(1)
    return hlp_max(obj,rows,lb,ub,[INF]*N)
def solveDS(c,Xk,Ys,As,bs,n):
    parts=[Xk]+Ys; sizes=[len(P) for P in parts]; off=np.cumsum([0]+sizes); N=off[-1]
    obj=np.zeros(N);obj[:sizes[0]]=c@Xk.T
    rows=[];lb=[];ub=[]
    for i,Ak in enumerate(As):
        for r_ in range(Ak.shape[0]):
            row=np.zeros(N);row[:off[1]]=Ak[r_]@Xk.T;row[off[i+1]:off[i+2]]=-(Ak[r_]@parts[i+1].T)
            rows.append(row);lb.append(0);ub.append(0)
    for i in range(len(parts)):
        row=np.zeros(N);row[off[i]:off[i+1]]=1;rows.append(row);lb.append(1);ub.append(1)
    return hlp_max(obj,rows,lb,ub,[INF]*N)
def solveR(c,Xk,Acop,bcop):
    N=len(Xk); obj=c@Xk.T
    rows=list(Acop@Xk.T);lb=[-INF]*len(bcop);ub=list(bcop)
    rows.append(np.ones(N));lb.append(1);ub.append(1)
    return hlp_max(obj,rows,lb,ub,[INF]*N)
def solveT2(c,Xk,As,bs,n):
    q=len(As); N=len(Xk)+n*q
    obj=np.zeros(N);obj[:len(Xk)]=c@Xk.T
    rows=[];lb=[];ub=[]
    for d in range(n):
        row=np.zeros(N);row[:len(Xk)]=q*Xk[:,d]
        for i in range(q): row[len(Xk)+n*i+d]=-1
        rows.append(row);lb.append(0);ub.append(0)
    row=np.zeros(N);row[:len(Xk)]=1;rows.append(row);lb.append(1);ub.append(1)
    for i,Ak in enumerate(As):
        for r_ in range(Ak.shape[0]):
            row=np.zeros(N);row[len(Xk)+n*i:len(Xk)+n*i+n]=Ak[r_];rows.append(row);lb.append(-INF);ub.append(bs[i][r_])
    return hlp_max(obj,rows,lb,ub,[INF]*len(Xk)+[1.0]*(n*q))

E=1e-6; keys=["configs","viol_P_D","viol_D_Dp","viol_mono","viol_D_DS","viol_DS_R","viol_p2","viol_ident","viol_supp","viol_cross","viol_R_T2","viol_Dp_T2","viol_identT2","viol_inj","cnt_generic","cnt_ident","cnt_supp","cnt_sAdb","cnt_cross","cnt_inj","cnt_p2","dpr_gt","dpr_generic","dpr_ident","dpr_supp","dpr_sAdb","dpr_inj","dp_gt_ds","ds_gt_dp","t2_gt_lp","t2_lt_lp"]
stats={k:0 for k in keys}
FAMKEY={"generic":"generic","identical":"ident","support":"supp","sameAdiffb":"sAdb","injective":"inj"}
types=["generic","identical","support","sameAdiffb","injective"]
for inst in range(N_INST):
    typ=types[inst%5]
    n=int(rng.integers(3,6)) if typ!="injective" else 2
    p=int(rng.integers(3,5)) if typ in("generic","identical") else 3
    if typ=="support": n=5
    As=[];bs=[]
    if typ=="identical":
        Ac=rng.integers(0,10,size=(int(rng.integers(1,3)),n)).astype(float)
        bc=np.array([max(1,int(rng.integers(1,max(2,int(Ac[r].sum()))))) for r in range(Ac.shape[0])],float)
        As=[Ac.copy() for _ in range(p-1)]; bs=[bc.copy() for _ in range(p-1)]
    elif typ=="support":
        for Jk in [[0,1,2],[2,3,4]]:
            A=np.zeros((1,n)); A[0,Jk]=rng.integers(1,10,size=len(Jk))
            As.append(A); bs.append(np.array([float(rng.integers(1,int(A.sum())))]))
    elif typ=="sameAdiffb":
        Ac=rng.integers(0,8,size=(1,n)).astype(float)
        if Ac.sum()<3: Ac[0,0]+=3
        As=[Ac.copy(),Ac.copy()]; bs=[np.array([float(rng.integers(1,int(Ac.sum())))]) for _ in range(2)]
    elif typ=="injective":
        for k in range(2):
            while True:
                A=rng.integers(0,6,size=(2,n)).astype(float)
                if abs(np.linalg.det(A))>1e-9: break
            As.append(A); bs.append(np.array([max(1,int(rng.integers(1,max(2,int(A[r].sum()))))) for r in range(2)],float))
    else:
        for k in range(p-1):
            mk=int(rng.integers(1,3))
            A=rng.integers(0,10,size=(mk,n)).astype(float)
            As.append(A); bs.append(np.array([max(0,int(rng.integers(0,max(1,int(A[r].sum()))))) for r in range(mk)],float))
    Akp=rng.integers(0,10,size=(1,n)).astype(float); bkp=np.array([max(1,int(rng.integers(1,max(2,int(Akp.sum())))))],float)
    c=rng.integers(1,10,size=n).astype(float)
    Xk=pts(Akp,bkp,n); Ys=[pts(As[k],bs[k],n) for k in range(len(As))]
    if len(Xk)==0 or any(len(Y)==0 for Y in Ys): continue
    q=len(Ys)
    Dfull=solveD(c,Xk,Ys,set(range(n)),n)
    S1=set(int(j) for j in rng.choice(n,size=int(rng.integers(0,n)),replace=False)); S2=S1|{int(rng.integers(0,n))}
    Dp0=solveD(c,Xk,Ys,set(),n); DpS1=solveD(c,Xk,Ys,S1,n); DpS2=solveD(c,Xk,Ys,S2,n)
    DS=solveDS(c,Xk,Ys,As,bs,n)
    R=solveR(c,Xk,np.vstack(As),np.concatenate(bs))
    T2=solveT2(c,Xk,As,bs,n)
    feas=[x for x in it.product([0,1],repeat=n) if np.all(Akp@np.array(x,float)<=bkp) and all(np.all(As[k]@np.array(x,float)<=bs[k]) for k in range(q))]
    VP=max((c@np.array(x,float) for x in feas),default=None)
    if None in (Dfull,Dp0,DpS1,DpS2,DS,R,T2): continue
    stats["configs"]+=1
    if VP is not None and VP>Dfull+E: stats["viol_P_D"]+=1
    if Dfull>Dp0+E or Dfull>DpS1+E or Dfull>DpS2+E: stats["viol_D_Dp"]+=1
    if DpS2>DpS1+E: stats["viol_mono"]+=1
    if Dfull>DS+E: stats["viol_D_DS"]+=1
    if DS>R+E: stats["viol_DS_R"]+=1
    if R>T2+E: stats["viol_R_T2"]+=1
    if Dp0>T2+E: stats["viol_Dp_T2"]+=1
    if Dp0>R+E: stats["dpr_gt"]+=1; stats["dpr_"+FAMKEY[typ]]+=1
    if typ in ("generic","sameAdiffb"): stats["cnt_"+FAMKEY[typ]]+=1
    if Dp0>DS+E: stats["dp_gt_ds"]+=1
    if DS>Dp0+E: stats["ds_gt_dp"]+=1
    # LP relaxation over all blocks
    Nlp=n; rowsL=[];lbL=[];ubL=[]
    Ablk=[Akp]+As; bblk=[bkp]+bs
    for Ak_,bk_ in zip(Ablk,bblk):
        for r_ in range(Ak_.shape[0]):
            rowsL.append(Ak_[r_]);lbL.append(-1e30);ubL.append(bk_[r_])
    LPv=hlp_max(c,rowsL,lbL,ubL,[1.0]*n)
    if LPv is not None:
        if T2>LPv+E: stats["t2_gt_lp"]+=1
        if T2<LPv-E: stats["t2_lt_lp"]+=1
    if typ=="identical":
        stats["cnt_ident"]+=1
        if abs(Dp0-Dfull)>E or abs(DpS1-Dfull)>E: stats["viol_ident"]+=1
        if abs(T2-R)>E: stats["viol_identT2"]+=1
    if typ=="support":
        stats["cnt_supp"]+=1
        DpSup=solveD(c,Xk,Ys,set([0,1,2])|set([2,3,4]),n)
        if DpSup is None or abs(DpSup-Dfull)>E: stats["viol_supp"]+=1
    if typ=="injective":
        stats["cnt_inj"]+=1
        if abs(DS-Dfull)>E: stats["viol_inj"]+=1
    if all(all(np.all(As[k]@y<=bs[k]+1e-9) for y in Ys[kk]) for k in range(q) for kk in range(q)):
        stats["cnt_cross"]+=1
        if Dp0>R+E: stats["viol_cross"]+=1
    Dp2=solveD(c,Xk,Ys[:1],set(),n); Df2=solveD(c,Xk,Ys[:1],set(range(n)),n)
    if Dp2 is not None and Df2 is not None:
        stats["cnt_p2"]+=1
        if abs(Dp2-Df2)>E: stats["viol_p2"]+=1
with open(OUT,"a") as f: f.write(json.dumps({"seed":SEED,"inst":N_INST,**stats})+"\n")
print(json.dumps(stats))
