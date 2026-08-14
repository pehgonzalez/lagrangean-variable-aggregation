"""Independent verification of the counterexample certificates used in the paper.

Witnesses may be discovered by linear programming, but every final check is
carried out in pure exact rational arithmetic (Python fractions): primal
feasibility, coupling equations as exact identities, and exact evaluation of
the relevant dual functions at the certifying multipliers. No check relies
on solver output. Deterministic; prints PASS/FAIL per certificate.
"""
import itertools as it, numpy as np
from fractions import Fraction as F
from scipy.optimize import linprog
PASS=[]
def frac(v,md=10**7): return F(v).limit_denominator(md)
def fv(vec,md=10**7): return [frac(x,md) for x in vec]

def exact_dual_D0(beta,cf,Xk,Yints):  # upper bound for D'(empty): max(c-q*beta)x + sum max beta*y
    q=len(Yints); n=len(cf)
    t0=max(sum((cf[j]-q*beta[j])*x[j] for j in range(n)) for x in Xk)
    return t0+sum(max(sum(beta[j]*y[j] for j in range(n)) for y in Y) for Y in Yints)
def exact_dual_R(nu,cf,Xk,Arf,brf):
    n=len(cf)
    return max(sum(cf[j]*x[j] for j in range(n))+sum(nu[i]*(brf[i]-sum(Arf[i][j]*x[j] for j in range(n))) for i in range(len(brf))) for x in Xk)
def exact_dual_DS(nus,cf,Xk,Yints,Arows):  # nus: list of vectors per copied block; value = max(c - sum nu_k A_k)x + sum max (nu_k A_k) y
    n=len(cf)
    red=[cf[j]-sum(sum(nus[k][r]*Arows[k][r][j] for r in range(len(nus[k]))) for k in range(len(nus))) for j in range(n)]
    t0=max(sum(red[j]*x[j] for j in range(n)) for x in Xk)
    s=0
    for k,Y in enumerate(Yints):
        s+=max(sum(sum(nus[k][r]*Arows[k][r][j] for r in range(len(nus[k])))*y[j] for j in range(n)) for y in Y)
    return t0+s

# ---------- CE1: minimal D'>R (n=1,p=3) ----------
# blocks: y<=0 ; y<=1 ; keep x<=1 ; c=1
# D' lower: witness x=1/2 in conv{0,1}, y1=0, y2=1, coupling 0+1=2*(1/2). value 1/2.
lower=F(1,2); okw=(F(0)+F(1)==2*F(1,2))
# R upper: nu=(1,0): max_x [x + 1*(0-x) + 0]=0
upper=max(F(x)+F(1)*(F(0)-F(x)) for x in (0,1))
# D' upper: beta=1/2 exact dual evaluation, so V(D')=1/2 on both sides
upDp1=exact_dual_D0([F(1,2)],[F(1)],[(0,),(1,)],[[(0,)],[(0,),(1,)]])
PASS.append(("CE1 D'(=1/2)>R(=0) minimal", okw and lower>upper and upper==0 and upDp1==F(1,2)))

# ---------- CE2: robust D'>R (n=5) ----------
A1=[[4,4,2,0,5],[8,0,8,8,2]];b1=[9,4];A2=[[7,3,0,9,4],[8,6,7,7,1]];b2=[8,13];A3=[[0,5,1,7,6]];b3=[17];c=[7,4,9,4,3];n=5
Xk=[x for x in it.product([0,1],repeat=n) if all(sum(A2[r][j]*x[j] for j in range(n))<=b2[r] for r in range(2))]
Y1=[y for y in it.product([0,1],repeat=n) if all(sum(A1[r][j]*y[j] for j in range(n))<=b1[r] for r in range(2))]
Y3=[y for y in it.product([0,1],repeat=n) if sum(A3[0][j]*y[j] for j in range(n))<=b3[0]]
cf=[F(v) for v in c]
# D' lower: discover witness by LP then verify exactly
P0=np.array(Xk,float);P1=np.array(Y1,float);P3=np.array(Y3,float)
NX,N1,N3=len(Xk),len(Y1),len(Y3);N=NX+N1+N3
obj=np.concatenate([-(np.array(c,float)@P0.T),np.zeros(N1+N3)])
Aeq=[];beq=[]
for d in range(n):
    row=np.zeros(N);row[:NX]=2*P0[:,d];row[NX:NX+N1]=-P1[:,d];row[NX+N1:]=-P3[:,d];Aeq.append(row);beq.append(0)
for off,ln in [(0,NX),(NX,N1),(NX+N1,N3)]:
    row=np.zeros(N);row[off:off+ln]=1;Aeq.append(row);beq.append(1)
r=linprog(obj,A_eq=np.array(Aeq),b_eq=beq,bounds=[(0,None)]*N,method='highs')
wx=fv(r.x[:NX]);w1=fv(r.x[NX:NX+N1]);w3=fv(r.x[NX+N1:])
for w in (wx,w1,w3):
    s=sum(w)
    for i in range(len(w)): w[i]/=s
xh=[sum(wx[i]*F(Xk[i][j]) for i in range(NX)) for j in range(n)]
y1h=[sum(w1[i]*F(Y1[i][j]) for i in range(N1)) for j in range(n)]
y3h=[sum(w3[i]*F(Y3[i][j]) for i in range(N3)) for j in range(n)]
ok=all(y1h[j]+y3h[j]==2*xh[j] for j in range(n)) and all(w>=0 for w in wx+w1+w3)
lowD=sum(cf[j]*xh[j] for j in range(n))
# R upper: nu*=(0,1,0) on rows (A1r1,A1r2,A3): exact eval
Arf=[[F(v) for v in row] for row in A1+A3]; brf=[F(v) for v in b1+b3]
upR=exact_dual_R([F(0),F(1),F(0)],cf,Xk,Arf,brf)
# D' upper: beta*=(0,0,3,0,0) exact dual evaluation, so V(D')=10 on both sides
upDp2=exact_dual_D0([F(0),F(0),F(3),F(0),F(0)],cf,Xk,[Y1,Y3])
PASS.append(("CE2 D'(=10)>R(=9) robust n=5", ok and lowD==10 and upR==9 and upDp2==10))

# ---------- CE3: DS>D' (n=2) ----------
# keep 4x1+x2<=2 ; copied x1+2x2<=1 and 2x1+4x2<=2 ; c=(3,4)
Xk=[(0,0),(0,1)];Y2=[(0,0),(1,0)];Y3=[(0,0),(1,0)];cf=[F(3),F(4)];n=2
lowDS=F(0)+F(4)*F(1,2)  # witness x=(0,1/2), y=(1,0) each: A2y=1=A2x, A3y=2=A3x
okDS=(F(1)+2*F(0)==F(0)+2*F(1,2)) and (2*F(1)+4*F(0)==2*F(0)+4*F(1,2))
upDp=exact_dual_D0([F(0),F(2)],cf,Xk,[Y2,Y3])
PASS.append(("CE3 DS(2)>D'(0)", okDS and lowDS==2 and upDp==0))

# ---------- CE4: same-A-diff-b D'>R (n=2) ----------
# copied 7y1+7y2<=8 and <=3 ; keep x1+x2<=1 ; c=(4,5)
Xk=[(0,0),(1,0),(0,1)];Y1=[(0,0),(1,0),(0,1)];Y2=[(0,0)];cf=[F(4),F(5)];n=2
# D' lower: x=(0,1/2), y1=(0,1), y2=(0,0): coupling (0,1)+(0,0)=2*(0,1/2) ok
lowDp=F(5)*F(1,2)
okc=(F(0)+F(0)==2*F(0)) and (F(1)+F(0)==2*F(1,2))
Arf=[[F(7),F(7)],[F(7),F(7)]];brf=[F(8),F(3)]
upR=exact_dual_R([F(0),F(5,7)],cf,Xk,Arf,brf)
# D' upper: beta=(2,5/2) exact dual evaluation, so V(D')=5/2 on both sides
upDp4=exact_dual_D0([F(2),F(5,2)],cf,Xk,[Y1,Y2])
PASS.append(("CE4 same-A D'(=5/2)>R(=15/7)", okc and lowDp==F(5,2) and upR==F(15,7) and lowDp>upR and upDp4==F(5,2)))

# ---------- CE5/CE6: continuous variant vs LP (example instance) ----------
A=[[49,40,31],[30,25,34],[12,19,30]];b=[76,57,46];c=[2,3,4];n=3
Af=[[F(v) for v in row] for row in A];bf=[F(v) for v in b];cf=[F(v) for v in c]
# LP value exact both sides: primal x=(7/82,1,71/82)? verify our known dual (0,1/41,13/123)+box(0,47/123,0) and find primal by LP
r=linprog(-np.array(c,float),A_ub=np.array(A,float),b_ub=np.array(b,float),bounds=[(0,1)]*3,method='highs')
xf=fv(r.x)
okp=all(sum(Af[i][j]*xf[j] for j in range(3))<=bf[i] for i in range(3)) and all(0<=v<=1 for v in xf)
lp_low=sum(cf[j]*xf[j] for j in range(3))
lam=[F(0),F(1,41),F(13,123)];ubox=[F(0),F(47,123),F(0)]
okd=all(sum(Af[i][j]*lam[i] for i in range(3))+ubox[j]>=cf[j] for j in range(3))
lp_up=sum(bf[i]*lam[i] for i in range(3))+sum(ubox)
okLP=okp and okd and lp_low==lp_up==F(272,41)
# CE5: T2 keep row1 lower witness 837/125
w=[(F(1,25),(1,0,0)),(F(467,500),(0,1,1)),(F(13,500),(0,1,0))]
xb=[sum(wi*F(pt[j]) for wi,pt in w) for j in range(3)]
y2=[F(0),F(23,25),F(1)];y3=[F(2,25),F(1),F(217,250)]
ok5=(sum(wi for wi,_ in w)==1 and all(sum(Af[0][j]*F(pt[j]) for j in range(3))<=bf[0] for _,pt in w)
     and sum(Af[1][j]*y2[j] for j in range(3))<=bf[1] and sum(Af[2][j]*y3[j] for j in range(3))<=bf[2]
     and all(0<=v<=1 for v in y2+y3) and all(y2[j]+y3[j]==2*xb[j] for j in range(3)))
t2low=sum(cf[j]*xb[j] for j in range(3))
PASS.append(("CE5 T2(837/125)>LP(272/41)", okLP and ok5 and t2low==F(837,125) and t2low>F(272,41)))
# CE6: T2 keep row2 upper via beta=(49/178,20/89,0) exact eval over X2 and K-vertex sets
X2=[x for x in it.product([0,1],repeat=3) if sum(A[1][j]*x[j] for j in range(3))<=b[1]]
beta=[F(49,178),F(20,89),F(0)]
def exact_max_K(beta,ai,bi):
    aif=[F(v) for v in ai];bif=F(bi);best=F(0);Vs=[]
    for pt in it.product([0,1],repeat=3): Vs.append([F(v) for v in pt])
    for pt in it.product([0,1],repeat=3):
        for d in range(3):
            if aif[d]==0: continue
            q=[F(v) for v in pt];t=(bif-sum(aif[k]*q[k] for k in range(3))+aif[d]*q[d])/aif[d]
            if 0<=t<=1:
                q2=q[:];q2[d]=t;Vs.append(q2)
    for v in Vs:
        if all(0<=vv<=1 for vv in v) and sum(aif[k]*v[k] for k in range(3))<=bif:
            best=max(best,sum(beta[k]*v[k] for k in range(3)))
    return best
t0=max(sum((cf[j]-2*beta[j])*x[j] for j in range(3)) for x in X2)
t2up=t0+exact_max_K(beta,A[0],b[0])+exact_max_K(beta,A[2],b[2])
PASS.append(("CE6 T2keep2(<=877/178)<LP(272/41)", t2up==F(877,178) and t2up<F(272,41)))
# CE5b: upper certificate making the keep-row1 value two-sided, beta=(4/5,13/10,2)
beta5=[F(4,5),F(13,10),F(2)]
X1=[x for x in it.product([0,1],repeat=3) if sum(A[0][j]*x[j] for j in range(3))<=b[0]]
t2up1=max(sum((cf[j]-2*beta5[j])*x[j] for j in range(3)) for x in X1) \
      +exact_max_K(beta5,A[1],b[1])+exact_max_K(beta5,A[2],b[2])
PASS.append(("CE5b T2keep1 upper = 837/125 at beta=(4/5,13/10,2)", t2up1==F(837,125)))
# CE6b: primal witness attaining 877/178 for the keep-row2 value
xw=[F(165,178),F(165,178),F(13,178)]
y1w=[F(76,89),F(76,89),F(0)];y3w=[F(1),F(1),F(13,89)]
ok6b=(all(y1w[j]+y3w[j]==2*xw[j] for j in range(3))
      and sum(Af[0][j]*y1w[j] for j in range(3))<=bf[0]
      and sum(Af[2][j]*y3w[j] for j in range(3))<=bf[2]
      and all(0<=v<=1 for v in y1w+y3w)
      and F(165,178)+F(13,178)==1
      and sum(A[1][j]*[1,1,0][j] for j in range(3))<=b[1]
      and sum(A[1][j]*[0,0,1][j] for j in range(3))<=b[1]
      and sum(cf[j]*xw[j] for j in range(3))==F(877,178))
PASS.append(("CE6b T2keep2 witness attains 877/178", ok6b))

# ---------- CE7: D'>DS on the example instance, block 2 retained ----------
Xk=[x for x in it.product([0,1],repeat=3) if sum(A[1][j]*x[j] for j in range(3))<=b[1]]
Y1=[y for y in it.product([0,1],repeat=3) if sum(A[0][j]*y[j] for j in range(3))<=b[0]]
Y3=[y for y in it.product([0,1],repeat=3) if sum(A[2][j]*y[j] for j in range(3))<=b[2]]
# D' lower: discover witness via LP, verify exact
P0=np.array(Xk,float);P1=np.array(Y1,float);P3=np.array(Y3,float)
NX,N1,N3=len(Xk),len(Y1),len(Y3);N=NX+N1+N3
obj=np.concatenate([-(np.array(c,float)@P0.T),np.zeros(N1+N3)])
Aeq=[];beq=[]
for d in range(3):
    row=np.zeros(N);row[:NX]=2*P0[:,d];row[NX:NX+N1]=-P1[:,d];row[NX+N1:]=-P3[:,d];Aeq.append(row);beq.append(0)
for off,ln in [(0,NX),(NX,N1),(NX+N1,N3)]:
    row=np.zeros(N);row[off:off+ln]=1;Aeq.append(row);beq.append(1)
r=linprog(obj,A_eq=np.array(Aeq),b_eq=beq,bounds=[(0,None)]*N,method='highs')
wx=fv(r.x[:NX]);w1=fv(r.x[NX:NX+N1]);w3=fv(r.x[NX+N1:])
for w in (wx,w1,w3):
    s=sum(w)
    for i in range(len(w)): w[i]/=s
xh=[sum(wx[i]*F(Xk[i][j]) for i in range(NX)) for j in range(3)]
y1h=[sum(w1[i]*F(Y1[i][j]) for i in range(N1)) for j in range(3)]
y3h=[sum(w3[i]*F(Y3[i][j]) for i in range(N3)) for j in range(3)]
okc=all(y1h[j]+y3h[j]==2*xh[j] for j in range(3))
lowDp=sum(cf[j]*xh[j] for j in range(3))
# DS upper: discover nu=(nu1,nu3) by the epigraph LP min t0+t1+t3 s.t.
#   t0 >= (c - nu1 A1 - nu3 A3) x  for every x in Xk   (retained generators)
#   t1 >= (nu1 A1) y  for y in Y1;  t3 >= (nu3 A3) y  for y in Y3
# then re-evaluate the DS dual at the rounded nu in exact arithmetic.
Nv=2+3
Aub=[];bub=[]
for x in Xk:
    a1x=sum(A[0][j]*x[j] for j in range(3)); a3x=sum(A[2][j]*x[j] for j in range(3)); cx=sum(c[j]*x[j] for j in range(3))
    Aub.append(np.array([-a1x,-a3x,-1,0,0],float)); bub.append(-cx)
for y in Y1:
    a1y=sum(A[0][j]*y[j] for j in range(3)); Aub.append(np.array([a1y,0,0,-1,0],float)); bub.append(0)
for y in Y3:
    a3y=sum(A[2][j]*y[j] for j in range(3)); Aub.append(np.array([0,a3y,0,0,-1],float)); bub.append(0)
obj=np.array([0,0,1,1,1],float)
r=linprog(obj,A_ub=np.array(Aub),b_ub=bub,bounds=[(None,None)]*5,method='highs')
nu=fv(r.x[:2])
upDS=exact_dual_DS([[nu[0]],[nu[1]]],cf,Xk,[Y1,Y3],[ [Af[0]], [Af[2]] ])
PASS.append((f"CE7 D'({lowDp})>DS(<={upDS}) example instance, block 2 retained",
             okc and lowDp==F(19,4) and upDS==F(136,29) and lowDp>upDS))
print()
allok=True
for name,ok in PASS:
    print(("PASS " if ok else "FAIL ")+name); allok&=ok
print("\nALL CERTIFICATES:", "PASS" if allok else "FAIL")
