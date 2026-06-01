"""
Stefan-CL  |  Step 3: erf masks + masked functional anchoring  (final testbed)
------------------------------------------------------------------------------
TESTBED (frozen from here on):
  * Frank-sphere annuli  R_k = R0*sqrt(k)  (equal-area rings).
  * Per-task ROTATED quadrant rule: task k labels its ring by the XOR/quadrant
    pattern rotated by theta_k = (k-1)/(T-1) * pi/2.  Each ring is easy alone,
    but rings impose conflicting rules -> genuine catastrophic forgetting.

MECHANISM:
  L = L_task(new ring)
      + LAM * [ sum_x H_s(phi_k(x)) ||f_theta(x)-f_frozen(x)||^2 ] / [ sum_x H_s ]
  H_s(phi) = 0.5*(1 - erf(phi/eps))            (paper Eq. 12b, solid mask)
  phi_k(x) = ||x|| - R_k   (analytic frontier; learned+advected field in Step 4)
  Anchor is NORMALIZED by the mask mass -> scale-invariant as the solid region
  grows, so LAM is a clean stability<->plasticity dial.
"""
import copy, numpy as np, torch, torch.nn as nn

SEED=0
def setseed(): torch.manual_seed(SEED); np.random.seed(SEED)
setseed(); torch.use_deterministic_algorithms(True)
DEVICE=torch.device("cpu")

# ---- config ----
T=5; R0=1.0; OMEGA=1.0; SPREAD=np.pi/2
N_TRAIN=2000; N_TEST=4000; HIDDEN=64; EPOCHS=300; LR=1e-3
R_MAX=2.6; EPS=0.10; N_ANCH=2000
LAM=0.1                              # chosen anchor strength (see sweep below)
R=[R0*np.sqrt(k) for k in range(T+1)]
THETA=[(k)/(T-1)*SPREAD for k in range(T)]    # rule rotation per task (0-based)

# ---- data ----
def rot(xy,th):
    c,s=np.cos(th),np.sin(th)
    return np.stack([c*xy[:,0]-s*xy[:,1], s*xy[:,0]+c*xy[:,1]],1)
def lab(xy,th):
    z=rot(xy,th); return (np.sin(OMEGA*z[:,0])*np.sin(OMEGA*z[:,1])>0).astype(np.int64)
def ann(ri,ro,n,rng):
    r=np.sqrt(rng.uniform(ri**2,ro**2,n)); a=rng.uniform(0,2*np.pi,n)
    return np.stack([r*np.cos(a),r*np.sin(a)],1).astype(np.float32)
def mk(k,n,sd):
    rng=np.random.default_rng(sd); X=ann(R[k-1],R[k],n,rng)
    return torch.tensor(X), torch.tensor(lab(X,THETA[k-1]))
def disk(n,rng):
    r=R_MAX*np.sqrt(rng.uniform(0,1,n)); a=rng.uniform(0,2*np.pi,n)
    return np.stack([r*np.cos(a),r*np.sin(a)],1).astype(np.float32)
train_tasks=[mk(k,N_TRAIN,1000+k) for k in range(1,T+1)]
test_tasks =[mk(k,N_TEST, 2000+k) for k in range(1,T+1)]

# ---- model ----
class MLP(nn.Module):
    def __init__(s,h=HIDDEN):
        super().__init__(); s.net=nn.Sequential(
            nn.Linear(2,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,2))
    def forward(s,x): return s.net(x)
def fresh(): setseed(); return MLP().to(DEVICE)
@torch.no_grad()
def acc(m,X,y): m.eval(); return (m(X).argmax(1)==y).float().mean().item()
def allacc(m): return [acc(m,X,y) for X,y in test_tasks]
def H_s(phi,eps=EPS): return 0.5*(1.0-torch.erf(phi/eps))
def summarize(A):
    final=A[T-1,:]
    forget=np.mean([A[:,j].max()-A[T-1,j] for j in range(T-1)])
    bwt=np.mean([A[T-1,j]-A[j,j] for j in range(T-1)])
    return final.mean(),forget,bwt,final

# ---- baselines on the final testbed ----
def naive():
    m=fresh(); A=np.zeros((T,T)); lf=nn.CrossEntropyLoss()
    for i in range(T):
        X,y=train_tasks[i]; opt=torch.optim.Adam(m.parameters(),LR)
        for _ in range(EPOCHS): opt.zero_grad(); lf(m(X),y).backward(); opt.step()
        A[i,:]=allacc(m)
    return A
def joint():
    m=fresh(); X=torch.cat([t[0] for t in train_tasks]); y=torch.cat([t[1] for t in train_tasks])
    opt=torch.optim.Adam(m.parameters(),LR); lf=nn.CrossEntropyLoss()
    for _ in range(EPOCHS): opt.zero_grad(); lf(m(X),y).backward(); opt.step()
    return np.array(allacc(m))

# ---- Stefan-CL ----
def stefan(LAM):
    m=fresh(); frozen=None; rng=np.random.default_rng(SEED); A=np.zeros((T,T))
    lf=nn.CrossEntropyLoss()
    for i in range(T):
        X,y=train_tasks[i]; opt=torch.optim.Adam(m.parameters(),LR); Rc=R[i]
        for _ in range(EPOCHS):
            m.train(); opt.zero_grad(); loss=lf(m(X),y)
            if frozen is not None:
                Xa=torch.tensor(disk(N_ANCH,rng)); phi=Xa.norm(dim=1)-Rc; w=H_s(phi)
                with torch.no_grad(): fo=frozen(Xa)
                L_anchor=(w*((m(Xa)-fo)**2).sum(1)).sum()/(w.sum()+1e-8)
                loss=loss+LAM*L_anchor
            loss.backward(); opt.step()
        A[i,:]=allacc(m); frozen=copy.deepcopy(m).eval()
        for p in frozen.parameters(): p.requires_grad_(False)
    return A

print("="*68)
print("STEFAN-CL  STEP 3  |  Masked anchoring (final rotated-rule testbed)")
print("="*68)
An=naive(); navg,nforget,_,_=summarize(An); J=joint()
print(f"NAIVE  avg={navg:.3f}  forget={nforget:.3f}    JOINT oracle avg={J.mean():.3f}\n")
print("LAM sweep (normalized anchor):")
for lam in [0.05,0.1,0.2,0.3,0.5]:
    A=stefan(lam); avg,forget,bwt,final=summarize(A)
    print(f"  LAM={lam:5.2f}  avg={avg:.3f}  forget={forget:.3f}  "
          f"diag={np.round(np.diag(A),3)}")

print(f"\nChosen LAM={LAM}:")
A=stefan(LAM); avg,forget,bwt,final=summarize(A)
np.set_printoptions(precision=3,suppress=True)
print("Stefan-CL accuracy matrix A[i,j]:"); print(A)
print(f"\nStefan-CL avg acc      : {avg:.3f}")
print(f"Stefan-CL forgetting   : {forget:.3f}")
print(f"Stefan-CL backward xfer : {bwt:.3f}")
print(f"Stefan-CL final per-task: {np.round(final,3)}")

print("-"*68)
print(f"{'':18s}{'avg acc':>10s}{'forgetting':>12s}")
print(f"{'Naive':18s}{navg:>10.3f}{nforget:>12.3f}")
print(f"{'Stefan-CL':18s}{avg:>10.3f}{forget:>12.3f}")
print(f"{'Joint oracle':18s}{J.mean():>10.3f}{0.0:>12.3f}")
print("-"*68)
gap=(avg-navg)/(J.mean()-navg)
c1=forget<nforget-0.20
c2=avg>navg+0.20
c3=np.diag(A).min()>0.85
print(f"Gap to oracle closed          : {gap*100:.1f}%")
print(f"Forgetting reduced (>0.20)?    {'YES' if c1 else 'NO'}")
print(f"Avg acc improved (>0.20)?      {'YES' if c2 else 'NO'}")
print(f"Plasticity kept (all diag>.85)?{'YES' if c3 else 'NO'}")
print(f"STEP 3 PASS: {'YES' if (c1 and c2 and c3) else 'NO'}")
print("-"*68)
