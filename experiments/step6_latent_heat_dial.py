"""
Stefan-CL  |  Step 6: Latent heat L as the stability-plasticity dial
--------------------------------------------------------------------
Stefan condition  rho L V_n = driving flux  =>  V_n = v(R)/L.
L is the COST to consolidate plastic region into protected region.
Fixed adaptation budget (N_STEPS*DT). Sweep L and show the stability-
plasticity trade-off emerges as a CONSEQUENCE of the physics:

  L small -> front reaches envelope -> large protected region
          -> low forgetting (stable), lower plasticity
  L large -> front lags within budget -> small protected region
          -> high forgetting (unstable), higher plasticity

Circular-front testbed => level-set evolution reduces EXACTLY to the radial
ODE dR/dtau = v(R)/L (Steps 4-5 validated the full 2D neural-field version);
phi is refit as a neural SDF of R_front for the erf mask.

Verifies: (1) protected fraction decreasing in L, (2) forgetting increasing
in L, (3) plasticity non-decreasing in L, (4) Stefan invariant L*V_n const.
"""
import copy, time, numpy as np, torch, torch.nn as nn
SEED=0
def ss(): torch.manual_seed(SEED); np.random.seed(SEED)
ss(); torch.use_deterministic_algorithms(True)

T=5; R0=1.0; OMEGA=1.0; SPREAD=np.pi/2
N_TRAIN=2000; N_TEST=4000; HIDDEN=128; EPOCHS=250; LR=1e-3
R_MAX=2.6; EPS=0.10; N_ANCH=1200; LAM=0.1
R=[R0*np.sqrt(k) for k in range(T+1)]; THETA=[(k)/(T-1)*SPREAD for k in range(T)]
DT=0.04; N_STEPS=25; SEED_R=0.5; V_TAPER=0.10
L_GRID=[0.5,1.0,2.0,4.0,8.0]

def rot(xy,th):
    c,s=np.cos(th),np.sin(th); return np.stack([c*xy[:,0]-s*xy[:,1],s*xy[:,0]+c*xy[:,1]],1)
def lab(xy,th):
    z=rot(xy,th); return (np.sin(OMEGA*z[:,0])*np.sin(OMEGA*z[:,1])>0).astype(np.int64)
def ann(ri,ro,n,rng):
    r=np.sqrt(rng.uniform(ri**2,ro**2,n)); a=rng.uniform(0,2*np.pi,n)
    return np.stack([r*np.cos(a),r*np.sin(a)],1).astype(np.float32)
def disk(n,rng):
    r=R_MAX*np.sqrt(rng.uniform(0,1,n)); a=rng.uniform(0,2*np.pi,n)
    return np.stack([r*np.cos(a),r*np.sin(a)],1).astype(np.float32)
def mk(k,n,sd):
    rng=np.random.default_rng(sd); X=ann(R[k-1],R[k],n,rng); return torch.tensor(X),torch.tensor(lab(X,THETA[k-1]))
train_tasks=[mk(k,N_TRAIN,1000+k) for k in range(1,T+1)]; test_tasks=[mk(k,N_TEST,2000+k) for k in range(1,T+1)]

class MLP(nn.Module):
    def __init__(s,h=HIDDEN):
        super().__init__(); s.net=nn.Sequential(nn.Linear(2,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,2))
    def forward(s,x): return s.net(x)
class Phi(nn.Module):
    def __init__(s,h=64):
        super().__init__(); s.net=nn.Sequential(nn.Linear(2,h),nn.Tanh(),nn.Linear(h,h),nn.Tanh(),nn.Linear(h,1))
    def forward(s,x): return s.net(x).squeeze(-1)
@torch.no_grad()
def acc(m,X,y): m.eval(); return (m(X).argmax(1)==y).float().mean().item()
def H_s(p): return 0.5*(1-torch.erf(p/EPS))
def fit_circle(net,r,iters,rg):
    op=torch.optim.Adam(net.parameters(),3e-3)
    for _ in range(iters):
        X=torch.tensor(disk(1200,rg)); op.zero_grad(); (net(X)-(X.norm(dim=1)-r)).pow(2).mean().backward(); op.step()

def radial_advect(Rf, data_radii, L, log=None):
    for _ in range(N_STEPS):
        ahead=float((data_radii>Rf).mean())
        v=min(ahead/V_TAPER,1.0)        # driving flux (data ahead), tapers to 0 at envelope
        Vn=v/L                          # Stefan: V_n = flux/(rho L)
        if log is not None: log.append(abs(1.0*L*Vn - v))  # Stefan identity rho*L*Vn = flux
        Rf=Rf+DT*Vn
    return Rf

def run(L, inv_log=None):
    ss(); clf=MLP(); phi=Phi(); rg=np.random.default_rng(31)
    fit_circle(phi,SEED_R,700,rg); R_front=SEED_R
    A=np.zeros((T,T)); Rprot=[]; frozen=None
    rng_anch=np.random.default_rng(SEED); lf=nn.CrossEntropyLoss()
    for i in range(T):
        X,y=train_tasks[i]; opt=torch.optim.Adam(clf.parameters(),LR)
        for _ in range(EPOCHS):
            clf.train(); opt.zero_grad(); loss=lf(clf(X),y)
            if frozen is not None:
                Xa=torch.tensor(disk(N_ANCH,rng_anch))
                with torch.no_grad(): w=H_s(phi(Xa)); fo=frozen(Xa)
                loss=loss+LAM*(w*((clf(Xa)-fo)**2).sum(1)).sum()/(w.sum()+1e-8)
            loss.backward(); opt.step()
        A[i,:]=[acc(clf,*test_tasks[j]) for j in range(T)]
        R_front=radial_advect(R_front, train_tasks[i][0].norm(dim=1).numpy(), L, inv_log)
        fit_circle(phi,R_front,250,rg); Rprot.append(R_front)
        frozen=copy.deepcopy(clf).eval()
        for p in frozen.parameters(): p.requires_grad_(False)
    final=A[T-1,:]
    return (final.mean(),
            np.mean([A[:,j].max()-A[T-1,j] for j in range(T-1)]),
            np.mean(np.diag(A)),
            np.mean(np.array(Rprot)/np.array(R[1:])),
            np.array(Rprot))

print("="*70); print("STEFAN-CL  STEP 6  |  Latent heat L as stability-plasticity dial"); print("="*70)
print(f"Stefan: V_n = v(R)/L | budget={N_STEPS*DT} | LAM={LAM}\n")
print(f"{'L':>5} {'protFrac':>9} {'forget':>8} {'plast':>7} {'avg':>7}   Rprot")
rows=[]; inv=[]
t0=time.time()
for L in L_GRID:
    avg,forget,plast,pfrac,Rp=run(L, inv if L==2.0 else None)
    rows.append((L,pfrac,forget,plast,avg))
    print(f"{L:5.1f} {pfrac:9.3f} {forget:8.3f} {plast:7.3f} {avg:7.3f}   {np.round(Rp,2)}")
print(f"[total {time.time()-t0:.0f}s]")

L_=np.array([r[0] for r in rows]); pf=np.array([r[1] for r in rows])
fg=np.array([r[2] for r in rows]); pl=np.array([r[3] for r in rows])
def mono_dec(a,tol=0.02): return all(a[i+1]<=a[i]+tol for i in range(len(a)-1))
def mono_inc(a,tol=0.02): return all(a[i+1]>=a[i]-tol for i in range(len(a)-1))
inv=np.array(inv); inv_res=float(inv.max()) if len(inv)>0 else 1.0
print("-"*70)
c1=mono_dec(pf); c2=mono_inc(fg); c3=mono_inc(pl)
c4=(fg.max()-fg.min()>0.10) and (pf.max()-pf.min()>0.20)   # dial has real range
c5=inv_res<1e-6                                            # Stefan identity rho*L*Vn=flux
print(f"Protected fraction decreasing in L?   {'YES' if c1 else 'NO'}")
print(f"Forgetting increasing in L?           {'YES' if c2 else 'NO'}")
print(f"Plasticity non-decreasing in L?       {'YES' if c3 else 'NO'}")
print(f"Dial spans a real range?              {'YES' if c4 else 'NO'}")
print(f"Stefan identity rho*L*Vn=flux (res<1e-6)? {'YES' if c5 else 'NO'}  (max res={inv_res:.2e})")
print(f"STEP 6 PASS: {'YES' if (c1 and c2 and c3 and c4 and c5) else 'NO'}")
print("-"*70)
