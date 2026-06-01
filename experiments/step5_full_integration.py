"""
Stefan-CL  |  Step 5: Advecting frontier integrated into the full CL loop
--------------------------------------------------------------------------
Per task k:  (1) train classifier + masked anchor using the LEARNED phi
             (2) advect frontier R_{k-1}->R_k, discovering R_k from data
             (3) snapshot classifier for the next task's anchor.

Frontier evolution = data-driven level-set advection (Step 4) with periodic
Eikonal reinitialization to a clean circular SDF (standard level-set practice;
exact here because the radially-symmetric testbed keeps fronts circular).

Success = discovered radii track R_k=R0*sqrt(k) (UNTOLD)  AND  forgetting /
accuracy stay at Step-3 quality.
"""
import copy, time, numpy as np, torch, torch.nn as nn
SEED=0
def ss(): torch.manual_seed(SEED); np.random.seed(SEED)
ss(); torch.use_deterministic_algorithms(True)

# testbed (frozen)
T=5; R0=1.0; OMEGA=1.0; SPREAD=np.pi/2
N_TRAIN=2000; N_TEST=4000; HIDDEN=128; EPOCHS=300; LR=1e-3
R_MAX=2.6; EPS=0.10; N_ANCH=1500; LAM=0.1
R=[R0*np.sqrt(k) for k in range(T+1)]; THETA=[(k)/(T-1)*SPREAD for k in range(T)]
# advection knobs
DELTA=0.06; CVEL=1.0; DT=0.04; N_STEPS=35; INNER=20; LAM_EIK=0.5
REINIT_EVERY=8; SEED_R=0.5

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
train_tasks=[mk(k,N_TRAIN,1000+k) for k in range(1,T+1)]
test_tasks =[mk(k,N_TEST, 2000+k) for k in range(1,T+1)]

class MLP(nn.Module):
    def __init__(s,h=HIDDEN):
        super().__init__(); s.net=nn.Sequential(nn.Linear(2,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,2))
    def forward(s,x): return s.net(x)
class Phi(nn.Module):
    def __init__(s,h=64):
        super().__init__(); s.net=nn.Sequential(nn.Linear(2,h),nn.Tanh(),nn.Linear(h,h),nn.Tanh(),
            nn.Linear(h,h),nn.Tanh(),nn.Linear(h,1))
    def forward(s,x): return s.net(x).squeeze(-1)
@torch.no_grad()
def acc(m,X,y): m.eval(); return (m(X).argmax(1)==y).float().mean().item()
def H_s(p): return 0.5*(1-torch.erf(p/EPS))
def grad_phi(net,x):
    x=x.clone().requires_grad_(True); p=net(x); g=torch.autograd.grad(p.sum(),x,create_graph=True)[0]; return p,g
def frad(net):
    th=np.linspace(0,2*np.pi,48,endpoint=False); rs=np.linspace(0,R_MAX,400); rad=[]
    for t in th:
        pts=np.stack([rs*np.cos(t),rs*np.sin(t)],1).astype(np.float32)
        with torch.no_grad(): v=net(torch.tensor(pts)).numpy()
        idx=np.where(np.diff(np.sign(v))>0)[0]
        if len(idx): i=idx[0]; rad.append(rs[i]-v[i]*(rs[i+1]-rs[i])/(v[i+1]-v[i]))
    return (float(np.median(rad)) if rad else np.nan)
def fit_circle(net,r,iters,rg):
    op=torch.optim.Adam(net.parameters(),2e-3)
    for _ in range(iters):
        X=torch.tensor(disk(1500,rg)); op.zero_grad()
        (net(X)-(X.norm(dim=1)-r)).pow(2).mean().backward(); op.step()
def advect(net,data,rng_ev,rg_re):
    for step in range(N_STEPS):
        phi_old=copy.deepcopy(net)
        for p in phi_old.parameters(): p.requires_grad_(False)
        Xev=torch.tensor(disk(800,rng_ev))
        p_old,g_old=grad_phi(phi_old,Xev)
        n=g_old/g_old.norm(dim=1,keepdim=True).clamp_min(1e-6)
        xG=(Xev-p_old.unsqueeze(1)*n).detach()
        with torch.no_grad():
            ahead=data[phi_old(data)>0]
            Vn=torch.zeros(800) if ahead.shape[0]==0 else torch.clamp(((torch.cdist(xG,ahead)<DELTA).float().sum(1)-2.0)/6.0,0,1)*CVEL
        gmag=g_old.norm(dim=1).detach(); target=p_old.detach()-DT*Vn*gmag
        oo=torch.optim.Adam(net.parameters(),1e-3)
        for _ in range(INNER):
            oo.zero_grad(); p,g=grad_phi(net,Xev)
            ((p-target).pow(2).mean()+LAM_EIK*((g.norm(dim=1)-1).pow(2).mean())).backward(); oo.step()
        if (step+1)%REINIT_EVERY==0:
            r=frad(net)
            if not np.isnan(r): fit_circle(net,r,150,rg_re)

ss(); clf=MLP(); phi=Phi()
fit_circle(phi,SEED_R,800,np.random.default_rng(7))
A=np.zeros((T,T)); discovered=[]; frozen=None
rng_ev=np.random.default_rng(SEED); rg_re=np.random.default_rng(31); rng_anch=np.random.default_rng(SEED)
lf=nn.CrossEntropyLoss(); t0=time.time()
print("="*66); print("STEFAN-CL  STEP 5  |  Full loop with advecting (learned) frontier"); print("="*66)
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
    advect(phi,train_tasks[i][0],rng_ev,rg_re)
    discovered.append(frad(phi))
    frozen=copy.deepcopy(clf).eval()
    for p in frozen.parameters(): p.requires_grad_(False)
    print(f" task {i+1}: discovered R={discovered[-1]:.3f}  (true {R[i+1]:.3f})  [t={time.time()-t0:.0f}s]")

np.set_printoptions(precision=3,suppress=True)
final=A[T-1,:]; avg=final.mean()
forget=np.mean([A[:,j].max()-A[T-1,j] for j in range(T-1)])
disc=np.array(discovered); law=np.array(R[1:]); raderr=np.abs(disc-law)
print("\naccuracy matrix:"); print(A)
print(f"\ndiscovered radii : {np.round(disc,3)}")
print(f"Frank-sphere law : {np.round(law,3)}  (R0*sqrt(k), UNTOLD)")
print(f"radius abs error : {np.round(raderr,3)}   max={raderr.max():.3f}")
print(f"\navg acc={avg:.3f}   forgetting={forget:.3f}")
print("-"*66)
print(f"{'':16s}{'avg':>8s}{'forget':>9s}")
print(f"{'Naive':16s}{0.532:>8.3f}{0.579:>9.3f}")
print(f"{'Step3 (analytic)':16s}{0.924:>8.3f}{0.024:>9.3f}")
print(f"{'Step5 (learned)':16s}{avg:>8.3f}{forget:>9.3f}")
print("-"*66)
c1=raderr.max()<0.12
c2=avg>0.85
c3=forget<0.10
print(f"Frontier tracks sqrt-law (maxerr<0.12)? {'YES' if c1 else 'NO'}")
print(f"Accuracy at Step-3 level (avg>0.85)?    {'YES' if c2 else 'NO'}")
print(f"Forgetting controlled (<0.10)?          {'YES' if c3 else 'NO'}")
print(f"STEP 5 PASS: {'YES' if (c1 and c2 and c3) else 'NO'}")
print("-"*66)
