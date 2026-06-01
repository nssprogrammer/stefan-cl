"""
Stefan-CL | Multi-seed latent-heat sweep (ROBUST build)
Environment-hardened: disables torch compile/dynamo import path and avoids
torch.optim entirely (manual SGD+momentum) so a broken sympy/_dynamo install
cannot crash the run. Numerically equivalent to multiseed_latent.py.
"""
import os
os.environ["TORCHDYNAMO_DISABLE"]="1"      # never import the _dynamo/sympy path
os.environ["TORCHINDUCTOR_DISABLE"]="1"
import copy, time, numpy as np, torch, torch.nn as nn

T=5; R0=1.0; OMEGA=1.0; SPREAD=np.pi/2
N_TRAIN=2000; N_TEST=4000; HIDDEN=128; LR=1e-3; EPOCHS=200
R_MAX=2.6; EPS=0.10; N_ANCH=1200; LAM=0.1
R=[R0*np.sqrt(k) for k in range(T+1)]; THETA=[(k)/(T-1)*SPREAD for k in range(T)]
DT=0.04; N_STEPS=25; SEED_R=0.5; V_TAPER=0.10
SEEDS=[0,1,2,3]; L_GRID=[0.5,1.0,2.0,4.0,8.0]

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
def make_tasks(seed):
    tr=[(torch.tensor(X),torch.tensor(lab(X,THETA[k-1]))) for k in range(1,T+1)
        for X in [ann(R[k-1],R[k],N_TRAIN,np.random.default_rng(10_000*seed+1000+k))]]
    te=[(torch.tensor(X),torch.tensor(lab(X,THETA[k-1]))) for k in range(1,T+1)
        for X in [ann(R[k-1],R[k],N_TEST, np.random.default_rng(10_000*seed+2000+k))]]
    return tr,te
class MLP(nn.Module):
    def __init__(s,h=HIDDEN):
        super().__init__(); s.net=nn.Sequential(nn.Linear(2,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,2))
    def forward(s,x): return s.net(x)
class Phi(nn.Module):
    def __init__(s,h=64):
        super().__init__(); s.net=nn.Sequential(nn.Linear(2,h),nn.Tanh(),nn.Linear(h,h),nn.Tanh(),nn.Linear(h,1))
    def forward(s,x): return s.net(x).squeeze(-1)
def new_mlp(seed): torch.manual_seed(seed); return MLP()
def new_phi(seed): torch.manual_seed(1000+seed); return Phi()
@torch.no_grad()
def acc(m,X,y): m.eval(); return (m(X).argmax(1)==y).float().mean().item()
def H_s(p): return 0.5*(1-torch.erf(p/EPS))

# ---- manual Adam (no torch.optim, so no _dynamo/sympy import) ----
class ManualAdam:
    def __init__(s, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        s.p=list(params); s.lr=lr; s.b1=b1; s.b2=b2; s.eps=eps
        s.m=[torch.zeros_like(x) for x in s.p]; s.v=[torch.zeros_like(x) for x in s.p]; s.t=0
    def zero_grad(s):
        for x in s.p:
            if x.grad is not None: x.grad.detach_(); x.grad.zero_()
    @torch.no_grad()
    def step(s):
        s.t+=1
        for i,x in enumerate(s.p):
            if x.grad is None: continue
            g=x.grad
            s.m[i]=s.b1*s.m[i]+(1-s.b1)*g
            s.v[i]=s.b2*s.v[i]+(1-s.b2)*g*g
            mh=s.m[i]/(1-s.b1**s.t); vh=s.v[i]/(1-s.b2**s.t)
            x.add_(-s.lr*mh/(vh.sqrt()+s.eps))

def fit_circle(net,r,iters,rg):
    op=ManualAdam(net.parameters(),3e-3)
    for _ in range(iters):
        X=torch.tensor(disk(1200,rg)); op.zero_grad()
        (net(X)-(X.norm(dim=1)-r)).pow(2).mean().backward(); op.step()
def radial_advect(Rf,dr,L):
    for _ in range(N_STEPS):
        v=min(float((dr>Rf).mean())/V_TAPER,1.0); Rf=Rf+DT*(v/L)
    return Rf
def run(seed,L):
    tr,te=make_tasks(seed); m=new_mlp(seed); phi=new_phi(seed)
    rg=np.random.default_rng(31+seed); fit_circle(phi,SEED_R,500,rg); Rf=SEED_R
    frozen=None; rng_anch=np.random.default_rng(seed); lf=nn.CrossEntropyLoss(); A=np.zeros((T,T)); Rprot=[]
    for i in range(T):
        X,y=tr[i]; opt=ManualAdam(m.parameters(),LR)
        for _ in range(EPOCHS):
            m.train(); opt.zero_grad(); loss=lf(m(X),y)
            if frozen is not None:
                Xa=torch.tensor(disk(N_ANCH,rng_anch))
                with torch.no_grad(): w=H_s(phi(Xa)); fo=frozen(Xa)
                loss=loss+LAM*(w*((m(Xa)-fo)**2).sum(1)).sum()/(w.sum()+1e-8)
            loss.backward(); opt.step()
        A[i,:]=[acc(m,*te[j]) for j in range(T)]
        Rf=radial_advect(Rf, tr[i][0].norm(dim=1).numpy(), L); fit_circle(phi,Rf,180,rg); Rprot.append(Rf)
        frozen=copy.deepcopy(m).eval()
        for p in frozen.parameters(): p.requires_grad_(False)
    final=A[T-1,:]
    forget=np.mean([A[:,j].max()-A[T-1,j] for j in range(T-1)])
    plast=np.mean(np.diag(A)); pfrac=np.mean(np.array(Rprot)/np.array(R[1:]))
    return pfrac,forget,plast
t0=time.time()
print("="*70); print("STEFAN-CL | MULTI-SEED LATENT-HEAT SWEEP (robust build)"); print("="*70)
print(f"seeds={SEEDS} epochs={EPOCHS}  [dynamo/optim disabled]\n")
print(f"{'L':>5} {'protFrac':>16} {'forget':>16} {'plast':>16}")
means={}
for L in L_GRID:
    rows=np.array([run(s,L) for s in SEEDS]); mu=rows.mean(0); sd=rows.std(0); means[L]=mu
    print(f"{L:5.1f} {mu[0]:7.3f} +/-{sd[0]:.3f} {mu[1]:7.3f} +/-{sd[1]:.3f} {mu[2]:7.3f} +/-{sd[2]:.3f}")
pf=np.array([means[L][0] for L in L_GRID]); fg=np.array([means[L][1] for L in L_GRID]); pl=np.array([means[L][2] for L in L_GRID])
def dec(a,t=0.02): return all(a[i+1]<=a[i]+t for i in range(len(a)-1))
def inc(a,t=0.02): return all(a[i+1]>=a[i]-t for i in range(len(a)-1))
print("-"*70)
print(f"protFrac decreasing in L (means)? {'YES' if dec(pf) else 'NO'}")
print(f"forgetting increasing in L (means)? {'YES' if inc(fg) else 'NO'}")
print(f"plasticity non-decreasing in L (means)? {'YES' if inc(pl) else 'NO'}")
print(f"trade-off survives error bars? {'YES' if (fg.max()-fg.min())>3*0.03 else 'NO'}")
print(f"[total {time.time()-t0:.0f}s]")
