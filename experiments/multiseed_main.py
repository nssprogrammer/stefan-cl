"""
Stefan-CL  |  Multi-seed error bars  (Part 1: naive, Step 3, Step 5)
--------------------------------------------------------------------
Each seed varies (a) network init, (b) data sampling, (c) advection/anchor RNG
TOGETHER, so the bars reflect true run-to-run variance. Per-seed rows are
printed (not hidden) so any blow-up is visible. Validated settings: 250 epochs,
hidden=128, LAM=0.1. Part 2 (latent-heat sweep) is a separate script.
"""
import copy, time, numpy as np, torch, torch.nn as nn

T=5; R0=1.0; OMEGA=1.0; SPREAD=np.pi/2
N_TRAIN=2000; N_TEST=4000; HIDDEN=128; LR=1e-3; EPOCHS=250
R_MAX=2.6; EPS=0.10; N_ANCH=1200; LAM=0.1
R=[R0*np.sqrt(k) for k in range(T+1)]; THETA=[(k)/(T-1)*SPREAD for k in range(T)]
DT=0.04; N_STEPS=25; SEED_R=0.5; V_TAPER=0.10
SEEDS=[0,1,2,3,4]

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
def metrics(A):
    final=A[T-1,:]; return final.mean(), np.mean([A[:,j].max()-A[T-1,j] for j in range(T-1)])

def run_naive(seed):
    tr,te=make_tasks(seed); m=new_mlp(seed); lf=nn.CrossEntropyLoss(); A=np.zeros((T,T))
    for i in range(T):
        X,y=tr[i]; opt=torch.optim.Adam(m.parameters(),LR)
        for _ in range(EPOCHS): opt.zero_grad(); lf(m(X),y).backward(); opt.step()
        A[i,:]=[acc(m,*te[j]) for j in range(T)]
    return metrics(A)

def run_step3(seed):
    tr,te=make_tasks(seed); m=new_mlp(seed); frozen=None
    rng_anch=np.random.default_rng(seed); lf=nn.CrossEntropyLoss(); A=np.zeros((T,T))
    for i in range(T):
        X,y=tr[i]; opt=torch.optim.Adam(m.parameters(),LR); Rc=R[i]
        for _ in range(EPOCHS):
            m.train(); opt.zero_grad(); loss=lf(m(X),y)
            if frozen is not None:
                Xa=torch.tensor(disk(N_ANCH,rng_anch)); w=H_s(Xa.norm(dim=1)-Rc)
                with torch.no_grad(): fo=frozen(Xa)
                loss=loss+LAM*(w*((m(Xa)-fo)**2).sum(1)).sum()/(w.sum()+1e-8)
            loss.backward(); opt.step()
        A[i,:]=[acc(m,*te[j]) for j in range(T)]
        frozen=copy.deepcopy(m).eval()
        for p in frozen.parameters(): p.requires_grad_(False)
    return metrics(A)

def fit_circle(net,r,iters,rg):
    op=torch.optim.Adam(net.parameters(),3e-3)
    for _ in range(iters):
        X=torch.tensor(disk(1200,rg)); op.zero_grad(); (net(X)-(X.norm(dim=1)-r)).pow(2).mean().backward(); op.step()
def radial_advect(Rf,dr,L):
    for _ in range(N_STEPS):
        v=min(float((dr>Rf).mean())/V_TAPER,1.0); Rf=Rf+DT*(v/L)
    return Rf
def frad_net(net):
    th=np.linspace(0,2*np.pi,48,endpoint=False); rs=np.linspace(0,R_MAX,400); rad=[]
    for t in th:
        pts=np.stack([rs*np.cos(t),rs*np.sin(t)],1).astype(np.float32)
        with torch.no_grad(): vv=net(torch.tensor(pts)).numpy()
        idx=np.where(np.diff(np.sign(vv))>0)[0]
        if len(idx): i=idx[0]; rad.append(rs[i]-vv[i]*(rs[i+1]-rs[i])/(vv[i+1]-vv[i]))
    return float(np.median(rad)) if rad else np.nan

def run_step5(seed, L=1.0):
    tr,te=make_tasks(seed); m=new_mlp(seed); phi=new_phi(seed)
    rg=np.random.default_rng(31+seed); fit_circle(phi,SEED_R,600,rg); Rf=SEED_R
    frozen=None; rng_anch=np.random.default_rng(seed); lf=nn.CrossEntropyLoss(); A=np.zeros((T,T)); Rprot=[]
    for i in range(T):
        X,y=tr[i]; opt=torch.optim.Adam(m.parameters(),LR)
        for _ in range(EPOCHS):
            m.train(); opt.zero_grad(); loss=lf(m(X),y)
            if frozen is not None:
                Xa=torch.tensor(disk(N_ANCH,rng_anch))
                with torch.no_grad(): w=H_s(phi(Xa)); fo=frozen(Xa)
                loss=loss+LAM*(w*((m(Xa)-fo)**2).sum(1)).sum()/(w.sum()+1e-8)
            loss.backward(); opt.step()
        A[i,:]=[acc(m,*te[j]) for j in range(T)]
        Rf=radial_advect(Rf, tr[i][0].norm(dim=1).numpy(), L); fit_circle(phi,Rf,200,rg)
        Rprot.append(frad_net(phi))
        frozen=copy.deepcopy(m).eval()
        for p in frozen.parameters(): p.requires_grad_(False)
    avg,forget=metrics(A)
    law=np.array(R[1:]); maxerr=np.abs(np.array(Rprot)-law).max()
    return avg,forget,maxerr

def msd(rows): 
    a=np.array(rows); return a.mean(0), a.std(0)
t0=time.time()
print("="*70); print("STEFAN-CL  |  MULTI-SEED ERROR BARS  (naive, Step 3, Step 5)"); print("="*70)
print(f"seeds={SEEDS}  epochs={EPOCHS}  (init+data+RNG all vary per seed)\n")

naive=[run_naive(s) for s in SEEDS]
step3=[run_step3(s) for s in SEEDS]
step5=[run_step5(s) for s in SEEDS]
print(f"per-seed naive (avg,forget):  {[ (round(a,3),round(f,3)) for a,f in naive ]}")
print(f"per-seed step3 (avg,forget):  {[ (round(a,3),round(f,3)) for a,f in step3 ]}")
print(f"per-seed step5 (avg,forget,raderr): {[ (round(a,3),round(f,3),round(e,3)) for a,f,e in step5 ]}")

nm,ns=msd([(a,f) for a,f in naive]); m3,s3=msd([(a,f) for a,f in step3])
s5arr=np.array(step5); m5=s5arr.mean(0); sd5=s5arr.std(0)
print("\n"+"-"*70)
print(f"{'method':16s}{'avg acc':>16s}{'forgetting':>16s}")
print(f"{'Naive':16s}{nm[0]:>9.3f} +/-{ns[0]:.3f}{nm[1]:>9.3f} +/-{ns[1]:.3f}")
print(f"{'Stefan-CL S3':16s}{m3[0]:>9.3f} +/-{s3[0]:.3f}{m3[1]:>9.3f} +/-{s3[1]:.3f}")
print(f"{'Stefan-CL S5':16s}{m5[0]:>9.3f} +/-{sd5[0]:.3f}{m5[1]:>9.3f} +/-{sd5[1]:.3f}")
print(f"\nStep5 frontier radius max-error: {m5[2]:.3f} +/- {sd5[2]:.3f}  (all seeds < {s5arr[:,2].max():.3f})")
print("-"*70)
c1=m5[0]-3*sd5[0] > nm[0]+3*ns[0]   # accuracy gain robust to 3 sigma
c2=sd5[1] < 0.03                     # forgetting std small
print(f"Accuracy gain robust (>3 sigma vs naive)? {'YES' if c1 else 'NO'}")
print(f"Forgetting std small (<0.03)?             {'YES' if c2 else 'NO'}")
print(f"[total {time.time()-t0:.0f}s]")
