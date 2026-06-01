"""
Stefan-CL  |  Baseline comparison  (EWC, SI, Replay vs Stefan-CL)
----------------------------------------------------------------
Same testbed, net, epochs for all methods. Each baseline reported at its
BEST operating point from a strength sweep (fair comparison, not strawman):
  EWC lambda=300, SI lambda=50, Replay buffer=200/task.
Stefan-CL = S5 learned advecting frontier, LAM=0.1, L=1.0.
Environment-hardened (TORCHDYNAMO_DISABLE + manual Adam).

Reports mean +/- std over seeds for avg-accuracy and forgetting, plus the
exemplar-storage cost of each method (Stefan-CL & EWC & SI store NO raw data).
"""
import os
os.environ["TORCHDYNAMO_DISABLE"]="1"; os.environ["TORCHINDUCTOR_DISABLE"]="1"
import copy, time, numpy as np, torch, torch.nn as nn

T=5; R0=1.0; OMEGA=1.0; SPREAD=np.pi/2
N_TRAIN=2000; N_TEST=4000; HIDDEN=128; LR=1e-3; EPOCHS=250
R_MAX=2.6; EPS=0.10; N_ANCH=1200; LAM_STEFAN=0.1
R=[R0*np.sqrt(k) for k in range(T+1)]; THETA=[(k)/(T-1)*SPREAD for k in range(T)]
DT=0.04; N_STEPS=25; SEED_R=0.5; V_TAPER=0.10
SEEDS=[0,1,2,3,4]
EWC_LAM=300.0; SI_LAM=50.0; REPLAY_BUF=200

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
class ManualAdam:
    def __init__(s,params,lr=1e-3,b1=0.9,b2=0.999,eps=1e-8):
        s.p=list(params); s.lr=lr;s.b1=b1;s.b2=b2;s.eps=eps
        s.m=[torch.zeros_like(x) for x in s.p]; s.v=[torch.zeros_like(x) for x in s.p]; s.t=0
    def zero_grad(s):
        for x in s.p:
            if x.grad is not None: x.grad.detach_(); x.grad.zero_()
    @torch.no_grad()
    def step(s):
        s.t+=1
        for i,x in enumerate(s.p):
            if x.grad is None: continue
            g=x.grad; s.m[i]=s.b1*s.m[i]+(1-s.b1)*g; s.v[i]=s.b2*s.v[i]+(1-s.b2)*g*g
            mh=s.m[i]/(1-s.b1**s.t); vh=s.v[i]/(1-s.b2**s.t); x.add_(-s.lr*mh/(vh.sqrt()+s.eps))

def run_naive(seed):
    tr,te=make_tasks(seed); m=new_mlp(seed); lf=nn.CrossEntropyLoss(); A=np.zeros((T,T))
    for i in range(T):
        X,y=tr[i]; opt=ManualAdam(m.parameters(),LR)
        for _ in range(EPOCHS): opt.zero_grad(); lf(m(X),y).backward(); opt.step()
        A[i,:]=[acc(m,*te[j]) for j in range(T)]
    return metrics(A)
def run_ewc(seed,lam=EWC_LAM):
    tr,te=make_tasks(seed); m=new_mlp(seed); lf=nn.CrossEntropyLoss(); A=np.zeros((T,T)); fishers=[]; stars=[]
    for i in range(T):
        X,y=tr[i]; opt=ManualAdam(m.parameters(),LR)
        for _ in range(EPOCHS):
            opt.zero_grad(); loss=lf(m(X),y)
            for F,star in zip(fishers,stars):
                loss=loss+0.5*lam*sum((F[n]*(p-star[n])**2).sum() for n,p in m.named_parameters())
            loss.backward(); opt.step()
        A[i,:]=[acc(m,*te[j]) for j in range(T)]
        F={n:torch.zeros_like(p) for n,p in m.named_parameters()}; ls=torch.log_softmax(m(X),1)
        for c in range(2):
            m.zero_grad(); (ls[:,c].mean()).backward(retain_graph=(c==0))
            for n,p in m.named_parameters():
                if p.grad is not None: F[n]+=p.grad.detach()**2/2
        fishers.append({n:f.clone() for n,f in F.items()}); stars.append({n:p.detach().clone() for n,p in m.named_parameters()})
    return metrics(A)
def run_si(seed,lam=SI_LAM,xi=1e-3):
    tr,te=make_tasks(seed); m=new_mlp(seed); lf=nn.CrossEntropyLoss(); A=np.zeros((T,T))
    Omega={n:torch.zeros_like(p) for n,p in m.named_parameters()}
    prev_star={n:p.detach().clone() for n,p in m.named_parameters()}
    for i in range(T):
        X,y=tr[i]; opt=ManualAdam(m.parameters(),LR)
        w={n:torch.zeros_like(p) for n,p in m.named_parameters()}
        p_old={n:p.detach().clone() for n,p in m.named_parameters()}
        for _ in range(EPOCHS):
            opt.zero_grad(); loss=lf(m(X),y)
            if i>0: loss=loss+lam*sum((Omega[n]*(p-prev_star[n])**2).sum() for n,p in m.named_parameters())
            loss.backward()
            grads={n:(p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for n,p in m.named_parameters()}
            opt.step()
            for n,p in m.named_parameters():
                w[n]+=-grads[n]*(p.detach()-p_old[n]); p_old[n]=p.detach().clone()
        A[i,:]=[acc(m,*te[j]) for j in range(T)]
        for n,p in m.named_parameters():
            star=p.detach().clone(); Omega[n]=Omega[n]+torch.clamp(w[n],min=0)/((star-prev_star[n])**2+xi); prev_star[n]=star
    return metrics(A)
def run_replay(seed,buf=REPLAY_BUF):
    tr,te=make_tasks(seed); m=new_mlp(seed); lf=nn.CrossEntropyLoss(); A=np.zeros((T,T)); bX=[]; bY=[]; rng=np.random.default_rng(seed)
    for i in range(T):
        X,y=tr[i]; opt=ManualAdam(m.parameters(),LR)
        BX=torch.cat(bX) if bX else None; BY=torch.cat(bY) if bY else None
        for _ in range(EPOCHS):
            opt.zero_grad(); loss=lf(m(X),y)
            if BX is not None: loss=loss+lf(m(BX),BY)
            loss.backward(); opt.step()
        A[i,:]=[acc(m,*te[j]) for j in range(T)]
        idx=rng.choice(len(X),size=min(buf,len(X)),replace=False); bX.append(X[idx].clone()); bY.append(y[idx].clone())
    return metrics(A)
def fit_circle(net,r,iters,rg):
    op=ManualAdam(net.parameters(),3e-3)
    for _ in range(iters):
        Xx=torch.tensor(disk(1200,rg)); op.zero_grad(); (net(Xx)-(Xx.norm(dim=1)-r)).pow(2).mean().backward(); op.step()
def radial_advect(Rf,dr,L):
    for _ in range(N_STEPS):
        v=min(float((dr>Rf).mean())/V_TAPER,1.0); Rf=Rf+DT*(v/L)
    return Rf
def run_stefan(seed,L=1.0):
    tr,te=make_tasks(seed); m=new_mlp(seed); phi=new_phi(seed)
    rg=np.random.default_rng(31+seed); fit_circle(phi,SEED_R,600,rg); Rf=SEED_R
    frozen=None; rng_a=np.random.default_rng(seed); lf=nn.CrossEntropyLoss(); A=np.zeros((T,T))
    for i in range(T):
        X,y=tr[i]; opt=ManualAdam(m.parameters(),LR)
        for _ in range(EPOCHS):
            m.train(); opt.zero_grad(); loss=lf(m(X),y)
            if frozen is not None:
                Xa=torch.tensor(disk(N_ANCH,rng_a))
                with torch.no_grad(): w=H_s(phi(Xa)); fo=frozen(Xa)
                loss=loss+LAM_STEFAN*(w*((m(Xa)-fo)**2).sum(1)).sum()/(w.sum()+1e-8)
            loss.backward(); opt.step()
        A[i,:]=[acc(m,*te[j]) for j in range(T)]
        Rf=radial_advect(Rf, tr[i][0].norm(dim=1).numpy(), L); fit_circle(phi,Rf,200,rg)
        frozen=copy.deepcopy(m).eval()
        for p in frozen.parameters(): p.requires_grad_(False)
    return metrics(A)

def agg(fn): a=np.array([fn(s) for s in SEEDS]); return a.mean(0),a.std(0)
t0=time.time()
print("="*74); print("STEFAN-CL  |  BASELINE COMPARISON  (each baseline at its best setting)"); print("="*74)
print(f"seeds={SEEDS}  epochs={EPOCHS}  net=MLP[128]  |  EWC lam={EWC_LAM} SI lam={SI_LAM} replay buf={REPLAY_BUF}/task\n")
res={}
for name,fn,store in [("Naive",run_naive,"none"),("EWC",run_ewc,"none (Fisher)"),
                      ("SI",run_si,"none (Omega)"),("Replay",run_replay,f"{REPLAY_BUF*T} exemplars"),
                      ("Stefan-CL",run_stefan,"none")]:
    mu,sd=agg(fn); res[name]=(mu,sd)
    print(f"  {name:11s} avg={mu[0]:.3f} +/-{sd[0]:.3f}   forget={mu[1]:.3f} +/-{sd[1]:.3f}   stored data: {store}")
print("-"*74)
sc=res["Stefan-CL"][0]; 
print(f"Stefan-CL vs regularization (EWC/SI):")
print(f"   beats EWC by {sc[0]-res['EWC'][0][0]:+.3f} acc, {res['EWC'][0][1]-sc[1]:+.3f} less forgetting")
print(f"   beats SI  by {sc[0]-res['SI'][0][0]:+.3f} acc, {res['SI'][0][1]-sc[1]:+.3f} less forgetting")
print(f"Stefan-CL vs Replay: {sc[0]-res['Replay'][0][0]:+.3f} acc  (Replay stores {REPLAY_BUF*T} raw points; Stefan-CL stores none)")
c1=sc[0]>res["EWC"][0][0]+0.05 and sc[0]>res["SI"][0][0]+0.05
c2=abs(sc[0]-res["Replay"][0][0])<0.03
print("-"*74)
print(f"Stefan-CL beats reg.-based methods (>0.05)?  {'YES' if c1 else 'NO'}")
print(f"Stefan-CL matches replay (within 0.03)?      {'YES' if c2 else 'NO'}")
print(f"[total {time.time()-t0:.0f}s]")
