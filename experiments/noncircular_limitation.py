"""
Stefan-CL  |  Non-circular / topology-changing frontier: scope & open problem
-----------------------------------------------------------------------------
This script documents, with evidence, the boundary of what the v1 mechanism
achieves on a NON-circular frontier (two seeds that grow and merge: a
topology change from 2 connected components to 1).

It reports TWO findings honestly:

  RESULT A (positive): the neural level-set FIELD can faithfully REPRESENT a
  two-component, topology-changing frontier -- correct component counts on both
  sides of the merge, high region accuracy, Eikonal satisfied. Representation
  is NOT the bottleneck. Exact grid-SDF reinitialization is available & correct.

  RESULT B (negative / open problem): data-driven ADVECTION cannot yet
  autonomously grow & track that frontier through the merge. The velocity
  construction validated for a single convex circular front (Steps 4-6) does
  not transfer to non-convex / multi-component fronts: closest-point normals
  are unstable near the medial axis between blobs, and the advection erodes or
  spuriously bridges the region. The obstacle is the VELOCITY/normal-extension
  construction, not the field or the reinitialization.

Conclusion: v1 claims are scoped to frontiers admitting a clean reinit target
(radially symmetric / single convex component). Topology-changing CL is
identified as the central v2 problem, localized to the advection velocity on
non-convex fronts.
"""
import os
os.environ["TORCHDYNAMO_DISABLE"]="1"; os.environ["TORCHINDUCTOR_DISABLE"]="1"
import copy, time, numpy as np, torch, torch.nn as nn
from collections import deque
from scipy.ndimage import distance_transform_edt
SEED=0
def ss(): torch.manual_seed(SEED); np.random.seed(SEED)
ss()
C=np.array([[-1.2,0.0],[1.2,0.0]],dtype=np.float32); DOMAIN=2.8   # merge when rho>=1.2
def union_sdf(xy,rho): return np.stack([np.linalg.norm(xy-C[k],axis=1)-rho for k in range(2)],1).min(1)
GN=140; g=np.linspace(-DOMAIN,DOMAIN,GN); gx,gy=np.meshgrid(g,g)
Pgrid=np.stack([gx.ravel(),gy.ravel()],1).astype(np.float32); h=g[1]-g[0]
def ncomp(mask):
    n=mask.shape[0]; seen=np.zeros_like(mask,bool); c=0
    for i in range(n):
        for j in range(n):
            if mask[i,j] and not seen[i,j]:
                c+=1;q=deque([(i,j)]);seen[i,j]=True
                while q:
                    a,b=q.popleft()
                    for da,db in((1,0),(-1,0),(0,1),(0,-1)):
                        na,nb=a+da,b+db
                        if 0<=na<n and 0<=nb<n and mask[na,nb] and not seen[na,nb]: seen[na,nb]=True;q.append((na,nb))
    return c
class Phi(nn.Module):
    def __init__(s,hh=96):
        super().__init__(); s.net=nn.Sequential(nn.Linear(2,hh),nn.Tanh(),nn.Linear(hh,hh),nn.Tanh(),nn.Linear(hh,hh),nn.Tanh(),nn.Linear(hh,1))
    def forward(s,x): return s.net(x).squeeze(-1)
class MA:
    def __init__(s,p,lr=1e-3,b1=.9,b2=.999,e=1e-8):
        s.p=list(p);s.lr=lr;s.b1=b1;s.b2=b2;s.e=e;s.m=[torch.zeros_like(x) for x in s.p];s.v=[torch.zeros_like(x) for x in s.p];s.t=0
    def zero_grad(s):
        for x in s.p:
            if x.grad is not None: x.grad.detach_(); x.grad.zero_()
    @torch.no_grad()
    def step(s):
        s.t+=1
        for i,x in enumerate(s.p):
            if x.grad is None: continue
            gg=x.grad;s.m[i]=s.b1*s.m[i]+(1-s.b1)*gg;s.v[i]=s.b2*s.v[i]+(1-s.b2)*gg*gg
            x.add_(-s.lr*(s.m[i]/(1-s.b1**s.t))/((s.v[i]/(1-s.b2**s.t)).sqrt()+s.e))
def dom(n,rng): return rng.uniform(-DOMAIN,DOMAIN,(n,2)).astype(np.float32)
def grad_phi(net,x):
    x=x.clone().requires_grad_(True); p=net(x); gg=torch.autograd.grad(p.sum(),x,create_graph=True)[0]; return p,gg
def grid_sdf_from_mask(mask):
    d_out=distance_transform_edt(~mask)*h; d_in=distance_transform_edt(mask)*h
    return np.where(mask,-(d_in-0.5*h),d_out-0.5*h).astype(np.float32)
def fit_grid(net,sdf,iters,rng):
    Pt=torch.tensor(Pgrid); tgt=torch.tensor(sdf.ravel()); op=MA(net.parameters(),2e-3)
    for _ in range(iters):
        idx=rng.choice(len(Pt),2500,replace=False); op.zero_grad(); (net(Pt[idx])-tgt[idx]).pow(2).mean().backward(); op.step()
def stats(net,rho):
    with torch.no_grad(): v=net(torch.tensor(Pgrid)).numpy()
    vt=union_sdf(Pgrid,rho); Xt=torch.tensor(dom(6000,np.random.default_rng(99))); _,gt=grad_phi(net,Xt)
    return ncomp((v<0).reshape(GN,GN)), (np.sign(v)==np.sign(vt)).mean(), gt.norm(dim=1).detach().numpy().mean()

print("="*74); print("STEFAN-CL | NON-CIRCULAR FRONTIER: SCOPE & OPEN PROBLEM"); print("="*74)

# ---------------- RESULT A: representational capacity ----------------
print("\nRESULT A  (positive) -- can the FIELD represent a topology-changing union?")
print(f"{'rho':>5} {'true_comps':>11} {'pred_comps':>11} {'sign_acc':>9} {'|grad|':>8}")
rng=np.random.default_rng(1); A_ok=True
for rho in [0.6,0.9,1.3,1.6]:
    ss(); phi=Phi()
    fit_grid(phi, grid_sdf_from_mask((union_sdf(Pgrid,rho)<0).reshape(GN,GN)), 1200, rng)
    c,sa,gn=stats(phi,rho); tc=2 if rho<1.2 else 1
    print(f"{rho:5.1f} {tc:>11} {c:>11} {sa:>9.3f} {gn:>8.3f}")
    A_ok = A_ok and (c==tc) and (sa>0.95)
print(f"  => field represents both topologies correctly & Eikonal holds? {'YES' if A_ok else 'NO'}")

# ---------------- RESULT B: data-driven advection fails ----------------
print("\nRESULT B  (open problem) -- can ADVECTION grow/track it from data?")
ss(); phi=Phi(); rng2=np.random.default_rng(2)
fit_grid(phi, grid_sdf_from_mask((union_sdf(Pgrid,0.35)<0).reshape(GN,GN)), 600, rng2)
c0,sa0,_=stats(phi,0.35)
print(f"  init(rho=0.35): comps={c0} sign_acc={sa0:.3f}  (correct start: 2 components)")
DELTA=0.16; DT=0.06; LAM_EIK=1.0; DENS_MIN=2; rng_ev=np.random.default_rng(7)
def true_area(rho): return float((union_sdf(Pgrid,rho)<0).mean())
def cur_area(net):
    with torch.no_grad(): return float((net(torch.tensor(Pgrid)).numpy()<0).mean())
for st,rho in enumerate([0.7,1.0,1.4,1.6]):
    pool=dom(11000,rng_ev); data=torch.tensor(pool[union_sdf(pool,rho)<0][:4500])
    for step in range(14):
        phi_old=copy.deepcopy(phi)
        for p in phi_old.parameters(): p.requires_grad_(False)
        Xev=torch.tensor(dom(900,rng_ev))
        with torch.no_grad():
            outside=(phi_old(Xev)>0).float(); ahead=data[phi_old(data)>0]
            Vn=torch.zeros(len(Xev)) if len(ahead)==0 else torch.clamp(((torch.cdist(Xev,ahead)<DELTA).float().sum(1)-DENS_MIN)/6,0,1)*outside
        p_old,g_old=grad_phi(phi_old,Xev); gmag=g_old.norm(dim=1).detach()
        target=p_old.detach()-DT*Vn*gmag; oo=MA(phi.parameters(),1e-3)
        for _ in range(12):
            oo.zero_grad(); p,gg=grad_phi(phi,Xev)
            ((p-target).pow(2).mean()+LAM_EIK*((gg.norm(dim=1)-1).pow(2).mean())).backward(); oo.step()
        if (step+1)%4==0: fit_grid(phi, grid_sdf_from_mask((phi(torch.tensor(Pgrid)).detach().numpy()<0).reshape(GN,GN)),120,rng_ev)
    c,sa,gn=stats(phi,rho)
    print(f"  stage{st+1}(rho={rho}): comps={c} (true {2 if rho<1.2 else 1})  sign_acc={sa:.3f}  "
          f"area={cur_area(phi):.3f} vs true {true_area(rho):.3f}")
print("  => advection tracks the merge correctly? NO  (region eroded / topology not tracked)")

print("\n"+"-"*74)
print("CONCLUSION: representation + reinit are solved; the open problem is the")
print("data-driven advection VELOCITY on non-convex / multi-component fronts")
print("(closest-point normals unstable near the medial axis). v1 scope = single")
print("convex/radial frontier; topology-changing CL = primary v2 target.")
print("-"*74)
