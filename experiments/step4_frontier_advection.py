"""
Stefan-CL  |  Step 4: Data-driven advection of the neural level-set field
--------------------------------------------------------------------------
Single transition R1 -> R2.  The frontier is NOT told R2; it discovers the
data's outer envelope by obeying the level-set evolution PDE.

Mechanism (paper Eqs. 15-21):
  * normal velocity (Stefan analogue):  Vn(p) = c * demand(p),
      demand = "is there task data within DELTA of p that is still OUTSIDE
                the front (phi>0)?"   -> front freezes available data, halts
                when none remains outside.
  * velocity extension (closest point): x_Gamma = x - phi * grad phi/|grad phi|,
      F(x) = Vn(x_Gamma)   (constant along normals).
  * advection + Eikonal: phi <- phi - DT * F * |grad phi|, keep |grad phi|=1.

Verifies: (1) frontier lands at R2 (untold), (2) Eikonal maintained,
(3) velocity constant along normals, (4) self-halting under extra steps.
"""
import copy, numpy as np, torch, torch.nn as nn
SEED=0
def ss(): torch.manual_seed(SEED); np.random.seed(SEED)
ss(); torch.use_deterministic_algorithms(True)

R0=1.0; R1=R0*np.sqrt(1); R2=R0*np.sqrt(2); R_MAX=2.6
DELTA=0.12; CVEL=1.0; DT=0.03; N_STEPS=40; INNER=40; LAM_EIK=0.5; N_EXTRA=15

def ann(ri,ro,n,rng):
    r=np.sqrt(rng.uniform(ri**2,ro**2,n)); a=rng.uniform(0,2*np.pi,n)
    return np.stack([r*np.cos(a),r*np.sin(a)],1).astype(np.float32)
def disk(n,rng):
    r=R_MAX*np.sqrt(rng.uniform(0,1,n)); a=rng.uniform(0,2*np.pi,n)
    return np.stack([r*np.cos(a),r*np.sin(a)],1).astype(np.float32)
DATA=torch.tensor(ann(R1,R2,1200,np.random.default_rng(1002)))

class Phi(nn.Module):
    def __init__(s,h=64):
        super().__init__(); s.net=nn.Sequential(
            nn.Linear(2,h),nn.Tanh(),nn.Linear(h,h),nn.Tanh(),
            nn.Linear(h,h),nn.Tanh(),nn.Linear(h,1))
    def forward(s,x): return s.net(x).squeeze(-1)
ss(); phi=Phi()

def grad_phi(net,x):
    x=x.clone().requires_grad_(True); p=net(x)
    g=torch.autograd.grad(p.sum(),x,create_graph=True)[0]; return p,g
def frontier_radius(net,n_rays=48,n_steps=400):
    th=np.linspace(0,2*np.pi,n_rays,endpoint=False); rs=np.linspace(0,R_MAX,n_steps); rad=[]
    for t in th:
        pts=np.stack([rs*np.cos(t),rs*np.sin(t)],1).astype(np.float32)
        with torch.no_grad(): v=net(torch.tensor(pts)).numpy()
        idx=np.where(np.diff(np.sign(v))>0)[0]
        if len(idx): i=idx[0]; rad.append(rs[i]-v[i]*(rs[i+1]-rs[i])/(v[i+1]-v[i]))
    return np.mean(rad) if rad else np.nan
def demand_at(points, phi_old):
    with torch.no_grad():
        ahead=DATA[phi_old(DATA)>0]
        if ahead.shape[0]==0: return torch.zeros(points.shape[0])
        cnt=(torch.cdist(points,ahead)<DELTA).float().sum(1)
    return torch.clamp(cnt/3.0,0,1)
def extended_velocity(net,X):
    p,g=grad_phi(net,X); n=g/g.norm(dim=1,keepdim=True).clamp_min(1e-6)
    xG=(X - p.unsqueeze(1)*n).detach()
    return demand_at(xG,net)*CVEL, g.norm(dim=1).detach(), p.detach()

# init as SDF(R1)
opt=torch.optim.Adam(phi.parameters(),2e-3); rinit=np.random.default_rng(7)
for _ in range(1500):
    X=torch.tensor(disk(2000,rinit))
    opt.zero_grad(); (phi(X)-(X.norm(dim=1)-R1)).pow(2).mean().backward(); opt.step()
print("="*66); print("STEFAN-CL  STEP 4  |  Level-set advection (R1 -> R2, untold)"); print("="*66)
print(f"init frontier R = {frontier_radius(phi):.3f}  (R1={R1})")

# --- (3) velocity-extension check: F constant along a normal ray (front moving) ---
th0=0.7; rs=np.linspace(R1-0.25,R1+0.25,40)
ray=torch.tensor(np.stack([rs*np.cos(th0),rs*np.sin(th0)],1).astype(np.float32))
Fray,_,_=extended_velocity(phi,ray)
Fray=Fray.numpy(); active=Fray[Fray>0.05]
ext_cv=(active.std()/(active.mean()+1e-9)) if len(active)>3 else 0.0
print(f"velocity-extension: F along normal ray  mean={active.mean():.3f} "
      f"CV={ext_cv:.3f}  (low CV => constant along normal)")

# --- advection ---
radii=[frontier_radius(phi)]; rng_ev=np.random.default_rng(SEED)
def advect_step():
    phi_old=copy.deepcopy(phi)
    for p in phi_old.parameters(): p.requires_grad_(False)
    Xev=torch.tensor(disk(1000,rng_ev))
    F,gmag,p_old=extended_velocity(phi_old,Xev)
    target=p_old - DT*F*gmag
    o=torch.optim.Adam(phi.parameters(),1e-3)
    for _ in range(INNER):
        o.zero_grad(); p,g=grad_phi(phi,Xev)
        (( p-target).pow(2).mean()+LAM_EIK*((g.norm(dim=1)-1).pow(2).mean())).backward(); o.step()
for _ in range(N_STEPS):
    advect_step(); radii.append(frontier_radius(phi))
radii=np.array(radii)
print(f"\nadvection: R {radii[0]:.3f} -> {radii[-1]:.3f}   (target R2={R2:.3f}, untold)")
print(f"trajectory (every 8): {np.round(radii[::8],3)}")

# --- (4) self-halting: extra steps shouldn't grow the front ---
r_before=frontier_radius(phi)
for _ in range(N_EXTRA): advect_step()
r_after=frontier_radius(phi)
print(f"self-halting: extra {N_EXTRA} steps  R {r_before:.3f} -> {r_after:.3f}  "
      f"(drift={abs(r_after-r_before):.4f})")

# --- (2) Eikonal final ---
Xt=torch.tensor(disk(8000,np.random.default_rng(11)))
_,gt=grad_phi(phi,Xt); gnorm=gt.norm(dim=1).detach().numpy()
print(f"final |grad phi| mean={gnorm.mean():.3f} std={gnorm.std():.3f}")

print("-"*66)
err=abs(radii[-1]-R2)
c1=err<0.05
c2=0.93<=gnorm.mean()<=1.07
c3=ext_cv<0.15
c4=abs(r_after-r_before)<0.05
print(f"Frontier discovered R2 (|err|<0.05)?  {'YES' if c1 else 'NO'}  (err={err:.4f})")
print(f"Eikonal maintained (|grad|~1)?        {'YES' if c2 else 'NO'}")
print(f"Velocity constant along normal (CV<.15)? {'YES' if c3 else 'NO'}")
print(f"Front self-halts (drift<0.05)?        {'YES' if c4 else 'NO'}")
print(f"STEP 4 PASS: {'YES' if (c1 and c2 and c3 and c4) else 'NO'}")
print("-"*66)
