"""
Stefan-CL  |  Step 2: Level-set field phi + Eikonal constraint  (v2)
-------------------------------------------------------------------
Learn a SIGNED distance field phi_theta(x) whose zero-set is ||x||=R.

Training losses (no regression to phi*):
    L = L_eikonal                       (|grad phi| = 1 over domain)
      + lam_d * L_interface             (phi = 0 on circle)
      + lam_n * L_normal                (grad phi = outward normal on circle)
      + lam_s * L_sign                  (region membership: inside<0, outside>0)

L_sign is the honest data term: in Stefan-CL we KNOW whether a point is
consolidated (inside) or plastic (outside). Eikonal turns that sign into a
true distance. Verified against analytic phi*(x)=||x||-R.
"""

import numpy as np
import torch
import torch.nn as nn

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)
torch.use_deterministic_algorithms(True)
DEVICE = torch.device("cpu")

# ---------------- config ----------------
R       = 1.0
R_MAX   = 2.6
N_DOM   = 4000
N_CIRC  = 512
HIDDEN  = 64
EPOCHS  = 2000
LR      = 2e-3
LAM_D   = 1.0
LAM_N   = 1.0
LAM_S   = 0.5     # sign / region weight
TAU     = 0.20    # sign sharpness

class Phi(nn.Module):
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def geometric_init(model, r_init):
    with torch.no_grad():
        layers = [m for m in model.net if isinstance(m, nn.Linear)]
        for lin in layers[:-1]:
            nn.init.normal_(lin.weight, 0.0, np.sqrt(2.0 / lin.weight.shape[0]))
            nn.init.zeros_(lin.bias)
        last = layers[-1]
        nn.init.normal_(last.weight, mean=0.0, std=1e-4)
        nn.init.constant_(last.bias, -r_init)
    return model

torch.manual_seed(SEED)
phi = geometric_init(Phi(), r_init=0.5 * R_MAX).to(DEVICE)

def sample_disk(n, rng):
    r = R_MAX * np.sqrt(rng.uniform(0, 1, n))
    th = rng.uniform(0, 2*np.pi, n)
    return np.stack([r*np.cos(th), r*np.sin(th)], 1).astype(np.float32)

def circle_points(n):
    th = np.linspace(0, 2*np.pi, n, endpoint=False)
    xy = np.stack([R*np.cos(th), R*np.sin(th)], 1).astype(np.float32)
    nrm = np.stack([np.cos(th), np.sin(th)], 1).astype(np.float32)
    return torch.tensor(xy), torch.tensor(nrm)

def grad_phi(x):
    x = x.clone().requires_grad_(True)
    p = phi(x)
    g = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
    return p, g

opt = torch.optim.Adam(phi.parameters(), lr=LR)
rng = np.random.default_rng(SEED)
Xc, Nc = circle_points(N_CIRC)
bce = nn.BCEWithLogitsLoss()

for ep in range(EPOCHS):
    Xd = torch.tensor(sample_disk(N_DOM, rng))
    r_d = Xd.norm(dim=1)
    target_out = (r_d > R).float()          # 1 outside, 0 inside
    opt.zero_grad()
    pd, gd = grad_phi(Xd)
    L_eik = ((gd.norm(dim=1) - 1.0) ** 2).mean()
    L_sign = bce(pd / TAU, target_out)        # sign(phi) matches region
    pc, gc = grad_phi(Xc)
    L_d = (pc ** 2).mean()
    L_n = ((gc - Nc) ** 2).sum(1).mean()
    loss = L_eik + LAM_D*L_d + LAM_N*L_n + LAM_S*L_sign
    loss.backward()
    opt.step()
    if (ep + 1) % 500 == 0:
        print(f"  epoch {ep+1:4d}  L_eik={L_eik.item():.4e}  L_int={L_d.item():.4e}  "
              f"L_nrm={L_n.item():.4e}  L_sign={L_sign.item():.4e}")

# ---------------- verification ----------------
print("\n" + "="*64)
print("STEFAN-CL  STEP 2  |  Eikonal SDF verification  (v2)")
print("="*64)
rng_t = np.random.default_rng(123)
Xt = torch.tensor(sample_disk(20000, rng_t))
with torch.no_grad():
    phi_pred = phi(Xt).numpy()
phi_star = (Xt.norm(dim=1) - R).numpy()

_, gt = grad_phi(Xt)
gnorm = gt.norm(dim=1).detach().numpy()
print(f"|grad phi|  mean={gnorm.mean():.4f}  std={gnorm.std():.4f}  (target 1.000)")
print(f"Eikonal residual mean((|grad|-1)^2) = {((gnorm-1)**2).mean():.4e}")
print(f"SDF error |phi-phi*|  MAE={np.abs(phi_pred-phi_star).mean():.4f}  "
      f"max={np.abs(phi_pred-phi_star).max():.4f}")

def recovered_radius(n_rays=64, n_steps=600):
    th = np.linspace(0, 2*np.pi, n_rays, endpoint=False)
    rs = np.linspace(0.0, R_MAX, n_steps); radii = []
    for t in th:
        pts = np.stack([rs*np.cos(t), rs*np.sin(t)], 1).astype(np.float32)
        with torch.no_grad():
            v = phi(torch.tensor(pts)).numpy()
        idx = np.where(np.diff(np.sign(v)) > 0)[0]
        if len(idx):
            i = idx[0]; r0,r1,v0,v1 = rs[i],rs[i+1],v[i],v[i+1]
            radii.append(r0 - v0*(r1-r0)/(v1-v0))
    radii = np.array(radii); return radii.mean(), radii.std()

rmean, rstd = recovered_radius()
print(f"Recovered frontier radius = {rmean:.4f} +/- {rstd:.4f}  (target R={R})")
sign_acc = (np.sign(phi_pred) == np.sign(phi_star)).mean()
print(f"Region sign accuracy = {sign_acc:.4f}")

print("-"*64)
c1 = 0.95 <= gnorm.mean() <= 1.05
c2 = np.abs(phi_pred - phi_star).mean() < 0.05
c3 = abs(rmean - R) < 0.03
c4 = sign_acc > 0.98
print(f"Eikonal satisfied (|grad|~1)?   {'YES' if c1 else 'NO'}")
print(f"SDF recovered (MAE<0.05)?       {'YES' if c2 else 'NO'}")
print(f"Frontier at R (|err|<0.03)?     {'YES' if c3 else 'NO'}")
print(f"Region sign correct (>0.98)?    {'YES' if c4 else 'NO'}")
print(f"STEP 2 PASS: {'YES' if (c1 and c2 and c3 and c4) else 'NO'}")
print("-"*64)
