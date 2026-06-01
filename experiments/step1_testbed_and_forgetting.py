"""
Stefan-CL  |  Step 1: Testbed + Catastrophic-Forgetting Baseline
----------------------------------------------------------------
Toy continual-learning problem that is a direct analogue of the 2D radial
Stefan problem (Frank-sphere growth R_k = R0 * sqrt(k)).

Deliverables:
  (A) Data generators: global checkerboard target, tasks = expanding annuli.
  (B) Naive sequential baseline  -> demonstrates catastrophic forgetting.
  (C) Joint oracle (all tasks at once) -> learnability upper bound.

Everything is CPU, deterministic, and runs in a few seconds.
"""

import numpy as np
import torch
import torch.nn as nn

# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.use_deterministic_algorithms(True)
DEVICE = torch.device("cpu")

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
T          = 5        # number of tasks
R0         = 1.0      # base radius;  R_k = R0 * sqrt(k)
OMEGA      = 2.0      # checkerboard frequency
N_TRAIN    = 2000     # train points per task (per annulus)
N_TEST     = 4000     # test points per task
HIDDEN     = 64
EPOCHS     = 300      # per task (naive) / total (joint)
LR         = 1e-3

R = [R0 * np.sqrt(k) for k in range(T + 1)]   # R[0]=0, R[1]..R[T] frontier radii

# ----------------------------------------------------------------------
# (A) Data
# ----------------------------------------------------------------------
def global_label(xy):
    """Global checkerboard target, defined everywhere. Returns {0,1}."""
    s = np.sin(OMEGA * xy[:, 0]) * np.sin(OMEGA * xy[:, 1])
    return (s > 0).astype(np.int64)

def sample_annulus(r_in, r_out, n, rng):
    """Uniform-in-area sampling of an annulus centered at origin."""
    r = np.sqrt(rng.uniform(r_in**2, r_out**2, size=n))   # area-uniform radius
    th = rng.uniform(0.0, 2*np.pi, size=n)
    xy = np.stack([r*np.cos(th), r*np.sin(th)], axis=1)
    return xy.astype(np.float32)

def make_task(k, n, seed):
    rng = np.random.default_rng(seed)
    xy = sample_annulus(R[k-1], R[k], n, rng)
    y = global_label(xy)
    return torch.tensor(xy), torch.tensor(y)

train_tasks = [make_task(k, N_TRAIN, 1000 + k) for k in range(1, T + 1)]
test_tasks  = [make_task(k, N_TEST,  2000 + k) for k in range(1, T + 1)]

# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )
    def forward(self, x):
        return self.net(x)

def fresh_model():
    torch.manual_seed(SEED)          # identical init every run
    return MLP().to(DEVICE)

# ----------------------------------------------------------------------
# Train / eval helpers
# ----------------------------------------------------------------------
def train_on(model, X, y, epochs=EPOCHS):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(model(X), y)
        loss.backward()
        opt.step()
    return model

@torch.no_grad()
def accuracy(model, X, y):
    model.eval()
    pred = model(X).argmax(1)
    return (pred == y).float().mean().item()

def acc_all_tasks(model):
    return [accuracy(model, Xt, yt) for (Xt, yt) in test_tasks]

# ----------------------------------------------------------------------
# (B) Naive sequential baseline  (the forgetting demo)
# ----------------------------------------------------------------------
def run_naive():
    model = fresh_model()
    A = np.zeros((T, T))            # A[i,j] = acc on task j after training task i
    for i in range(T):
        Xtr, ytr = train_tasks[i]
        train_on(model, Xtr, ytr)
        A[i, :] = acc_all_tasks(model)
    return A

# ----------------------------------------------------------------------
# (C) Joint oracle  (learnability upper bound)
# ----------------------------------------------------------------------
def run_joint():
    model = fresh_model()
    X = torch.cat([t[0] for t in train_tasks], 0)
    y = torch.cat([t[1] for t in train_tasks], 0)
    train_on(model, X, y, epochs=EPOCHS)     # same total epoch budget per task-equiv
    return acc_all_tasks(model)

# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def summarize(A):
    final = A[T-1, :]                       # accuracy on every task at the end
    avg_acc = final.mean()
    # Forgetting: max acc ever seen on task j (i>=j) minus final acc
    forget = np.mean([A[:, j].max() - A[T-1, j] for j in range(T-1)])  # exclude last task
    # Backward transfer
    bwt = np.mean([A[T-1, j] - A[j, j] for j in range(T-1)])
    return avg_acc, forget, bwt, final

# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------
print("="*64)
print("STEFAN-CL  STEP 1  |  Testbed + Forgetting Baseline")
print("="*64)
print(f"Tasks={T}  R_k=R0*sqrt(k) ->", [f"{r:.3f}" for r in R[1:]])
print(f"checkerboard omega={OMEGA}  hidden={HIDDEN}  epochs/task={EPOCHS}\n")

A = run_naive()
np.set_printoptions(precision=3, suppress=True)
print("NAIVE sequential -- accuracy matrix A[i,j] (row=after task i, col=task j):")
print(A)
avg, forget, bwt, final = summarize(A)
print(f"\nNaive  final per-task acc : {np.round(final,3)}")
print(f"Naive  avg acc (all tasks): {avg:.3f}")
print(f"Naive  forgetting (old)   : {forget:.3f}   (higher = worse)")
print(f"Naive  backward transfer  : {bwt:.3f}   (negative = forgetting)")

joint = np.array(run_joint())
print(f"\nJOINT oracle per-task acc : {np.round(joint,3)}")
print(f"JOINT oracle avg acc      : {joint.mean():.3f}")

print("\n" + "-"*64)
gap = joint.mean() - avg
print(f"GAP (oracle - naive)      : {gap:.3f}")
fc = "YES" if forget > 0.10 else "NO"
lb = "YES" if joint.mean() > 0.85 else "NO"
print(f"Forgetting demonstrated (>0.10)?  {fc}")
print(f"Problem learnable (oracle>0.85)?  {lb}")
print("-"*64)
