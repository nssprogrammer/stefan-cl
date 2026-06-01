"""The analytically-grounded continual-learning benchmark.

Tasks are concentric annuli following the Frank-sphere growth law
R_k = R0 * sqrt(k). Labels follow a quadrant/XOR rule rotated by a per-task
angle, so rings impose conflicting rules (genuine forgetting) while each ring
stays individually learnable. Ground truth (the frontier radii) is analytic,
which is what lets every claim be verified exactly.
"""
import numpy as np
import torch


def frank_sphere_radii(T=5, R0=1.0):
    """Return [R_0, R_1, ..., R_T] with R_k = R0*sqrt(k)  (R_0 = 0)."""
    return [R0 * np.sqrt(k) for k in range(T + 1)]


def _rotate(xy, th):
    c, s = np.cos(th), np.sin(th)
    return np.stack([c * xy[:, 0] - s * xy[:, 1],
                     s * xy[:, 0] + c * xy[:, 1]], 1)


def _label(xy, th, omega=1.0):
    z = _rotate(xy, th)
    return (np.sin(omega * z[:, 0]) * np.sin(omega * z[:, 1]) > 0).astype(np.int64)


def _annulus(r_in, r_out, n, rng):
    r = np.sqrt(rng.uniform(r_in ** 2, r_out ** 2, n))   # area-uniform radius
    a = rng.uniform(0, 2 * np.pi, n)
    return np.stack([r * np.cos(a), r * np.sin(a)], 1).astype(np.float32)


def make_tasks(seed=0, T=5, R0=1.0, omega=1.0, spread=np.pi / 2,
               n_train=2000, n_test=4000):
    """Build train/test task lists for a given seed.

    Returns (train_tasks, test_tasks), each a list of (X, y) tensors. The seed
    varies both the data sampling and (when used) downstream stochastic
    components, so multi-seed runs reflect true run-to-run variance.
    """
    R = frank_sphere_radii(T, R0)
    theta = [(k) / (T - 1) * spread for k in range(T)]
    train, test = [], []
    for k in range(1, T + 1):
        Xtr = _annulus(R[k - 1], R[k], n_train, np.random.default_rng(10_000 * seed + 1000 + k))
        Xte = _annulus(R[k - 1], R[k], n_test,  np.random.default_rng(10_000 * seed + 2000 + k))
        train.append((torch.tensor(Xtr), torch.tensor(_label(Xtr, theta[k - 1], omega))))
        test.append((torch.tensor(Xte), torch.tensor(_label(Xte, theta[k - 1], omega))))
    return train, test


@torch.no_grad()
def accuracy_matrix(model_after_task, test_tasks):
    """Given a callable that returns the model snapshot after task i, build the
    T x T accuracy matrix A[i, j] = acc on task j after training task i."""
    T = len(test_tasks)
    A = np.zeros((T, T))
    for i in range(T):
        m = model_after_task(i)
        m.eval()
        for j, (X, y) in enumerate(test_tasks):
            A[i, j] = (m(X).argmax(1) == y).float().mean().item()
    return A


def forgetting(A):
    """Mean over tasks j<T of (max_i A[i,j] - A[T-1,j])."""
    T = A.shape[0]
    return float(np.mean([A[:, j].max() - A[T - 1, j] for j in range(T - 1)]))
