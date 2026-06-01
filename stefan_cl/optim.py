"""Dependency-light Adam, avoiding torch.optim (which can pull in the
torch._dynamo/sympy import path that is fragile in some environments)."""
import torch


class ManualAdam:
    """Hand-rolled Adam. Numerically equivalent to torch.optim.Adam for the
    settings used here, but constructs no optimizer state machinery that would
    trigger torch's lazy compile/_dynamo import."""

    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.p = list(params)
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [torch.zeros_like(x) for x in self.p]
        self.v = [torch.zeros_like(x) for x in self.p]
        self.t = 0

    def zero_grad(self):
        for x in self.p:
            if x.grad is not None:
                x.grad.detach_()
                x.grad.zero_()

    @torch.no_grad()
    def step(self):
        self.t += 1
        for i, x in enumerate(self.p):
            if x.grad is None:
                continue
            g = x.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            x.add_(-self.lr * mh / (vh.sqrt() + self.eps))
