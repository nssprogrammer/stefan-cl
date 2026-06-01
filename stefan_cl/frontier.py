"""Level-set frontier machinery: gradients, Eikonal regularization,
closest-point projection, and a single data-driven advection step.

These implement Eqs. (3)-(6) of the paper:
  - eikonal_loss:   (||grad phi|| - 1)^2          [Eq. 6]
  - closest_point:  x_Gamma = x - phi * n          [normal projection]
  - advect_step:    phi <- phi - dt * F * ||grad phi||  via a fit-to-target step [Eqs. 4-5]
"""
import torch
from .optim import ManualAdam


def grad_field(net, x):
    """Return (phi(x), grad_x phi(x)) with a differentiable graph."""
    x = x.clone().requires_grad_(True)
    p = net(x)
    g = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
    return p, g


def eikonal_loss(net, x):
    """Mean squared Eikonal residual (||grad phi|| - 1)^2 over points x."""
    _, g = grad_field(net, x)
    return ((g.norm(dim=1) - 1.0) ** 2).mean()


def closest_point(phi_vals, grad_vals, x):
    """Closest-point projection onto the frontier: x_Gamma = x - phi * n,
    with n = grad phi / ||grad phi|| (valid when phi is a signed-distance fn)."""
    n = grad_vals / grad_vals.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return (x - phi_vals.unsqueeze(1) * n).detach()


def advect_step(net, x_eval, velocity_fn, dt=0.04, inner=15,
                lam_eik=0.5, lr=1e-3):
    """One level-set advection step driven by an extended velocity field.

    velocity_fn(x_gamma, phi_old) -> tensor of normal speeds V_n at the
    projected interface points. The new field is obtained by fitting phi to the
    semi-Lagrangian target  phi_old - dt * V_n * ||grad phi||  while enforcing
    the Eikonal constraint.

    net is updated in place.
    """
    import copy
    phi_old = copy.deepcopy(net)
    for p in phi_old.parameters():
        p.requires_grad_(False)

    p_old, g_old = grad_field(phi_old, x_eval)
    x_gamma = closest_point(p_old, g_old, x_eval)
    Vn = velocity_fn(x_gamma, phi_old)
    gmag = g_old.norm(dim=1).detach()
    target = p_old.detach() - dt * Vn * gmag

    opt = ManualAdam(net.parameters(), lr)
    for _ in range(inner):
        opt.zero_grad()
        p, g = grad_field(net, x_eval)
        loss = (p - target).pow(2).mean() + lam_eik * ((g.norm(dim=1) - 1) ** 2).mean()
        loss.backward()
        opt.step()
    return net
