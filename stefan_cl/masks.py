"""Smooth phase masks built from the error function (Eq. 2 in the paper).

H_s  ~ 1 deep inside the consolidated (solid) region {phi < 0}, ~0 outside.
H_ell ~ 1 in the plastic (liquid) region {phi > 0}, ~0 inside.
band_weight is the derivative dH/dphi, a Gaussian band localized on the frontier.

eps controls the transition-band width.
"""
import torch


def H_s(phi, eps=0.10):
    """Solid mask: freezes the consolidated interior."""
    return 0.5 * (1.0 - torch.erf(phi / eps))


def H_ell(phi, eps=0.10):
    """Liquid mask: marks the plastic exterior."""
    return 0.5 * (1.0 + torch.erf(phi / eps))


def band_weight(phi, eps=0.10):
    """Gaussian weight localized on the frontier (proportional to dH/dphi)."""
    return torch.exp(-(phi ** 2) / (eps ** 2))
