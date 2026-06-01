"""
Stefan-CL core library
======================
Reusable components for the moving-boundary formulation of continual learning:

  - testbed:   the Frank-sphere rotated-rule benchmark (analytic ground truth)
  - models:    classifier MLP and frontier signed-distance field
  - masks:     erf phase masks H_s / H_ell
  - frontier:  Eikonal regularizer, closest-point projection, advection step
  - optim:     ManualAdam (dependency-light optimizer, avoids torch.optim/_dynamo)

The standalone scripts in experiments/ are self-contained reproductions; this
module factors out the shared pieces for reuse and import.
"""
from .models import ClassifierMLP, FrontierField
from .masks import H_s, H_ell, band_weight
from .frontier import eikonal_loss, grad_field, closest_point, advect_step
from .optim import ManualAdam
from .testbed import make_tasks, frank_sphere_radii, accuracy_matrix, forgetting

__all__ = [
    "ClassifierMLP", "FrontierField",
    "H_s", "H_ell", "band_weight",
    "eikonal_loss", "grad_field", "closest_point", "advect_step",
    "ManualAdam",
    "make_tasks", "frank_sphere_radii", "accuracy_matrix", "forgetting",
]

__version__ = "1.0.0"
