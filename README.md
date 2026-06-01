# Continual Learning as a Multiphase Moving-Boundary Problem

A level-set formulation of continual learning in which the **stability–plasticity
dilemma** is recast as a classical **Stefan (melting/solidification) problem**, with the
**latent heat** of fusion serving as a single calibrated stability–plasticity dial.

> Consolidated knowledge is "solid"; unused capacity is "liquid". Learning a task
> *freezes* a region of capacity by advancing a moving interface — the **knowledge
> frontier** — represented implicitly as the zero level-set of a learned signed-distance
> field. How expensive it is to freeze (the latent heat) trades stability against
> plasticity.

📄 **[Read the paper](paper/stefan_cl.pdf)** &nbsp;|&nbsp; 🧪 **[Reproduce everything](#reproducing-the-results)**

---

## The physics → ML mapping, in one table

| Stefan / level-set physics | Continual-learning meaning |
|---|---|
| Solid phase `Ω_s = {φ<0}` | Consolidated knowledge (protected) |
| Liquid phase `Ω_ℓ = {φ>0}` | Unused / plastic capacity |
| Interface `Γ = {φ=0}` | Knowledge frontier |
| Signed-distance field `φ(x)` | Learned scalar field; sign = which region |
| Eikonal eq. `‖∇φ‖=1` | Regularizer making `φ` a true distance |
| Stefan condition `ρL·V_n = flux` | Frontier speed = consolidation demand / latent heat |
| Latent heat `L` | **Stability–plasticity dial** (cost to consolidate) |
| Level-set advection `φ_t + F‖∇φ‖=0` | Frontier moves to engulf the new task |
| erf phase masks `H_s, H_ℓ` | Smooth freeze of the consolidated interior |
| Closest-point projection `x_Γ = x − φ∇φ` | Extend the frontier velocity off the interface |

The paper (`paper/stefan_cl.pdf`) explains each of these in plain language for an ML
audience in Section 2.

---

## Headline results

On an analytically-grounded benchmark (concentric annuli following the Frank-sphere
growth law `R_k = R₀√k`, with per-task rotated labels to force conflict), over **10 seeds**:

| Method | Avg. accuracy ↑ | Forgetting ↓ | Stores raw data? |
|---|---|---|---|
| Naive sequential | 0.514 ± 0.006 | 0.603 ± 0.008 | no |
| SI (λ=50, best) | 0.701 ± 0.022 | 0.241 ± 0.051 | no |
| EWC (λ=300, best) | 0.716 ± 0.027 | 0.287 ± 0.045 | no |
| **Stefan-CL** | **0.923 ± 0.004** | **0.021 ± 0.003** | **no** |
| Replay (200/task) | 0.940 ± 0.004 | 0.056 ± 0.006 | yes |
| Joint oracle | 0.95 | — | — |

- **30× less forgetting** than naive; ~95% of the gap to the joint oracle closed.
- **Beats regularization baselines** (EWC, SI) by >0.20 accuracy with far smaller variance.
- **Matches replay** while storing **no raw data** (and forgetting less).
- The self-driven frontier **recovers the analytic growth law `R_k=R₀√k`** from data alone
  (max radius error 0.03), and the **latent heat `L`** traces a clean monotone
  stability–plasticity trade-off (forgetting 0.02 → 0.21 across `L`).

---

## Repository layout

```
stefan-cl/
├── stefan_cl/                 # importable core library
│   ├── models.py              #   classifier MLP + frontier SDF field
│   ├── masks.py               #   erf phase masks H_s, H_ell
│   ├── frontier.py            #   Eikonal, closest-point, advection step
│   ├── optim.py               #   ManualAdam (dependency-light)
│   └── testbed.py             #   Frank-sphere rotated-rule benchmark
├── experiments/              # self-contained reproduction scripts (Steps 1–6 + studies)
│   ├── step1_testbed_and_forgetting.py
│   ├── step2_eikonal_sdf.py
│   ├── step3_masked_anchoring.py
│   ├── step4_frontier_advection.py
│   ├── step5_full_integration.py
│   ├── step6_latent_heat_dial.py
│   ├── multiseed_main.py
│   ├── multiseed_latent_heat.py
│   ├── baselines_comparison.py
│   ├── baselines_hparam_sweep.py
│   └── noncircular_limitation.py
├── figures/
│   └── make_figures.py        # regenerate all paper figures
├── paper/
│   ├── stefan_cl.tex          # arXiv-submittable LaTeX source
│   ├── refs.bib
│   ├── figs/                  # compiled figure PDFs
│   └── stefan_cl.pdf          # compiled paper
├── requirements.txt
└── LICENSE
```

Each `experiments/` script is **standalone and self-verifying**: it prints its own
pass/fail checks against analytic ground truth. The `stefan_cl/` package factors the shared
pieces out for reuse.

---

## Installation

```bash
git clone https://github.com/<your-org>/stefan-cl.git
cd stefan-cl
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

CPU-only; no GPU required. Everything runs in minutes.

> **Environment note.** Some PyTorch builds lazily import `torch._dynamo`, which can fail
> against a mismatched `sympy` install when constructing an optimizer. The scripts set
> `TORCHDYNAMO_DISABLE=1` and use a dependency-light `ManualAdam` to sidestep this. If you
> still hit a `sympy`-related error, restart your Python process or
> `pip install --force-reinstall "sympy>=1.13"`.

---

## Reproducing the results

Run the steps in order; each is independently verifiable and prints `STEP n PASS: YES/NO`.

```bash
# The six-step construction (each ~1-3 min)
python experiments/step1_testbed_and_forgetting.py   # induce + measure catastrophic forgetting
python experiments/step2_eikonal_sdf.py              # learn a valid signed-distance frontier (|∇φ|=1)
python experiments/step3_masked_anchoring.py         # erf-masked freezing  -> forgetting collapses
python experiments/step4_frontier_advection.py       # frontier self-advects to the data envelope
python experiments/step5_full_integration.py         # full loop; recovers R_k=R₀√k from data
python experiments/step6_latent_heat_dial.py         # latent heat = stability–plasticity dial

# Rigor studies
python experiments/multiseed_main.py                 # 10-seed error bars (naive / Stefan-CL)
python experiments/multiseed_latent_heat.py          # 10-seed latent-heat sweep
python experiments/baselines_hparam_sweep.py         # tune EWC / SI / replay (fair comparison)
python experiments/baselines_comparison.py           # head-to-head at each method's best setting
python experiments/noncircular_limitation.py         # scope: topology-change study (open problem)

# Regenerate paper figures
python figures/make_figures.py
```

To use more/fewer seeds, edit the `SEEDS = [...]` list at the top of the multiseed/baseline
scripts.

---

## Building the paper

```bash
cd paper
pdflatex stefan_cl.tex && bibtex stefan_cl && pdflatex stefan_cl.tex && pdflatex stefan_cl.tex
```

The `paper/` directory is also a self-contained arXiv source bundle (`.tex`, `.bib`, and
figure PDFs).

---

## Scope and the open problem

We are explicit about what v1 does **not** yet do. The signature advantage of a level-set
representation is handling **topology change** (e.g. two consolidated regions that grow and
merge). `experiments/noncircular_limitation.py` documents two findings honestly:

- **Result A (positive):** the frontier field *can represent* a topology-changing union —
  correct connected-component counts on both sides of a merge, region accuracy 0.98–0.99,
  Eikonal satisfied. Representation is **not** the bottleneck.
- **Result B (open problem):** the data-driven advection *cannot yet* grow and track such a
  frontier from data — it erodes. The obstacle localizes to the **velocity construction on
  non-convex fronts**: the closest-point normal `∇φ/‖∇φ‖` is unstable near the *medial axis*
  between components.

The present claims are therefore scoped to frontiers admitting a clean reinitialization
target (radially symmetric / single convex components). **Medial-axis-stable velocity
extension** is the central problem for future work (per-component velocities; PDE-based
extension velocities on a grid; smoothed normal estimation).

---

## Citation

```bibtex
@article{stefancl2026,
  title  = {Stefan-CL: Continual Learning as a Moving-Boundary Problem},
  author = {(author list withheld for review)},
  year   = {2026},
  note   = {Level-set formulation with latent heat as a stability--plasticity dial}
}
```

The neural level-set machinery (phase masks, Eikonal regularization, closest-point velocity
extension) is adapted from Pandea, Minnikanti & Karagadde, *A coupled Kolmogorov–Arnold
Network and Level-Set framework for evolving interfaces* (arXiv:2601.09818, 2026).

## License

MIT — see [LICENSE](LICENSE).
