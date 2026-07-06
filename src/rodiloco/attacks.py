"""Byzantine attacks — Phase 3.

An attack corrupts the pseudo-gradients of the byzantine subset of workers *before*
aggregation. Two families:

* **Δ-falsification** (``sign_flip``, ``scaled_noise``) — the worker trains honestly but
  reports a tampered Δ_k. Cheap, often devastating against a naive mean.
* **inner poisoning** (``targeted_drift``) — the worker trains on corrupted data, so its
  Δ_k has a *plausible norm* but points the wrong way. Stealthier: norm-based defenses
  see nothing anomalous.

``targeted_drift`` is applied inside the training loop (label permutation), so here it is
represented by a marker; the diloco loop reads it and corrupts that worker's targets.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class AttackSpec:
    name: str
    n_byzantine: int = 0  # f: how many of the workers are adversarial
    lam: float = 1.0  # sign-flip scale λ
    sigma_mult: float = 1.0  # scaled-noise σ multiplier (× honest Δ norm)
    z: float = 1.5  # ALIE deviation multiplier (× honest per-coordinate std)

    @property
    def is_inner(self) -> bool:
        return self.name == "targeted_drift"

    @property
    def is_omniscient(self) -> bool:
        """Attacks that read the honest updates to craft an evasive, colluding vector."""
        return self.name in ("alie", "min_max")


def apply_delta_attack(spec: AttackSpec, deltas: list[Tensor], byz_idx: list[int]) -> list[Tensor]:
    """Return a new list of deltas with the byzantine ones corrupted.

    Attacks split into *blind* (sign_flip, scaled_noise — tamper only with the worker's own
    Δ) and *omniscient* (alie, min_max — the colluding adversaries observe the honest Δ's and
    craft a single malicious vector designed to shift the aggregate while staying inside the
    honest cloud, i.e. evading distance/clustering defenses). The latter are the stress test
    that separates a real robust aggregator from one that only survives crude attacks.
    """
    if spec.n_byzantine == 0 or spec.name in ("none", "targeted_drift"):
        return deltas

    honest = [d for i, d in enumerate(deltas) if i not in byz_idx]
    honest_norm = torch.stack([d.norm() for d in honest]).mean() if honest else deltas[0].norm()

    out = list(deltas)

    if spec.is_omniscient:
        mal = _craft_omniscient(spec, torch.stack(honest, dim=0))
        for i in byz_idx:
            out[i] = mal.clone()
        return out

    for i in byz_idx:
        d = deltas[i]
        if spec.name == "sign_flip":
            out[i] = -spec.lam * d
        elif spec.name == "scaled_noise":
            noise = torch.randn_like(d)
            noise = noise / noise.norm().clamp_min(1e-8) * honest_norm * spec.sigma_mult
            out[i] = noise
        else:
            raise KeyError(f"unknown delta attack '{spec.name}'")
    return out


def _craft_omniscient(spec: AttackSpec, honest: Tensor) -> Tensor:
    """Build one colluding malicious Δ from the honest updates (honest: (h, D))."""
    mu = honest.mean(dim=0)
    std = honest.std(dim=0)

    if spec.name == "alie":
        # "A Little Is Enough" (Baruch et al., 2019): shift each coordinate a fraction of a
        # std below the honest mean — small enough per-coordinate to look benign, jointly
        # enough to bias the aggregate.
        return mu - spec.z * std

    # "min_max" agnostic optimal attack (Shejwalkar & Houmansadr, 2021): mal = mu + γ·pert,
    # with γ the largest scale keeping mal within the honest cloud —
    #   max_i ||mal - h_i||  ≤  max_{i,j} ||h_i - h_j||.
    pert = -std  # deviation direction; unit-normalized below
    pert = pert / pert.norm().clamp_min(1e-8)
    hh = torch.cdist(honest, honest).max()  # honest cloud diameter
    lo, hi = 0.0, 100.0
    for _ in range(30):  # binary search on γ
        gamma = 0.5 * (lo + hi)
        mal = mu + gamma * pert
        worst = (mal[None, :] - honest).norm(dim=1).max()
        if worst <= hh:
            lo = gamma
        else:
            hi = gamma
    return mu + lo * pert


def poison_targets(targets: Tensor, vocab_size: int, seed: int) -> Tensor:
    """Label permutation for the targeted-drift inner attack.

    A fixed random permutation of the vocabulary is applied to the targets, so the worker
    optimizes a coherent-but-wrong objective (norm stays plausible).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(vocab_size, generator=g).to(targets.device)
    return perm[targets]


def choose_byzantine(n_workers: int, n_byzantine: int, seed: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(n_workers, generator=g)[:n_byzantine].tolist()
