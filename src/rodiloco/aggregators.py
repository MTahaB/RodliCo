"""The one interface the whole project extends: ``aggregate(deltas) -> Tensor``.

Each aggregator maps a list of per-worker pseudo-gradients ``Δ_k`` (flat vectors) to a
single aggregated Δ that the outer optimizer consumes. Phase 2 uses ``mean``; Phase 4
adds the robust aggregators.

Research caveat baked into the docstrings: these defenses were designed for *gradients*.
Here they aggregate *pseudo-gradients after H steps of AdamW, then consumed by outer
Nesterov momentum*, at ``n = 8`` workers. Two effects are measured, not assumed:
(a) defense × outer-momentum interaction, (b) small-``n`` behaviour of Krum.
"""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor


class Aggregator(Protocol):
    stateful: bool

    def __call__(self, deltas: list[Tensor]) -> Tensor: ...


def _stack(deltas: list[Tensor]) -> Tensor:
    return torch.stack(deltas, dim=0)  # (n, D)


# --- Phase 2: the baseline ----------------------------------------------------
class Mean:
    """Plain average — the DiLoCo-vanilla aggregation. The thing attacks break."""

    stateful = False

    def __call__(self, deltas: list[Tensor]) -> Tensor:
        return _stack(deltas).mean(dim=0)


# --- Phase 4: classical robust baselines --------------------------------------
class TrimmedMean:
    """Coordinate-wise trimmed mean: drop the ``b`` largest & smallest per coordinate."""

    stateful = False

    def __init__(self, trim: int = 1):
        self.trim = trim

    def __call__(self, deltas: list[Tensor]) -> Tensor:
        x = _stack(deltas)  # (n, D)
        n = x.shape[0]
        b = self.trim
        if 2 * b >= n:
            b = max(0, (n - 1) // 2)
        sorted_x, _ = torch.sort(x, dim=0)
        kept = sorted_x[b : n - b] if b > 0 else sorted_x
        return kept.mean(dim=0)


class Krum:
    """(Multi-)Krum. Score_k = sum of squared distances to the n-f-2 nearest Δ.

    Keep the ``m`` lowest-scoring deltas and average them (m=1 is vanilla Krum).
    NOTE: pathological at small n — with n=8 and f=1, each score uses only the 5
    nearest neighbours; this small-n cost is exactly what Phase 4 quantifies.
    """

    stateful = False

    def __init__(self, n_byzantine: int = 1, multi: int = 1):
        self.f = n_byzantine
        self.multi = multi

    def __call__(self, deltas: list[Tensor]) -> Tensor:
        x = _stack(deltas)  # (n, D)
        n = x.shape[0]
        # pairwise squared distances
        d2 = torch.cdist(x, x, p=2).pow(2)  # (n, n)
        k = max(1, n - self.f - 2)  # neighbours counted (excluding self)
        scores = torch.empty(n, device=x.device)
        for i in range(n):
            dists = d2[i].clone()
            dists[i] = float("inf")  # exclude self
            nearest, _ = torch.topk(dists, k, largest=False)
            scores[i] = nearest.sum()
        m = min(self.multi, n)
        chosen = torch.topk(scores, m, largest=False).indices
        return x[chosen].mean(dim=0)


class GeometricMedian:
    """Geometric median via Weiszfeld iteration (~the third standard baseline)."""

    stateful = False

    def __init__(self, iters: int = 32, eps: float = 1e-8):
        self.iters = iters
        self.eps = eps

    def __call__(self, deltas: list[Tensor]) -> Tensor:
        x = _stack(deltas)  # (n, D)
        y = x.mean(dim=0)
        for _ in range(self.iters):
            dist = (x - y).norm(dim=1).clamp_min(self.eps)  # (n,)
            w = 1.0 / dist
            y_new = (w[:, None] * x).sum(dim=0) / w.sum()
            if (y_new - y).norm() < self.eps:
                y = y_new
                break
            y = y_new
        return y


class CenteredClipping:
    """Centered clipping (Karimireddy et al., 2021) — a modern, momentum-aware defense.

    Iteratively clips each Δ around a running center ``v`` that persists across outer rounds
    (the center acts like server momentum, which is why this defense is a natural fit for
    DiLoCo's outer-Nesterov regime rather than an awkward transplant):

        v <- v + (1/n) Σ_i (Δ_i - v) · min(1, r / ||Δ_i - v||)

    repeated ``iters`` times. The clip radius ``r`` is set scale-adaptively to
    ``tau · median_i ||Δ_i - v||`` so ``tau`` stays O(1) across model/LR scales.
    """

    stateful = True

    def __init__(self, tau: float = 1.0, iters: int = 3, eps: float = 1e-8):
        self.tau = tau
        self.iters = iters
        self.eps = eps
        self.center: Tensor | None = None

    def reset(self) -> None:
        self.center = None

    def __call__(self, deltas: list[Tensor]) -> Tensor:
        x = _stack(deltas)  # (n, D)
        v = self.center if self.center is not None else x.mean(dim=0)
        for _ in range(self.iters):
            diff = x - v  # (n, D)
            norms = diff.norm(dim=1, keepdim=True).clamp_min(self.eps)  # (n, 1)
            radius = self.tau * norms.median().clamp_min(self.eps)
            scale = torch.clamp(radius / norms, max=1.0)  # (n, 1)
            v = v + (diff * scale).mean(dim=0)
        self.center = v.detach().clone()
        return v


# --- Phase 4: the contribution ------------------------------------------------
class TrustWeighted:
    """Trust-weighted outer aggregation (the proposed contribution).

    Per-worker trust score = EMA of the cosine similarity between Δ_k(t) and the previous
    robust aggregate Δ̄(t-1):

        s_k(t) = β·s_k(t-1) + (1-β)·cos(Δ_k(t), Δ̄(t-1))

    Aggregation weights = softmax(s / τ). Unlike Krum, it *weights instead of excludes* —
    it never throws away information — and it exploits the temporal dimension that the
    outer setting (few rounds, rich Δ) makes informative.

    Design intent — "zero tax": at f=0 all workers stay consistent, scores converge, and
    softmax weights approach uniform ⇒ it should reduce to the plain mean.

    Cold-start: the trust score needs a *reference* aggregate to compare against, and on the
    very first round there is no history. Seeding that reference with the plain mean is a
    trap — under a strong attack the round-0 mean is already poisoned, so the trust signal is
    corrupted from birth and never recovers (empirically, this is what made the method fail at
    f=2). We instead seed from a robust aggregate (geometric median), which gives the trust
    dynamics an uncorrupted starting point.
    """

    stateful = True

    def __init__(self, beta: float = 0.9, tau: float = 0.1, normalize: bool = True,
                 bootstrap: str = "geometric_median"):
        self.beta = beta
        self.tau = tau
        self.normalize = normalize
        self._boot = GeometricMedian() if bootstrap == "geometric_median" else TrimmedMean(trim=1)
        self.scores: Tensor | None = None
        self.prev_agg: Tensor | None = None

    def reset(self) -> None:
        self.scores = None
        self.prev_agg = None

    def __call__(self, deltas: list[Tensor]) -> Tensor:
        x = _stack(deltas)  # (n, D)
        n = x.shape[0]
        if self.scores is None:
            self.scores = torch.zeros(n, device=x.device)

        if self.prev_agg is None:
            # first round: no history yet, seed the reference from a robust aggregate so the
            # trust signal is not poisoned by a corrupted round-0 mean.
            agg = self._boot(deltas)
        else:
            ref = self.prev_agg
            cos = torch.nn.functional.cosine_similarity(x, ref[None, :].expand_as(x), dim=1)
            self.scores = self.beta * self.scores + (1 - self.beta) * cos
            w = torch.softmax(self.scores / self.tau, dim=0)  # (n,)
            src = torch.nn.functional.normalize(x, dim=1) if self.normalize else x
            agg = (w[:, None] * src).sum(dim=0)
            if self.normalize:
                # Rescale to a *robust* target magnitude. Using the mean norm was a bug: under
                # a norm-inflating attack (e.g. sign-flip with λ≫1) the byzantine deltas blow
                # up the mean, so the aggregate magnitude grows every round until it diverges
                # (NaN at f=2). The median norm ignores those outliers and keeps the outer
                # step at the honest scale.
                target_norm = x.norm(dim=1).median()
                agg = agg / agg.norm().clamp_min(1e-8) * target_norm

        self.prev_agg = agg.detach().clone()
        return agg


_REGISTRY: dict[str, type] = {
    "mean": Mean,
    "trimmed_mean": TrimmedMean,
    "krum": Krum,
    "geometric_median": GeometricMedian,
    "centered_clipping": CenteredClipping,
    "trust_weighted": TrustWeighted,
}


def build_aggregator(name: str, **kwargs) -> Aggregator:
    if name not in _REGISTRY:
        raise KeyError(f"unknown aggregator '{name}'. options: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)
