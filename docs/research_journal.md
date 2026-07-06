# Research journal

Dated technical notes on what was built and what ran.
Rule: **a non-reproducible result does not exist** — every entry names seed, config, commit.

---

## 2026-07-03 — repo scaffolded

Full codebase skeleton for all phases:

- `model.py` — from-scratch decoder Transformer (RMSNorm, RoPE, hand-written causal MHA,
  SwiGLU, weight tying). Tests cover shapes, attention-vs-reference equivalence (atol
  1e-4), causality (future tokens don't leak), RoPE norm preservation, tiny-batch overfit.
- `optim.py` — AdamW from scratch (decoupled weight decay) + cosine-with-warmup. Reused as
  the DiLoCo inner optimizer.
- `diloco.py` — sequential worker simulation, Δ_k = θ_before − θ_after, outer Nesterov,
  analytic communication accounting; plus `run_synchronous` (the reference-high baseline).
- `aggregators.py` — the single `aggregate(deltas) -> Tensor` interface: mean / trimmed /
  krum / geometric_median / trust_weighted.
- `attacks.py` — sign_flip, scaled_noise, targeted_drift (inner poisoning via label perm).

**Smoke checks (CPU, tiny scale — NOT results):**
- Phase 1: loss 5.58 → 0.01 in 150 steps; samples reproduce the training text. ✔
- DiLoCo baseline (mean, f=0): val ppl → ~1.03. ✔
- Attack (mean, sign_flip λ=10, f=1): val ppl → ~4261 — a single of 8 workers destroys the
  run. The plot-#1 story ("one liar in eight kills pretraining") already appears in
  miniature. ✔
- 23/23 unit + integration tests pass.

## 2026-07-03 — synchronous baseline + paper draft

- **Synchronous baseline** (`run_synchronous`): trains one model on the pooled corpus for the
  same token budget as DiLoCo but "communicates" every step — the reference-high curve for
  the perplexity-vs-communication plot. Wired into `reproduce_phase2_baseline.sh` + a `comm`
  plot mode. Verified end-to-end at tiny scale (sync ppl ~1.04, high comm).
- **Paper** `paper/rodiloco.tex` — full 6-section arXiv-style draft; compiles clean (5 pp
  with `article`, → ~8 with figures). Every number is a `\todo{}` mapped to a run output.
  Bibliography resolves (14 refs, bibtex clean).
- **Novelty positioning** captured in `docs/related_work.md`: the intersection
  "Byzantine-robust DiLoCo outer optimizer" is unpublished; nearest neighbours (Decoupled
  DiLoCo = failures not adversaries; DeMoA = federated not DiLoCo; outer-optimizer local SGD
  = no adversaries). Framing: *resilient (failures) ≠ robust (adversaries)*.

Next (needs compute): real runs on TinyStories / FineWeb-Edu subset at 10–20 M params →
fill the `\todo{}`s → drop PDF figures into `paper/figures/` → uncomment `\includegraphics`.

---

## 2026-07-06 — extended the science

Broadened the attack/defense space beyond the crude baselines:

- **Centered clipping** aggregator (`centered_clipping`) — Karimireddy et al. 2021; iterative
  clipping around a running center that persists across rounds (momentum-aware). Fifth
  baseline; a natural fit for the outer-Nesterov step.
- **Omniscient adaptive attacks** — `alie` (Baruch et al. 2019) and `min_max`
  (Shejwalkar & Houmansadr 2021). The colluding adversaries read the honest Δ's and emit one
  crafted vector inside the honest cloud (min_max verified to stay within the honest
  diameter). These evade distance/clustering defenses by construction.
- **Trust-weighted ablation harness** (`scripts/ablate_trust.py`) — sweeps β × τ × normalize,
  reporting perplexity-under-attack and the f=0 tax; writes a ranked CSV.

Smoke observation (trivial corpus, tiny scale — illustrative, not a result): under min_max at
f=2/8, **Krum underperforms the naive mean** (ppl 1.33 vs 1.04), while centered_clipping and
trust_weighted match the mean. This is the expected small-n Krum + adaptive-attack story; the
real-data runs will quantify it. Ablation confirmed the near-zero tax (+0.02) at smoke scale.

Tests now 27/27. Paper updated (background, threat model, defenses, results table) and
recompiles clean with the three new citations.

## 2026-07-06 — first full result set (P4 milestone)

Ran the full reproduce → attack → defend campaign on a free T4 (Kaggle) via the one-click
notebook. Config: ~1.8M-param char model (192/4L), TinyStories, 8 workers, H=50, 15 outer
rounds, sign-flip λ=10, seed 0, fp16.

**Final perplexity:**

| aggregator | f=0 | f=1 | f=2 |
|---|---|---|---|
| mean | 3.37 | diverged | diverged |
| trimmed_mean | 3.37 | 3.53 | diverged |
| krum | 3.99 | 3.98 | 4.08 |
| geometric_median | 3.35 | 3.45 | 3.52 |
| centered_clipping | 3.36 | 3.44 | 3.39 |
| trust_weighted | 3.30 | 3.33 | 3.27 |

- Plot #1 (fragility): one liar of 8 diverges the mean — confirmed.
- Plots #2/#3: classical defenses transfer partially; trimmed-mean dies at f=2 (trim<f); Krum
  highest tax (+18%); trust_weighted cheapest AND flattest — wins on both axes.
- Two fixes were decisive for trust_weighted, each from an observed failure: robust
  (geometric-median) cold-start [[trust-weighted-cold-start]] and median-norm rescale (moved
  f=2 from NaN to 3.27).

**Perf note:** the GPU run was launch-overhead-bound (~10 steps/s). Fixes: `_foreach`-batched
AdamW, fp16 mixed precision, branchless grad clipping. Reproduction hardened into
`notebooks/rodiloco_kaggle.ipynb` with hard asserts on GPU + dataset presence.

Results written into `README.md` (single canonical README) and `paper/rodiloco.tex`.

Next: seed replication (SEED=1,2 → mean±std); `min_max` as the main stressor; then submit.

## Template for future entries

```
## YYYY-MM-DD — <phase / milestone>
- what changed / what ran (config, seed, commit)
- result (number + plot path)
- decision / next
```
