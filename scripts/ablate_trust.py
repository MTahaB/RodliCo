"""Ablation harness for the trust-weighted outer optimizer.

Sweeps the method's hyperparameters and checks its two design claims:
  * **effectiveness** under attack (does it recover perplexity?),
  * **zero robustness tax** at f=0 (does it reduce to the mean when no one cheats?).

Grid: beta x tau x normalize, evaluated at the attacked f and at f=0. Emits a CSV
(`outputs/ablation/trust_ablation.csv`) and prints a ranked table.

Usage:
    python scripts/ablate_trust.py --config configs/defense_trustweighted.yaml
    python scripts/ablate_trust.py --quick          # tiny CPU settings, for a smoke pass
    python scripts/ablate_trust.py --attack min_max --f 2
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import yaml

from rodiloco.diloco import run_diloco

BETAS = [0.5, 0.7, 0.9, 0.99]
TAUS = [0.05, 0.1, 0.3]
NORMS = [True, False]

QUICK = dict(
    seq_len=32, batch_size=8, n_workers=4, H=8, outer_rounds=6, eval_every=3, eval_batches=5,
    model=dict(d_model=64, n_layers=2, n_heads=4, max_seq_len=32),
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/defense_trustweighted.yaml")
    ap.add_argument("--attack", default=None, help="override the attack (default: config's)")
    ap.add_argument("--f", type=int, default=1, help="byzantine fraction under attack")
    ap.add_argument("--quick", action="store_true", help="tiny CPU settings for a smoke pass")
    args = ap.parse_args()

    base = yaml.safe_load(open(args.config))
    base["aggregator"] = "trust_weighted"
    if args.attack:
        base["attack"] = args.attack
    if args.quick:
        base.update(QUICK)
    base["out_dir"] = None

    rows = []
    for beta, tau, norm in itertools.product(BETAS, TAUS, NORMS):
        kw = dict(beta=beta, tau=tau, normalize=norm)
        # (a) under attack
        atk = {**base, "n_byzantine": args.f, "aggregator_kwargs": kw}
        ppl_atk = run_diloco(atk)["final_ppl"]
        # (b) tax check: same settings, no attacker
        clean = {**base, "attack": "none", "n_byzantine": 0, "aggregator_kwargs": kw}
        ppl_tax = run_diloco(clean)["final_ppl"]
        rows.append(dict(beta=beta, tau=tau, normalize=norm, ppl_attacked=ppl_atk, ppl_f0=ppl_tax))
        print(f"beta={beta:<4} tau={tau:<4} norm={str(norm):<5} | "
              f"attacked ppl={ppl_atk:8.2f} | f0 ppl={ppl_tax:8.2f}")

    # reference: naive mean, to frame the tax (mean takes no kwargs — clear them)
    mean_cfg = {**base, "aggregator": "mean", "attack": "none", "n_byzantine": 0, "aggregator_kwargs": {}}
    mean_f0 = run_diloco(mean_cfg)["final_ppl"]
    print(f"\nreference mean f=0 ppl = {mean_f0:.2f}  (tax = trust_weighted f0 ppl - this)")

    rows.sort(key=lambda r: r["ppl_attacked"])
    out = Path("outputs/ablation")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "trust_ablation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["beta", "tau", "normalize", "ppl_attacked", "ppl_f0"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}/trust_ablation.csv")
    best = rows[0]
    print(f"best under attack: beta={best['beta']} tau={best['tau']} normalize={best['normalize']} "
          f"(ppl {best['ppl_attacked']:.2f}, tax {best['ppl_f0'] - mean_f0:+.2f})")


if __name__ == "__main__":
    main()
