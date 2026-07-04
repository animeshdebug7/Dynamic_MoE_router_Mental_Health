# Compare the two routing modes (heuristic nucleus vs deterministic conformal)
# on the full eval set.

import numpy as np
from MoE import route
from eval_queries import QUERIES


def run(mode):
    rows = []
    for text, gt, tier in QUERIES:
        r = route(text, mode=mode)
        chosen = set(r["chosen_idx"])
        if gt is None:
            rows.append((tier, r["abstain"], np.nan, r["k"], r["abstain"]))
        else:
            hit = not r["abstain"]
            recall = len(chosen & gt) / len(gt) if hit else 0.0
            rows.append((tier, hit and chosen == gt, recall, r["k"], False))
    return rows


def summary(rows, label):
    print(f"\n{label}")
    for t, name in [("A", "A named"), ("B", "B symptom"), ("C", "C multi"),
                    ("D", "D complex"), ("E", "E OOD")]:
        sub = [r for r in rows if r[0] == t]
        if t == "E":
            print(f"  {name:<12} abstain={np.mean([r[4] for r in sub])*100:.0f}%")
        else:
            print(f"  {name:<12} exact={np.mean([r[1] for r in sub])*100:>3.0f}%  "
                  f"recall={np.mean([r[2] for r in sub])*100:>3.0f}%  "
                  f"k={np.mean([r[3] for r in sub]):.2f}")
    ind = [r for r in rows if r[0] != "E"]
    print(f"  in-domain exact={np.mean([r[1] for r in ind])*100:.1f}%  "
          f"recall={np.mean([r[2] for r in ind])*100:.1f}%")


if __name__ == "__main__":
    summary(run("heuristic"), "heuristic (nucleus TAU=0.90)")
    summary(run("conformal"), "conformal (deterministic APS)")
