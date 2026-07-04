import json
import numpy as np

from MoE import route, route_mlp_baseline
from eval_queries import QUERIES


def bootstrap_ci(vals, n=5000, seed=0):
    vals = np.asarray(vals, dtype=float)
    rng = np.random.RandomState(seed)
    means = vals[rng.randint(0, len(vals), size=(n, len(vals)))].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]) * 100)


def score_one(r, gt):
    chosen = set(r["chosen_idx"])
    if gt is None:                       # OOD -> correct means abstained
        return r["abstain"], float("nan"), float("nan")
    hit = not r["abstain"]
    recall = len(chosen & gt) / len(gt) if hit else 0.0
    precision = len(chosen & gt) / len(chosen) if (hit and chosen) else 0.0
    return (hit and chosen == gt), recall, precision


def run(router, ood_key):
    rows = []
    for text, gt, tier in QUERIES:
        r = router(text)
        exact, recall, precision = score_one(r, gt)
        rows.append({"tier": tier, "text": text, "gt": sorted(gt) if gt else None,
                     "chosen": r["chosen"], "k": r["k"], "H_total": r["H_total"],
                     "ood": r[ood_key], "abstain": r["abstain"],
                     "exact": bool(exact), "recall": recall, "precision": precision})
    return rows


def tier_table(rows, title):
    print(f"\n  {title}")
    print(f"  {'tier':<28} {'exact':>7} {'recall':>7} {'k':>5}")
    labels = {"A": "A simple (named)", "B": "B symptom (unnamed)",
              "C": "C multi-disorder", "D": "D complex/indirect",
              "E": "E out-of-distribution"}
    for t, lab in labels.items():
        sub = [r for r in rows if r["tier"] == t]
        if t == "E":
            print(f"  {lab:<28} {'':>7} {'':>7} {'':>5}   "
                  f"abstain={np.mean([r['abstain'] for r in sub])*100:.0f}%")
        else:
            print(f"  {lab:<28} {np.mean([r['exact'] for r in sub])*100:>6.0f}% "
                  f"{np.mean([r['recall'] for r in sub])*100:>6.0f}% "
                  f"{np.mean([r['k'] for r in sub]):>5.2f}")
    ind = [r for r in rows if r["tier"] != "E"]
    print(f"  in-domain exact={np.mean([r['exact'] for r in ind])*100:.1f}%  "
          f"recall={np.mean([r['recall'] for r in ind])*100:.1f}%  "
          f"precision={np.nanmean([r['precision'] for r in ind])*100:.1f}%")


def claims(rows):
    byN = {}
    for r in rows:
        if r["gt"] is not None:
            byN.setdefault(len(r["gt"]), []).append(r["H_total"])
    print("\n  entropy vs #disorders:")
    prev, mono = None, True
    for n in sorted(byN):
        h = np.mean(byN[n])
        print(f"    {n}: H={h:.3f} (n={len(byN[n])})")
        mono = mono and (prev is None or h >= prev - 1e-6)
        prev = h
    print(f"    monotonic increasing: {mono}")

    ind = [r["ood"] for r in rows if r["gt"] is not None]
    ood = [r["ood"] for r in rows if r["gt"] is None]
    print(f"\n  OOD score  in-domain max={np.max(ind):.0f}  OOD min={np.min(ood):.0f}"
          f"  separable={np.min(ood) > np.max(ind)}")


def main():
    print(f"routing eval | {len(QUERIES)} queries, 5 tiers")
    prod = run(route, "ood_score")
    base = run(route_mlp_baseline, "I_epistemic")

    def agg(rows):
        ind = [r for r in rows if r["tier"] != "E"]
        e = [r for r in rows if r["tier"] == "E"]
        exact = [r["exact"] for r in ind]
        return (np.mean(exact) * 100, bootstrap_ci(exact),
                np.mean([r["recall"] for r in ind]) * 100,
                np.mean([r["abstain"] for r in e]) * 100)

    for name, rows in [("MLP baseline", base), ("zero-shot", prod)]:
        e, (lo, hi), rec, ab = agg(rows)
        print(f"  {name:<14} exact={e:.1f}% [{lo:.1f}, {hi:.1f}]  "
              f"recall={rec:.1f}%  OOD-abstain={ab:.0f}%")

    tier_table(base, "baseline (trained MLP)")
    tier_table(prod, "production (zero-shot)")
    claims(prod)

    json.dump({"production": prod, "baseline": base}, open("eval_results.json", "w"),
              indent=2)


if __name__ == "__main__":
    main()
