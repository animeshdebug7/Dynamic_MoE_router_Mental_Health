"""
Conformal dynamic-k routing (split-conformal APS, Romano-Sesia-Candes 2020):
replace the heuristic nucleus threshold with a calibrated q_hat guaranteeing
P(correct_expert in set) >= 1 - alpha. Sweeps temperature and validates coverage
on held-out same-distribution splits before saving. Finding: does not beat the
heuristic here (the router is ~99% accurate, so top-1 already over-covers).
"""

import os
import numpy as np

from MoE import _zero_shot_probs, ZS_TEMPERATURE, nucleus_select
from eval_queries import QUERIES

CACHE = "conformal_cache.npz"
QHAT_FILE = "conformal_qhat.npz"

NAMES = {
    0: ["depression", "major depression", "clinical depression"],
    1: ["anxiety", "an anxiety disorder", "generalized anxiety disorder"],
    2: ["bipolar disorder", "bipolar", "manic depression"],
    3: ["ocd", "obsessive compulsive disorder", "obsessive-compulsive disorder"],
    4: ["schizophrenia", "schizophrenic disorder"],
}
TEMPLATES = [
    "what is {n}?", "what does {n} mean?", "can you explain {n}?",
    "give an overview of {n}", "how is {n} diagnosed?", "how is {n} treated?",
    "what treatments exist for {n}?", "what causes {n}?",
    "what are the warning signs of {n}?", "is {n} hereditary?",
    "how common is {n}?", "what medication helps {n}?",
    "describe {n}", "what are the early signs of {n}?",
]
SYMPTOMS = {
    0: ["nothing brings me joy anymore and I feel exhausted all the time",
        "I cry for no reason and feel like a burden",
        "I feel numb and hopeless about the future",
        "everything feels pointless and heavy",
        "I sleep all day and still feel drained"],
    1: ["my mind races with worst-case scenarios constantly",
        "I feel on edge and cannot relax at all",
        "my chest tightens whenever I have to speak up",
        "I overthink every small decision and feel dread",
        "I get shaky and sweaty before meetings"],
    2: ["some weeks I barely sleep and feel invincible then crash",
        "my mood flips between wild productivity and despair",
        "I go on spending sprees during my high phases",
        "I talk fast and start dozens of projects then lose all energy",
        "periods of euphoria followed by weeks of emptiness"],
    3: ["I have to arrange things perfectly or I feel intense dread",
        "unwanted disturbing thoughts intrude and I do rituals to neutralize them",
        "I count steps and tap objects a set number of times",
        "I re-read messages dozens of times fearing a mistake",
        "I wash surfaces repeatedly afraid of contamination"],
    4: ["I believe strangers are sending me secret messages",
        "I hear a voice narrating my actions",
        "my thoughts feel scattered and others say I make no sense",
        "I think the television is talking directly to me",
        "I feel detached from reality and suspicious of everyone"],
}


def build_calibration():
    qs, ys = [], []
    for c, names in NAMES.items():
        for t in TEMPLATES:
            for n in names:
                qs.append(t.format(n=n)); ys.append(c)
        for s in SYMPTOMS[c]:
            qs.append(s); ys.append(c)
    return qs, np.array(ys)


def single_label_eval():
    qs, ys = [], []
    for text, gt, _ in QUERIES:
        if gt is not None and len(gt) == 1:
            qs.append(text); ys.append(next(iter(gt)))
    return qs, np.array(ys)


def get_probs():
    cal_q, cal_y = build_calibration()
    ev_q, ev_y = single_label_eval()
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        if len(d["P_cal"]) == len(cal_q) and len(d["P_ev"]) == len(ev_q):
            return d["P_cal"], cal_y, d["P_ev"], ev_y
    print(f"  computing router softmax for {len(cal_q)} cal + {len(ev_q)} eval...",
          flush=True)
    P_cal = np.stack([_zero_shot_probs(q) for q in cal_q])
    P_ev = np.stack([_zero_shot_probs(q) for q in ev_q])
    np.savez(CACHE, P_cal=P_cal, P_ev=P_ev)
    return P_cal, cal_y, P_ev, ev_y


def retemp(P, T_target, T_cache=ZS_TEMPERATURE):
    """Re-temperature a cached softmax (cached at T_cache) to T_target."""
    Q = np.power(np.clip(P, 1e-12, 1.0), T_cache / T_target)
    return Q / Q.sum(axis=1, keepdims=True)


def aps_scores(P, y):
    return np.array([P[i][P[i] >= P[i][y[i]]].sum() for i in range(len(y))])


def aps_qhat(P, y, alpha):
    s = aps_scores(P, y)
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return 1.0 if k > n else float(np.sort(s)[k - 1])


def aps_set(p, qhat):
    order = np.argsort(p)[::-1]
    cum, chosen = 0.0, []
    for j in order:
        chosen.append(int(j)); cum += p[j]
        if cum >= qhat:
            break
    return chosen


def coverage_size(P, y, qhat):
    cov = [y[i] in aps_set(P[i], qhat) for i in range(len(y))]
    size = [len(aps_set(P[i], qhat)) for i in range(len(y))]
    return np.mean(cov), np.mean(size)


def rank_of_true(P, y):
    return np.array([int(np.where(np.argsort(P[i])[::-1] == y[i])[0][0]) + 1
                     for i in range(len(y))])


# randomized APS gives exact (non-conservative) coverage
def aps_scores_rand(P, y, rng):
    s = np.empty(len(y))
    for i in range(len(y)):
        p = P[i]; py = p[y[i]]
        s[i] = p[p > py].sum() + rng.uniform() * py
    return s


def aps_qhat_rand(P, y, alpha, rng):
    s = aps_scores_rand(P, y, rng)
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return 1.0 if k > n else float(np.sort(s)[k - 1])


def aps_set_rand(p, qhat, u):
    order = np.argsort(p)[::-1]
    cum, chosen = 0.0, []
    for j in order:
        if cum + p[j] < qhat:
            chosen.append(int(j)); cum += p[j]
        else:
            if p[j] > 0 and u <= (qhat - cum) / p[j]:
                chosen.append(int(j))
            break
    return chosen if chosen else [int(order[0])]


def coverage_size_rand(P, y, qhat, rng):
    cov, size = [], []
    for i in range(len(y)):
        S = aps_set_rand(P[i], qhat, rng.uniform())
        cov.append(y[i] in S); size.append(len(S))
    return np.mean(cov), np.mean(size)


def main():
    print("=" * 70)
    print("CONFORMAL DYNAMIC-K ROUTING  (split-conformal APS)")
    print("=" * 70)
    P_cal, y_cal, P_ev, y_ev = get_probs()
    print(f"  calibration: {len(y_cal)} single-label queries {np.bincount(y_cal)}")
    print(f"  eval (A+B) : {len(y_ev)} single-label queries")

    ranks = rank_of_true(P_cal, y_cal)
    print(f"\n  router rank-of-true on calibration (T={ZS_TEMPERATURE}):")
    print(f"    top-1 acc = {(ranks==1).mean()*100:.1f}%   "
          f"rank dist = {np.bincount(ranks, minlength=6)[1:].tolist()}")
    print(f"  -> {(ranks>=4).mean()*100:.0f}% of queries put the true expert at "
          f"rank >=4  (this is what forces q_hat toward 1.0)")

    print("\n" + "-" * 70)
    print("TEMPERATURE SWEEP  (alpha=0.10, target coverage 0.90)")
    print(f"  {'T':>5} {'cal-top1':>9} | {'same-dist cov':>13} {'|set|':>6} | "
          f"{'eval cov':>9} {'|set|':>6} {'q_hat':>7}")
    rng = np.random.RandomState(0)
    n = len(y_cal)
    best = None
    for T in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
        Pc, Pe = retemp(P_cal, T), retemp(P_ev, T)
        acc = (rank_of_true(Pc, y_cal) == 1).mean()
        covs, sizes = [], []
        for _ in range(50):
            perm = rng.permutation(n); h = n // 2
            q = aps_qhat(Pc[perm[:h]], y_cal[perm[:h]], 0.10)
            c, s = coverage_size(Pc[perm[h:]], y_cal[perm[h:]], q)
            covs.append(c); sizes.append(s)
        qf = aps_qhat(Pc, y_cal, 0.10)
        ec, es = coverage_size(Pe, y_ev, qf)
        print(f"  {T:>5.1f} {acc*100:>8.1f}% | {np.mean(covs):>13.3f} "
              f"{np.mean(sizes):>6.2f} | {ec:>9.3f} {es:>6.2f} {qf:>7.3f}")
        # prefer smallest same-dist set that still covers >= target
        if np.mean(covs) >= 0.90 - 1e-9:
            cand = (np.mean(sizes), T, qf)
            if best is None or cand[0] < best[0]:
                best = cand

    print("\n" + "-" * 70)
    print("RANDOMIZED APS SWEEP  (exact coverage; alpha=0.10)")
    print(f"  {'T':>5} | {'same-dist cov':>13} {'|set|':>6} | {'eval cov':>9} {'|set|':>6}")
    best_r = None
    for T in (0.5, 1.0, 2.0, 3.0, 5.0):
        Pc, Pe = retemp(P_cal, T), retemp(P_ev, T)
        covs, sizes = [], []
        for _ in range(50):
            perm = rng.permutation(n); h = n // 2
            q = aps_qhat_rand(Pc[perm[:h]], y_cal[perm[:h]], 0.10, rng)
            c, s = coverage_size_rand(Pc[perm[h:]], y_cal[perm[h:]], q, rng)
            covs.append(c); sizes.append(s)
        qf = aps_qhat_rand(Pc, y_cal, 0.10, rng)
        ec, es = coverage_size_rand(Pe, y_ev, qf, rng)
        print(f"  {T:>5.1f} | {np.mean(covs):>13.3f} {np.mean(sizes):>6.2f} | "
              f"{ec:>9.3f} {es:>6.2f}")
        if np.mean(covs) >= 0.90 - 0.015:      # near target (randomization noise)
            cand = (np.mean(sizes), T, qf)
            if best_r is None or cand[0] < best_r[0]:
                best_r = cand

    print("\n" + "-" * 70)
    print("BASELINE  heuristic nucleus TAU=0.90 (production T=0.5, no guarantee)")
    cov = [y_ev[i] in set(int(x) for x in nucleus_select(P_ev[i], 0.90)[0])
           for i in range(len(y_ev))]
    size = [len(nucleus_select(P_ev[i], 0.90)[0]) for i in range(len(y_ev))]
    heur_cov, heur_size = np.mean(cov), np.mean(size)
    print(f"  eval coverage={heur_cov:.3f}   avg |set|={heur_size:.2f}")

    print("\n" + "=" * 70)
    print("VERDICT")
    if best_r is None:
        print("  Randomized conformal could not reach target coverage. NOT saving.")
    else:
        size_r, T_r, q_r = best_r
        print(f"  best randomized conformal: T={T_r}, q_hat={q_r:.3f}, "
              f"|set|={size_r:.2f} at ~90% coverage (with guarantee)")
        print(f"  heuristic nucleus         : |set|={heur_size:.2f} at "
              f"{heur_cov*100:.0f}% coverage (no guarantee)")
        # Save deterministic q_hat (reproducible) + randomized q_hat at same T.
        q_det = aps_qhat(retemp(P_cal, T_r), y_cal, 0.10)
        np.savez(QHAT_FILE, qhat_det=q_det, qhat_rand=q_r, temperature=T_r,
                 alpha=0.10)
        print(f"  saved -> {QHAT_FILE}  (T={T_r}, det q_hat={q_det:.3f}, "
              f"rand q_hat={q_r:.3f})")
        # matches the heuristic + adds a guarantee, but the guarantee is loose:
        # a ~99%-accurate router already over-covers with top-1.


if __name__ == "__main__":
    main()
