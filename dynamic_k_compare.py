# Dynamic-k selection rules on the same production router distributions:
#   nucleus / top-p  (cumulative-mass threshold; = Harder Tasks, ACL 2024)
#   relative-gap     (MINE: cut at the largest relative drop between sorted probs;
#                     derived independently, distinct from the paper's cumulative rule)
#   conformal APS    (calibrated threshold with a coverage guarantee)
# All three consume the same distribution and only decide how many experts (k).

import numpy as np
from conformal_router import get_probs, aps_qhat, aps_set


def nucleus(p, tau=0.90):
    order = np.argsort(p)[::-1]
    k = min(int(np.searchsorted(np.cumsum(p[order]), tau) + 1), len(p))
    return set(int(i) for i in order[:k])


def relative_gap(p):  # my own rule (see header)
    order = np.argsort(p)[::-1]
    ps = p[order]
    rel = (ps[:-1] - ps[1:]) / np.clip(ps[:-1], 1e-9, None)
    k = int(np.argmax(rel)) + 1
    return set(int(i) for i in order[:k])


def evaluate(rule, P, y):
    cov = size = exact = 0
    for i in range(len(y)):
        s = rule(P[i])
        cov += y[i] in s
        size += len(s)
        exact += (s == {int(y[i])})
    n = len(y)
    return cov / n, size / n, exact / n


def main():
    P_cal, y_cal, P_ev, y_ev = get_probs()
    qhat = aps_qhat(P_cal, y_cal, 0.10)
    rules = {
        "nucleus / top-p (Harder Tasks)": lambda p: nucleus(p, 0.90),
        "relative-gap (mine)":            relative_gap,
        "conformal APS":                  lambda p: set(aps_set(p, qhat)),
    }
    print(f"dynamic-k rules on {len(y_ev)} single-label eval queries (correct k=1)")
    print(f"  {'rule':<32} {'coverage':>9} {'avg k':>7} {'exact':>7}")
    for name, rule in rules.items():
        c, s, e = evaluate(rule, P_ev, y_ev)
        print(f"  {name:<32} {c*100:>8.0f}% {s:>7.2f} {e*100:>6.0f}%")


if __name__ == "__main__":
    main()
