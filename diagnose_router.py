# Why does the MLP router fail on short queries? Compare the MLP against
# nearest-prototype routing, and test Mahalanobis distance as an OOD detector.

import os
import numpy as np
import torch

from train_mc_router import load_data
from eval_queries import QUERIES
from MoE import get_llama_embedding, mc_router

QCACHE = "eval_query_embeddings.npz"
NE = 5


def load_train():
    X = np.load("llama_embeddings_cache.npz")["X"]
    _, labels = load_data()  # same seed/order as the cached embeddings
    y = np.array(labels)
    assert len(y) == len(X)
    return X, y


def load_query_embeddings():
    texts = [q[0] for q in QUERIES]
    if os.path.exists(QCACHE):
        Q = np.load(QCACHE)["Q"]
        if len(Q) == len(texts):
            return Q
    Q = np.stack([get_llama_embedding(t).numpy()[0] for t in texts])
    np.savez(QCACHE, Q=Q)
    return Q


def l2norm(a, axis=-1, eps=1e-8):
    return a / (np.linalg.norm(a, axis=axis, keepdims=True) + eps)


def scores_mlp(Q):
    mc_router.eval()
    with torch.no_grad():
        return torch.softmax(mc_router(torch.FloatTensor(Q)), -1).numpy()


def scores_prototype(Q, protos):
    return l2norm(Q) @ l2norm(protos).T


def mahalanobis_min(Q, X, y, shrink=1e-2):
    mus = np.stack([X[y == c].mean(0) for c in range(NE)])
    cov = np.cov(X - mus[y], rowvar=False)
    cov += shrink * np.eye(cov.shape[0]) * np.trace(cov) / cov.shape[0]
    prec = np.linalg.inv(cov)
    dmin = np.full(len(Q), np.inf)
    for c in range(NE):
        diff = Q - mus[c]
        dmin = np.minimum(dmin, np.einsum("ij,jk,ik->i", diff, prec, diff))
    return dmin


def top1_AB(scores):
    c = t = 0
    for i, (_, gt, tier) in enumerate(QUERIES):
        if gt is None or tier not in ("A", "B") or len(gt) != 1:
            continue
        c += int(np.argmax(scores[i]) == next(iter(gt))); t += 1
    return c / t


def recall_at_k(scores):
    r = []
    for i, (_, gt, _) in enumerate(QUERIES):
        if gt is None:
            continue
        topk = set(np.argsort(scores[i])[::-1][:len(gt)].tolist())
        r.append(len(topk & gt) / len(gt))
    return float(np.mean(r))


def main():
    X, y = load_train()
    Q = load_query_embeddings()
    protos = np.stack([X[y == c].mean(0) for c in range(NE)])

    print(f"  {'method':<24} {'top1(A+B)':>9} {'recall@k':>9}")
    for name, sc in [("MLP", scores_mlp(Q)),
                     ("prototype cosine", scores_prototype(Q, protos))]:
        print(f"  {name:<24} {top1_AB(sc)*100:>8.1f}% {recall_at_k(sc)*100:>8.1f}%")

    dmin = mahalanobis_min(Q, X, y)
    ind = np.array([gt is not None for (_, gt, _) in QUERIES])
    thr = np.percentile(dmin[ind], 95)
    print(f"\n  Mahalanobis OOD: caught {(dmin[~ind] > thr).mean()*100:.0f}% "
          f"at 95th-pct threshold (in-domain mean {dmin[ind].mean():.0f}, "
          f"OOD mean {dmin[~ind].mean():.0f})")


if __name__ == "__main__":
    main()
