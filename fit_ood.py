# Fit the Mahalanobis OOD detector (Lee et al. 2018): class-conditional Gaussians
# with shared covariance on the paragraph embeddings. Threshold = 95th percentile
# of in-domain query distances. Saves ood_stats.npz.

import numpy as np
from train_mc_router import load_data
from eval_queries import QUERIES

NE = 5

X = np.load("llama_embeddings_cache.npz")["X"].astype(np.float64)
_, labels = load_data()
y = np.array(labels)

mus = np.stack([X[y == c].mean(0) for c in range(NE)])
Xc = X - mus[y]
cov = np.cov(Xc, rowvar=False)
cov += 1e-2 * np.eye(cov.shape[0]) * np.trace(cov) / cov.shape[0]  # shrinkage
prec = np.linalg.inv(cov)


def maha_min(Q):
    dmin = np.full(len(Q), np.inf)
    for c in range(NE):
        diff = Q - mus[c]
        dmin = np.minimum(dmin, np.einsum("ij,jk,ik->i", diff, prec, diff))
    return dmin


Q = np.load("eval_query_embeddings.npz")["Q"].astype(np.float64)
ind = np.array([gt is not None for (_, gt, _) in QUERIES])
dq = maha_min(Q)
thr = float(np.percentile(dq[ind], 95))

np.savez("ood_stats.npz", mus=mus, prec=prec, threshold=thr)
print(f"threshold={thr:.1f}  OOD caught={(dq[~ind] > thr).mean()*100:.0f}%  "
      f"false-abstain={(dq[ind] > thr).mean()*100:.0f}%")
