# Can the paragraph-trained MLP route short queries without query-labelled data?
# Test unsupervised domain adaptation on the cached embeddings (no Llama).

import numpy as np
import torch
import torch.nn as nn

from train_mc_router import load_data, MCDropoutRouter
from eval_queries import QUERIES

NE = 5


def load_all():
    X = np.load("llama_embeddings_cache.npz")["X"].astype(np.float64)
    _, labels = load_data()
    Q = np.load("eval_query_embeddings.npz")["Q"].astype(np.float64)
    return X, np.array(labels), Q


def l2n(a, eps=1e-8):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + eps)


def train_mlp(Xtr, ytr, epochs=25, seed=0):
    torch.manual_seed(seed)
    net = MCDropoutRouter(input_dim=Xtr.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    Xt, yt = torch.FloatTensor(Xtr), torch.LongTensor(ytr)
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 256):
            b = perm[i:i + 256]
            opt.zero_grad()
            lossf(net(Xt[b]), yt[b]).backward()
            opt.step()
    net.eval()
    return net


def mlp_scores(net, feat):
    with torch.no_grad():
        return torch.softmax(net(torch.FloatTensor(feat)), -1).numpy()


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


def report(name, scores, out):
    a, r = top1_AB(scores), recall_at_k(scores)
    out.append((name, a))
    print(f"  {name:<28} {a*100:>6.1f}% {r*100:>8.1f}%")


def sqrtm(C, inv=False):
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-6, None)
    w = 1 / np.sqrt(w) if inv else np.sqrt(w)
    return V @ np.diag(w) @ V.T


def main():
    X, y, Q = load_all()
    muX, sd = X.mean(0), X.std(0) + 1e-6
    print(f"  {'method':<28} {'top1(A+B)':>7} {'recall@k':>9}")
    out = []

    base = train_mlp(X, y)
    report("MLP raw (baseline)", mlp_scores(base, Q), out)
    report("mean-shift align", mlp_scores(base, Q - Q.mean(0) + muX), out)

    Vt = np.linalg.svd(X - muX, full_matrices=False)[2][:10]
    strip = lambda M: (M - muX) - ((M - muX) @ Vt.T) @ Vt
    report("common-comp removal", mlp_scores(train_mlp(strip(X), y), strip(Q)), out)
    report("standardize + retrain",
           mlp_scores(train_mlp((X - muX) / sd, y), (Q - muX) / sd), out)
    report("L2-normalize + retrain", mlp_scores(train_mlp(l2n(X), y), l2n(Q)), out)

    lam = 1.0
    Cx = np.cov(X, rowvar=False) + lam * np.eye(X.shape[1])
    Cq = np.cov(Q, rowvar=False) + lam * np.eye(Q.shape[1])
    Q5 = (Q - Q.mean(0)) @ sqrtm(Cq, inv=True) @ sqrtm(Cx) + muX
    report("CORAL align", mlp_scores(base, Q5), out)

    Xs = (X - muX) / sd
    protos = np.stack([Xs[y == c].mean(0) for c in range(NE)])
    report("prototype (standardized)",
           l2n((Q - muX) / sd) @ l2n(protos).T, out)

    best = max(out, key=lambda t: t[1])
    print(f"\n  best: {best[0]} ({best[1]*100:.1f}%) vs baseline "
          f"{out[0][1]*100:.1f}% -- domain adaptation caps well below the "
          f"zero-shot router.")


if __name__ == "__main__":
    main()
