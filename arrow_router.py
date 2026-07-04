"""Arrow baseline (Ostapenko et al. 2024): route by cosine similarity to each
LoRA's top singular direction. Compared against the zero-shot and MLP routers."""

import numpy as np
import torch
from safetensors import safe_open

from MoE import (model, tokenizer, DEVICE, NUM_EXPERTS, num2adapter, exp_path,
                 _zero_shot_probs, mc_router, get_llama_embedding)
from eval_queries import QUERIES

NUM_LAYERS = model.config.num_hidden_layers


def prototype(A, B):
    # top right singular vector of dW = B@A, via a small r x r eigenproblem
    M = (B.T @ B) @ (A @ A.T)
    w, V = np.linalg.eig(M)
    z = np.real(V[:, int(np.argmax(np.real(w)))])
    v = A.T @ z
    return v / (np.linalg.norm(v) + 1e-8)


def load_prototypes(path):
    mods = {}
    with safe_open(f"{path}/adapter_model.safetensors", "pt") as f:
        for k in f.keys():
            if "lora_A" not in k and "lora_B" not in k:
                continue
            parts = k.split(".")
            layer = int(parts[parts.index("layers") + 1])
            target = parts[parts.index("self_attn") + 1]
            slot = mods.setdefault((layer, target), {})
            slot["A" if "lora_A" in k else "B"] = f.get_tensor(k).float().numpy()
    return {lt: prototype(m["A"], m["B"]) for lt, m in mods.items()}


PROTOS = [load_prototypes(exp_path[num2adapter[a]]) for a in range(NUM_EXPERTS)]


def layer_reps(query):
    inp = tokenizer(query, return_tensors="pt").to(DEVICE)
    mask = inp["attention_mask"].unsqueeze(-1).float()
    reps = {}
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
        for L in range(NUM_LAYERS):
            hn = model.model.layers[L].input_layernorm(out.hidden_states[L])
            reps[L] = ((hn.float() * mask).sum(1) / mask.sum(1))[0].cpu().numpy()
    return reps


def arrow_probs(query):
    reps = layer_reps(query)
    scores = np.zeros(NUM_EXPERTS)
    for a in range(NUM_EXPERTS):
        for (L, _), v in PROTOS[a].items():
            h = reps[L]
            scores[a] += abs(h @ v) / (np.linalg.norm(h) + 1e-8)
    e = np.exp(scores - scores.max())
    return e / e.sum()


def mlp_probs(query):
    mc_router.eval()
    with torch.no_grad():
        return torch.softmax(mc_router(get_llama_embedding(query)), -1).numpy()[0]


ROUTERS = {"zero-shot (ours)": _zero_shot_probs, "Arrow": arrow_probs,
           "MLP baseline": mlp_probs}


def tier_top1(P, tier):
    hit = tot = 0
    for i, (_, gt, t) in enumerate(QUERIES):
        if gt is None or t != tier or len(gt) != 1:
            continue
        hit += int(np.argmax(P[i]) == next(iter(gt))); tot += 1
    return hit / tot


def recall_at_k(P):
    r = []
    for i, (_, gt, _) in enumerate(QUERIES):
        if gt is None:
            continue
        topk = set(np.argsort(P[i])[::-1][:len(gt)].tolist())
        r.append(len(topk & gt) / len(gt))
    return float(np.mean(r))


def main():
    print(f"routing comparison | {len(QUERIES)} queries")
    print(f"  {'router':<22} {'tierA':>6} {'tierB':>6} {'recall@k':>9}")
    for name, fn in ROUTERS.items():
        P = np.stack([fn(q) for q, _, _ in QUERIES])
        print(f"  {name:<22} {tier_top1(P,'A')*100:>5.0f}% {tier_top1(P,'B')*100:>5.0f}%"
              f" {recall_at_k(P)*100:>8.0f}%")

    sims = []
    for lt in PROTOS[0]:
        V = np.stack([PROTOS[a][lt] for a in range(NUM_EXPERTS)])
        sims.append(np.abs(V @ V.T)[np.triu_indices(NUM_EXPERTS, 1)].mean())
    print(f"\n  Arrow prototype |cos| between adapters = {np.mean(sims):.2f} "
          f"(near 1 -> can't discriminate similar domains)")


if __name__ == "__main__":
    main()
