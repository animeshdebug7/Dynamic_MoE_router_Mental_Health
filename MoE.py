import os
import numpy as np
import torch
import torch.nn.functional as F
from colorama import Fore
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_mc_router import MCDropoutRouter

BASE = "/Users/animeshsingh/Desktop/Models/Llama 3.2 1b"
OOD_STATS = "ood_stats.npz"
CONFORMAL_STATS = "conformal_qhat.npz"

TAU = 0.90
ZS_TEMPERATURE = 0.50
N_MC_SAMPLES = 30
NUM_EXPERTS = 5

num2adapter = {0: "depression", 1: "anxiety", 2: "bipolar", 3: "ocd", 4: "schiz"}
_root = "/Users/animeshsingh/Desktop/Projects/MoE_exp/adapters"
exp_path = {name: f"{_root}/{name}" for name in num2adapter.values()}

model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16,
                                             device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(BASE)
tokenizer.pad_token = tokenizer.eos_token
DEVICE = next(model.parameters()).device


def entropy(p, axis=-1, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p), axis=axis)


def nucleus_select(p, tau=TAU):
    order = np.argsort(p)[::-1]
    k = min(int(np.searchsorted(np.cumsum(p[order]), tau) + 1), len(p))
    chosen = order[:k]
    return chosen, p[chosen] / p[chosen].sum()


# 5-shot prompt: base model scores the disorder words after "Topic:"
_WORDS = {0: " depression", 1: " anxiety", 2: " bipolar",
          3: " OCD", 4: " schizophrenia"}
_TOK = {k: tokenizer.encode(w, add_special_tokens=False)[0] for k, w in _WORDS.items()}
_PROMPT = (
    "Classify the mental health topic of the text into exactly one of: "
    "depression, anxiety, bipolar, OCD, schizophrenia.\n"
    'Text: "I feel restless, my heart pounds, and I fear the worst"\nTopic: anxiety\n'
    'Text: "I feel empty, tired, and have lost interest in life"\nTopic: depression\n'
    'Text: "I repeat rituals to feel safe and get intrusive urges"\nTopic: OCD\n'
    'Text: "my energy cycles between euphoric highs and deep lows"\nTopic: bipolar\n'
    'Text: "I see and hear things others cannot and feel watched"\nTopic: schizophrenia\n'
    'Text: "{q}"\nTopic:'
)

_ood_mus = _ood_prec = None
OOD_THR = np.inf
if os.path.exists(OOD_STATS):
    d = np.load(OOD_STATS)
    _ood_mus, _ood_prec, OOD_THR = d["mus"], d["prec"], float(d["threshold"])

_conf_qhat = _conf_t = None
if os.path.exists(CONFORMAL_STATS):
    c = np.load(CONFORMAL_STATS)
    _conf_qhat, _conf_t = float(c["qhat_det"]), float(c["temperature"])


def _last_logits(prompt):
    inp = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        return model(**inp).logits[0, -1].float().cpu().numpy()


def _zero_shot_probs(query, temperature=ZS_TEMPERATURE):
    logits = _last_logits(_PROMPT.format(q=query))
    z = np.array([logits[_TOK[k]] for k in range(NUM_EXPERTS)]) / temperature
    p = np.exp(z - z.max())
    return p / p.sum()


def _ood_distance(query):
    if _ood_mus is None:
        return 0.0
    emb = get_llama_embedding(query).numpy()[0].astype(np.float64)
    return min(float((emb - _ood_mus[c]) @ _ood_prec @ (emb - _ood_mus[c]))
               for c in range(NUM_EXPERTS))


def route(query, tau=TAU, temperature=ZS_TEMPERATURE, mode="heuristic"):
    if mode == "conformal" and _conf_qhat is not None:
        p = _zero_shot_probs(query, _conf_t)
        chosen, weights = nucleus_select(p, _conf_qhat)
    else:
        p = _zero_shot_probs(query, temperature)
        chosen, weights = nucleus_select(p, tau)

    ood = _ood_distance(query)
    H = float(entropy(p))
    return {
        "text": query,
        "mean_p": p,
        "chosen_idx": [int(i) for i in chosen],
        "chosen": [num2adapter[int(i)] for i in chosen],
        "weights": weights,
        "k": len(chosen),
        "k_eff": float(np.exp(H)),
        "H_total": H,
        "ood_score": ood,
        "abstain": bool(ood > OOD_THR),
    }


def generate_moe_response(query, tau=TAU):
    r = route(query, tau=tau)
    chosen = set(r["chosen_idx"])
    print(f"\n{Fore.BLUE}Routing (T={ZS_TEMPERATURE}):{Fore.RESET}")
    for i in range(NUM_EXPERTS):
        mark = f"{Fore.GREEN}activate" if i in chosen else f"{Fore.RED}skip"
        print(f"  {num2adapter[i]:<12} {r['mean_p'][i]*100:>6.2f}%  {mark}{Fore.RESET}")
    print(f"\n  H={r['H_total']:.3f}  k_eff={r['k_eff']:.2f}  "
          f"OOD={r['ood_score']:.0f} (thr {OOD_THR:.0f})")
    if r["abstain"]:
        print(f"  {Fore.YELLOW}OOD -> abstain{Fore.RESET}")
        return [], np.array([])
    print(f"{Fore.MAGENTA}experts (k={r['k']}): {r['chosen']}  "
          f"weights {r['weights'].round(3)}{Fore.RESET}")
    return r["chosen"], r["weights"]


# Baseline: the earlier MLP + MC-Dropout router on paragraph embeddings.
mc_router = MCDropoutRouter()
mc_router.load_state_dict(torch.load("mc_router.pt", map_location="cpu",
                                     weights_only=True))


def get_llama_embedding(text, max_length=64):
    inp = tokenizer(text, truncation=True, padding=True, max_length=max_length,
                    return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    mask = inp["attention_mask"].unsqueeze(-1).float()
    return ((out.hidden_states[-1].float() * mask).sum(1) / mask.sum(1)).cpu()


def mc_route(x, n_samples=N_MC_SAMPLES):
    mc_router.train()  # dropout stays on for MC sampling
    with torch.no_grad():
        samples = np.stack([F.softmax(mc_router(x), -1).cpu().numpy()[0]
                            for _ in range(n_samples)])
    mean_p = samples.mean(0)
    epistemic = entropy(mean_p) - entropy(samples, -1).mean()  # BALD
    return {"mean_p": mean_p, "H_total": float(entropy(mean_p)),
            "I_epistemic": max(float(epistemic), 0.0)}


def route_mlp_baseline(query, tau=TAU):
    s = mc_route(get_llama_embedding(query))
    chosen, weights = nucleus_select(s["mean_p"], tau)
    return {"text": query, "mean_p": s["mean_p"],
            "chosen_idx": [int(i) for i in chosen],
            "chosen": [num2adapter[int(i)] for i in chosen], "weights": weights,
            "k": len(chosen), "H_total": s["H_total"],
            "I_epistemic": s["I_epistemic"], "k_eff": float(np.exp(s["H_total"])),
            "abstain": False}


if __name__ == "__main__":
    for q in ["what is depression?", "what is ocd?",
              "I check the door lock twenty times before I can leave",
              "what is ocd, bipolar disorder and anxiety?",
              "what is the capital of France?"]:
        print(f"\n{'=' * 56}\n{q}")
        generate_moe_response(q)
