import sys
import numpy as np
import torch
from colorama import Fore, Style
from peft import PeftModel

from MoE import model, tokenizer, route, exp_path, num2adapter, NUM_EXPERTS, OOD_THR

ORDER = ["depression", "anxiety", "bipolar", "ocd", "schiz"]
print("loading adapters...", flush=True)
peft_model = PeftModel.from_pretrained(model, exp_path["depression"],
                                       adapter_name="depression")
for name in ORDER[1:]:
    peft_model.load_adapter(exp_path[name], adapter_name=name)
peft_model.disable_adapter_layers()  # routing runs on the base model
peft_model.eval()


def select_adapter(labels, weights):
    if "moe_combo" in peft_model.peft_config:
        peft_model.delete_adapter("moe_combo")
    if len(labels) == 1:
        peft_model.set_adapter(labels[0])
        return labels[0]
    peft_model.add_weighted_adapter(list(labels), [float(w) for w in weights],
                                    "moe_combo", combination_type="linear")
    peft_model.set_adapter("moe_combo")
    return "moe_combo"


def generate(query, max_new_tokens=160):
    prompt = f"Answer the question in a short paragraph.\nQuestion: {query}\nAnswer:"
    inp = tokenizer(prompt, return_tensors="pt").to(next(peft_model.parameters()).device)
    with torch.no_grad():
        out = peft_model.generate(**inp, max_new_tokens=max_new_tokens,
                                  min_new_tokens=24, do_sample=True,
                                  temperature=0.7, top_p=0.9,
                                  repetition_penalty=1.2, no_repeat_ngram_size=3,
                                  pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inp["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def show_routing(r):
    chosen = set(r["chosen_idx"])
    wmap = dict(zip(r["chosen"], r["weights"]))
    print(f"\n{Fore.CYAN}routing{Style.RESET_ALL}")
    for i in range(NUM_EXPERTS):
        name = num2adapter[i]
        on = i in chosen
        w = f"{wmap.get(name, 0):.2f}" if on else "-"
        tag = f"{Fore.GREEN}ON{Style.RESET_ALL}" if on else f"{Fore.RED}off{Style.RESET_ALL}"
        print(f"  {name:<12} {r['mean_p'][i]*100:>6.1f}%  {w:>5}  {tag}")
    print(f"  k={r['k']}  H={r['H_total']:.2f}  OOD={r['ood_score']:.0f} "
          f"(thr {OOD_THR:.0f})")


def answer(query, max_new_tokens=160, verbose=True):
    peft_model.disable_adapter_layers()
    r = route(query)
    if verbose:
        print(f"\n{'=' * 52}\n{Fore.YELLOW}{query}{Style.RESET_ALL}")
        show_routing(r)
    try:
        if r["abstain"]:
            active = "base (none)"
        else:
            active = select_adapter(r["chosen"], r["weights"])
            peft_model.enable_adapter_layers()
        response = generate(query, max_new_tokens)
    finally:
        peft_model.disable_adapter_layers()

    if verbose:
        used = [] if r["abstain"] else r["chosen"]
        print(f"\n{Fore.GREEN}adapters:{Style.RESET_ALL} {used} ({active})")
        print(f"{Fore.GREEN}answer:{Style.RESET_ALL} {response}\n")
    return {"query": query, "adapters": [] if r["abstain"] else r["chosen"],
            "k": 0 if r["abstain"] else r["k"], "abstain": r["abstain"],
            "response": response}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        answer(" ".join(sys.argv[1:]))
    else:
        print('type a query (or "quit")')
        while True:
            try:
                q = input(f"{Fore.YELLOW}> {Style.RESET_ALL}").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() in {"quit", "exit", "q"}:
                break
            answer(q)
