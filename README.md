# Dynamic Mixture-of-Experts Routing over LoRA Adapters

An inference-time Mixture-of-Experts that decides **how many** of five domain
LoRA adapters to activate per query — not a fixed top-k — using the entropy of a
zero-shot LLM router's distribution, with an out-of-distribution abstention gate.
Base model: Llama-3.2-1B. Domain: five mental-health topics (depression,
anxiety, bipolar, OCD, schizophrenia).

The point of the project is **the routing + evaluation methodology**, not the
generated text. The core methods are established (see *Prior work*); the
contribution is an evaluation-driven design and a set of honest ablations.

---

## Method

Two cheap forward passes per query, no gate training and no task-specific
routing dataset:

1. **Routing (pass 1).** A 5-shot prompt asks the base model to classify the
   query's topic; we read the next-token logits for the five disorder words and
   apply a calibrated temperature → a routing distribution `p`.
2. **Dynamic-k.** Activate the smallest expert set whose cumulative mass ≥ τ
   (nucleus / top-p). Peaked `p` → k=1; spread `p` → k grows. The predictive
   entropy `H(p)` is the confidence signal.
3. **OOD gate (pass 2).** Mahalanobis distance to class-conditional Gaussians
   fit on the base model's embeddings; abstain (fall back to the base model)
   when the query is far from all experts.

Selected adapters are then blended by their routing weights
(`peft.add_weighted_adapter`) for generation — see `moe_demo.py`.

```
python3 moe_demo.py "what is ocd, bipolar disorder and anxiety?"
```

---

## Results (250 queries, 50 per tier, bootstrap 95% CIs)

| Router | in-domain exact (95% CI) | recall | OOD-abstain |
|---|---|---|---|
| MLP baseline (trained) | 10.0% [6.0, 14.5] | 66.8% | 0% |
| **Zero-shot (this work)** | **49.5% [42.5, 56.5]** | 76.2% | 72% |

The confidence intervals do not overlap → the improvement is statistically
significant. Per-tier (production router):

| Tier | description | exact | recall | avg k |
|---|---|---|---|---|
| A | single disorder, **named** | 92% | 98% | 1.06 |
| B | single disorder, **symptom-only** | 44% | 78% | 1.52 |
| C | explicit multi-disorder | 40% | 69% | 1.96 |
| D | indirect comorbidity | 22% | 60% | 1.78 |
| E | out-of-distribution | — | — | abstain 72% |

Dynamic-k behaves as intended (k grows 1.06 → 1.96 with the number of
disorders), and routing entropy increases monotonically with the number of
ground-truth disorders.

### Routing baselines (top-1 accuracy, same 250 queries)

| Router | tier A | tier B | recall@k |
|---|---|---|---|
| **Zero-shot (this work)** | **100%** | **82%** | **82%** |
| Arrow (SVD-prototype routing) | 20% | 20% | 30% |
| Trained MLP | 60% | 38% | 53% |

Arrow scores near chance here because the five adapters were trained on similar
text, so their prototypes are **0.88 cosine-similar** — weight-geometry routing
cannot separate closely-related domains. (This is a query-level reimplementation
of Arrow's routing signal, not PEFT's per-token Arrow; see *Limitations*.)

### Ablations (what did **not** work — kept as honest negatives)

- **Unsupervised domain adaptation** to rescue the paragraph-trained MLP
  (CORAL, mean-shift, standardize, L2-norm, common-component removal,
  prototypes) capped at ~60% top-1 — a linear realignment cannot recover signal
  that is weak in the embedding. (`experiment_adapt.py`)
- **MC-Dropout mutual information** for OOD detection: 0% caught. A **yes/no LLM
  gate**: 0% (a 1B model answers "yes" to everything). **Mahalanobis**: 86%.
- **Conformal dynamic-k** (split-conformal APS): correctly implemented and
  coverage-validated, but it does **not** beat the heuristic here — the router
  is ~99% accurate, so top-1 already over-covers and conformal over-selects
  (~1.9 vs ~1.1 experts). Available as `route(mode="conformal")`, not default.
  (`conformal_router.py`)
- **Dynamic-k selection rules** (`dynamic_k_compare.py`): nucleus/top-p (the
  cumulative-mass rule of *Harder Tasks Need More Experts*, ACL 2024), **my own
  relative-gap rule** (derived independently, before I found the paper — a
  distinct max-gap criterion, not the paper's cumulative threshold), and
  conformal, all on the same distributions. Nucleus is best (97% coverage,
  k=1.29); **my relative-gap rule** is competitive (97%, k=1.57); conformal
  over-covers (k=3.55). Nucleus is the default.

---

## Dynamic-k selection: an original rule

The number of active experts is set by a *dynamic-k rule* applied to the routing
distribution. The common choice (and the production default here) is a
cumulative-probability threshold — top-p / nucleus — as used by *Harder Tasks
Need More Experts* (ACL 2024).

This repo also introduces a **parameter-free rule I derived independently**: cut
at the **largest relative gap** between consecutive sorted expert probabilities,

```
k = argmax_i  (p_i - p_{i+1}) / p_i
```

It needs **no threshold to tune**. On the 250-query benchmark it **matches the
cumulative rule's coverage (97%)** at a slightly larger average set size
(k = 1.57 vs 1.29); see `dynamic_k_compare.py`. It is a *distinct* criterion from
the paper's cumulative threshold — not a re-implementation — and to my knowledge
has not been applied to MoE / LoRA expert-count selection in this form.

## Limitations (read this)

- **Scale is small:** a 1B base model, five adapters, one narrow domain.
- **The evaluation is synthetic and self-labelled.** Queries and ground-truth
  labels were generated by the author; there is no external benchmark. Tier A
  overlaps the method's mechanism (the disorder name is in the query), so **tier
  B (44%) is the more honest capability measure** than tier A.
- **The MLP baseline is distribution-mismatched** (trained on paragraphs, tested
  on queries). The 10%→49.5% gap is best read as *"a naive baseline breaks under
  distribution shift; the zero-shot design does not"* — a process result, not a
  claim that the method is intrinsically 5× better.
- **No methodological novelty.** Dynamic-k-by-confidence and training-free LoRA
  routing are established (see below).
- **Multi-disorder labels are subjective**, so C/D use recall@k rather than
  exact-set as the primary metric.

---

## Prior work

- Huang et al., *Harder Tasks Need More Experts: Dynamic Routing in MoE Models*,
  ACL 2024 — dynamic expert count by cumulative-confidence threshold (the same
  idea family as the nucleus dynamic-k here).
- Ostapenko et al., *Towards Modular LLMs by Building and Reusing a Library of
  LoRAs*, 2024 — **Arrow**, training-free LoRA routing (baseline here).
- Lee et al., 2018 — Mahalanobis OOD detection. Romano/Sesia/Candès, 2020 —
  conformal APS prediction sets.

---

## Repository

| File | Purpose |
|---|---|
| `MoE.py` | production zero-shot router + MLP baseline + conformal mode |
| `moe_demo.py` | interactive demo: query → adapters activated → generated answer |
| `eval_queries.py` | 250-query, 5-tier evaluation set (reproducible generator) |
| `eval_routing.py` | main evaluation, production vs baseline, bootstrap CIs |
| `arrow_router.py` | Arrow baseline + routing comparison |
| `conformal_router.py` | split-conformal APS calibration + coverage validation |
| `dynamic_k_compare.py` | compares dynamic-k rules: nucleus / relative-gap / conformal |
| `experiment_adapt.py` | unsupervised domain-adaptation ablations |
| `diagnose_router.py` | router / OOD-detector diagnostics |
| `fit_ood.py` | fits the Mahalanobis OOD detector |
| `train_mc_router.py` | trains the MLP baseline |
| `Adapters.ipynb` | trains the 5 domain LoRA adapters |

```
python3 train_mc_router.py     # baseline MLP + cached embeddings
python3 fit_ood.py             # Mahalanobis OOD stats
python3 eval_routing.py        # main results (250 queries, CIs)
python3 arrow_router.py        # routing comparison vs Arrow / MLP
```
