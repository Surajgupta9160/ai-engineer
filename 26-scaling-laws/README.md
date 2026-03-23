# 26 — Scaling Laws: How to Train LLMs Optimally

Scaling laws are empirical relationships between compute budget, model size, dataset size, and model performance. Understanding them is essential for making training budget decisions and advising on model selection.

**Builds on:** Module 09 (training process — forward/backward passes, FLOPs per step, why training cost scales with parameters × tokens). Module 05 (backpropagation — why each training step costs 6 FLOPs per parameter per token: 2 for forward, 4 for backward). Module 22 (fine-tuning — scaling laws for fine-tuning differ from pre-training).

**Quick FLOPs refresher:** Each training step for one token against one parameter costs ~6 FLOPs: 2 for the forward pass (multiply + add) and 4 for backpropagation (gradient computation is ~2× the forward pass). So total training FLOPs ≈ 6 × parameters × tokens.

---

## 1. The Compute Formula

```
TRAINING COMPUTE (FLOPs):
  C ≈ 6 × N × D
  where:
    C = total FLOPs
    N = number of model parameters
    D = number of training tokens

  Why "6"?
    Each forward pass: ~2 FLOPs per weight per token (multiply + add)
    Each backward pass: ~4 FLOPs per weight per token
    Total per token per weight: 6 FLOPs

EXAMPLES:
  GPT-3 (175B):   C = 6 × 175B × 300B  = 3.15 × 10²³ FLOPs
  LLaMA-2 (70B):  C = 6 × 70B  × 2000B = 8.4  × 10²³ FLOPs
  GPT-4 (est 1T): C ≈ 6 × 1T   × 2000B = 1.2  × 10²⁵ FLOPs

HARDWARE REFERENCE:
  A100 (BF16):   312 TFLOP/s = 3.12 × 10¹⁴ FLOPs/second
  H100 (BF16):   989 TFLOP/s ≈ 10¹⁵ FLOPs/second
  MFU (model flop utilization): typically 30-50% in practice

  GPT-3 training time at 50% MFU on 1000 A100s:
    3.15 × 10²³ / (1000 × 3.12 × 10¹⁴ × 0.5) ≈ 2M seconds ≈ 23 days
```

---

## 2. Kaplan et al. Scaling Laws (OpenAI, 2020)

```
Original scaling laws paper: "Scaling Laws for Neural Language Models"

KEY FINDINGS:
  1. Performance scales as a power law with each resource:
     L(N) ∝ N^(-0.076)   (loss vs parameters, holding data constant)
     L(D) ∝ D^(-0.095)   (loss vs data, holding model constant)
     L(C) ∝ C^(-0.050)   (loss vs compute)

  2. Larger models are more sample-efficient:
     A 10B model needs fewer tokens to reach a given loss than a 1B model
     → For a fixed compute budget, use the LARGEST model you can

  KAPLAN CONCLUSION:
     Given fixed compute budget C:
       Optimal model size: N ∝ C^0.73
       Optimal token count: D ∝ C^0.27
     This says: scale parameters faster than data.

  WHAT GPT-3 DID (following Kaplan):
     175B parameters, 300B tokens
     N:D ratio = 175B/300B ≈ 0.58 parameters per token
```

---

## 3. Chinchilla Scaling Laws (DeepMind, 2022)

Hoffmann et al., "Training Compute-Optimal Large Language Models" — the paper that changed the industry.

```
WHAT CHINCHILLA FOUND:
  Kaplan's laws were wrong — they undertrained the data dimension.
  The optimal allocation is:
    N_optimal ∝ C^0.5   (scale parameters with sqrt of compute)
    D_optimal ∝ C^0.5   (scale data with sqrt of compute)
  This means: N_optimal ≈ D_optimal / 20
              or equivalently: D_optimal ≈ 20 × N_optimal

  "For compute-optimal training, every 10× increase in compute should
   produce a ~3.1× increase in model size AND a ~3.1× increase in tokens"

CHINCHILLA vs GPT-3 COMPARISON:
  GPT-3:       175B params, 300B tokens  (undertrained — too few tokens)
  Chinchilla:  70B params, 1.4T tokens   (same compute, smaller + more data)

  Result: Chinchilla (70B) OUTPERFORMED Gopher (280B) on almost all tasks.
  A 4x smaller model, trained with 4x more data, with the SAME compute budget.

THE 20:1 RULE:
  For compute-optimal training: train on ~20 tokens per parameter.
  70B model: train on ~1.4T tokens (= 70B × 20)
  7B model:  train on ~140B tokens (= 7B × 20)

  This is a starting point — actual optimal varies by data quality
  and task distribution. The "20" is for general-purpose LM.

GPT-3 WAS UNDERTRAINED:
  GPT-3: 175B params × 300B tokens
  Chinchilla-optimal for 175B: 175B × 20 = 3.5T tokens
  GPT-3 only trained on 300B / 3500B = 8.6% of the optimal token count.
  → GPT-3 could have been much smaller for the same performance,
    OR much better for the same parameter count.
```

---

## 4. Inference Cost Changes Everything

Chinchilla optimal is NOT production optimal.

```
THE INFERENCE ARGUMENT (LLaMA insight, Meta 2023):
  Chinchilla says: use a large model trained on few tokens
  Production reality: you run inference billions of times

  TOTAL COST = training cost + N_queries × inference_cost_per_query
  inference_cost ∝ N (model size)

  For high-traffic products, inference cost DOMINATES training cost.
  Optimal strategy: train a SMALLER model with MORE tokens than
  Chinchilla-optimal → lower inference cost per query.

  LLaMA-1 (Meta, 2023) explicitly used this insight:
    LLaMA-7B:  trained on 1T tokens  (vs Chinchilla-optimal 140B)
    LLaMA-13B: trained on 1T tokens  (vs Chinchilla-optimal 260B)
    LLaMA-65B: trained on 1.4T tokens (approximately Chinchilla-optimal)

    Result: LLaMA-7B matched GPT-3 (175B) on many benchmarks.
    A 25x smaller model reached the same quality — by training longer.

  LLaMA-2 (Meta, 2023):
    LLaMA-2-70B: trained on 2T tokens (vs Chinchilla-optimal 1.4T)
    Still undertrained for inference-optimal use case.

  LLaMA-3 (Meta, 2024):
    LLaMA-3-8B: trained on 15T tokens (vs Chinchilla-optimal 160B)
    That's 94× more tokens than Chinchilla-optimal for 8B params.
    Reasoning: for billions of inference queries, the extra training
    investment is massively ROI-positive.

PRACTICAL FRAMEWORK — choosing N and D for your budget:
  If you train once, deploy once (research): follow Chinchilla
  If you train once, deploy at scale (product): overtrain vs Chinchilla
  If you retrain frequently (fast-changing data): use Chinchilla or less
```

---

## 5. Data Quality > Data Quantity

```
THE QUALITY CEILING:
  Scaling laws assume IID data sampled from a fixed distribution.
  Real-world scaling is bounded by DATA QUALITY, not just quantity.

  After pre-training on all high-quality internet text:
  - More tokens requires lower-quality data
  - Diminishing returns set in before theoretical compute-optimal point
  - Fine-tuning on 1K high-quality examples often outperforms
    pre-training on 1M low-quality examples for specific tasks

SYNTHETIC DATA:
  Growing use of LLM-generated synthetic data to overcome quality ceilings.
  Phi-1.5 (Microsoft, 2023): 1.3B model trained on "textbook-quality"
  synthetic data, outperformed much larger models on reasoning benchmarks.

  Risk: model collapse — training on synthetic data generated by a model
  leads to degenerate outputs over multiple generations (Shumailov et al. 2023).
  Mitigation: always blend synthetic with real data.

TOKEN QUALITY vs QUANTITY HIERARCHY:
  1. Human expert-written text (best)
  2. High-quality web text (filtered Common Crawl)
  3. Synthetic data from frontier models
  4. Raw web text (unfiltered)
  5. Low-quality/repetitive text (can harm training)
```

---

## 6. Scaling Laws in Practice — Advisor's Guide

```
SCENARIO 1: "We have 10²³ FLOPs budget. How do we allocate?"
  Chinchilla-optimal: N ≈ sqrt(C/6) × 0.5 = ~130B params, 2T tokens
  Inference-optimal (high-traffic product):
    N ≈ 7-13B, D ≈ 2-5T tokens (smaller model, much more training)
  Decision gate: what is your expected query volume?
    If N_queries × inference_savings > training_cost_increase → go smaller

SCENARIO 2: "Our 7B model isn't good enough. Should we train a larger model?"
  First: check if you've reached Chinchilla-optimal token count.
  7B Chinchilla-optimal: 140B tokens.
  If you trained on <140B tokens: increase training data first (cheaper).
  If already Chinchilla-optimal: then scale model size.
  Rule: more data per parameter is often the cheapest path forward.

SCENARIO 3: "Should we use a 70B model or a 7B model in production?"
  7B: ~4GB memory (int4), fast inference (~50 tok/s on A100)
  70B: ~35GB memory (int4), slower (~5-10 tok/s on single A100)
  Quality difference: ~10-15% on benchmarks (varies by task)
  Cost difference: ~5-10x at inference time
  Decision: route 80% of easy queries to 7B, 20% hard to 70B.
            Combined quality ≈ 70B at 80% reduced cost.

SCENARIO 4: "How much data do we need to fine-tune?"
  Scaling laws for fine-tuning are different from pre-training.
  Empirical finding: quality matters much more than quantity.
  50-500 high-quality examples often sufficient for SFT.
  1K-10K examples for significant capability improvement.
  100K+ examples only needed for behavioral alignment (RLHF/DPO).

HALLMARKS OF A PRINCIPAL-LEVEL ANSWER ON SCALING:
  ✓ Distinguishes training-optimal from inference-optimal
  ✓ Knows the 20:1 Chinchilla rule
  ✓ Understands that GPT-3/early models were undertrained
  ✓ Knows LLaMA series operationalized "train small, train long"
  ✓ Considers data quality ceiling in real scenarios
  ✓ Can calculate rough FLOP counts using 6ND formula
```

---

## Practice Questions

1. What is the Chinchilla scaling law finding? How does it differ from Kaplan et al. (2020)?
2. You have a 10²³ FLOP budget. How many parameters and tokens are compute-optimal?
3. Why is inference-optimal different from training-optimal? Give a concrete example.
4. How did LLaMA-1 use Chinchilla insights to build a smaller but competitive model?
5. GPT-3 was trained with 175B params, 300B tokens. Was it Chinchilla-optimal? Why?
6. What is the approximate token count for Chinchilla-optimal training of a 70B model?
7. A company wants to retrain their LLM monthly on fresh data. Should they follow Chinchilla?
8. What is synthetic data and what is the "model collapse" risk?
9. At what point does data quality become the binding constraint, rather than quantity?
10. Given a fixed deployment budget, when does it make sense to route to a 7B vs 70B model?

---
*Previous: [25 — AI App Architecture](../25-ai-app-architecture/README.md)*
