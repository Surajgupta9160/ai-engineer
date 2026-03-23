# 27 — LLM Inference Serving: vLLM, Quantization, and Production Optimization

This module covers the engineering of high-throughput, low-latency LLM inference — the layer between the model weights and production traffic that determines cost and latency.

**Prerequisites:** This module assumes you understand the transformer architecture (Module 09), specifically the KV cache (Module 09, Section 6) and attention math. It also assumes you know what model parameters are and how forward passes work (Module 05).

---

## 0. GPU Hardware Primer

Before diving into inference optimization, you need a mental model of the hardware.

```
GPU MEMORY HIERARCHY (fast → slow, small → large):
  Registers:      ~256KB, sub-nanosecond access — per-thread, tiny
  L2 Cache:       ~40MB, nanosecond access — shared across cores
  HBM (High-Bandwidth Memory): 40-80GB, ~3TB/s bandwidth — main GPU memory
  CPU RAM:        128-1000GB, ~100GB/s — much slower than HBM
  NVMe SSD:       several TB, ~10GB/s — slowest

WHY THIS MATTERS FOR LLM INFERENCE:
  All model weights and KV cache live in HBM.
  Each decode step: GPU must READ all attended KV tensors from HBM.
  At 3TB/s bandwidth, reading 140GB (LLaMA-3-70B weights) takes ~47ms.
  This is the FUNDAMENTAL BOTTLENECK for large model decode speed.
  Adding more compute cores doesn't help — we're bandwidth-limited.

COMPUTE vs MEMORY-BANDWIDTH BOTTLENECK:
  Compute-bound: the math takes longer than the memory reads
    → Happens during prefill (large batch, dense matrix multiply)
    → More CUDA cores helps; larger batch helps
  Memory-bandwidth-bound: reading weights takes longer than computing
    → Happens during single-request decode
    → More HBM bandwidth helps (H100 > A100 > V100)
    → Smaller model (quantization) helps: less data to read

TENSOR CORES:
  Specialized GPU circuits for matrix multiply (the core operation in attention + FFN)
  A100: 312 TFLOP/s (BF16) — 312 trillion operations per second
  H100: 989 TFLOP/s (BF16) — ~3× faster than A100
  But at decode: Tensor Core utilization often <10% because we're waiting for HBM reads
```

---

## 1. The Inference Bottleneck

```
LLM INFERENCE IS DIFFERENT FROM NORMAL DEEP LEARNING INFERENCE:

Normal model inference (image classifier):
  Input → single forward pass → output
  Fully parallelizable; batch size doesn't change computation structure.

LLM inference (autoregressive):
  Input → token_1 → token_2 → ... → token_N
  Sequential by nature; each new token requires one full forward pass
  over ALL previous tokens (unless KV-cached).

TWO PHASES:
  Prefill phase: process the entire prompt in one parallel forward pass
    - Compute-bound: benefits from large batch size
    - All prompt tokens processed simultaneously (like a regular forward pass)

  Decode phase: generate one token at a time, autoregressively
    - Memory-bandwidth-bound: reading all KV cache tensors from HBM each step
    - The bottleneck for most production systems
    - Latency ∝ model size (more weights to read per token)

Key metric: MFU (Model FLOPs Utilization) = actual TFLOPs / theoretical peak TFLOPs
  Typical decode MFU: 10-30% (memory bandwidth limited, not compute limited)
  Typical prefill MFU: 50-70% (compute limited, more efficient)
```

---

## 2. KV Cache — Mechanics and Memory

```
WHAT THE KV CACHE STORES:
  Every transformer layer computes K and V tensors for all tokens.
  For token i, K_i and V_i never change once computed.
  KV cache stores all K, V tensors from all previous tokens × all layers.

  Memory per token per layer:
    2 (K + V) × num_heads × head_dim × bytes_per_element
    For LLaMA-3-70B: 2 × 64 × 128 × 2 bytes = 32KB per token per layer
    Total per token (all 80 layers): 32KB × 80 = 2.56MB per token

  For 2K context with a batch of 100 requests:
    100 × 2048 × 2.56MB = ~525GB — this is the KV cache problem

NAIVE KV CACHE ALLOCATION (pre-vLLM):
  Allocate max_context memory per request up front (e.g., 4096 tokens × 2.56MB)
  Problem 1: most requests finish early → wasted memory
  Problem 2: fragmentation prevents multiple requests from sharing GPU
  Problem 3: can't dynamically serve mixed-length requests
```

---

## 3. vLLM and Paged Attention

```
PagedAttention (Kwon et al. 2023, vLLM) solves KV cache memory fragmentation
by borrowing the virtual memory / paging concept from OS design.

CORE INSIGHT: instead of one contiguous KV cache block per sequence,
allocate KV cache in small fixed-size pages (blocks) of 16 tokens.

PAGED ATTENTION MECHANISM:
  Physical KV cache: a pool of fixed-size blocks (pages), each holds K tokens
  Block table: per-sequence mapping of logical → physical block numbers

  Example (block size = 4 tokens):
    Sequence: "The cat sat on the mat" (6 tokens so far)
    Logical blocks: [0: "The cat sat on"], [1: "the mat __"]
    Physical blocks: [47: block 0], [12: block 1]  ← non-contiguous is OK!

  During attention:
    Standard attention: read contiguous memory at offset seq_start
    Paged attention: follow block table to gather scattered pages

  Benefits:
    ✓ Near-zero internal fragmentation (blocks are small, ~16 tokens)
    ✓ External fragmentation eliminated (reuse freed blocks immediately)
    ✓ Prefix caching: share KV pages for common prefixes across requests
    ✓ Parallel sampling: multiple outputs share the same prompt KV pages (copy-on-write)

MEMORY EFFICIENCY:
  Naive allocation: ~20% actual token utilization (memory reserved but unused)
  vLLM paged: >96% utilization
  Real-world result: 2-4× higher throughput at the same memory budget

vLLM ARCHITECTURE:
  LLMEngine: orchestrates scheduling, memory management, execution
  Scheduler: decides which requests to run given available KV blocks
  Worker: executes the GPU forward pass (one per GPU in tensor-parallel setup)
  Block Manager: allocates/frees/swaps KV cache blocks

  Continuous batching (iteration-level scheduling):
    Traditional: static batching — group N requests, wait for ALL to finish
    Continuous: after each forward pass step, check if any requests finished;
                if yes, insert new requests immediately into the next batch
    → GPU never idles waiting for slow requests; throughput increases 2-4×
```

---

## 4. Quantization — AWQ, GPTQ, GGUF

Quantization reduces model weight precision to decrease memory and increase inference speed.

```
WHY QUANTIZE:
  LLaMA-3-70B in BF16: 70B × 2 bytes = 140GB → requires 2× A100 80GB
  LLaMA-3-70B in int4: 70B × 0.5 bytes = 35GB → fits on 1× A100 80GB
  Speedup: decode is memory-bandwidth-bound; less data read = faster

NAIVE QUANTIZATION (round to nearest int8):
  weight_quantized = round(weight / scale)
  weight_dequantized = weight_quantized × scale
  Problem: large outlier weights cause high quantization error for most weights

GPTQ (Post-Training Quantization, Frantar et al. 2022):
  Layer-by-layer weight quantization using a calibration dataset.
  Algorithm: for each layer, iteratively quantize one weight at a time,
  compensating for the quantization error by adjusting remaining weights
  using the inverse Hessian of the loss w.r.t. weights.

  Result: 4-bit models with <1% quality loss on most benchmarks
  Cost: requires ~10 GPU-hours calibration for a 70B model
  Toolchain: AutoGPTQ, optimum.quanto

AWQ (Activation-Aware Weight Quantization, Lin et al. 2023):
  Key insight from GPTQ: only ~1% of channels are "salient" (have large activations).
  Protecting those channels during quantization preserves most model quality.

  Algorithm:
    1. Measure activation magnitudes per channel using calibration data
    2. Scale salient channels UP before quantization (protects them)
    3. Scale activations DOWN correspondingly during inference
    4. Quantize remaining weights with less precision loss

  Result: better quality than GPTQ at same bit width; faster quantization
  Toolchain: AutoAWQ, llm-compressor

GGUF (GPT-Generated Unified Format) — for CPU inference:
  Used by llama.cpp. A file format that stores quantized weights in various
  precision levels (Q4_K_M, Q5_K_M, Q8_0, etc.).

  "K" variants: mixed-precision quantization (some layers in higher precision)
  "M" suffix: medium mix (balances quality and size)

  Commonly used levels:
    Q4_K_M: 4-bit mixed; best quality-size tradeoff for CPU inference
    Q5_K_M: 5-bit; better quality, ~25% larger
    Q8_0: 8-bit; near-original quality, 2× FP16 size

  Use case: running LLMs locally on MacBooks (Apple Silicon) or CPU servers
  LM Studio, Ollama both use GGUF under the hood

QUANTIZATION COMPARISON:
┌─────────┬──────────────┬─────────────┬───────────────────┬────────────────┐
│ Method  │ Target HW    │ Quality     │ Speed (vs BF16)   │ Best for       │
├─────────┼──────────────┼─────────────┼───────────────────┼────────────────┤
│ GPTQ    │ GPU          │ Good        │ 2-3× decode       │ Batch inference│
│ AWQ     │ GPU          │ Better      │ 2-3× decode       │ Prod inference │
│ GGUF    │ CPU/Apple    │ Good (Q4KM) │ Varies            │ Local / edge   │
│ BF16    │ GPU          │ Baseline    │ Baseline          │ Training/eval  │
│ FP8     │ H100 only    │ ~BF16       │ ~1.5× decode      │ H100 clusters  │
└─────────┴──────────────┴─────────────┴───────────────────┴────────────────┘

INT4 MEMORY SAVINGS (LLaMA-3 family):
  Model   BF16 size  INT4 size  GPU needed (BF16)  GPU needed (INT4)
  8B      16GB       4GB        1× A100 40GB       Single A10G
  70B     140GB      35GB       2× A100 80GB       1× A100 80GB
  405B    810GB      202GB      11× A100 80GB      3× A100 80GB
```

---

## 5. Tensor Parallelism and Model Parallelism

```
WHEN A MODEL DOESN'T FIT ON ONE GPU:
  LLaMA-3-70B in BF16 = 140GB > single A100 80GB → must split across GPUs

TENSOR PARALLELISM (Megatron-LM style):
  Split individual weight matrices across GPUs along one dimension.

  For attention: split attention heads across GPUs
    GPU 0: heads 0-7 (1/4 of d_model)
    GPU 1: heads 8-15
    GPU 2: heads 16-23
    GPU 3: heads 24-31
  After each attention/FFN sublayer: all-reduce to sum results.

  Communication: one all-reduce per sublayer forward pass (expensive)
  Benefit: linear memory reduction; each GPU holds 1/N of each layer

PIPELINE PARALLELISM:
  Split model layers across GPUs (GPU 0: layers 0-19, GPU 1: layers 20-39...)
  Each GPU processes a "micro-batch" then passes activations to next GPU.

  Pipeline bubble: GPU k idles waiting for GPU k-1 to finish.
  Solution: micro-batching — run multiple micro-batches to fill the pipeline.
  Bubble fraction = (number of GPUs - 1) / micro_batches

COMBINED STRATEGIES (production 70B+ serving):
  Typically: Tensor Parallel × 2-4 GPUs for memory + KV cache sharing
  Then Data Parallel across nodes for throughput
  vLLM: built-in tensor parallel via --tensor-parallel-size flag

SERVING CONFIGURATION EXAMPLES:
  LLaMA-3-70B (BF16) production:
    2× A100 80GB: tensor parallel 2
    Average latency: ~15ms per token (decode)
    Throughput: ~500 tokens/sec at batch=1, ~4000 tok/sec at batch=32

  LLaMA-3-70B (AWQ int4):
    1× A100 80GB: fits entirely on one GPU
    Average latency: ~8ms per token (decode, faster due to less HBM reads)
    Throughput: ~1000 tokens/sec at batch=1
```

---

## 6. Continuous Batching and Scheduling

```
STATIC BATCHING (naive approach):
  Group N requests → run until ALL finish → accept new batch
  Problem: one slow request (long output) delays all others
  GPU utilization: often <50% due to waiting

CONTINUOUS BATCHING (iteration-level scheduling):
  After each token generation step, check if any requests are done.
  If done: free KV cache blocks, insert new requests into the batch.
  The batch composition changes at every step.

  Result:
    GPU utilization: 80-95% (near continuous)
    Throughput: 2-4× higher than static batching
    Used by: vLLM, TGI (text-generation-inference), TensorRT-LLM

CHUNKED PREFILL:
  Problem: a long prompt (4K tokens) takes many seconds to prefill,
  blocking all decode requests during that time (head-of-line blocking).

  Solution: split the long prompt into chunks (e.g., 256 tokens each);
  interleave prefill chunks with decode steps.
  → New requests don't block existing decode traffic
  → More consistent latency for all requests

PRIORITY SCHEDULING:
  Short requests: higher priority (finish fast, free memory)
  Long requests: lower priority or separate queue
  SLA-aware: premium users get guaranteed latency

SPECULATIVE DECODING IN PRODUCTION (recap):
  Draft model generates K tokens; target model verifies in parallel.
  In continuous batching: use draft model for all decode steps,
  verify in target model as a batched verification forward pass.
  Works best when requests are similar in content (e.g., code completion).
```

---

## 7. Inference Frameworks Comparison

```
┌───────────────────┬─────────────────┬──────────────────┬──────────────────┐
│ Framework         │ Backend         │ Key feature      │ Best for         │
├───────────────────┼─────────────────┼──────────────────┼──────────────────┤
│ vLLM              │ Python/CUDA     │ PagedAttention   │ OSS production   │
│ TGI (HuggingFace) │ Rust/Python     │ Continuous batch │ HF model zoo     │
│ TensorRT-LLM      │ TensorRT/C++    │ Fused kernels    │ NVIDIA production│
│ llama.cpp         │ C++/GGUF        │ CPU+Apple Silicon│ Local inference  │
│ Ollama            │ llama.cpp based │ Easy local setup │ Developer local  │
│ SGLang            │ Python/CUDA     │ RadixAttention   │ Structured gen   │
└───────────────────┴─────────────────┴──────────────────┴──────────────────┘

SGLang RadixAttention:
  Extension of PagedAttention: tree-structured KV sharing.
  When multiple requests share a common prefix (e.g., same system prompt +
  few-shot examples), they can share the EXACT same physical KV blocks.
  Benefit: massive memory savings for shared-prefix workloads (RAG, agents).

TensorRT-LLM specifics:
  Compiles the model into fused CUDA kernels for specific GPU architecture.
  Operator fusion: combines multiple operations (LayerNorm + linear +
  activation) into a single kernel → fewer memory reads → faster
  Requires re-compilation per model architecture and GPU type.
  2-3× faster than vLLM on equivalent hardware for throughput workloads.
```

---

## 8. Production Inference SLA Design

```
LATENCY COMPONENTS FOR A RAG CHATBOT REQUEST:
  1. Embed query:            ~10ms  (OpenAI API) / ~5ms (local)
  2. Vector search:          ~5-20ms
  3. LLM prefill (prompt):   ~100-500ms (depends on prompt length + model)
  4. LLM decode (output):    ~500ms-5s (depends on output length)
  Total: 1-6 seconds typical

OPTIMIZING EACH COMPONENT:
  Embedding: cache embeddings for repeated queries (Redis)
  Vector search: use ANN (HNSW); keep index in GPU/CPU RAM not disk
  Prefill: prefix caching for shared system prompts; chunked prefill for long prompts
  Decode: speculative decoding (2-3×); smaller model via routing

DEFINING SLAs:
  TTFT (Time to First Token): latency until user sees first output character
    → Matters most for user experience (perceived responsiveness)
    → Improved by: fast prefill, batching priority for short prompts

  TPOT (Time Per Output Token): average latency between generated tokens
    → Matters for streaming experience (smooth vs. chunky)
    → Improved by: speculative decoding, quantization, smaller models

  E2E latency p95: 95th percentile of full response time
    → Service SLA metric; include in alerts

COST PER REQUEST CALCULATION:
  GPU cost = (request TTFT + response_tokens × TPOT) × GPU_cost_per_second
  Amortized over batch size and concurrency

  Example: A100 cloud ~$3/hour = $0.00083/second
  50-token response at 100 tok/s decode: 0.5s × $0.00083 = $0.00042/request
  Equivalent to gpt-4o-mini at this scale → self-hosting only wins at scale
```

---

## Practice Questions

1. What is the difference between prefill and decode phases? Which is compute-bound vs memory-bound?
2. What problem does PagedAttention solve? How does it differ from naive KV cache allocation?
3. What is continuous batching and why does it improve throughput vs static batching?
4. Explain AWQ quantization. Why is it better than naive int4 rounding?
5. What is GGUF and when would you use it over AWQ?
6. What is tensor parallelism and how does it differ from pipeline parallelism?
7. Calculate the KV cache memory for LLaMA-3-70B with batch=32, context=4K.
8. What is speculative decoding and what is the acceptance rate requirement for it to be beneficial?
9. Why does quantization speed up decode more than prefill?
10. What is TTFT and TPOT? Which is more important for a streaming chat interface?
11. You're serving a model at 10M requests/day with p95 latency SLA of 800ms. Walk through your serving architecture choices.
12. What is chunked prefill and when does it help?
13. Compare vLLM vs TensorRT-LLM. When would you choose each?
14. What is SGLang RadixAttention and how does it improve on PagedAttention?
15. A user reports the first word appears after 3 seconds but then tokens stream fast. What metric is high and what would you fix?

---
*Previous: [26 — Scaling Laws](../26-scaling-laws/README.md)*
