# AI Engineer — Complete Study Notes

> A structured path from **zero to expert** in AI Engineering.
> Covers the full evolution of AI — from classical ML to modern LLMs and agentic systems.
> Each section builds on the previous. Read sequentially or jump to your level.

---

## Where to Start

| Your Background | Start Here |
|-----------------|-----------|
| Complete beginner (no ML/programming) | Section 01 → read all sequentially |
| Programmer, no ML background | Section 01 → can skim 01, deep-read 02–08 |
| ML practitioner, new to LLMs | Section 09, reference 02–08 as needed |
| LLM user, want to build with AI | Section 14 |
| Building AI apps, want production skills | Section 22 |

---

## Phase 1 — Foundations
### *Complete Beginner → ML-Literate*

> Build the intellectual bedrock. Understand why deep learning works before learning how LLMs work.
> Sections 01–08 give you the full evolutionary story from 1950 to the Transformer.

| # | Section | Skill Gained |
|---|---------|-------------|
| 01 | [What is AI Engineering](./01-what-is-ai-engineering/README.md) | Role, responsibilities, tech stack, career paths |
| 02 | [AI History](./02-ai-history/README.md) | Full timeline: Turing → AI winters → deep learning → ChatGPT |
| 03 | [Math Foundations](./03-math-foundations/README.md) | Linear algebra, calculus, probability, information theory |
| 04 | [Classical Machine Learning](./04-classical-ml/README.md) | Regression, decision trees, SVMs, k-means, PCA, evaluation |
| 05 | [Neural Networks](./05-neural-networks/README.md) | Perceptrons, backprop, activation functions, optimizers, regularization |
| 06 | [Deep Learning](./06-deep-learning/README.md) | CNNs, AlexNet → ResNets, skip connections, GANs, diffusion models |
| 07 | [Sequence Models](./07-sequence-models/README.md) | RNNs, LSTMs, GRUs, vanishing gradient, language modeling |
| 08 | [Pre-Transformer Era](./08-pre-transformer-era/README.md) | Word2Vec, GloVe, seq2seq, Bahdanau attention, ELMo |

---

## Phase 2 — Modern LLMs: Theory
### *ML-Literate → Understanding How LLMs Actually Work*

> Understand the Transformer architecture from first principles.
> Know what tokens, context windows, and sampling actually mean.
> Be able to explain how GPT-4 and Claude work at a technical level.

| # | Section | Skill Gained |
|---|---------|-------------|
| 09 | [How LLMs Work](./09-how-llms-work/README.md) | Transformers, self-attention, positional encoding, CoT, sampling |
| 10 | [LLM Terminology](./10-llm-terminology/README.md) | Tokens, context windows, temperature, top-p, open vs closed models |
| 11 | [Core Concepts](./11-core-concepts/README.md) | Inference, training, pretraining, RLHF, RAG/agents overview |
| 12 | [Pre-trained Models](./12-pretrained-models/README.md) | GPT-4o, Claude 3, Gemini, Llama 3, Mistral — compare capabilities |
| 13 | [Accessing Models](./13-accessing-models/README.md) | OpenAI API, Anthropic API, HuggingFace, Ollama, local inference |

---

## Phase 3 — Building with AI
### *Understanding → Shipping AI Applications*

> Build real AI-powered systems: RAG pipelines, agents, multimodal apps.
> Know every tool in the modern AI engineer's stack.

| # | Section | Skill Gained |
|---|---------|-------------|
| 14 | [Prompt Engineering](./14-prompt-engineering/README.md) | System prompts, few-shot, chain-of-thought, structured output |
| 15 | [Embeddings & Semantic Search](./15-embeddings/README.md) | Dense vectors, cosine similarity, embedding models, semantic search |
| 16 | [Vector Databases](./16-vector-databases/README.md) | Chroma, Pinecone, FAISS, Qdrant — choose, configure, query |
| 17 | [RAG](./17-rag/README.md) | Document ingestion, chunking, retrieval, generation, RAGAS evaluation |
| 18 | [Tools & Function Calling](./18-tools-function-calling/README.md) | Tool use, structured JSON output, OpenAI/Anthropic tool APIs |
| 19 | [AI Agents](./19-ai-agents/README.md) | ReAct pattern, agentic loops, multi-agent orchestration |
| 20 | [Multimodal AI](./20-multimodal-ai/README.md) | Vision, audio, video understanding, image generation |
| 21 | [Model Context Protocol (MCP)](./21-mcp/README.md) | MCP servers, clients, building and using tool integrations |

---

## Phase 4 — Expert Engineering
### *Shipping → Production-Grade AI Systems*

> Fine-tune your own models. Deploy, monitor, and optimize at scale.
> Handle safety, alignment, cost, and reliability in production.

| # | Section | Skill Gained |
|---|---------|-------------|
| 22 | [Fine-Tuning](./22-fine-tuning/README.md) | LoRA, QLoRA, PEFT, SFT, RLHF, DPO — when and how to fine-tune |
| 23 | [AI Safety & Ethics](./23-ai-safety-ethics/README.md) | Bias, prompt injection, content moderation, alignment, red-teaming |
| 24 | [LLMOps](./24-llmops/README.md) | Evaluation, observability, A/B testing, deployment, latency optimization |
| 25 | [AI App Architecture](./25-ai-app-architecture/README.md) | Production patterns, streaming, caching, cost optimization, scaling |

---

## The Full Learning Path

```
PHASE 1 — FOUNDATIONS (Beginner)
  01 What is AI Engineering
     ↓
  02 AI History (1950s → 2025)
     ↓
  03 Math Foundations
     ↓
  04 Classical Machine Learning
     ↓
  05 Neural Networks
     ↓
  06 Deep Learning (CNNs, ResNets)
     ↓
  07 Sequence Models (RNNs, LSTMs)
     ↓
  08 Pre-Transformer Era (Word2Vec, Attention)

PHASE 2 — MODERN LLMs (Intermediate)
  09 How LLMs Work (Transformers)
     ↓
  10 LLM Terminology
     ↓
  11 Core Concepts
     ↓
  12 Pre-trained Models
     ↓
  13 Accessing Models

PHASE 3 — BUILDING WITH AI (Intermediate → Advanced)
  14 Prompt Engineering
     ↓
  15 Embeddings & Semantic Search
     ↓
  16 Vector Databases
     ↓
  17 RAG
     ↓
  18 Tools & Function Calling
     ↓
  19 AI Agents
     ↓
  20 Multimodal AI
     ↓
  21 MCP

PHASE 4 — EXPERT ENGINEERING (Expert)
  22 Fine-Tuning
     ↓
  23 AI Safety & Ethics
     ↓
  24 LLMOps
     ↓
  25 AI App Architecture
```

---

## Certification Coverage

| Exam | Sections to Focus On |
|------|---------------------|
| OpenAI API Certification | 09, 10, 12, 13, 14, 18 |
| Google Cloud Professional ML | 03, 04, 05, 09, 15, 16, 17, 24 |
| AWS AI Practitioner | 02, 09, 11, 12, 20, 23, 24 |
| Hugging Face Course | 05, 07, 09, 13, 15, 22 |
| DeepLearning.AI Specializations | 03–08, 09, 17, 19, 14 |

---

## Key Cross-References

| If you're studying… | Also read… |
|--------------------|------------|
| 09 (Transformers) | 07 (RNNs, what transformers replaced) + 08 (attention origins) |
| 22 (Fine-Tuning / RLHF) | 04 (RL basics) + 02 (history: InstructGPT section) |
| 15 (Embeddings) | 08 (Word2Vec — origin of dense embeddings) |
| 17 (RAG) | 15 (embeddings) + 16 (vector databases) |
| 19 (Agents) | 18 (function calling) + 21 (MCP) |
| 20 (Diffusion / Image Gen) | 06 (GANs, diffusion fundamentals) |
| 22 (LoRA) | 03 (SVD — the math behind LoRA) |

---

## The AI Evolution Timeline

```
1943  First mathematical neuron (McCulloch-Pitts)
1956  Dartmouth Conference — AI as a field is born
1957  Perceptron — first trainable neural network
1969  Perceptron limitations proven → First AI Winter
1980s Expert systems boom
1986  Backpropagation rediscovered (Rumelhart, Hinton, Williams)
1987  Second AI Winter
1990s Statistical ML (SVMs, Random Forests, Gradient Boosting)
2006  Deep Belief Nets — first successful deep networks
2012  AlexNet wins ImageNet by 10pp → Deep Learning Revolution
2013  Word2Vec — dense semantic word representations
2014  Seq2Seq + Attention mechanism
2015  ResNet — superhuman image recognition
2017  "Attention Is All You Need" — Transformer paper
2018  BERT (bidirectional) + GPT (autoregressive)
2020  GPT-3 (175B) + Scaling Laws
2022  InstructGPT / RLHF + ChatGPT (Nov 30)
2023  GPT-4, Claude, Llama, Mistral — multi-model race
2024  Reasoning models (o1, R1), agents, MCP
2025  Agentic AI at scale
```

---

*25 sections | Beginner to Expert | Updated 2025*

## New Module

- [26 — Scaling Laws](26-scaling-laws/README.md) — Chinchilla, compute-optimal training, inference cost tradeoffs
