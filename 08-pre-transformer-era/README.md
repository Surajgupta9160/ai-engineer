# 00f — Pre-Transformer Era: Word Embeddings, Seq2Seq & Attention

> **The direct precursors to LLMs.** This section covers the developments
> from 2013–2017 that created the conceptual building blocks of the Transformer:
> dense word representations, encoder-decoder models, and the attention mechanism
> that eventually became self-attention.

---

## Table of Contents

1. [From Symbolic to Distributed Representations](#1-symbolic-to-distributed)
2. [Word2Vec in Depth](#2-word2vec-in-depth)
3. [GloVe and FastText](#3-glove-and-fasttext)
4. [Sequence-to-Sequence Models](#4-sequence-to-sequence-models)
5. [The Attention Mechanism (Bahdanau, 2014)](#5-the-attention-mechanism)
6. [Transformer-XL and Memory Augmentation](#6-transformer-xl-and-memory)
7. [ELMo: Contextualized Embeddings (2018)](#7-elmo)
8. [The Road to Self-Attention](#8-road-to-self-attention)
9. [Recap: The Building Blocks](#9-building-blocks-recap)

---

## 1. Symbolic to Distributed Representations

### The Old Way: One-Hot Encoding

Before word embeddings, words were represented as one-hot vectors:

```
Vocabulary: ["cat", "dog", "apple", "orange"]
           → size 4

"cat"   = [1, 0, 0, 0]
"dog"   = [0, 1, 0, 0]
"apple" = [0, 0, 1, 0]
```

**Problems:**
1. Dimensionality: 100K word vocabulary → 100K-dimensional vectors
2. No similarity: cat and dog are as different as cat and spaceship
   - dot product of any two different one-hot vectors = 0
3. No generalization: if you see "the orange cat", model learns nothing useful about "the tabby cat"

### The Distributed Representation Hypothesis

Hinton (1986): represent words as dense, low-dimensional vectors where **meaning is distributed across dimensions**.

```
Ideal word vector space:
  "cat"  ≈ [0.2, 0.8, -0.1, 0.5, ...]
  "dog"  ≈ [0.3, 0.7, -0.2, 0.4, ...]  ← similar to cat
  "car"  ≈ [-0.5, -0.1, 0.9, -0.3, ...]  ← very different

Cosine similarity:
  sim("cat", "dog") ≈ 0.9  (high similarity)
  sim("cat", "car") ≈ 0.1  (low similarity)
```

This allows generalization: a model that learns about cats can immediately apply that knowledge to similar animals.

---

## 2. Word2Vec in Depth

### Mikolov et al., Google, 2013

Two prediction tasks, both train neural networks whose hidden layer weights become the word vectors.

### Architecture 1: Skip-gram

Given a target word, predict its context words.

```
Input: target word "jumped"
Context window (size 2): ["The", "fox", "over", "the"]

Training pairs:
  (jumped, The), (jumped, fox), (jumped, over), (jumped, the)

Network:
  input: one-hot target word
  → hidden: 300-dim embedding layer (the word vectors!)
  → output: softmax over vocabulary

Train to predict context words.
```

### Architecture 2: Continuous Bag of Words (CBOW)

Predict target word from context words (reverse of skip-gram).

```
Input: context ["The", "fox", "over", "the"]
Target: "jumped"

Average the embeddings of context words → predict target
```

CBOW is faster; skip-gram works better for rare words.

### Negative Sampling

The softmax over 100K words is expensive. Negative sampling approximates it:

```
Instead of:
  train to assign high probability to all context words

Do:
  for each positive pair (word, context):
    + train to increase similarity for this pair
    - train to decrease similarity for k random (negative) words
```

k = 5–20 negative samples works well. This makes training ~100x faster.

### Subsampling Frequent Words

Common words like "the", "a", "is" appear so frequently they add little information.
Randomly discard high-frequency words during training with probability:
```
P(discard) ∝ 1 - √(t / frequency)
```

This improves quality and speeds training.

### The Famous Analogies

Word2Vec's key result: **analogies emerge as vector arithmetic**:

```
vec("king") - vec("man") + vec("woman") ≈ vec("queen")
vec("Paris") - vec("France") + vec("Germany") ≈ vec("Berlin")
vec("walked") - vec("walk") + vec("swim") ≈ vec("swam")
```

This shows that relationships (gender, nationality, tense) are encoded as consistent vector directions.

**Why this happens**: the skip-gram objective causes words appearing in similar contexts to have similar vectors. "King" and "queen" appear in similar contexts except for gender-specific context words, so their vectors differ mainly along a "gender" dimension.

### Evaluating Word Embeddings

1. **Word similarity tasks**: rate word pairs for similarity (WordSim-353)
   - Compare cosine similarity of embeddings to human ratings

2. **Analogy tasks**: a:b :: c:d (king:queen :: man:?)
   - Evaluate using 3CosAdd: argmax cos(d, b-a+c)

3. **Downstream NLP tasks**: use embeddings as features, measure task performance

### Pre-trained Word Vectors

After training, the embedding matrix can be saved and reused:
```python
# Use pre-trained Word2Vec embeddings in a model
import gensim.downloader as api
wv = api.load('word2vec-google-news-300')

embedding = torch.tensor(wv['cat'])  # 300-dim vector
```

This became the standard practice: use someone else's pre-trained word vectors as initialization → significant improvement over random initialization.

---

## 3. GloVe and FastText

### GloVe (Global Vectors, Pennington et al., Stanford, 2014)

Word2Vec is a predictive model — learn vectors by predicting context.

GloVe is a count-based model — learn vectors by factoring the global word co-occurrence matrix.

**Training objective:**
```
For word pairs (i, j) with co-occurrence count Xᵢⱼ:
  minimize: (vᵢ · vⱼ + bᵢ + bⱼ - log Xᵢⱼ)²

Weighted by f(Xᵢⱼ) to down-weight very common pairs
```

**Result**: vectors where dot products predict log co-occurrence probabilities.

**Properties:**
- Captures both local context (like Word2Vec) and global statistics
- Often performs similarly to Word2Vec on benchmarks
- Easier to parallelize training
- Available as pre-trained vectors for Wikipedia, Common Crawl

### FastText (Facebook, Joulin & Mikolov, 2016)

Extends Word2Vec to handle subword information.

**Key idea**: represent each word as a **bag of character n-grams**.

```
"where" → {"<wh", "whe", "her", "ere", "re>", "<where>"}
(substrings + full word)

Embedding of "where" = sum of n-gram embeddings
```

**Why this matters:**
- Out-of-vocabulary words: can represent "unrecognizable" by combining known n-grams
- Morphologically related words share n-grams: "run", "running", "runner" share "run"
- Language-agnostic: works for morphologically rich languages (German, Finnish, Turkish)
- Handles typos: "whree" and "where" share many character n-grams

FastText is still widely used for text classification at production scale (fast, no GPU needed).

---

## 4. Sequence-to-Sequence Models

### The Problem: Variable-Length to Variable-Length

Machine translation, summarization, question answering:
```
"Bonjour, comment ça va?" (length 4)
→ "Hello, how are you?"   (length 5)
```

Input and output are both sequences, potentially different lengths.

### The Encoder-Decoder Architecture

Sutskever, Vinyals, Le (Google, 2014) — "Sequence to Sequence Learning with Neural Networks"

```
Encoder LSTM:
  Process input sequence one token at a time
  Final hidden state = context vector c (fixed size, e.g., 512-dim)

Decoder LSTM:
  Initialize with context vector c
  Generate output one token at a time
  Each step: use previous output token + hidden state → next output token
```

**Training (teacher forcing):**
```
Input: "Bonjour comment ça va"
Target: [<SOS>, "Hello", "how", "are", "you", <EOS>]

For each step, feed the GROUND TRUTH previous token to the decoder,
not the predicted token (prevents compounding errors during training)
```

**Inference (beam search):**
```
Maintain K hypotheses (beams) at each step
For each hypothesis, expand with top-K next tokens
Keep top-K overall
Return highest-probability complete sequence
```

### The Context Vector Bottleneck

For a sentence of length 50:
- Encoder compresses 50 tokens into one 512-dimensional vector
- Decoder must reconstruct output entirely from this vector

For longer sentences, this is a severe bottleneck:
- Early words get "compressed away" by later words
- The model must store everything relevant in 512 numbers

This is precisely the problem that the attention mechanism solves.

### Seq2Seq Success Stories

Before attention, seq2seq worked well for:
- Short to medium sentences in machine translation (WMT benchmarks improved significantly)
- Dialogue systems (feed conversation, decode response)
- Summarization

Google used seq2seq at the core of Google Translate starting in 2016.

---

## 5. The Attention Mechanism

### Bahdanau et al. (2014) — The Pivotal Paper

*"Neural Machine Translation by Jointly Learning to Align and Translate"*

**The key insight**: instead of compressing the entire source into a fixed vector, at each decoding step, let the decoder **look back at all encoder hidden states** and focus on the most relevant ones.

### How Attention Works

```
Encoder produces: h₁, h₂, h₃, ..., hₙ  (one hidden state per input token)

At decoding step t:
  sₜ = decoder's current state

  Attention scores:
    eₜᵢ = alignment_model(sₜ, hᵢ)   for each i = 1..n
    (how relevant is input position i to current decoding state?)

  Attention weights (normalized):
    αₜᵢ = softmax(eₜ₁, eₜ₂, ..., eₜₙ)ᵢ
    → values sum to 1, higher for more relevant positions

  Context vector:
    cₜ = Σᵢ αₜᵢ × hᵢ
    → weighted sum of encoder hidden states

  Output:
    ŷₜ = generate(sₜ, cₜ)
```

The **alignment model** computes relevance between decoder state and each encoder state:

**Additive attention (Bahdanau):**
```
eₜᵢ = vᵀ × tanh(Wₛ × sₜ + Wₕ × hᵢ)
```
Where v, Wₛ, Wₕ are learned parameters.

### Visualization

Attention weights can be visualized as a matrix (source positions × target positions):

```
Attention matrix for "La Zone Economique Europeenne" → "The European Economic Area":

           The  European  Economic  Area
La          0.9    0.1      0.0      0.0
Zone        0.0    0.0      0.3      0.7
Economique  0.0    0.1      0.8      0.1
Europeenne  0.0    0.7      0.1      0.2
```

High values on the diagonal mean the model learns to align source and target in roughly the same order. Off-diagonal values capture re-ordering.

**This was interpretable AI before interpretability was a concept.**

### Luong Attention (2015)

Simplified version by Luong et al.:

**Multiplicative (dot-product) attention:**
```
eₜᵢ = sₜ · hᵢ   (dot product, no parameters!)
```

or with a learned transformation:
```
eₜᵢ = sₜᵀ × Wₐ × hᵢ
```

Dot-product attention is O(n) computations — faster than additive attention.

This is the version that directly generalizes to the Transformer's self-attention.

### Why Attention is Powerful

1. **Solves the bottleneck**: each decoding step accesses all encoder positions directly
2. **Dynamic focus**: attention weights change at each step based on current state
3. **Parallelizable context access**: can compute all attention weights in parallel
4. **Interpretable**: we can see which input positions influenced each output

### Attention Beyond Translation

The attention mechanism quickly spread beyond translation:

**Image captioning (Show and Tell + attention):**
- Encoder: CNN producing spatial feature maps (HxW feature vectors, not just one)
- Decoder: LSTM with attention over spatial positions
- The model learns to look at the right part of the image when generating each word

**Reading Comprehension:**
- Attend from question tokens to document tokens
- Find relevant passages to answer the question

**Self-Attention (a crucial generalization):**
What if the query and the key-value memory come from the SAME sequence?

```
standard: query from decoder, keys/values from encoder
self-attention: query, keys, values ALL from the same sequence
```

This allows each position to directly incorporate information from any other position in the same sequence. This is the core of the Transformer.

---

## 6. Transformer-XL and Memory Augmentation

### The Segment-Level Recurrence Problem

Before Transformer-XL (Dai et al., 2019), transformers processed fixed-length segments independently. Information couldn't flow between segments.

Transformer-XL introduced segment-level recurrence:
- Cache the hidden states of the previous segment
- Attend over both current + cached states

This allowed transformers to effectively model longer sequences than their context window.

**Influenced**: memory mechanisms in later models, extended context strategies.

---

## 7. ELMo: Contextualized Embeddings (2018)

### The Limitation of Word2Vec

Word2Vec gives each word a **single** vector regardless of context:

```
"I deposited money at the bank."
"The boat docked at the river bank."
```

"bank" gets the same vector in both sentences. The model cannot know which meaning is intended.

### ELMo (Embeddings from Language Models)

Peters et al. (Allen Institute, 2018)

Train a deep bidirectional LSTM language model on large text corpus.

For a given word in a sentence, its ELMo representation is computed by:
1. Running the entire sentence through the bidirectional LSTM
2. Taking the activations at each layer for that word's position
3. Weighting the layer activations (task-specific weights learned during fine-tuning)

```
For word wₜ in sentence w₁...wₙ:
  Layer 0: character-based embedding (not context-dependent)
  Layer 1: first BiLSTM layer activations (syntactic information)
  Layer 2: second BiLSTM layer activations (semantic information)

ELMo(wₜ) = γ × Σₖ sₖ × hₜₖ
```

Where sₖ are task-specific softmax-normalized weights, γ is a scalar.

### Why ELMo Mattered

1. **Contextualized**: same word → different representation depending on context
2. **Transfer learning**: pre-train once, use in any downstream task
3. **Different layers encode different information**: layer 1 = syntax (POS, NER), layer 2 = semantics (word sense, coreference)
4. Achieved state-of-the-art on 6 NLP benchmarks simultaneously in 2018

**ELMo was the proof of concept** that pre-trained language model representations could dramatically improve NLP tasks — directly motivating BERT.

---

## 8. The Road to Self-Attention

### Progression of Ideas

```
One-hot vectors (1990s)
    ↓
Word2Vec (2013): static, context-free word embeddings
    ↓
RNN Language Models (2010s): sequential, context-dependent processing
    ↓
Seq2Seq (2014): encoder-decoder for variable-length sequences
    ↓
Attention over Encoder (2014): dynamic focus on input positions
    ↓
Self-Attention (2015, as component): attend within one sequence
    ↓
Transformer (2017): ONLY attention, no recurrence
    ↓
BERT (2018): bidirectional transformer, masked language modeling
    ↓
GPT (2018): autoregressive transformer, causal language modeling
    ↓
GPT-3 (2020): scaled to 175B params → emergent capabilities
    ↓
RLHF + InstructGPT (2022): aligned LLMs
    ↓
ChatGPT, GPT-4, Claude, Gemini...
```

### What Each Step Contributed

| Step | Key Idea | Problem Solved |
|------|----------|---------------|
| Word2Vec | Dense vectors from context prediction | One-hot inefficiency, no similarity |
| Seq2Seq | Encoder-decoder for sequences | Variable-length outputs |
| Attention | Dynamic soft focus on all input positions | Context vector bottleneck |
| Self-attention | Attend to all positions within same sequence | Sequential processing bottleneck |
| Positional encoding | Inject position information without recurrence | Permutation invariance of attention |
| Multi-head attention | Multiple attention heads in parallel | Capture different types of relationships |
| Transformer | All of the above, remove recurrence | Full parallelization, scale |

---

## 9. Building Blocks Recap

This section connects all the pre-transformer components into a coherent picture.

### The Conceptual Flow

**Problem**: represent text for computers
→ **Solution 1** (1990s): one-hot → expensive, no similarity
→ **Solution 2** (2013): Word2Vec → static vectors, polysemy problem
→ **Solution 3** (2018): ELMo → contextualized, but sequential and slow
→ **Solution 4** (2018+): BERT/GPT → parallel, scalable, contextualized

**Problem**: handle variable-length output sequences
→ **Solution 1** (2014): Seq2Seq LSTM → works but bottleneck for long sequences
→ **Solution 2** (2014): Add attention → look at all encoder states
→ **Solution 3** (2017): Transformer — generalize attention to replace recurrence entirely

### Core Concepts That Survived

These ideas from the pre-transformer era are still central to modern LLMs:

| Concept | Pre-Transformer | Modern LLM |
|---------|----------------|------------|
| Word embeddings | Word2Vec, GloVe | Token embeddings (learned) |
| Language modeling | LSTM LM | Causal transformer LM |
| Contextual representations | ELMo (BiLSTM) | Transformer hidden states |
| Attention | Seq2Seq decoder attention | Multi-head self-attention |
| Teacher forcing | Seq2Seq training | Autoregressive training |
| Beam search | Seq2Seq decoding | Sampling / beam search in LLMs |
| Temperature | Character RNN sampling | LLM sampling |
| Perplexity | LSTM LM evaluation | LLM evaluation |

---

## Key Points for Exams

1. **Word2Vec**: skip-gram or CBOW to learn 300-dim embeddings; analogies via vector arithmetic
2. **Negative sampling**: efficient approximation to softmax; sample k negative words per positive pair
3. **GloVe**: factorizes global co-occurrence matrix; similar quality to Word2Vec
4. **FastText**: n-gram based; handles OOV words and morphologically rich languages
5. **Seq2Seq**: encoder → context vector → decoder; bottleneck for long sequences
6. **Bahdanau attention**: at each decoding step, soft-weight all encoder hidden states; eₜᵢ = alignment_model(sₜ, hᵢ)
7. **Attention weights = softmax(alignment scores)**: sum to 1; differentiable; backpropagable
8. **Context vector = weighted sum** of encoder hidden states
9. **Self-attention**: query, key, value all from the same sequence; enables direct position-to-position connections
10. **ELMo**: first contextualized word embeddings; different layers capture syntax vs semantics
11. The progression: Word2Vec → Seq2Seq → Attention → Self-Attention → Transformer

---

## Practice Questions

1. What are the limitations of one-hot word representations?
2. Explain how Word2Vec learns word embeddings (skip-gram objective).
3. What is negative sampling and why is it used in Word2Vec?
4. What does the king - man + woman ≈ queen analogy demonstrate about word embeddings?
5. What is the context vector bottleneck in seq2seq models?
6. Describe the Bahdanau attention mechanism step by step.
7. What are attention weights and what do they represent geometrically?
8. How is self-attention different from seq2seq attention?
9. What problem does ELMo solve that Word2Vec doesn't?
10. Draw the progression from one-hot encodings to transformers, naming the key contribution at each step.
