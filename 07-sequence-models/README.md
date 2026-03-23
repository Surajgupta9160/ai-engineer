# 00e — Sequence Models: RNNs, LSTMs, and GRUs

> **The era before Transformers.** Recurrent networks were the dominant approach
> for language, speech, and time series from the 1980s through 2017. They introduced
> the concept of sequential state — the ideas that LSTMs handle are still relevant in
> modern architectures (attention evolved directly from these limitations).

---

## Table of Contents

1. [The Sequence Problem](#1-the-sequence-problem)
2. [Recurrent Neural Networks (RNNs)](#2-recurrent-neural-networks)
3. [The Vanishing Gradient Problem in RNNs](#3-the-vanishing-gradient-problem)
4. [Long Short-Term Memory (LSTM)](#4-long-short-term-memory)
5. [Gated Recurrent Units (GRU)](#5-gated-recurrent-units)
6. [Bidirectional RNNs](#6-bidirectional-rnns)
7. [Deep Stacked RNNs](#7-deep-stacked-rnns)
8. [Language Modeling with RNNs](#8-language-modeling-with-rnns)
9. [Applications and Limitations](#9-applications-and-limitations)
10. [The Bridge to Transformers](#10-the-bridge-to-transformers)

---

## 1. The Sequence Problem

### Why Standard Neural Networks Don't Handle Sequences

A fixed-size MLP or CNN:
- Expects fixed-size input
- Has no concept of ordering
- Cannot share information across time steps

**Sequences are different:**
- Variable length: "Hi" and "The quick brown fox jumps" are both valid sentences
- Order matters: "The dog bit the man" ≠ "The man bit the dog"
- Long-range dependencies: the subject of a verb may be many tokens ago

**What makes sequence modeling hard:**
```
"The trophy didn't fit in the suitcase because it was too big."
               ↑                          ↑
          what does "it" refer to? → need to understand "trophy", not "suitcase"
```

To resolve this, the model needs to track information about words seen many steps ago.

### Types of Sequence Tasks

| Task | Input | Output | Example |
|------|-------|--------|---------|
| Sequence classification | Sequence | Single label | Sentiment: "I loved it" → Positive |
| Sequence labeling | Sequence | Sequence | POS tagging: "The/DT cat/NN sat/VBD" |
| Seq2Seq | Sequence | Sequence | Translation: "Bonjour" → "Hello" |
| Language modeling | Sequence prefix | Next token | "The quick brown..." → "fox" |
| Speech recognition | Audio sequence | Text sequence | wav → transcript |

---

## 2. Recurrent Neural Networks (RNNs)

### Core Idea

Process one element at a time, maintaining a **hidden state** that carries information about the sequence so far.

```
At each time step t:
  input:         xₜ
  previous state: hₜ₋₁

  hₜ = tanh(Wₕ × hₜ₋₁ + Wₓ × xₜ + b)
  ŷₜ = Wᵧ × hₜ + bᵧ   (optional output at each step)
```

**Key insight**: the same weight matrices (Wₕ, Wₓ, Wᵧ) are used at every time step.

This is **weight sharing across time** — analogous to CNNs sharing weights across space.

### The Unrolled View

An RNN can be "unrolled" over time:

```
x₁ → [h₁] → [h₂] ← x₂ → [h₃] ← x₃ → [h₄] ← x₄
       ↓       ↓              ↓              ↓
      ŷ₁      ŷ₂             ŷ₃             ŷ₄
```

The hidden state hₜ is a summary of the sequence up to time t.

For a sequence of length T:
- T forward computations (sequential, not parallelizable)
- The final hidden state hₜ is commonly used as a sequence representation

### Parameter Count

A simple RNN with input dim d and hidden dim h:
- Wₓ: (h, d)
- Wₕ: (h, h)
- bₕ: (h,)
- Wᵧ: (output_dim, h)

Total: h×d + h×h + h + output_dim×h — same parameters used at every step.

### Backpropagation Through Time (BPTT)

Unroll the RNN over T steps → compute loss at each step → backpropagate through the unrolled graph.

```python
# Simplified BPTT
for t in range(T):
    h[t] = tanh(Wh @ h[t-1] + Wx @ x[t])
    y_pred[t] = Wy @ h[t]
    loss[t] = cross_entropy(y_pred[t], y_true[t])

total_loss = sum(loss)
total_loss.backward()  # PyTorch handles the graph automatically
```

Gradient flows backward through each unrolled step, multiplying by Wₕ at each step.

### Simple RNN in PyTorch

```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, hidden = self.rnn(x)
        # out: (batch, seq_len, hidden_size) — outputs at each time step
        # hidden: (1, batch, hidden_size) — final hidden state

        # For classification: use final hidden state
        return self.fc(hidden.squeeze(0))
```

---

## 3. The Vanishing Gradient Problem

### Why RNNs Fail on Long Sequences

During backpropagation, gradient at step t requires multiplying by Wₕ at each previous step:

```
∂L/∂h₁ = ∂L/∂hₜ × (Wₕᵀ)^(t-1) × diag(f'(h₂)) × ...
```

If the largest eigenvalue of Wₕ is λ:
- λ < 1: gradients decay exponentially → **vanishing gradient**
- λ > 1: gradients grow exponentially → **exploding gradient**

For a sequence of length 100:
```
λ = 0.9: 0.9^100 ≈ 0.000027  (practically zero)
λ = 1.1: 1.1^100 ≈ 13780    (exploding)
```

### Consequences

**Vanishing gradient:**
- Early time steps receive near-zero gradient updates
- RNN "forgets" long-range dependencies
- The model cannot learn to connect information that appeared 50 steps ago with the current output

**Exploding gradient:**
- Training becomes unstable
- Solution: gradient clipping (clip gradients to max norm)

### The Long-Range Dependency Problem

```
"The chef who trained in Paris for several years and won numerous awards ___ the meal."
                                                                          ^
        What verb tense? Singular/plural? → depends on "chef", not "awards"
```

Simple RNNs fail to preserve the subject "chef" over 10+ tokens. LSTMs were designed to fix this.

---

## 4. Long Short-Term Memory (LSTM)

### Hochreiter & Schmidhuber (1997)

LSTM adds a **cell state** (separate from hidden state) and **gating mechanisms** that control information flow.

### LSTM Architecture

Three gates control what information is kept or discarded:

```
Forget gate:  fₜ = σ(Wf × [hₜ₋₁, xₜ] + bf)
               → how much of previous cell to forget (0 = forget all, 1 = keep all)

Input gate:   iₜ = σ(Wi × [hₜ₋₁, xₜ] + bi)
               → how much new information to add

Cell gate:    g̃ₜ = tanh(Wg × [hₜ₋₁, xₜ] + bg)
               → what new information to potentially add

Output gate:  oₜ = σ(Wo × [hₜ₋₁, xₜ] + bo)
               → how much to expose to output

Cell update:  cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ g̃ₜ
               → forget old, add new

Hidden state: hₜ = oₜ ⊙ tanh(cₜ)
               → output = gated cell state
```

Where σ = sigmoid, ⊙ = element-wise multiply.

### Intuition for Each Gate

**Forget gate** (fₜ):
- "Should I forget what I've remembered about the previous context?"
- Example: when processing "he said ... she", the gender information should change → forget gate fires high for gender feature
- Sigmoid output → values between 0 and 1 → smooth, differentiable forgetting

**Input gate** (iₜ) + Cell gate (g̃ₜ):
- iₜ: "Should I update my memory with this new information?"
- g̃ₜ: "What do I want to update my memory with?"
- Together: gated write to cell state

**Cell state** (cₜ):
- The "long-term memory" — can persist unchanged for many steps
- Gradient path: gradients can flow directly through cell state with minimal transformation
- This is the key to solving vanishing gradients — the cell state provides a "highway" for gradients

**Output gate** (oₜ):
- "What portion of my memory should I expose as output?"
- The hidden state hₜ is a filtered version of the cell state

### Why LSTM Solves Vanishing Gradients

The cell state update:
```
cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ g̃ₜ
```

Gradient through cell state:
```
∂cₜ/∂cₜ₋₁ = fₜ  (element-wise)
```

If the forget gate is close to 1, the gradient flows through unchanged. No matrix multiplication with Wₕ — the vanishing gradient bottleneck is broken.

The LSTM can maintain information for hundreds of steps by keeping the forget gate near 1.

### LSTM in PyTorch

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len) token ids
        embedded = self.embedding(x)           # (batch, seq_len, embed_dim)
        out, (hidden, cell) = self.lstm(embedded)
        # out: (batch, seq_len, hidden_dim)  — all hidden states
        # hidden: (1, batch, hidden_dim)     — final hidden state
        # cell: (1, batch, hidden_dim)       — final cell state

        return self.fc(hidden.squeeze(0))      # Use final hidden state

# Multi-layer LSTM
lstm = nn.LSTM(input_size, hidden_size, num_layers=2,
               dropout=0.3, batch_first=True)
```

---

## 5. Gated Recurrent Units (GRU)

### Cho et al. (2014)

Simplified version of LSTM — merges cell state and hidden state, uses only two gates.

```
Reset gate: rₜ = σ(Wr × [hₜ₋₁, xₜ] + br)
             → how much of previous hidden state to reset

Update gate: zₜ = σ(Wz × [hₜ₋₁, xₜ] + bz)
              → how much to update (vs keep previous)

Candidate:   h̃ₜ = tanh(W × [rₜ ⊙ hₜ₋₁, xₜ])
              → potential new hidden state

New state:   hₜ = (1 - zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ
              → interpolate between old and new
```

### GRU vs LSTM

| Aspect | LSTM | GRU |
|--------|------|-----|
| States | 2 (hidden + cell) | 1 (hidden only) |
| Gates | 3 (forget, input, output) | 2 (reset, update) |
| Parameters | More | ~25% fewer |
| Performance | Generally comparable | Often similar or slightly worse |
| When to use | Long sequences, complex dependencies | Shorter sequences, computational budget |

**Rule of thumb**: LSTM is more powerful; GRU is faster to train. Try both and pick the best for your task.

---

## 6. Bidirectional RNNs

### Problem with Standard RNNs

Standard RNNs only use **past context**. For the word "bat" in "He hit the ball with a bat", we need future context ("ball") to disambiguate meaning.

### Bidirectional Architecture

Run two RNNs: one forward (left to right), one backward (right to left). Concatenate their hidden states.

```
Forward:  x₁ → h→₁ → h→₂ → h→₃ → h→₄
Backward: x₁ ← h←₁ ← h←₂ ← h←₃ ← h←₄

At each step t: [h→ₜ; h←ₜ]   (concatenated)
```

The representation at each position now contains information from both directions.

**Limitation**: cannot be used for language *generation* (need future tokens which don't exist yet). Only for encoding/understanding tasks.

```python
lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
# Output: (batch, seq_len, 2 * hidden_size)  — both directions concatenated
```

**Used in**: BERT (bidirectional transformer), Named Entity Recognition, classification.

---

## 7. Deep Stacked RNNs

Multiple LSTM layers stacked, each receiving the hidden states of the layer below:

```
Input: x₁, x₂, ..., xₜ
Layer 1: h¹₁, h¹₂, ..., h¹ₜ
Layer 2: h²₁, h²₂, ..., h²ₜ   (takes h¹ₜ as input)
Layer 3: h³₁, h³₂, ..., h³ₜ   (takes h²ₜ as input)
...
```

Each layer learns higher-level representations of the sequence.

```python
# 3-layer LSTM with dropout between layers
lstm = nn.LSTM(input_size, hidden_size, num_layers=3, dropout=0.3, batch_first=True)
```

**Typical depth**: 2-4 layers for most tasks. Deep stacked LSTMs were used in Google Translate's model before transformers.

---

## 8. Language Modeling with RNNs

### Character-Level RNN Language Model

Andrej Karpathy's "The Unreasonable Effectiveness of RNNs" (2015 blog post):

Train an LSTM to predict the next character given previous characters:

```python
# Training
for t in range(seq_len):
    # x_t is the current character
    # y_t is the next character
    loss += cross_entropy(lstm_output[t], y[t])

# Generation (sampling):
current_char = start_char
for _ in range(num_chars):
    logits = model(current_char)
    probs = softmax(logits / temperature)
    next_char = sample_from(probs)
    current_char = next_char
```

### Temperature Sampling

```python
probs = softmax(logits / T)
```

- T = 1.0: normal sampling
- T < 1.0 (e.g., 0.5): sharper distribution, more repetitive, "safer"
- T > 1.0 (e.g., 1.5): flatter distribution, more diverse, "creative" but more errors

This sampling strategy is identical to what modern LLMs use.

### Word-Level Language Models

Same idea but at word level:

```
Input:  "The quick brown"
Target: "quick brown fox"   (shift by 1)
```

Train LSTM to maximize P(word_t | word_{t-1}, word_{t-2}, ...).

At generation time: sample from predicted distribution at each step.

This was state-of-the-art NLP until 2018.

### Perplexity as Language Model Metric

```
Perplexity = exp(-(1/T) × Σₜ log P(wₜ | w₁...wₜ₋₁))
```

For a character-level model on Shakespeare, Karpathy's LSTM achieved ~1.3 bits per character.
Lower = better.

This metric is directly used to evaluate LLMs today.

---

## 9. Applications and Limitations

### Where RNNs/LSTMs Excelled

| Application | Approach |
|-------------|----------|
| Language modeling | Word/character level LSTM |
| Machine translation | Encoder-decoder LSTM (before attention) |
| Speech recognition | Bidirectional LSTM (CTC loss) |
| Text classification | LSTM → final hidden state → classifier |
| Named entity recognition | Bidirectional LSTM + CRF |
| Time series prediction | LSTM with regression head |
| Music generation | Character-level LSTM on MIDI |

### Key Limitations That Led to Transformers

**1. Sequential computation prevents parallelization:**
```
LSTM must process x₁ before x₂, x₂ before x₃...
Cannot use modern GPU parallelism effectively
Training is slow
```

**2. Fundamental long-range dependency bottleneck:**
Even LSTMs struggle with very long sequences. The hidden state is a fixed-size vector that must encode all past context. Information can still get "compressed away."

**3. Fixed-length bottleneck in encoder-decoder:**
```
Encode: x₁, x₂, ..., xₙ → single context vector c
Decode: c → y₁, y₂, ..., yₘ
```
For long sequences, the context vector is insufficient. Solution → attention (covered in 00f).

**4. Recency bias:**
The hidden state is most strongly influenced by recent tokens. Information from early in the sequence is diluted even with LSTM gating.

---

## 10. The Bridge to Transformers

### LSTM's Legacy

The concepts LSTMs introduced are still present in modern architectures:

| LSTM Concept | Transformer Equivalent |
|-------------|----------------------|
| Gating (learned information routing) | Multi-head attention (adaptive weighting) |
| Cell state as long-term memory | Key-Value cache |
| Hidden state | Hidden states / residual stream |
| Forget gate | Attention score ≈ 0 (attending away) |

### The Attention Solution

LSTMs with attention (2014, Bahdanau) were the crucial bridge:
- Instead of compressing all past information into a single vector
- At each decoding step, directly look at all encoder hidden states
- Weight them by relevance (learned attention scores)

This is the direct precursor to the Transformer.

Full coverage in **`08-pre-transformer-era/README.md`**.

### Why Transformers Won

Transformers replaced LSTMs for almost everything because:

1. **Fully parallel**: process all positions simultaneously → 10-100x faster training
2. **Direct connections**: every position can attend to every other position → no information bottleneck
3. **Scales better**: parallelism allows training on vastly more data
4. **Context window**: modern transformers handle 100K+ tokens; LSTMs were practical up to ~512

**LSTM niche areas (still in use):**
- On-device / embedded inference (lower memory than transformers)
- Time series with strict sequential ordering constraints
- When training data is very limited (transformers need more data)

---

## Key Points for Exams

1. **RNN**: processes sequences step-by-step, maintains hidden state hₜ = f(hₜ₋₁, xₜ)
2. **Vanishing gradient**: gradient multiplied by Wₕ^T at each step → decays exponentially for long sequences
3. **LSTM** uses 3 gates (forget, input, output) + separate cell state → information highway for gradients
4. **Forget gate**: σ × previous cell → smooth, learnable forgetting
5. **Cell state** is the key LSTM innovation: additive gradient path prevents vanishing gradient
6. **GRU**: 2 gates, merged cell+hidden state, fewer params, similar performance to LSTM
7. **Bidirectional RNN**: run forward AND backward, concatenate → context from both directions; cannot generate
8. **Language modeling**: train to predict next token; evaluate with perplexity
9. **Temperature sampling**: T<1 conservative, T>1 diverse; same concept used in modern LLMs
10. **Main LSTM limitation**: sequential computation, no parallelism → Transformers are 10-100x faster to train

---

## Practice Questions

1. What problem do RNNs solve that MLPs can't handle?
2. Why does the vanishing gradient problem affect RNNs more than MLPs?
3. What is the cell state in an LSTM and why is it important for long-range dependencies?
4. Explain the forget gate. What does it do mathematically?
5. How does LSTM solve the vanishing gradient problem?
6. What is the difference between LSTM and GRU? When would you use each?
7. What is a bidirectional RNN? What tasks is it well-suited for? Why can't it be used for text generation?
8. What is perplexity and how is it calculated?
9. What is temperature sampling and how does it affect text generation?
10. What were the main limitations of LSTM that led to the development of Transformers?
