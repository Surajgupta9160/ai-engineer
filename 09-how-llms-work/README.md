# 05 — How LLMs Work: Deep Technical Understanding

---

## Why AI Engineers Need to Understand the Internals

You might think: "I just call the API, why do I need to know how it works internally?"

Here's why it matters:

```
Understanding internals helps you:

1. DEBUG problems
   "Why does the model keep ignoring my instruction?"
   → Because your instruction is buried in a long context (attention dilution)

2. OPTIMISE prompts
   "Why does 'think step by step' help so much?"
   → Because it forces the model to use its hidden states for intermediate reasoning

3. UNDERSTAND failure modes
   "Why does the model make up facts?"
   → Because it's predicting likely text, not retrieving from a knowledge base

4. MAKE COST DECISIONS
   "When is it worth using o1 vs GPT-4o?"
   → Because o1 does internal chain-of-thought before outputting

5. PASS CERTIFICATION EXAMS
   Most AI certifications test conceptual understanding
```

---

## 1. Neural Networks: The Foundation

### What Is a Neuron? (Starting From Zero)

Biological neurons:
```
A real brain neuron:
  Receives signals from many other neurons (dendrites)
  If signals are strong enough → fires (sends signal forward)
  If signals are too weak → stays silent

An artificial neuron (mathematical model):
  Takes multiple numbers as input
  Multiplies each by a "weight" (how important is it?)
  Adds them all up
  Applies an "activation function" (should I fire?)
  Outputs one number
```

Mathematical neuron:

```
Inputs:  x1=0.5, x2=0.3, x3=0.8
Weights: w1=0.2, w2=0.7, w3=-0.1
Bias:    b=0.1

Step 1: Weighted sum
  sum = (x1×w1) + (x2×w2) + (x3×w3) + b
      = (0.5×0.2) + (0.3×0.7) + (0.8×-0.1) + 0.1
      = 0.10 + 0.21 + (-0.08) + 0.1
      = 0.33

Step 2: Activation function (ReLU: output 0 if negative, else output unchanged)
  output = max(0, 0.33) = 0.33

This neuron outputs 0.33.
```

### Why We Need Many Neurons

A single neuron can only learn a linear boundary (a straight line). To learn complex patterns (like recognising cats in images), we need:
- Many neurons arranged in **layers**
- Multiple layers create a **deep neural network**

```
Simple neural network for text:

INPUT LAYER          HIDDEN LAYER 1       HIDDEN LAYER 2       OUTPUT
(text features)      (96 neurons each     (complex patterns)   (next word probs)
                      looking for simple
                      patterns)

["the",   0.2]  ──► [neuron 1: 0.7] ──► [neuron 97: 0.3] ──► "cat: 85%"
["cat",   0.8]  ──► [neuron 2: 0.1] ──► [neuron 98: 0.9] ──► "dog: 8%"
["sat",   0.1]  ──► [neuron 3: 0.4] ──► [neuron 99: 0.5] ──► "mat: 4%"
["on",    0.3]  ──► [neuron 4: 0.6] ──► [neuron100: 0.2] ──► ...
["the",   0.5]  ──► ...               ──► ...
            ⋮               ⋮                   ⋮
(many more inputs)  (many more neurons)   (many more)

Each layer transforms the representation.
Final layer outputs probability for each possible next word.
```

---

## 2. The Transformer Architecture

### What Existed Before Transformers

Before 2017, language models used **RNNs** (Recurrent Neural Networks):

```
RNN Processing (Sequential):
  "The cat sat on the mat"
       │
       ▼
  [Process "The"] → hidden state h1
       │
       ▼
  [Process "cat"] + h1 → hidden state h2
       │
       ▼
  [Process "sat"] + h2 → hidden state h3
       │
       ▼
  ... and so on, word by word

Problems with RNNs:
  ✗ Sequential: Can't parallelise → SLOW training
  ✗ "Forgetting": Information from 50 words ago gets diluted by the time
    we process the 100th word (vanishing gradient problem)
  ✗ Can't look back: By the time we're at word 100, word 1 is mostly forgotten
```

### The Transformer Revolution (2017)

The paper "Attention Is All You Need" introduced a completely different architecture:

```
Transformer Processing (Parallel):
  "The cat sat on the mat"
   │    │   │   │   │   │
   ▼    ▼   ▼   ▼   ▼   ▼
  [ALL tokens processed simultaneously]
   │    │   │   │   │   │
   ▼    ▼   ▼   ▼   ▼   ▼
  [Each token ATTENDS to all other tokens]
  "cat" can directly look at "The" and "mat" and "sat"
  No forgetting, no sequential bottleneck

Advantages:
  ✓ Massively parallelisable → FAST training (use GPUs to full potential)
  ✓ No forgetting: every token can attend to every other token equally
  ✓ Scales beautifully: more hardware = bigger models = better results
```

### The GPT (Decoder-Only) Architecture

Modern LLMs like GPT-4, Claude, and Llama use a "decoder-only" transformer:

```
Decoder-Only Transformer (GPT-style):

INPUT TOKENS:   [The] [cat] [sat] [on] [the]
                  │     │     │    │    │
                  ▼     ▼     ▼    ▼    ▼
EMBEDDING:      [vec] [vec] [vec] [vec] [vec]  ← Convert token IDs to vectors
                  │     │     │    │    │
POSITIONAL:     +pos  +pos  +pos  +pos  +pos    ← Add position information
                  │     │     │    │    │
                  ▼     ▼     ▼    ▼    ▼
TRANSFORMER    ┌──────────────────────────────┐
BLOCK 1:       │  Self-Attention              │  ← Tokens look at each other
               │  Feed-Forward Network        │  ← Process each token
               │  Layer Normalisation         │  ← Stabilise
               └──────────────────────────────┘
                  (repeat 32-96 times)
                  │     │     │    │    │
                  ▼     ▼     ▼    ▼    ▼
OUTPUT HEAD:    probabilities for next token
                "mat": 45%, "floor": 20%, ...

KEY PROPERTY: Causal masking
  "sat" can attend to: "The", "cat", "sat" (itself and before)
  "sat" CANNOT attend to: "on", "the" (future tokens)
  → Model only sees past, predicts future
  → This allows next-token prediction during training
```

---

## 3. Tokenization: Text to Numbers

### Why Tokenization Is Needed

Computers can't process raw text directly. Everything must become numbers. Tokenization is the bridge.

```
Step 1: Build vocabulary (done once during model training)
  Collect huge text corpus
  Run BPE (Byte Pair Encoding) algorithm
  Result: vocabulary of ~50,000-100,000 sub-words

Step 2: Encode text (done for every input/output)
  "Hello world" → ["Hello", " world"] → [9906, 1917]

Step 3: Decode (done when generating output)
  [9906, 1917, 0] → ["Hello", " world", "!"] → "Hello world!"
```

### Byte Pair Encoding (BPE) in Detail

```python
# Simplified BPE algorithm (conceptual)

# Start: each character is its own token
vocabulary = {"h", "e", "l", "o", " ", "w", "r", "d"}

# Training corpus (simplified)
corpus = "hello world hello hello world"

# Count pairs in corpus
pair_counts = {
    ("h", "e"): 3,   # "he" appears 3 times
    ("e", "l"): 3,
    ("l", "l"): 3,
    ("l", "o"): 3,
    ("o", " "): 2,   # "o " appears 2 times
    (" ", "w"): 2,
    ("w", "o"): 2,
    ("o", "r"): 2,
    ("r", "l"): 2,
    ("l", "d"): 2,
}

# Merge most common pair: ("h", "e") → "he"
vocabulary.add("he")
# Re-tokenize: "hello" is now ["he", "l", "l", "o"]

# Next iteration: ("he", "l") is most common → "hel"
vocabulary.add("hel")
# "hello" is now ["hel", "l", "o"]

# Continue until vocabulary size is reached (e.g., 50,000)
# Eventually "hello" becomes a single token: "hello"
# Common words → single tokens (efficient)
# Rare words → multiple tokens (handles any text)
```

### Special Tokens

Every model has special tokens with specific meanings:

```
[BOS] = Beginning of Sequence — marks the start
[EOS] = End of Sequence — model outputs this to stop generating
[PAD] = Padding — used to make batches same length during training
[UNK] = Unknown — for characters not in vocabulary
[SEP] = Separator — between segments in some models
[CLS] = Classification — used in BERT-style models

For LLaMA 3.1, special tokens look like:
  <|begin_of_text|>     ← Start of conversation
  <|eot_id|>            ← End of turn
  <|start_header_id|>   ← Start of role header
  <|end_header_id|>     ← End of role header

A full formatted prompt:
  <|begin_of_text|>
  <|start_header_id|>system<|end_header_id|>
  You are a helpful assistant.<|eot_id|>
  <|start_header_id|>user<|end_header_id|>
  What is Python?<|eot_id|>
  <|start_header_id|>assistant<|end_header_id|>
```

```python
import tiktoken

# Count tokens accurately
enc = tiktoken.encoding_for_model("gpt-4o")

def token_analysis(text: str):
    """Analyse tokenization of any text."""
    tokens = enc.encode(text)
    decoded_tokens = [enc.decode([t]) for t in tokens]

    print(f"Text: {text!r}")
    print(f"Token count: {len(tokens)}")
    print(f"Token IDs: {tokens}")
    print(f"Decoded tokens: {decoded_tokens}")
    print()

# See how different texts tokenize
token_analysis("Hello, world!")
token_analysis("supercalifragilistic")
token_analysis("def add(a, b): return a + b")
token_analysis("2024-01-15")
token_analysis("$1,234.56")
```

---

## 4. Self-Attention: The Core Mechanism

### The Coreference Problem

Before attention mechanisms, models struggled with this:

```
"The animal didn't cross the street because it was too tired."

What does "it" refer to? → The animal (not the street)

Humans immediately understand:
  "it" = "animal" because animals get tired, streets don't

Pre-attention models (RNNs):
  By the time they process "it" (token 10),
  information about "animal" (token 2) has faded
  → Often got this wrong

With attention:
  When processing "it", the model looks at ALL previous tokens
  and asks "which of these is 'it' referring to?"
  It computes that "animal" is highly relevant → correct answer
```

### Query, Key, Value — The Library Analogy

Self-attention works like a library search:

```
LIBRARY SEARCH ANALOGY:

Imagine you're looking for information about "it" (you are a QUERY)
Each book in the library has a TITLE (the KEY)
Each book has CONTENT (the VALUE)

You search by matching your query against all titles:
  Your query: "what animal is this?"
  Book titles:
    "The" → poor match (score: 0.02)
    "animal" → excellent match (score: 0.78)  ← HIGH ATTENTION
    "didn't" → poor match (score: 0.01)
    "cross" → poor match (score: 0.03)
    "street" → okay match (score: 0.11)
    "because" → poor match (score: 0.02)
    "it" → self-match (score: 0.03)
    "was" → poor match (score: 0.01)
    "too" → poor match (score: 0.01)
    "tired" → some match (score: 0.08)  ← some attention

  Attention weights (normalised by softmax):
    [0.02, 0.78, 0.01, 0.03, 0.11, 0.02, 0.03, 0.01, 0.01, 0.08]

  Weighted sum of VALUES:
    New representation of "it" = 0.02×v_The + 0.78×v_animal + ...
    ← Dominated by "animal" information!

Result: "it" now carries information about being an animal.
        The model understands coreference!
```

### Mathematical Formulation (Simplified)

```
For each token, compute three vectors:
  Q (Query):   "What am I looking for?"
  K (Key):     "What information do I have?"
  V (Value):   "What information do I give?"

These are computed by multiplying the token embedding by
three learned weight matrices (Wq, Wk, Wv).

Attention score between token i and token j:
  score(i, j) = dot_product(Q_i, K_j) / sqrt(dimension)

The scaling by sqrt(dimension) prevents the scores from
getting too large as dimension grows.

Softmax normalisation:
  attention_weights = softmax(scores)
  → All weights sum to 1.0

Output for token i:
  output_i = sum(attention_weight(i,j) × V_j  for all j)
  → Weighted combination of values from all tokens
```

### Multi-Head Attention

Instead of one attention mechanism, transformers use many in parallel:

```
Multi-Head Attention with 8 heads:

Each head learns to attend to different relationships:
  Head 1: Syntactic relationships (subject-verb agreement)
           "cats" → "are" (plural agreement)

  Head 2: Coreference resolution
           "it" → "animal" (what does it refer to?)

  Head 3: Positional patterns
           Recent tokens vs distant tokens

  Head 4: Semantic similarity
           "happy" close attention to "joyful", "pleased"

  Head 5: Dependency parsing
           Subject of sentence pays attention to object

  Head 6: Entity relationships
           Company names attend to products they make

  Head 7: Temporal patterns
           Before/after relationships in text

  Head 8: Semantic role labeling
           Who is doing what to whom

All 8 heads run in PARALLEL (efficient)
Their outputs are CONCATENATED, then projected
→ Richer representation than any single head
```

---

## 5. Feed-Forward Network

### What It Does

After self-attention, each token goes through a two-layer feed-forward network:

```
Feed-Forward Network (applied independently to each token):

token_vector (768 dim)
     │
     ▼
Linear transformation: 768 → 3072 (expand 4×)
     │
     ▼
Activation function (GELU or ReLU)
     │
     ▼
Linear transformation: 3072 → 768 (compress back)
     │
     ▼
Updated token_vector (768 dim)

Why expand then compress?
  The expansion creates a "working space" where the network
  can compute more complex transformations.
  Think of it like: spreading papers on a desk to work,
  then filing them neatly.
```

---

## 6. Autoregressive Token Generation

### The Generation Loop in Detail

```python
# Pseudocode showing how text generation actually works

def generate_text(prompt: str, max_tokens: int = 100) -> str:
    """
    This is conceptually what the model does at inference time.
    Real implementation is much more optimised.
    """
    # Step 1: Tokenize the input
    tokens = tokenizer.encode(prompt)
    # tokens = [1, 464, 3797, 3332, 319, 262, 2603]

    generated = []

    for _ in range(max_tokens):
        # Step 2: Run the full transformer forward pass
        # ALL previous tokens are processed simultaneously
        logits = transformer_forward_pass(tokens)
        # logits: probability scores for each vocabulary token

        # Step 3: Sample the next token
        next_token = sample_from_distribution(
            logits,
            temperature=0.7,
            top_p=0.9
        )

        # Step 4: Stop if we hit end-of-sequence token
        if next_token == EOS_TOKEN:
            break

        # Step 5: Append to our sequence and repeat
        tokens.append(next_token)
        generated.append(next_token)

        # NOTE: We re-process ALL tokens each iteration
        # This is inefficient... which is why KV-cache exists

    # Step 6: Decode token IDs back to text
    return tokenizer.decode(generated)
```

### KV-Cache: The Efficiency Optimisation

```
Problem with naive generation:
  To generate token 100, we re-compute attention for tokens 1-99
  To generate token 101, we re-compute attention for tokens 1-100
  ← Wasteful! Tokens 1-99 haven't changed.

KV-Cache solution:
  Store the Key (K) and Value (V) vectors from previous tokens
  When generating new token, only compute attention for the NEW token
  Reuse stored K, V for all previous tokens

  Memory trade-off:
    GPT-4 with 128K context: KV-cache takes ~several GB
    But saves re-computation → much faster generation

  This is why:
    First token is slower (building cache from scratch)
    Subsequent tokens are faster (using cached K, V)
```

---

## 7. Sampling Strategies

### Temperature: Full Mathematical Explanation

```python
import numpy as np

def temperature_sampling(logits: list, temperature: float) -> int:
    """
    Sample a token from logits using temperature.

    logits: raw scores from model (can be any value, positive or negative)
    temperature: controls randomness

    Returns: index of selected token
    """
    logits = np.array(logits)

    # Apply temperature: divide logits by temperature
    # Low temp (0.1) → sharp, high temp (2.0) → flat
    scaled_logits = logits / temperature

    # Softmax: convert to probabilities (sum to 1)
    # exp() makes all values positive
    # Dividing by sum normalises to [0, 1]
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))  # Subtract max for numerical stability
    probabilities = exp_logits / exp_logits.sum()

    # Sample from the probability distribution
    token_index = np.random.choice(len(probabilities), p=probabilities)
    return token_index

# Example: What different temperatures do
logits = [5.2, 3.1, 1.8, 0.5, -1.2]  # Raw model scores for 5 words
words = ["Paris", "France", "city", "the", "beautiful"]

for temp in [0.1, 0.5, 1.0, 1.5, 2.0]:
    probs = []
    scaled = np.array(logits) / temp
    exp_s = np.exp(scaled - np.max(scaled))
    probs = exp_s / exp_s.sum()

    print(f"\nTemperature = {temp}:")
    for word, prob in zip(words, probs):
        bar = "█" * int(prob * 40)
        print(f"  {word:12} {prob*100:5.1f}%  {bar}")

# Output shows: Low temp makes Paris dominate; high temp flattens distribution
```

### Top-P (Nucleus) Sampling Implementation

```python
def top_p_sampling(logits: list, top_p: float = 0.9, temperature: float = 1.0) -> int:
    """
    Sample using nucleus (top-p) sampling.

    Algorithm:
    1. Apply temperature
    2. Sort tokens by probability (highest first)
    3. Take smallest set whose cumulative probability >= top_p
    4. Sample from that set only (renormalise)
    """
    logits = np.array(logits)

    # Apply temperature
    scaled = logits / temperature

    # Softmax
    exp_s = np.exp(scaled - np.max(scaled))
    probs = exp_s / exp_s.sum()

    # Sort by probability (descending)
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]

    # Find nucleus: smallest set with cumulative prob >= top_p
    cumulative_probs = np.cumsum(sorted_probs)
    # Include one extra token past the threshold
    nucleus_mask = cumulative_probs <= top_p
    nucleus_mask[np.argmax(~nucleus_mask)] = True  # Always include at least one token past threshold

    # Zero out non-nucleus tokens
    nucleus_probs = sorted_probs.copy()
    nucleus_probs[~nucleus_mask] = 0

    # Renormalise
    nucleus_probs /= nucleus_probs.sum()

    # Sample from nucleus
    selected_sorted_idx = np.random.choice(len(sorted_probs), p=nucleus_probs)
    return sorted_indices[selected_sorted_idx]
```

---

## 8. Chain of Thought (CoT)

### Why Step-by-Step Thinking Works

```
The key insight: LLMs generate the answer token-by-token.
When they output intermediate reasoning steps, those steps become
part of the context for generating the final answer.

WITHOUT CoT:
  Input:  "What is 15% of $847.50?"
  LLM generates: "127.125" or sometimes wrong values

  The model must "compress" all reasoning into the first few tokens.
  This is hard! The answer often comes out wrong.

WITH CoT (adding "Think step by step"):
  Input:  "What is 15% of $847.50? Think step by step."
  LLM generates:
    "Step 1: To find 15% of a number, multiply by 0.15
     Step 2: 847.50 × 0.15
     Step 3: 847.50 × 0.1 = 84.75
             847.50 × 0.05 = 42.375
     Step 4: 84.75 + 42.375 = 127.125
     Answer: $127.13"

Why CoT helps:
  Each generated step becomes context for the next
  The model can "show its work" and verify at each step
  Intermediate tokens serve as a "scratch pad"
  The final answer is generated AFTER the reasoning, not instead of it
```

### Zero-Shot CoT vs Few-Shot CoT

```python
from openai import OpenAI

client = OpenAI()

# Zero-shot CoT (just add magic phrase, no examples needed)
def zero_shot_cot(question: str) -> str:
    """
    Add "Let's think step by step" to any question.
    Works surprisingly well without any examples.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"{question}\n\nLet's think step by step."
        }],
        temperature=0
    )
    return response.choices[0].message.content

# Few-shot CoT (give examples of step-by-step reasoning)
def few_shot_cot(question: str) -> str:
    """
    Provide example reasoning before the actual question.
    More reliable for specific types of problems.
    """
    system = """You solve problems step by step.

Example:
Q: If a train travels at 60mph for 2.5 hours, how far does it go?
A: Step 1: Identify the formula: distance = speed × time
   Step 2: Speed = 60 mph
   Step 3: Time = 2.5 hours
   Step 4: Distance = 60 × 2.5 = 150 miles
   Answer: The train travels 150 miles.

Now solve the next problem the same way."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question}
        ],
        temperature=0
    )
    return response.choices[0].message.content

# Test
result = zero_shot_cot("Sarah has 3 boxes. Each box has 4 rows of 5 apples. How many apples total?")
print(result)
# Step 1: Each box has 4 rows × 5 apples = 20 apples
# Step 2: 3 boxes × 20 apples = 60 apples
# Answer: 60 apples
```

---

## 9. ReAct: Reasoning + Acting

ReAct is a prompting strategy that interleaves reasoning and action for complex tasks that require accessing external information.

```
ReAct Template:
─────────────────────────────────────────────────────────────────

Question: What is the GDP per capita of the country where the
          Eiffel Tower is located?

Thought: I need to find which country has the Eiffel Tower, then
         find that country's GDP per capita.

Action: search("Eiffel Tower location country")

Observation: The Eiffel Tower is located in Paris, France.

Thought: Now I need to find France's GDP per capita.

Action: search("France GDP per capita 2024")

Observation: France GDP per capita 2024: approximately $46,000 USD

Thought: I now have both pieces of information needed to answer.

Action: finish("The Eiffel Tower is in France, which has a GDP per
               capita of approximately $46,000 USD.")
```

```python
# ReAct implementation
import json
from openai import OpenAI

client = OpenAI()

REACT_SYSTEM = """You are a helpful assistant that solves problems step by step.
For each step, output in this EXACT format:
THOUGHT: [your reasoning about what to do]
ACTION: tool_name({"arg": "value"})

Available tools:
- search({"query": "search terms"}) - Search the internet
- calculate({"expression": "math expression"}) - Calculate math
- finish({"answer": "final answer"}) - Provide the final answer

Always THOUGHT before ACTION. Use finish() when you have the answer."""

def react_agent(question: str, tools: dict, max_steps: int = 6) -> str:
    """Run ReAct agent to answer a question using tools."""

    messages = [
        {"role": "system", "content": REACT_SYSTEM},
        {"role": "user", "content": f"Question: {question}"}
    ]

    for step in range(max_steps):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0,
            stop=["Observation:"]  # Stop before writing its own observation
        )

        output = response.choices[0].message.content
        messages.append({"role": "assistant", "content": output})

        print(f"\nStep {step + 1}:\n{output}")

        # Parse and execute action
        if "ACTION: finish(" in output:
            # Extract final answer
            start = output.find("ACTION: finish(") + len("ACTION: finish(")
            end = output.rfind(")")
            try:
                args = json.loads(output[start:end])
                return args.get("answer", "No answer found")
            except:
                return output.split("finish(")[-1].rstrip(")")

        # Find and execute tool call
        if "ACTION:" in output:
            action_line = [l for l in output.split("\n") if l.startswith("ACTION:")][0]
            tool_name = action_line.split("(")[0].replace("ACTION:", "").strip()

            try:
                args_str = action_line[action_line.index("(") + 1:action_line.rindex(")")]
                args = json.loads(args_str)
            except:
                args = {}

            # Execute tool
            if tool_name in tools:
                result = tools[tool_name](**args)
                observation = f"Observation: {result}"
                messages.append({"role": "user", "content": observation})
                print(f"\n{observation}")

    return "Reached max steps without answer"
```

---

## 10. The Training Process in Full Detail

### Pre-training Data Pipeline

```
1. DATA COLLECTION
   Common Crawl (internet): 1.6TB compressed text
   Books (BookCorpus, Project Gutenberg): millions of books
   Wikipedia: all languages
   GitHub: entire public code repos
   ArXiv: scientific papers
   StackOverflow: Q&A pairs

2. DATA CLEANING
   Remove duplicate content
   Filter low-quality text (spam, gibberish)
   Remove offensive/illegal content
   Detect and tag language
   Normalise unicode, fix encoding issues

3. TOKENIZATION
   Build vocabulary using BPE (done once)
   Convert all text to token IDs
   Create training batches

4. TRAINING
   Take batch of text, slide context window
   For each position: predict next token
   Calculate cross-entropy loss
   Backpropagate gradients
   Update parameters with Adam optimiser
   Repeat for ALL data (often multiple epochs for smaller models)

Training compute:
  FLOPs (floating point operations) ≈ 6 × N × D
  where N = parameters, D = training tokens
  GPT-3: 6 × 175B × 300B = 3×10^23 FLOPs
  At A100 GPU speed (312 TFLOP/s): ~100,000 GPU-days
  With 1,000 GPUs: ~100 days training
```

### Learning Rate Schedule

```
Learning Rate Schedule during Training:

                    Peak LR
                    (e.g., 3e-4)
                       ▲
         Warm-up        │        Cosine Decay
              /‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\
             /                        \
            /                          \      Final LR
           /                            \    (e.g., 3e-5)
──────────/                              \──────────────
         │                               │
       Start                           End of training
       (LR=0)

Why warm-up? Starting with high LR causes instability early in training.
Why decay? Prevents overfitting as training completes.
```

---

## 11. Model Architecture Variants

```
┌──────────────────────────────────────────────────────────────────────┐
│                    TRANSFORMER VARIANTS                               │
│                                                                      │
│  ENCODER-ONLY (BERT-style):                                         │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Input: ["I", "love", "AI"]                                 │    │
│  │  ALL tokens see ALL other tokens (bidirectional attention)  │    │
│  │  Output: contextualised representation of each token        │    │
│  │                                                             │    │
│  │  Best for: Classification, NER, Embeddings, Understanding  │    │
│  │  Examples: BERT, RoBERTa, sentence-transformers            │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  DECODER-ONLY (GPT-style):                                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Input: ["I", "love"]                                       │    │
│  │  Each token only sees PREVIOUS tokens (causal attention)    │    │
│  │  Output: probability distribution for next token            │    │
│  │                                                             │    │
│  │  Best for: Text generation, chat, completion, code         │    │
│  │  Examples: GPT-4, Claude, Llama, Mistral                  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ENCODER-DECODER (T5/BART-style):                                   │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Encoder: reads full input (bidirectional)                  │    │
│  │  Decoder: generates output one token at a time              │    │
│  │                                                             │    │
│  │  Best for: Translation, Summarisation, Question Answering   │    │
│  │  Examples: T5, BART, mT5, MarianMT                         │    │
│  └─────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘

Modern trend: Decoder-only dominates for LLMs because:
  - Simpler architecture (one component, not two)
  - Can do everything encoders can do with appropriate prompting
  - Scales better with more compute and data
```

---

## Key Points & Practice Questions

### Exam-Ready Summary

```
LLM INTERNALS CHEAT SHEET:

ARCHITECTURE
  - Transformer: self-attention + feed-forward, repeated N times
  - Decoder-only (GPT): causal masking, predicts next token
  - Self-attention: Q, K, V matrices; softmax(QK^T/√d)V
  - Multi-head: multiple attention mechanisms in parallel

TOKENIZATION
  - BPE: merge most common character pairs repeatedly
  - ~50K-100K vocabulary size
  - English is most efficient; other languages cost more tokens

GENERATION
  - Autoregressive: one token at a time
  - KV-Cache: store K,V from previous tokens for efficiency
  - TTFT: time for first token; subsequent tokens faster with cache

SAMPLING
  - Temperature=0: deterministic (always pick highest prob)
  - Temperature scaling: divides logits before softmax
  - Top-P: nucleus, sample from tokens summing to P
  - Top-K: sample from top K tokens only

PROMPTING TECHNIQUES
  - Chain of Thought: intermediate steps become context for answer
  - Zero-shot CoT: just add "Let's think step by step"
  - ReAct: Thought → Action → Observation loop
  - Few-shot: examples in prompt guide output format

TRAINING PHASES
  1. Pre-training: predict next token on internet data
  2. SFT: learn to follow instructions
  3. RLHF/DPO: align with human preferences
```

### Practice Questions

**Beginner:**
1. What is a transformer and what problem did it solve compared to RNNs?
2. Explain in simple terms what self-attention does.
3. Why does "think step by step" improve math performance?
4. What is the difference between a token and a word?
5. Why is temperature=0 not truly random?

**Intermediate:**
6. Explain the Query, Key, Value mechanism using the library analogy.
7. What is KV-cache and why is it important for inference efficiency?
8. Why does multi-head attention use multiple heads instead of one?
9. What is the causal masking in decoder-only transformers and why is it needed?
10. Explain why zero-shot CoT works even without providing examples.

**Advanced:**
11. Describe how backpropagation and gradient descent work together to update model weights.
12. Why does the scaling by sqrt(d_k) in the attention formula prevent numerical instability?
13. How does the three-phase training (pre-training → SFT → RLHF) ensure the model is both capable AND aligned?
14. Explain the trade-offs between sampling strategies: when would you use each?
15. What is Mixture of Experts and how does it allow models to have more total parameters than active parameters?

---

## 12. Flash Attention: IO-Aware Attention

Standard attention materializes the full N×N attention matrix in GPU High Bandwidth Memory (HBM). For long sequences this is the memory bottleneck, not FLOPs.

```
STANDARD ATTENTION — HBM bottleneck:
  1. Load Q, K from HBM to SRAM     ← slow HBM read
  2. Compute S = Q K^T               ← write N×N matrix to HBM
  3. Load S, compute softmax(S)       ← slow HBM read + write
  4. Load S, V, compute output        ← another slow HBM read

Flash Attention — IO-aware tiling:
  Observation: softmax(Q K^T) V can be computed in TILES
  without ever materializing the full N×N matrix.

  Algorithm:
  - Split Q into blocks that fit in fast SRAM (on-chip cache)
  - For each Q block, iterate over all K,V blocks
  - Compute partial softmax contributions on-chip, accumulate
  - Never write the N×N matrix to HBM at all

  Result:
    HBM reads/writes: O(N) instead of O(N²)
    Speed:  2-4x faster than standard attention
    Memory: O(N) instead of O(N²) — enables much longer contexts
    FLOPs:  Same total FLOPs (attention is still quadratic in compute)
    Key insight: Flash Attention solves the MEMORY BANDWIDTH problem,
                 not the FLOPs problem.

  Flash Attention 2 (2023):
    - Better parallelism across attention heads
    - ~2x more efficient than Flash Attention 1
    - Used in virtually all production LLM inference

  Flash Attention 3 (2024):
    - Targets H100 with async execution and FP8 support
```

---

## 13. Attention Head Redundancy and Pruning

Not all attention heads contribute equally — many are redundant.

```
Michel et al. (2019) "Are Sixteen Heads Really Better than One?":
  Experiment: remove heads one at a time, measure accuracy drop
  Finding: most heads can be pruned with minimal accuracy loss
  In BERT: up to 70% of heads can be removed with <1% accuracy drop
  Implication: models are overparameterized in attention heads

Voita et al. (2019) "Analyzing Multi-Head Self-Attention":
  Identified head specializations via gradient-based importance:
  - Positional heads: attend to adjacent tokens (next, previous)
  - Syntactic heads: attend based on syntactic structure (subject→verb)
  - Rare word heads: attend to rare or unusual tokens in context

  The majority of heads are "redundant" — their patterns are not
  specialized and can be pruned without task harm.

Head pruning algorithm (structured pruning):
  1. Compute head importance scores:
     I_h = |∑_{x in data} ∂L/∂head_h(x)| * |head_h(x)|
     (gradient × activation, averaged over data)
  2. Rank heads by importance
  3. Prune lowest-importance heads (zero out and mask)
  4. Fine-tune remaining heads to recover accuracy

Practical implication for inference:
  Pruned models run faster because skipped heads = fewer MACs
  Important for deploying LLMs in latency-constrained environments
  Modern distillation pipelines often combine head pruning with KD

Why models have redundant heads:
  Training with gradient descent doesn't enforce specialization
  Multiple heads converge to similar patterns (degenerate solutions)
  The original 8-head design was empirically chosen, not analytically optimal
```

---

## 14. Long-Context Architectures Compared

For 1M+ token contexts, standard O(N²) attention is infeasible. Key alternatives:

```
APPROACH 1: Sliding Window / Sparse Attention
  Examples: Longformer, BigBird, Mistral (sliding window)
  Mechanism: each token attends to a local window of W tokens
             plus a small set of global tokens
  Complexity: O(N × W) where W << N
  Tradeoff:
    ✓ Good for tasks where relevant context is local
    ✗ Long-range dependencies must propagate through many layers
    ✗ Bounded receptive field limits cross-document reasoning

  Mistral/LLaMA variant: Sliding window of 4096 tokens with
  "attention sinks" at position 0 (the sink token absorbs
  excessive attention from distant positions)

APPROACH 2: Linear / Kernel Attention
  Examples: Performer, cosFormer, RWKV attention mode
  Mechanism: kernel trick to approximate softmax(QK^T)V
             without materializing the N×N matrix
  Complexity: O(N)
  Tradeoff:
    ✓ Linear in sequence length
    ✗ Approximation error — not exact softmax
    ✗ Quality often degrades vs standard attention on recall tasks

APPROACH 3: State Space Models (SSMs)
  Examples: Mamba, S4, Mamba-2, Jamba (hybrid)
  Mechanism: recurrent state compression — all past context
             is compressed into a fixed-size state vector
  Complexity: O(N) time, O(1) per-step memory at inference
  Tradeoff:
    ✓ Linear inference — scales to arbitrarily long sequences
    ✗ Information compression: precise token recall degrades
      as sequences grow (similar to LSTM vanishing issue)
    ✗ In-context learning weaker than transformer attention

  Mamba selectivity: unlike SSMs, Mamba learns input-dependent
  state transitions — better filtering of what to compress/retain

APPROACH 4: Ring Attention (sequence parallelism)
  Used for: training very long sequences across many GPUs
  Mechanism: split sequence across devices; each device holds
             a slice of Q, K, V and communicates via ring topology
  Not an architecture change — uses full softmax attention
  Enables: 100K+ context training on A100 clusters

APPROACH 5: Flash Attention + Long Chunked Prefill
  Practical approach used in most production systems:
  Use Flash Attention 2/3 with chunked prefill to process
  long prompts in memory-efficient blocks — not a new architecture,
  just efficient implementation of standard attention

CHOOSING:
  For production RAG (retrieved chunks in 32-128K context):
    → Standard attention + Flash Attention is sufficient
    → No need for architectural changes; chunked retrieval avoids
      the need for native 1M-token context

  For genuine 1M-token native context:
    → Sparse/sliding window attention (Mistral-class) is most
      production-ready today
    → Mamba-transformer hybrids (Jamba) promising but less mature
```

---

## 15. Positional Encodings — Sinusoidal, RoPE, ALiBi

Transformers have no inherent sense of order — positional encodings inject token position information.

```
APPROACH 1: Sinusoidal (Original Transformer, Vaswani et al. 2017)
  PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
  Added directly to token embeddings before layer 1.

  Properties:
    - Unique encoding per position
    - Different frequencies per dimension → relative positions are learnable
    - Theoretically extrapolates; in practice quality degrades beyond train length

APPROACH 2: Learned Absolute Positional Embeddings (GPT-2, BERT)
  Learn a separate embedding per position: Embedding(max_seq_len, d_model)
  Advantage: jointly optimized
  Disadvantage: hard limit at max_seq_len; no generalization beyond it

APPROACH 3: RoPE (Rotary Position Embedding) — LLaMA, Mistral, DeepSeek
  Su et al. 2022. The dominant approach for modern LLMs.

  Key idea: ROTATE Q and K vectors by an angle proportional to their position,
  then compute attention. The dot product becomes relative-position-sensitive.

  For each dimension pair (2i, 2i+1) at position m:
    q_m = R(m × θ_i) · q     θ_i = 10000^(-2i/d_model)
    k_n = R(n × θ_i) · k

  Attention score: q_m · k_n = q^T · R((m-n) × θ_i) · k
  → Score depends ONLY on relative offset (m-n), not absolute positions.

  Why RoPE wins:
    ✓ Relative position baked into attention score directly
    ✓ Zero extra parameters
    ✓ Context extension via frequency scaling (YaRN, NTK-aware)
    ✓ Used by every major modern open-weight model

  Context extension with RoPE:
    Linear scaling: divide θ_i by scale factor λ → stretches to λ× context
    YaRN (Peng et al. 2023): per-frequency scale + temperature scaling;
      achieves 128K context from 4K-trained model with minimal fine-tuning
    LLaMA-3.1: RoPE scaling + long-context fine-tuning → 128K context

APPROACH 4: ALiBi (Attention with Linear Biases) — MPT, BLOOM
  Adds a linear penalty to pre-softmax attention scores:
    score(i, j) = q_i · k_j / sqrt(d) − m × |i − j|
  where m is a fixed head-specific slope (no learning).

  Properties:
    ✓ Zero overhead — one scalar subtraction per score
    ✓ Strong extrapolation: train at 1K tokens, infer at 2-4K
    ✗ Weaker at very long-range token recall vs RoPE + YaRN

COMPARISON:
┌────────────────────┬──────────┬──────────────────┬──────────────────────────┐
│ Method             │ Params   │ Context extension │ Models                   │
├────────────────────┼──────────┼──────────────────┼──────────────────────────┤
│ Sinusoidal         │ 0        │ Poor             │ Original Transformer      │
│ Learned absolute   │ len × d  │ None             │ GPT-2, BERT               │
│ RoPE               │ 0        │ Excellent (YaRN) │ LLaMA, Mistral, DeepSeek  │
│ ALiBi              │ 0        │ Good (2-4×)      │ MPT, BLOOM                │
└────────────────────┴──────────┴──────────────────┴──────────────────────────┘
```

---

## 16. Mixture of Experts (MoE)

MoE allows models to have more total parameters than active parameters — more capacity at the same compute cost.

**Why this section builds on what you already know:** You've just seen how the FFN layer in a transformer block (Section 5) processes every single token through the same large weight matrix. MoE replaces that single FFN with N specialized FFN "experts" and routes each token to only K of them. The router is just a small linear layer + softmax — the same building blocks you already know.

```
THE CORE IDEA:
  Dense FFN: every token processed by ALL neurons in one big FFN
  MoE FFN:   N expert FFN layers; each token routed to K experts (K << N)
  → Same FLOPs per token, N× more parameters → more capacity per FLOP

ARCHITECTURE:
  Each MoE layer replaces one dense FFN with:
    1. Router (gating): x → softmax(W_gate · x) → shape (N,)
    2. Top-K selection: pick K experts with highest gate scores
    3. Weighted sum: output = Σ_{i in top_K} G[i] × Expert_i(x)

  Mixtral 8x7B: N=8 experts, K=2 (top-2 per token per layer)
    Total params: ~56B; active params per token: ~14B
    Training FLOPs ≈ 14B dense model; quality ≈ 30-40B dense model

EXPERT COLLAPSE — THE KEY FAILURE MODE:
  Router always prefers 1-2 experts → rest never trained → wasted capacity.
  Root cause: initial randomness → slight early advantage → more tokens →
  more gradient updates → wider advantage → full collapse.

  Fix: Auxiliary load balancing loss:
    L_balance = α × N × Σ_i (f_i × p_i)
    f_i = fraction of tokens going to expert i (empirical, non-differentiable)
    p_i = mean routing probability for expert i (differentiable)
    α ≈ 0.01 (typical); minimizing when f_i = p_i = 1/N for all i

  Expert-choice routing (alternative):
    Each expert selects its top-B tokens (not the other way around).
    Guarantees perfect load balance; risk: some tokens skipped entirely.

INFERENCE CHALLENGES:
  Memory: ALL N experts loaded simultaneously (even if only K active)
    Mixtral 8x7B: ~56B × 2 bytes (BF16) = ~112GB → requires 2× A100 80GB

  Batch-size sensitivity:
    Small batches: most experts idle → poor GPU utilization
    Large batches: all experts get tokens → near-full utilization
    → MoE thrives at data-center scale, hurts at edge deployment

  Expert parallelism: dedicate one GPU per expert, route via all-to-all

MoE vs DENSE:
  ✓ Use MoE: training compute limited; high-throughput inference; large GPU fleet
  ✗ Use Dense: memory-constrained; low-latency single-request; simpler ops
```

---

## 17. Training Stability — Large-Scale LLM Training

```
MIXED PRECISION — BF16 vs FP16:

  FP16 (16-bit IEEE 754): 5 exponent bits, 10 mantissa bits
    Max value: ±65504 — attention logits can exceed this → NaN → crash

  BF16 (Brain Float 16): 8 exponent bits, 7 mantissa bits
    Same exponent range as FP32 (max ~3.4 × 10^38)
    Less precision but virtually never overflows → stable for transformers

  AMP (Automatic Mixed Precision) standard recipe:
    Master weights: FP32 (precision for optimizer updates)
    Forward + backward: BF16 (fast Tensor Core compute)
    Gradient accumulation: FP32 (prevent precision loss)
    → Near-FP32 quality at ~2× throughput; BF16 needs no loss scaling

GRADIENT CHECKPOINTING:
  Problem: backprop needs all intermediate activations in memory.
  70B model, 4K context → activation memory ≈ 70-100+ GB → OOM.

  Solution: store only checkpoint activations at block boundaries;
  recompute internals during backward pass as needed.

  Cost: ~33% extra compute (one extra forward pass per block)
  Benefit: ~70-80% activation memory saved
  Code: model.gradient_checkpointing_enable() (HuggingFace Transformers)

ZeRO OPTIMIZER (DeepSpeed, Rajbhandari et al. 2020):
  Problem: Adam optimizer states for 70B:
    70B × (FP32 weight + m + v) × 4 bytes ≈ 840GB → no single GPU holds this

  ZeRO partitions states across N data-parallel GPUs:
    Stage 1: optimizer states only → 4× memory reduction per GPU
    Stage 2: + gradients → 8× reduction
    Stage 3: + model parameters → N× reduction (linear in GPU count)
             Weights gathered on-the-fly; more communication overhead

  Production setup: ZeRO Stage 2-3 + Tensor Parallelism (for layer-wide ops)

LOSS SPIKES:
  See Section 6 of this module for full diagnostic.
  Additional: z-loss regularization (PaLM) adds penalty on logit magnitude:
    L_z = ε × log(Σ exp(logits))²   (ε ≈ 1e-4)
  Prevents logits from growing unboundedly → reduces catastrophic spikes.
```

---
*Next: [13 — Accessing Models](../13-accessing-models/README.md)*
