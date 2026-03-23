# 00 — History of AI: From Turing to Today

> **The "why" behind modern AI.** Understanding where AI came from is essential for
> knowing why certain architectures exist, why certain problems are hard, and where
> the field is likely heading. This section traces the full intellectual lineage.

---

## Table of Contents

1. [What is Intelligence? The Philosophical Start](#1-what-is-intelligence)
2. [The Birth of AI (1940s–1950s)](#2-birth-of-ai)
3. [The First AI Boom (1950s–1960s)](#3-first-ai-boom)
4. [The First AI Winter (1970s)](#4-first-ai-winter)
5. [Expert Systems & Second Boom (1980s)](#5-expert-systems)
6. [The Second AI Winter (Late 1980s–1990s)](#6-second-ai-winter)
7. [Statistical ML & Data Era (1990s–2000s)](#7-statistical-ml)
8. [The Deep Learning Revolution (2006–2012)](#8-deep-learning-revolution)
9. [ImageNet & the GPU Moment (2012)](#9-imagenet-moment)
10. [The Rise of Word Embeddings (2013)](#10-word-embeddings)
11. [Sequence-to-Sequence & Attention (2014–2015)](#11-seq2seq-attention)
12. [The Transformer Era Begins (2017)](#12-transformer-era)
13. [BERT, GPT, and Transfer Learning (2018–2019)](#13-bert-gpt)
14. [GPT-3 and the Scale Hypothesis (2020)](#14-gpt3-scale)
15. [Instruction Tuning & RLHF (2021–2022)](#15-instruction-rlhf)
16. [ChatGPT and the Public Moment (2022–2023)](#16-chatgpt)
17. [The Multi-Model Race (2023–Present)](#17-multi-model-race)
18. [The Agentic Era (2024–Present)](#18-agentic-era)
19. [Key Themes & Recurring Patterns](#19-key-themes)

---

## 1. What is Intelligence?

Before building it, people had to define it.

**Alan Turing's Imitation Game (1950):**
- Paper: *"Computing Machinery and Intelligence"*
- Proposed: instead of defining intelligence, ask "can a machine behave indistinguishably from a human?"
- The **Turing Test** — if a human interrogator cannot distinguish the machine from a human in text conversation, the machine passes
- Still debated: does passing the Turing Test mean the machine *understands*, or just *mimics*?

**John McCarthy coined "Artificial Intelligence" (1956):**
- Dartmouth Conference, summer 1956
- McCarthy, Minsky, Shannon, and others gathered to work on making machines that "use language, form abstractions and concepts, solve problems now reserved for humans, and improve themselves"
- Estimated 2 months. Took 70+ years.

**Key philosophical debates that still matter:**
- **Strong AI vs Weak AI**: Does the machine truly think, or just compute?
- **The Chinese Room (Searle, 1980)**: Syntactic manipulation of symbols ≠ semantic understanding
- **Emergent behavior**: Can understanding arise from scale alone?

---

## 2. Birth of AI (1940s–1950s)

### Foundational Work

| Year | Person | Contribution |
|------|--------|-------------|
| 1943 | McCulloch & Pitts | First mathematical model of a neuron (binary threshold logic) |
| 1945 | Von Neumann | Stored-program computer architecture |
| 1948 | Norbert Wiener | *Cybernetics* — feedback loops, control systems, self-regulation |
| 1950 | Alan Turing | *"Computing Machinery and Intelligence"* — Turing Test |
| 1951 | Minsky & Edmonds | SNARC — first neural network computer (40 neurons, vacuum tubes) |
| 1956 | Dartmouth Conference | Birth of AI as a field |
| 1957 | Frank Rosenblatt | **Perceptron** — first trainable neural network |
| 1958 | McCarthy | **Lisp** — the language of AI for 30 years |

### The Perceptron (1957)

Rosenblatt's perceptron could:
- Take binary inputs
- Apply weights
- Threshold to produce 0 or 1

```
inputs: x1, x2, x3
output: 1 if (w1*x1 + w2*x2 + w3*x3) > threshold, else 0
```

**Learning rule**: if output wrong, adjust weights in the direction that would have made it right.

Claimed to be able to learn ANY pattern. This created massive hype.

---

## 3. The First AI Boom (1950s–1960s)

### Optimism Peaked

- **Logic Theorist (1955)**: Newell & Simon's program that proved 38 of 52 theorems from Principia Mathematica
- **General Problem Solver (1957)**: First attempt at general AI — extract problem-solving from domain knowledge
- **ELIZA (1966)**: Joseph Weizenbaum's chatbot using pattern matching. Simulated a therapist. Users attached to it emotionally — alarmed Weizenbaum himself
- **SHRDLU (1970)**: Terry Winograd's natural language system that could manipulate virtual blocks in a miniature world. Showed NLP could work in constrained domains

### Famous Predictions

> "Within 10 years a digital computer will be the world's chess champion" — Herbert Simon, 1957
> "Within 10 years machines will be capable of doing any work a man can do" — Herbert Simon, 1965
> "In from three to eight years we will have a machine with the general intelligence of an average human being" — Marvin Minsky, 1970

None came true. This set up the first winter.

### The Perceptron Limitation (1969)

Minsky & Papert's book *"Perceptrons"* proved:
- A single perceptron **cannot** learn XOR
- XOR is not linearly separable
- Implied fundamental limitations on neural networks

This single finding devastated neural network research funding for ~15 years.

```
XOR truth table:
0 XOR 0 = 0
0 XOR 1 = 1    ← Cannot draw a line that separates 0s from 1s
1 XOR 0 = 1
1 XOR 1 = 0
```

---

## 4. The First AI Winter (1974–1980)

### What Happened

**The Lighthill Report (1973):** UK government commissioned Sir James Lighthill to evaluate AI. Conclusion: AI was not delivering on its promises. Result: UK cut most AI funding.

**DARPA pulled funding from speech recognition (1974)**: After years of work, the gap between demos and real-world performance was too large.

### Root Causes

1. **Combinatorial explosion**: Problems that worked on 10 examples failed on 1000. Search space grows exponentially
2. **No common sense**: AI systems knew facts but couldn't reason about the world
3. **Brittleness**: Systems broke completely on inputs slightly outside training distribution
4. **Hardware limitations**: Computers were too slow and had insufficient memory

### The Connectionism vs Symbolism Divide

Two competing paradigms emerged that still echo today:

| Symbolism (GOFAI) | Connectionism |
|-------------------|--------------|
| Explicit rules and logic | Learned representations |
| Interpretable | Black box |
| Expert systems | Neural networks |
| Good at constrained domains | Generalizes better |
| 1950s–1980s dominant | 2010s–present dominant |

---

## 5. Expert Systems & Second Boom (1980s)

### What Are Expert Systems?

Programs that encode expert knowledge as IF-THEN rules:

```
IF patient_has_fever AND patient_has_cough
THEN possible_diagnosis = flu (confidence: 0.7)

IF possible_diagnosis = flu AND duration > 14 days
THEN escalate_to_doctor
```

### Key Systems

| System | Year | Domain |
|--------|------|--------|
| MYCIN | 1972 | Medical diagnosis (bacterial infections) |
| XCON/R1 | 1980 | Configure DEC computer orders — saved $40M/year |
| PROSPECTOR | 1978 | Mineral exploration |
| DENDRAL | 1965 | Chemical analysis |

### The Economic Boom

- By 1985, expert systems were a billion-dollar industry
- Every major corporation was building them
- New AI hardware companies (Lisp Machines) founded
- Japan's Fifth Generation Computer Project (1982–1992): government initiative to build AI supercomputers

### Backpropagation Rediscovered (1986)

Rumelhart, Hinton, and Williams published *"Learning representations by back-propagating errors"* — the algorithm that makes multi-layer neural networks trainable.

Key insight: chain rule of calculus allows error to be propagated backwards through multiple layers, adjusting all weights.

This was the technical unlock for deep networks — but computers were still too slow.

---

## 6. The Second AI Winter (Late 1980s–1990s)

### The Expert System Problem

Expert systems had fundamental flaws:
1. **Brittleness**: Outside the rule base, they failed completely
2. **Knowledge acquisition bottleneck**: Expert knowledge is hard to extract and encode
3. **Maintenance**: Rules became inconsistent as the system grew
4. **No learning**: They couldn't improve from new data

XCON, the most successful expert system, eventually required 50 full-time engineers to maintain 10,000 rules.

### The Lisp Machine Market Collapse (1987)

Specialized AI hardware companies failed when general-purpose workstations (Sun, DEC) became faster and cheaper. A billion-dollar industry disappeared in 2 years.

### Japan's Fifth Generation Project Failure (1992)

Ended without achieving its goals. Symbolic AI hit a ceiling.

### Neural Network Limitations Remained

- Backprop worked on paper but training was slow
- Vanishing gradients made deep networks hard to train (>3 layers was difficult)
- No large datasets existed to train on

---

## 7. Statistical ML & Data Era (1990s–2000s)

### The Paradigm Shift

Instead of encoding rules, **learn patterns from data**.

Rejection of hand-crafted rules → embrace of statistical methods.

### Key Algorithms That Emerged

**Support Vector Machines (SVMs) — Vapnik, 1995**
- Find the maximum-margin hyperplane that separates classes
- Used kernel trick to handle non-linear boundaries
- Theoretically grounded (PAC learning, VC dimension)
- Dominated classification through 2010s

**Decision Trees & Random Forests**
- Quinlan's ID3/C4.5 (1986, 1993)
- Breiman's Random Forests (2001): ensemble of decision trees with feature randomness
- Interpretable, handles mixed data types, still widely used

**Gradient Boosting (1999)**
- Friedman's GBM → AdaBoost (Freund & Schapire)
- Sequentially build weak learners, each correcting the last
- XGBoost (2014) became the dominant tabular ML algorithm

**Bayesian Methods**
- Naive Bayes: powerful for text classification, email spam filtering
- Hidden Markov Models: dominated speech recognition until ~2012

### NLP in the Statistical Era

The dominant paradigm was **n-gram language models**:
- Count occurrences of word sequences
- Predict next word based on previous N words
- Google's translation system used 5-gram models over 230 billion words

Problem: can't generalize to unseen word combinations; no understanding of meaning.

### The Data Availability Shift

**ImageNet (2009):** Fei-Fei Li's project — 14 million labeled images, 1000 categories. Created the benchmark that forced the deep learning breakthrough.

**The Web as training data:** By 2000s, the internet provided text at unprecedented scale. Statistical language models could be trained on billions of words.

---

## 8. The Deep Learning Revolution (2006–2012)

### Geoffrey Hinton's Comeback

In 2006, Hinton, Osindero, and Teh published *"A Fast Learning Algorithm for Deep Belief Nets"*:
- Used **layer-by-layer pretraining** to initialize deep networks
- Then fine-tuned with backpropagation
- Could train deeper networks than previously possible

This was a critical bridge: it showed deep networks *could* be trained, even if the mechanism (unsupervised pretraining as initialization) was later replaced.

### Why Now? Three Factors

1. **GPUs**: NVidia's CUDA (2007) allowed parallel matrix multiplication — the core operation of neural networks — at 10–100x speedup over CPUs
2. **Data**: ImageNet, web text, digitized books provided training sets at scale
3. **Algorithms**: Dropout, ReLU, better initialization schemes solved vanishing gradient

### The ReLU Activation Function

Old: Sigmoid and tanh saturate, causing vanishing gradients in deep networks
New (Glorot & Bengio, 2011): **Rectified Linear Unit (ReLU)**

```
ReLU(x) = max(0, x)
```

- No saturation for positive values
- Gradient is 1 for x > 0 — no vanishing
- Computationally simple
- Enabled training of much deeper networks

### Dropout (Srivastava et al., 2014)

Regularization technique: randomly zero out neurons during training (probability p = 0.5 typically).

Effect: prevents co-adaptation of neurons, forces network to learn redundant representations. Acts like training an ensemble of exponentially many networks.

---

## 9. The ImageNet Moment (2012)

### AlexNet: The Inflection Point

**ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2012:**

| Team | Top-5 Error |
|------|------------|
| Best non-deep-learning | 26.1% |
| **AlexNet (Krizhevsky, Sutskever, Hinton)** | **16.4%** |

A **10 percentage point gap**. The community was stunned.

**AlexNet Architecture:**
- 8 layers (5 convolutional + 3 fully connected)
- 60 million parameters
- Trained on 2 GPUs for 5–6 days
- Used ReLU, dropout, data augmentation, local response normalization
- Won by such a margin that nearly every subsequent ILSVRC winner used CNNs

This single result convinced most of the computer vision and ML community to abandon hand-crafted features.

### The GPU Gold Rush Begins

- NVidia stock price correlation with AI research became visible
- Cloud providers began offering GPU instances
- Deep learning frameworks: Theano (2008), then Caffe (2014), then TensorFlow (2015), then PyTorch (2017)

### Year-Over-Year ImageNet Progress

| Year | Winner | Top-5 Error |
|------|--------|------------|
| 2012 | AlexNet | 16.4% |
| 2013 | ZFNet | 11.7% |
| 2014 | VGGNet | 7.3% |
| 2014 | GoogLeNet/Inception | 6.7% |
| 2015 | ResNet | 3.57% |
| 2016 | Ensemble | 2.99% |

Human performance estimated at ~5%. ResNet surpassed it in 2015.

---

## 10. The Rise of Word Embeddings (2013)

### The Problem with Symbolic Representations

In traditional NLP, words were represented as one-hot vectors:
```
vocabulary size = 100,000 words
"cat" = [0, 0, 0, ..., 1, ..., 0]  (only one 1, the rest zeros)
```

Problems:
- 100,000 dimensional space
- No notion of similarity — "cat" and "dog" are as different as "cat" and "airplane"
- Cannot generalize across similar words

### Word2Vec (Mikolov et al., Google, 2013)

Idea: train a neural network to predict words from context (or context from words). The hidden layer weights become the word representations.

Two architectures:
- **CBOW**: predict target word from context words
- **Skip-gram**: predict context words from target word

Result: dense 300-dimensional vectors where similar words cluster together.

Famous analogy:
```
king - man + woman ≈ queen
Paris - France + Germany ≈ Berlin
```

This showed that **meaning could be encoded as geometry in vector space**.

### GloVe (Pennington et al., Stanford, 2014)

Global Vectors for Word Representation:
- Used global word co-occurrence statistics
- Faster to train than Word2Vec
- Better on word analogy tasks

### FastText (Facebook, 2016)

Extension: represent words as bags of character n-grams.
- Can represent rare words and misspellings
- Generates vectors for out-of-vocabulary words

### Impact

Word embeddings became the default "layer 1" of every NLP system from 2013–2018. Pre-trained word vectors were downloaded millions of times.

---

## 11. Sequence-to-Sequence & Attention (2014–2015)

### The Sequence Modeling Problem

RNNs and LSTMs (covered fully in `07-sequence-models/`) could handle sequences, but had a fundamental bottleneck in machine translation:

**The Encoder-Decoder Bottleneck:**
- Encoder reads entire input sentence
- Compresses to a fixed-size vector (the "context vector")
- Decoder generates output from that vector

Problem: for long sentences, one vector cannot encode all relevant information.

### Attention Mechanism (Bahdanau et al., 2014)

*"Neural Machine Translation by Jointly Learning to Align and Translate"*

Key innovation: at each decoding step, **look back at ALL encoder hidden states**, weighted by relevance.

```
For each output word:
  attention_scores = how_relevant(each_input_position, current_decoding_state)
  context = weighted_sum(encoder_hidden_states, attention_weights)
  output_word = generate(context, previous_output)
```

This was a watershed moment. The model could learn to **align** source and target words without being told how.

Visualization of attention weights showed that the model learned to attend to the right source words when generating each target word — interpretable and impressive.

### Why This Mattered

Attention:
1. Solved the bottleneck problem for long sequences
2. Provided interpretability (which input words influenced which output words)
3. Contained the core idea that would later become the Transformer

### Image Caption Generation (2014–2015)

Google's Show and Tell: combine CNN for image encoding + LSTM with attention for caption generation. Demonstrated that attention could work across modalities.

---

## 12. The Transformer Era Begins (2017)

### "Attention Is All You Need"

Vaswani et al. (Google Brain, 2017) — arguably the most important ML paper of the decade.

**The insight**: RNNs process sequences step-by-step (sequential computation). What if you use **only attention**, applied to the full sequence in parallel?

**Multi-Head Self-Attention:**
- Every token attends to every other token
- Computed in parallel (not sequential like RNNs)
- Multiple "heads" capture different types of relationships

**Key benefits over RNNs:**
1. **Parallelizable**: can train on GPUs much faster
2. **No vanishing gradient over long distances**: direct connections between any two positions
3. **Better at capturing long-range dependencies**

Full coverage of the Transformer architecture is in **`09-how-llms-work/README.md`**.

### First Applications

Initial use case: machine translation (the paper was framed as MT). Achieved state-of-the-art with less computation than previous models.

Researchers quickly realized the Transformer architecture was broadly applicable.

---

## 13. BERT, GPT, and Transfer Learning in NLP (2018–2019)

### The ImageNet Moment for NLP

In computer vision, ImageNet pretraining became standard: pretrain on ImageNet, fine-tune on your task. This transfer learning paradigm transformed CV.

In 2018, two papers did this for NLP simultaneously, but with different approaches.

### BERT (Google, 2018)

*Bidirectional Encoder Representations from Transformers*

Approach: **Masked Language Model (MLM)**
- Mask 15% of tokens
- Predict the masked tokens
- Forces model to use bidirectional context

Training tasks:
1. **MLM**: predict masked tokens (left AND right context)
2. **Next Sentence Prediction (NSP)**: predict if two sentences are consecutive

Result: a rich contextual representation of text. Fine-tune on downstream tasks with minimal task-specific layers.

Achieved state-of-the-art on 11 NLP benchmarks simultaneously.

**Key innovation**: bidirectional context (GPT was left-to-right only).

### GPT (OpenAI, 2018)

*Generative Pre-trained Transformer*

Approach: **Causal Language Modeling (CLM)**
- Predict the next token given all previous tokens
- Left-to-right (autoregressive)

Same pretraining + fine-tuning paradigm, but generated text rather than just representing it.

GPT-1 (117M parameters) showed that pretraining on diverse text then fine-tuning worked well.

### GPT-2 (OpenAI, 2019)

1.5 billion parameters. OpenAI released it in stages, fearing misuse — that decision itself became news.

**The surprise**: GPT-2 showed emergent capabilities. Zero-shot task performance appeared — the model could summarize, translate, and answer questions without being explicitly trained for these tasks.

Quote from paper: *"language models are unsupervised multitask learners"*

---

## 14. GPT-3 and the Scale Hypothesis (2020)

### Scaling Laws (Kaplan et al., OpenAI, 2020)

Neural language model performance follows predictable power laws with:
- Model size (parameters)
- Dataset size (tokens)
- Compute budget (FLOPs)

```
Loss ∝ (Parameters)^-0.076
Loss ∝ (Data)^-0.095
Loss ∝ (Compute)^-0.050
```

This was the key empirical finding: **just make it bigger and it gets better, predictably**.

### GPT-3 (OpenAI, 2020)

175 billion parameters. Trained on ~300 billion tokens.

The qualitative leap:
- **In-context learning**: provide a few examples in the prompt, and the model learns the pattern without weight updates
- **Emergent capabilities**: translation, arithmetic, code generation, chain-of-thought reasoning — none explicitly trained for

Few-shot performance rivaled fine-tuned models on many benchmarks.

**API access** was the business model: OpenAI didn't release weights, but provided API access. This planted the seed for the modern AI API ecosystem.

### The Scale Hypothesis

Proposed by Gwern Branwen and others: general intelligence might simply emerge from enough scale. No architectural innovation needed — just more data, more compute, more parameters.

Controversial but empirically supported.

---

## 15. Instruction Tuning & RLHF (2021–2022)

### The Alignment Problem with Base LLMs

GPT-3 was a powerful text completer. Ask it a question, it might:
- Answer it
- Generate more questions
- Write a news article about the topic
- Produce harmful content

Base LLMs optimize for predicting the next token in their training data — not for being helpful assistants.

### InstructGPT (OpenAI, 2022)

The paper that made LLMs useful:

**Three-stage process:**

**Stage 1: Supervised Fine-Tuning (SFT)**
- Contractors wrote high-quality responses to prompts
- Fine-tuned GPT-3 on these examples
- Creates a model that "knows" what good responses look like

**Stage 2: Reward Model Training**
- Show contractors model responses
- Rank them from best to worst
- Train a separate reward model to predict human preferences

**Stage 3: Reinforcement Learning from Human Feedback (RLHF)**
- Use PPO (Proximal Policy Optimization) to fine-tune the SFT model
- Reward signal: the reward model's score
- KL-divergence penalty prevents model from going too far from base model

Result: InstructGPT was preferred by human evaluators 77% of the time over GPT-3, even though it was 100x smaller (1.3B vs 175B parameters).

**The insight**: alignment data efficiency is extraordinary. A little human feedback goes very far.

### Constitutional AI (Anthropic, 2022)

Claude's training approach:
1. **SFT** on helpful, harmless, honest responses
2. **CAI** (Constitutional AI): model critiques and revises its own outputs based on written principles
3. **RL from AI Feedback (RLAIF)**: use AI rather than humans to generate preference labels

This reduced dependence on expensive human labeling.

### DPO (Direct Preference Optimization, Rafailov et al., 2023)

Simplified RLHF: optimize directly on the human preference data without a separate reward model or RL loop. Same outcome, much simpler training.

Became widely used in open-source fine-tuning.

---

## 16. ChatGPT and the Public Moment (2022–2023)

### November 30, 2022

OpenAI released ChatGPT as a "research preview." Within 5 days: 1 million users. Within 2 months: 100 million users — the fastest consumer product growth in history.

**What made it different from the API:**
- Chat interface (multi-turn conversation)
- Memory of conversation history
- Polished instruction following
- Free to use

The general public experienced, for the first time, a conversational AI that felt genuinely useful.

### Why It Was a Shock

Previous chatbots (ELIZA, Siri, Alexa, Google Assistant) worked on pattern matching and hand-crafted responses. ChatGPT could:
- Explain complex topics clearly
- Write code that worked
- Draft emails, essays, arguments
- Translate languages
- Role-play characters

### The Economic Response

- Microsoft invested $10 billion in OpenAI, integrated into Bing and Azure
- Google declared a "code red" — threat to core search business
- Google rushed Bard to market (initially struggled)
- Thousands of AI startups founded
- Every major tech company announced AI strategies

### Models Released in 2023

| Model | Organization | Parameters | Key Feature |
|-------|-------------|-----------|------------|
| GPT-4 | OpenAI | Unknown (~1T) | Vision, 128K context |
| Claude 2 | Anthropic | Unknown | 200K context |
| Gemini Ultra | Google | Unknown | Best on benchmarks |
| Llama 2 | Meta | 7B–70B | Open weights |
| Mistral 7B | Mistral | 7B | Efficient, open |
| Falcon | TII | 7B–180B | Open, multilingual |

---

## 17. The Multi-Model Race (2023–Present)

### Open vs Closed Model Divide

**Closed models (API only):**
- GPT-4, GPT-4o (OpenAI)
- Claude 3 family (Anthropic)
- Gemini family (Google)
- Proprietary, best performance, expensive

**Open-weight models:**
- LLaMA 2, LLaMA 3 (Meta) — most important
- Mistral, Mixtral (Mistral AI) — showed small models can be competitive
- Phi series (Microsoft) — showed data quality > quantity
- Qwen (Alibaba), Gemma (Google)

### Key Technical Advances 2023–2024

**Mixture of Experts (MoE):**
- Mixtral 8x7B: 8 expert FFN layers, 2 activated per token
- Same quality as dense model, 4x cheaper inference
- GPT-4 rumored to be MoE
- Became standard for efficient large models

**Context Length Expansion:**
- 2022: 4K tokens (GPT-3.5)
- 2023: 128K tokens (GPT-4), 200K (Claude 2)
- 2024: 1M tokens (Gemini 1.5 Pro)
- Enabled: process entire books, codebases, hours of transcripts

**Multimodal Models:**
- GPT-4V: vision understanding
- GPT-4o: audio + vision + text simultaneously
- Gemini 1.5 Pro: video understanding
- Claude 3: vision + text

**Reasoning Models:**
- OpenAI o1 (2024): "thinking" before answering using chain-of-thought
- DeepSeek R1: open-weight reasoning model that matched o1
- Showed that computation at inference time improves quality

### Chinchilla Scaling Laws (DeepMind, 2022)

Revised the scaling laws: for a given compute budget, the original GPT-3 was under-trained.

Optimal ratio: approximately **20 tokens of data per parameter**.

GPT-3 (175B params): should have been trained on 3.5T tokens, not 300B.

Chinchilla (70B params, 1.4T tokens) matched GPT-3 quality with 4x fewer parameters.

Implication: smaller models trained on more data can match or exceed larger models trained on less data.

---

## 18. The Agentic Era (2024–Present)

### From Text Generation to Autonomous Action

LLMs began to be used not just for generating text, but for taking actions in the world:
- Browsing the web
- Writing and executing code
- Calling APIs
- Managing files
- Operating computers

### Key Developments

**Function/Tool Calling (OpenAI, 2023)**
- Models could output structured JSON that called external functions
- Made AI integration into applications much easier
- Became standard across all major models

**Code Interpreters**
- GPT-4's Code Interpreter: execute Python in a sandbox
- Claude's computer use (2024): control mouse and keyboard
- Enabled end-to-end task completion

**Multi-Agent Systems**
- AutoGPT, BabyAGI (2023): early experiments in autonomous agents
- LangGraph, CrewAI: frameworks for coordinating multiple agents
- Each agent specializes, passes results to others

**Model Context Protocol (MCP)**
- Anthropic's protocol for standardized tool integration
- Covered fully in `21-mcp/README.md`

### The Infrastructure Layer

New category of companies building agent infrastructure:
- LangChain, LlamaIndex: frameworks
- Langfuse, Helicone: observability
- Weights & Biases: experiment tracking
- Modal, Replicate: model serving
- Together.ai, Groq: fast inference

---

## 19. Key Themes & Recurring Patterns

### Pattern 1: AI Winters Follow Hype

Every major AI wave has been followed by a winter when promises weren't delivered:
- Perceptron hype → Minsky/Papert critique → winter
- Expert system hype → brittleness → winter
- **Current question**: will LLM hype lead to a third winter?

### Pattern 2: Hardware Unlocks Algorithms

Algorithms often existed before the hardware to run them:
- Backpropagation (1986) → needed GPUs (2012) to be useful at scale
- Transformers (2017) → needed tensor cores (2018+) for efficient training
- Key insight: often the bottleneck is compute, not ideas

### Pattern 3: Data > Architecture

Time and again, more data has beaten architectural innovations:
- N-gram models + web scale data defeated linguistically-motivated models
- GPT-2/3 simple architecture + massive data defeated task-specific models
- Chinchilla showed training duration matters as much as model size

### Pattern 4: Scale Produces Emergence

Capabilities that appear suddenly at certain scales:
- In-context learning appeared around 10B parameters
- Chain-of-thought reasoning appeared around 100B
- These weren't predicted; they emerged

### Pattern 5: Transfer Learning Always Wins

Pre-train on large general dataset → fine-tune on specific task:
- ImageNet → Computer Vision fine-tuning
- BERT/GPT → NLP fine-tuning
- Foundation models → Everything

### Pattern 6: Simpler Architectures Win at Scale

Complex models often give way to simpler ones trained with more data:
- LSTM with many engineering tricks → Transformer (simpler, more parallelizable)
- Complex RL algorithms → PPO + simple reward models
- Hypothesis: the "right" architecture is the one that parallelizes training

---

## Timeline Summary

```
1943  McCulloch-Pitts neuron
1950  Turing Test proposed
1956  Dartmouth Conference — AI founded
1957  Perceptron (Rosenblatt)
1966  ELIZA — first chatbot
1969  Perceptron limitations (Minsky & Papert) → First AI Winter
1974  First AI Winter
1980s Expert Systems boom
1986  Backpropagation rediscovered (Rumelhart, Hinton, Williams)
1987  Lisp Machine collapse → Second AI Winter
1995  Support Vector Machines (Vapnik)
1997  Deep Blue beats Kasparov in chess
2001  Random Forests (Breiman)
2006  Deep Belief Nets, layer-wise pretraining (Hinton)
2007  CUDA — GPU computing becomes accessible
2009  ImageNet dataset released (Fei-Fei Li)
2011  AlexNet built (tested 2012)
2012  AlexNet wins ImageNet by 10pp → Deep Learning Revolution
2013  Word2Vec (Mikolov) — word embeddings
2014  Attention mechanism (Bahdanau) — seq2seq translation
2014  GANs (Goodfellow)
2015  ResNet — superhuman image recognition
2015  TensorFlow released (Google)
2017  "Attention Is All You Need" — Transformer paper
2017  PyTorch released (Facebook)
2018  BERT (Google) and GPT (OpenAI)
2019  GPT-2, Transformer-XL
2020  GPT-3 (175B), Scaling Laws paper
2021  GitHub Copilot
2022  InstructGPT, RLHF, ChatGPT (Nov 30)
2023  GPT-4, Claude 2, Llama 2, Mixtral, function calling
2024  Claude 3, GPT-4o, Llama 3, reasoning models (o1, R1)
2025  Agentic AI, MCP, multimodal at scale
```

---

## Key Points for Exams

1. The **Turing Test** (1950) proposed behavioral intelligence criterion — still debated
2. **First AI Winter** (1974) caused by combinatorial explosion + Perceptron limitations
3. **Second AI Winter** (late 1980s) caused by expert system brittleness
4. **AlexNet (2012)** won ImageNet by 10pp gap → triggered deep learning revolution
5. **Word2Vec (2013)** showed meaning can be encoded as vector geometry
6. **Attention mechanism (2014)** solved the bottleneck problem in seq2seq
7. **Transformer (2017)** replaced RNNs with parallel self-attention
8. **BERT vs GPT**: bidirectional encoder vs autoregressive decoder
9. **Scaling Laws (2020)**: performance predictably scales with parameters, data, compute
10. **InstructGPT / RLHF (2022)**: alignment gap between base LLM and useful assistant
11. **ChatGPT (Nov 2022)**: fastest consumer product adoption in history
12. **Chinchilla (2022)**: smaller model + more data can match larger model

---

## Practice Questions

1. What was the Dartmouth Conference and why does it matter?
2. Why did the Minsky-Papert paper cause an AI winter? What did it actually prove?
3. What made expert systems ultimately fail?
4. What three factors converged to enable the deep learning revolution around 2012?
5. Explain the difference between BERT and GPT pretraining objectives.
6. What is RLHF and why was it critical for making LLMs useful?
7. What are the Chinchilla scaling laws and what did they change?
8. What is the difference between a base language model and an instruction-tuned model?
9. Name three emergent capabilities that appeared at scale in LLMs.
10. What is the "scale hypothesis" and what evidence supports or challenges it?
