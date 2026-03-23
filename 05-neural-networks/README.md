# 00c — Neural Network Fundamentals

> **The building blocks of all modern AI.** Every LLM, every image model, every
> speech system is built on these foundations. Master this and the rest becomes
> pattern recognition.

---

## Table of Contents

1. [Biological Inspiration](#1-biological-inspiration)
2. [The Perceptron](#2-the-perceptron)
3. [Multi-Layer Perceptrons (MLPs)](#3-multi-layer-perceptrons)
4. [Activation Functions](#4-activation-functions)
5. [Loss Functions](#5-loss-functions)
6. [Forward Pass](#6-forward-pass)
7. [Backpropagation](#7-backpropagation)
8. [Gradient Descent and Optimizers](#8-gradient-descent-and-optimizers)
9. [Initialization](#9-initialization)
10. [Regularization Techniques](#10-regularization-techniques)
11. [Batch Normalization and Layer Normalization](#11-normalization-layers)
12. [Universal Approximation](#12-universal-approximation)
13. [Neural Network Architectures Overview](#13-architectures-overview)

---

## 1. Biological Inspiration

### The Neuron Analogy

A biological neuron:
- Receives signals through **dendrites**
- Sums those signals in the **cell body**
- Fires an output signal through the **axon** if the sum exceeds a threshold
- Connects to other neurons via **synapses**

The artificial neuron (1943, McCulloch-Pitts):
```
inputs → weighted sum → activation function → output
```

**Important caveat**: biological and artificial neurons are only loosely analogous. The "inspiration" is metaphorical. Modern neural networks are better understood as parameterized function approximators than as simulations of the brain.

---

## 2. The Perceptron

### Rosenblatt's Perceptron (1957)

The simplest learning machine:

```
Single perceptron:
  inputs: x₁, x₂, ..., xₙ
  weights: w₁, w₂, ..., wₙ
  bias: b

  pre-activation: z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b = wᵀx + b
  output: ŷ = 1 if z > 0, else 0
```

In vector form: `ŷ = step(wᵀx + b)`

### The Perceptron Learning Rule

```
For each training example (xᵢ, yᵢ):
  ŷ = predict(xᵢ)
  error = yᵢ - ŷ
  w = w + α × error × xᵢ
  b = b + α × error
```

If error = 0: no update. If error = 1 (predicted 0, actual 1): increase weights. If error = -1: decrease.

**Perceptron convergence theorem**: if the data is linearly separable, the algorithm converges to a solution in finite steps.

**XOR limitation**: cannot learn XOR because XOR is not linearly separable. A single hyperplane cannot separate the positive and negative examples.

This critical limitation required the move to multi-layer networks.

---

## 3. Multi-Layer Perceptrons (MLPs)

### Architecture

Stack multiple layers of neurons:

```
Input Layer → Hidden Layer(s) → Output Layer

[x₁]     [h₁]
[x₂] →  [h₂]  → [o₁]
[x₃]     [h₃]
         [h₄]
```

Each arrow represents a learned weight. Each node (except input) also has a bias.

**Layer sizes** (hyperparameters):
- Input layer: fixed by data dimensions
- Hidden layers: design choices
- Output layer: fixed by task (1 for binary, K for K-class)

### The Hidden Layer

A hidden layer with 4 units and ReLU activation:
```
h₁ = ReLU(w₁₁x₁ + w₁₂x₂ + w₁₃x₃ + b₁)
h₂ = ReLU(w₂₁x₁ + w₂₂x₂ + w₂₃x₃ + b₂)
h₃ = ReLU(w₃₁x₁ + w₃₂x₂ + w₃₃x₃ + b₃)
h₄ = ReLU(w₄₁x₁ + w₄₂x₂ + w₄₃x₃ + b₄)
```

In matrix form: `h = ReLU(Wx + b)`

where W is a (4, 3) weight matrix and b is a (4,) bias vector.

**Why hidden layers?** They learn intermediate representations. XOR example:
- Layer 1: learns to detect "at least one input is 1" and "both inputs are 1"
- Layer 2: computes "at least one" AND NOT "both" = XOR

### Depth vs Width

**Width**: number of neurons per layer
**Depth**: number of layers

- **Shallow and wide**: can approximate many functions, but may need exponentially many neurons
- **Deep and narrow**: can represent complex functions more efficiently
- Deep networks learn **hierarchical representations**: low layers learn simple features, high layers combine them into complex features

---

## 4. Activation Functions

Without nonlinear activations, stacking linear layers is still just a linear function. Activation functions introduce the nonlinearity that makes neural networks powerful.

### Step Function (Perceptron)

```
f(x) = 1 if x > 0, else 0
```

Not differentiable → can't use gradient descent.

### Sigmoid

```
σ(x) = 1 / (1 + e^(-x))
Range: (0, 1)
Derivative: σ(x) × (1 - σ(x))
```

**Used for**: binary classification output (probability of class 1).
**Problem for hidden layers**: saturation. When |x| is large, σ'(x) ≈ 0 → vanishing gradients through deep networks. Gradients near 0 prevent weights from updating.

### Tanh

```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
Range: (-1, 1)
Derivative: 1 - tanh²(x)
```

Better than sigmoid (zero-centered), but still saturates.

### ReLU (Rectified Linear Unit)

```
ReLU(x) = max(0, x)
Range: [0, ∞)
Derivative: 0 if x < 0, 1 if x > 0
```

**The dominant activation function since 2012.**

**Advantages:**
- No saturation for positive values → no vanishing gradient
- Computationally cheap (just a max operation)
- Sparse activations (many neurons output exactly 0)

**Problem**: "Dying ReLU" — if a neuron gets a very negative input, it outputs 0 and has gradient 0. It may never recover if this happens consistently. The neuron "dies."

### Leaky ReLU

```
f(x) = x if x > 0, else α × x   (α typically 0.01)
```

Fixes dying ReLU by giving a small gradient for negative inputs.

### ELU (Exponential Linear Unit)

```
f(x) = x if x > 0, else α(e^x - 1)
```

Smooth, negative values saturate to -α.

### GELU (Gaussian Error Linear Unit)

```
GELU(x) = x × Φ(x)
```
where Φ is the cumulative distribution function of the standard normal.

Approximately: `GELU(x) ≈ 0.5x × (1 + tanh(√(2/π) × (x + 0.044715x³)))`

**Used in BERT, GPT, and most modern transformers.**
Advantage: smooth, stochastic regularization interpretation.

### SiLU / Swish

```
SiLU(x) = x × σ(x)
```

Self-gated activation. Used in LLaMA and many modern LLMs.

### Softmax (Output Layer Only)

```
softmax(xᵢ) = e^xᵢ / Σⱼ e^xⱼ
```

Converts a vector of raw scores (logits) to probabilities that sum to 1. Used for multi-class classification output.

**Never use in hidden layers** (all units' gradients depend on each other).

### Activation Function Summary

| Function | Range | When to Use |
|----------|-------|-------------|
| Sigmoid | (0,1) | Binary output probability |
| Softmax | (0,1), sums to 1 | Multi-class output |
| Tanh | (-1,1) | RNNs (historical), LSTM gates |
| ReLU | [0,∞) | Default for hidden layers (CNNs, MLPs) |
| Leaky ReLU | (-∞,∞) | When dying ReLU is a problem |
| GELU | (-∞,∞) | Transformers (BERT, GPT) |
| SiLU | (-∞,∞) | LLaMA, modern LLMs |
| Linear (none) | (-∞,∞) | Regression output |

---

## 5. Loss Functions

The loss function measures how wrong the model is. We minimize it.

### Mean Squared Error (MSE) — Regression

```
MSE = (1/n) × Σ (yᵢ - ŷᵢ)²
```

- Penalizes large errors heavily (squared)
- Assumes Gaussian noise (MLE under Gaussian)
- Sensitive to outliers

**Mean Absolute Error (MAE):**
```
MAE = (1/n) × Σ |yᵢ - ŷᵢ|
```
- Robust to outliers
- Not differentiable at 0 (use subgradients)

**Huber Loss**: smooth approximation that's MSE near 0 and MAE far from 0. Best of both.

### Binary Cross-Entropy — Binary Classification

```
BCE = -(1/n) × Σ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]
```

Derived from MLE under Bernoulli distribution. ŷ is sigmoid output.

**Intuition**: if yᵢ=1 and model predicts ŷᵢ≈1 → loss ≈ 0. If yᵢ=1 and ŷᵢ≈0 → loss → ∞ (heavily penalized).

### Categorical Cross-Entropy — Multi-class Classification

```
CCE = -(1/n) × Σᵢ Σₖ yᵢₖ × log(ŷᵢₖ)
```

For one-hot y, simplifies to: `-log(ŷ_correct_class)`.

Derived from MLE under Categorical distribution. ŷ is softmax output.

**NLL (Negative Log-Likelihood)** is the same thing with a different name.

### Loss Functions for Language Models

**Perplexity** — standard language model evaluation metric:
```
Perplexity = exp(average cross-entropy per token)
           = exp(-(1/T) × Σₜ log P(wₜ | w₁...wₜ₋₁))
```

Lower perplexity = better model (predicting tokens more confidently).

---

## 6. Forward Pass

### Computing the Output

For an L-layer MLP:

```python
# Layer 0: input
a[0] = x

# For each layer l:
z[l] = W[l] @ a[l-1] + b[l]   # pre-activation (linear)
a[l] = activation(z[l])         # post-activation (nonlinear)

# Output
ŷ = a[L]
```

This is a **computation graph**: a directed acyclic graph where:
- Nodes are values (tensors)
- Edges are operations

Modern frameworks (PyTorch, JAX) build this graph automatically during the forward pass.

### PyTorch Example

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

model = MLP(784, 256, 10)
x = torch.randn(32, 784)  # batch of 32 images
output = model(x)          # shape: (32, 10)
```

---

## 7. Backpropagation

### The Core Algorithm

Backpropagation computes the gradient of the loss with respect to all weights using the chain rule.

**The Chain Rule in Neural Networks:**

For loss L, through layers L → l+1 → ... → l:
```
∂L/∂W[l] = ∂L/∂a[l] × ∂a[l]/∂z[l] × ∂z[l]/∂W[l]
```

Where:
- ∂L/∂a[l]: "upstream gradient" — how loss changes with layer l's output
- ∂a[l]/∂z[l]: derivative of activation function
- ∂z[l]/∂W[l]: this is just the layer's input a[l-1]

### Step-by-Step

```
Forward pass: compute all a[l] and z[l], store them
Backward pass: compute gradients starting from output

δ[L] = ∂L/∂z[L]  (output layer error)

For l = L-1, L-2, ..., 1:
  δ[l] = (W[l+1]^T × δ[l+1]) ⊙ f'(z[l])

  ∂L/∂W[l] = δ[l] × a[l-1]^T
  ∂L/∂b[l] = δ[l]
```

Where ⊙ is element-wise multiplication.

### Automatic Differentiation (Autograd)

In practice, you never implement backprop manually. PyTorch/JAX/TensorFlow do it automatically.

**How autograd works:**
1. During forward pass, record all operations in a computation graph
2. Each operation node stores how to compute its local gradient
3. `loss.backward()` traverses the graph in reverse, multiplying local gradients (chain rule)

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1  # y = x² + 3x + 1
y.backward()
print(x.grad)  # dy/dx at x=2 = 2x + 3 = 7
```

### The Vanishing Gradient Problem

In deep networks, gradients can become exponentially small as they propagate backward:

```
δ[l] = (W[l+1]^T × δ[l+1]) ⊙ f'(z[l])
```

If the activation function's derivative f'(z) < 1 everywhere, multiplying many such values:
```
(0.5)^50 ≈ 10^(-15)  →  effectively zero gradient
```

**Causes:**
- Sigmoid/tanh saturate (derivative ≈ 0 for large inputs)
- Deep networks multiply many small derivatives

**Solutions:**
- ReLU: derivative is 1 for positive inputs (no saturation)
- Residual connections (ResNets): skip connections provide gradient highways
- Careful initialization (Xavier, He)
- Batch/Layer normalization
- LSTM gates in recurrent networks

### The Exploding Gradient Problem

Opposite problem: gradients grow exponentially.
```
(2.0)^50 → ∞
```

**Solution**: gradient clipping
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 8. Gradient Descent and Optimizers

Covered mathematically in `03-math-foundations`. Here's the practical perspective.

### Learning Rate: The Most Important Hyperparameter

Too high → diverge (loss increases or oscillates)
Too low → train forever
Just right → smooth convergence

**How to set it:**
1. Start with standard defaults (Adam: 3e-4 for general, 1e-4 for fine-tuning LLMs)
2. Learning rate finder: increase lr from tiny to large, plot loss, find where it starts increasing, pick 10x lower
3. Use a schedule

### Mini-Batch Training

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()          # Clear previous gradients
        output = model(batch['x'])     # Forward pass
        loss = criterion(output, batch['y'])  # Compute loss
        loss.backward()                # Backward pass
        optimizer.step()               # Update weights
```

Batch size trade-offs:
- Small batch (8-32): noisier gradients, but acts as regularization
- Large batch (512+): more stable gradients, but needs larger learning rate; sometimes generalizes worse

### Adam vs SGD

**Adam**: adaptive learning rates, bias correction, works well out-of-the-box. Default for most tasks.

**SGD with momentum**: simple, often generalizes better for CNNs, but requires more hyperparameter tuning.

**AdamW**: Adam + decoupled weight decay. Standard for transformers.
```
weight decay: subtract λ × weight at each step (separate from gradient update in AdamW)
```

---

## 9. Initialization

### Why It Matters

If weights are too large: activations explode, gradients explode.
If weights are too small: activations shrink to zero, gradients vanish.

**Symmetry problem**: all-zero initialization → all hidden units learn the same function. Break symmetry with random initialization.

### Xavier / Glorot Initialization (for sigmoid/tanh)

```
W ~ Uniform(-√(6/(nᵢₙ + nₒᵤₜ)), +√(6/(nᵢₙ + nₒᵤₜ)))
```
Or Normal with:
```
σ = √(2 / (nᵢₙ + nₒᵤₜ))
```

Designed so that the variance of activations stays constant across layers.

### He / Kaiming Initialization (for ReLU)

```
σ = √(2 / nᵢₙ)
```

Accounts for the fact that ReLU zeros out half of its inputs (effectively halves the variance).

**Default in PyTorch for Linear layers with ReLU.**

### Initialization in Practice

Most frameworks handle this automatically. But knowing this prevents mysterious divergence when building custom architectures.

---

## 10. Regularization Techniques

Prevent overfitting (reduce generalization gap).

### L2 Regularization (Weight Decay)

Add penalty on weight magnitude to loss:
```
L_total = L_task + λ × Σ wᵢ²
```

Effect: gradient includes `-2λw`, which shrinks weights toward zero.

**In practice**: set via `weight_decay` parameter in optimizer.
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
```

### L1 Regularization

```
L_total = L_task + λ × Σ |wᵢ|
```

Effect: encourages sparsity (many weights become exactly 0). Useful for feature selection.

### Dropout

During training, randomly zero out each neuron with probability p (typically 0.1–0.5):

```python
x = nn.Dropout(p=0.5)(x)  # Each neuron independently zeroed with p=0.5
```

At inference: all neurons active, outputs multiplied by (1-p) to keep expected value same.

**Why it works:**
- Forces network to learn redundant representations
- Effectively trains an ensemble of 2^n thinned networks
- Reduces co-adaptation between neurons

**For transformers**: dropout applied to attention weights and to FFN outputs. Lower rates (0.1) typical.

### Early Stopping

Track validation loss during training. Stop when validation loss starts increasing:

```
Epoch 1: train_loss=0.8, val_loss=0.9
Epoch 5: train_loss=0.3, val_loss=0.4
Epoch 10: train_loss=0.1, val_loss=0.3    ← minimum
Epoch 15: train_loss=0.05, val_loss=0.35  ← getting worse
→ Stop and use checkpoint from epoch 10
```

**Patience**: how many epochs to wait before stopping (typically 5–10).

### Data Augmentation

For image/audio data, apply random transformations during training:
- Images: rotation, flip, crop, color jitter, random erasing
- Text: synonym replacement, back-translation, random deletion
- Audio: time stretching, pitch shifting, noise addition

Effectively increases dataset size without collecting new data.

---

## 11. Normalization Layers

Normalize activations to have mean ≈ 0 and std ≈ 1. Critical for training stability.

### Batch Normalization (BN)

Normalize across the batch dimension:

```
For each feature j in a batch of n examples:
  μⱼ = (1/n) × Σᵢ xᵢⱼ
  σⱼ² = (1/n) × Σᵢ (xᵢⱼ - μⱼ)²
  x̂ᵢⱼ = (xᵢⱼ - μⱼ) / √(σⱼ² + ε)
  yᵢⱼ = γⱼ × x̂ᵢⱼ + βⱼ    (learned scale and shift)
```

During inference: use running statistics from training.

**Benefits:**
- Allows higher learning rates
- Acts as regularization (reduces need for dropout)
- Reduces sensitivity to initialization

**Problem**: depends on batch size. Doesn't work well for small batches or online learning.

**Used in**: CNNs (ResNets, etc.)

### Layer Normalization (LN)

Normalize across the feature dimension (not batch):

```
For a single example x with d features:
  μ = (1/d) × Σᵢ xᵢ
  σ² = (1/d) × Σᵢ (xᵢ - μ)²
  x̂ᵢ = (xᵢ - μ) / √(σ² + ε)
  yᵢ = γᵢ × x̂ᵢ + βᵢ
```

**Does not depend on batch size** — works for any batch size, including 1.

**Used in**: Transformers (BERT, GPT, all modern LLMs).

### RMSNorm

Simplified LayerNorm — just normalize by RMS, no mean subtraction:
```
x̂ᵢ = xᵢ / RMS(x) = xᵢ / √((1/d) × Σⱼ xⱼ²)
yᵢ = γᵢ × x̂ᵢ
```

Faster and simpler. Used in LLaMA, Gemma, and most modern LLMs.

### Where to Place Normalization

**Pre-norm** (normalize before the layer): more stable training, now standard for transformers.
```
output = x + Layer(LayerNorm(x))
```

**Post-norm** (normalize after): original transformer used this, less stable.
```
output = LayerNorm(x + Layer(x))
```

---

## 12. Universal Approximation

**Universal Approximation Theorem (Cybenko, 1989; Hornik, 1991):**

A feedforward neural network with:
- A single hidden layer
- Enough neurons
- A non-polynomial activation function

Can approximate any continuous function on a compact domain to arbitrary precision.

### What This Means

Neural networks are universal function approximators. Any smooth mapping from inputs to outputs can be learned given enough capacity.

**Important caveats:**
1. It says a solution *exists*, not that gradient descent will *find* it
2. "Enough neurons" may mean exponentially many
3. Deep networks are often much more efficient than shallow + wide

### Why Depth Helps

For certain functions:
- Shallow network requires exponentially many neurons
- Deep network can represent the same function with polynomial neurons

Example: representing n-bit parity function requires 2^n neurons in one hidden layer, but only O(n) neurons spread across O(log n) layers.

Deep networks compose simple functions → complex hierarchical representations.

---

## 13. Architectures Overview

Neural networks come in many architectural flavors, each suited to different data modalities:

| Architecture | Data | Key Innovation | Example Tasks |
|-------------|------|----------------|--------------|
| **MLP** (Feedforward) | Tabular, embeddings | Multi-layer compositions | Classification, regression |
| **CNN** (Convolutional) | Images, audio spectrograms | Local filters + spatial sharing | Image classification, detection |
| **RNN** | Sequences (text, time series) | Hidden state carries context | Language modeling, speech |
| **LSTM** | Sequences | Gating mechanisms | Long sequences, translation |
| **Transformer** | Sequences (primarily) | Self-attention | LLMs, BERT, GPT |
| **GNN** (Graph) | Graph-structured data | Message passing | Social networks, molecules |
| **Diffusion** | Images, audio | Iterative denoising | Image generation |
| **VAE** | Any | Latent space sampling | Generation, representation |
| **GAN** | Any | Adversarial training | Image synthesis |

Each architecture is covered in depth in subsequent sections.

---

## Key Points for Exams

1. **Perceptron**: weighted sum + step function; cannot learn XOR (not linearly separable)
2. **Multi-layer networks** solve XOR because hidden layers create non-linear boundaries
3. **Activation functions**: ReLU is default (no saturation); GELU for transformers; sigmoid only at output for binary
4. **Cross-entropy loss** = negative log-likelihood = MLE for classification
5. **Backpropagation** = chain rule through the computation graph; computed automatically by autograd
6. **Vanishing gradients**: sigmoid/tanh derivatives < 1 → gradients shrink through layers → ReLU solves this
7. **Dropout**: randomly zeros neurons during training → prevents co-adaptation → ensemble effect
8. **Batch norm** normalizes across batch; **Layer norm** normalizes across features — Layer norm used in transformers
9. **Xavier init** for sigmoid/tanh; **He init** for ReLU — designed to keep activation variance constant
10. **Universal approximation**: a single hidden layer can approximate any function, but depth is more efficient
11. **AdamW** = Adam optimizer + weight decay; standard for transformers

---

## Practice Questions

1. Why can a single perceptron not learn XOR?
2. What does an activation function do and why is it necessary?
3. Compare sigmoid, tanh, and ReLU as activation functions. What's the main advantage of ReLU?
4. What is the vanishing gradient problem and how does it arise?
5. Explain backpropagation in one paragraph. What mathematical principle does it use?
6. What is dropout and why does it prevent overfitting?
7. What is the difference between batch normalization and layer normalization? When would you use each?
8. What is weight decay / L2 regularization? How is it different from L1?
9. What does the Universal Approximation Theorem say? What does it NOT guarantee?
10. What is the dying ReLU problem and how can it be addressed?
11. Why does initialization matter? What happens if all weights are initialized to zero?
12. What is the difference between Adam and AdamW?
