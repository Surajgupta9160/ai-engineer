# 00a — Mathematical Foundations for AI

> **You don't need a PhD, but you do need this.** This section covers the mathematical
> concepts that appear repeatedly across all of ML and deep learning. Focus on
> building intuition first, then formalism. You will recognize these ideas everywhere.

---

## Table of Contents

1. [Linear Algebra](#1-linear-algebra)
2. [Calculus & Optimization](#2-calculus--optimization)
3. [Probability & Statistics](#3-probability--statistics)
4. [Information Theory](#4-information-theory)
5. [Numerical Considerations](#5-numerical-considerations)
6. [How the Math Shows Up in Practice](#6-how-the-math-shows-up-in-practice)

---

## 1. Linear Algebra

The language of data in AI. Every forward pass through a neural network is matrix multiplication.

### Scalars, Vectors, Matrices, Tensors

| Object | Symbol | Shape | Example |
|--------|--------|-------|---------|
| Scalar | x | () | A single temperature value |
| Vector | **x** | (n,) | One data point with n features |
| Matrix | **X** | (m, n) | m data points, n features each |
| Tensor | **T** | (d1, d2, ...) | A batch of images: (batch, height, width, channels) |

```python
import numpy as np

scalar = 3.14                             # shape: ()
vector = np.array([1, 2, 3])             # shape: (3,)
matrix = np.array([[1,2],[3,4],[5,6]])   # shape: (3, 2)
tensor = np.zeros((32, 224, 224, 3))     # shape: (32, 224, 224, 3) — batch of images
```

### Vector Operations

**Dot Product (inner product):**
```
a · b = a₁b₁ + a₂b₂ + ... + aₙbₙ = |a| |b| cos(θ)
```
- Geometric meaning: projection of one vector onto another
- If θ = 0° (parallel): maximum positive value
- If θ = 90° (perpendicular): 0
- If θ = 180° (opposite): maximum negative value
- Critical in attention mechanisms: Q · K^T computes similarity

**Norms (vector length):**
```
L1 norm: ||x||₁ = Σ|xᵢ|
L2 norm: ||x||₂ = √(Σxᵢ²)   ← most common, "Euclidean length"
L∞ norm: ||x||∞ = max(|xᵢ|)
```

**Cosine Similarity:**
```
cos_sim(a, b) = (a · b) / (||a||₂ × ||b||₂)
```
Range: [-1, 1]. Used everywhere in embedding similarity (embeddings section).

### Matrix Operations

**Matrix Multiplication (the core operation):**
```
C = AB
C[i,j] = Σₖ A[i,k] × B[k,j]
```
Shapes: (m, k) × (k, n) → (m, n)

Intuition: each element of C is a dot product between a row of A and a column of B.

**What matrix multiplication represents:**
- Linear transformation (rotation, scaling, projection)
- A neural network layer: output = W × input + b
- Attention: softmax(Q × K^T / √d) × V

**Transpose:**
```
A^T[i,j] = A[j,i]
```
Swaps rows and columns. Critical in attention: attention_scores = Q × K^T

**Inverse:**
```
A × A⁻¹ = I   (identity matrix)
```
Only square matrices can be inverted. In practice, we avoid explicit inversion (use pseudoinverse or solve linear systems).

**Eigenvalues and Eigenvectors:**
```
A × v = λ × v
```
- v is an eigenvector: matrix multiplication only scales it, doesn't rotate
- λ is the eigenvalue: the scaling factor
- Intuition: eigendecomposition reveals the "principal axes" of a transformation
- Used in PCA (Principal Component Analysis) for dimensionality reduction

### Matrix Decompositions

**SVD (Singular Value Decomposition):**
```
A = U × Σ × V^T
```
- U: left singular vectors (m×m orthonormal)
- Σ: diagonal matrix of singular values (sorted descending)
- V^T: right singular vectors (n×n orthonormal)

Applications in AI:
- PCA is a special case of SVD
- Truncated SVD for dimensionality reduction
- **LoRA fine-tuning**: weight updates decomposed as UV^T (covered in section 22)

**Key Properties to Know:**
- Matrix rank: number of linearly independent rows/columns
- Orthogonal matrix: Q^T Q = I (rotation/reflection, no distortion)
- Positive definite matrix: all eigenvalues > 0 (used in optimization)

### Broadcasting

NumPy/PyTorch allow operations between different-shaped arrays:
```python
# vector + scalar: adds scalar to each element
np.array([1,2,3]) + 5 = [6,7,8]

# matrix + vector: adds vector to each row
np.ones((3,4)) + np.array([1,2,3,4]) = each row becomes [2,3,4,5]
```

Understanding broadcasting is essential for reading ML code efficiently.

---

## 2. Calculus & Optimization

Training neural networks is all about minimizing a loss function. Calculus tells us how.

### Derivatives (1D)

```
f'(x) = lim[h→0] (f(x+h) - f(x)) / h
```

Intuition: rate of change. If f'(x) > 0, f is increasing at x.

**Common derivatives in ML:**
```
d/dx [xⁿ] = n·xⁿ⁻¹
d/dx [eˣ] = eˣ
d/dx [ln(x)] = 1/x
d/dx [sigmoid(x)] = sigmoid(x) × (1 - sigmoid(x))
d/dx [tanh(x)] = 1 - tanh²(x)
d/dx [ReLU(x)] = 0 if x < 0, 1 if x > 0
```

### Gradients (Multi-dimensional)

For a function f(x₁, x₂, ..., xₙ), the gradient is the vector of partial derivatives:
```
∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
```

- Points in the direction of **steepest ascent**
- Negative gradient points toward the nearest minimum

### The Chain Rule (Critical)

If y = f(g(x)), then:
```
dy/dx = (dy/dg) × (dg/dx)
```

For composition of many functions (like a neural network):
```
y = f₄(f₃(f₂(f₁(x))))
dy/dx = (∂f₄/∂f₃) × (∂f₃/∂f₂) × (∂f₂/∂f₁) × (∂f₁/∂x)
```

This is the mathematical basis of **backpropagation** — compute gradients through composed functions by multiplying local derivatives.

### Gradient Descent

Find the minimum of a function by taking steps in the negative gradient direction:

```
θ_new = θ_old - α × ∇L(θ)

where:
  θ = parameters (weights)
  α = learning rate (step size)
  L = loss function
  ∇L = gradient of loss with respect to parameters
```

**Variants:**

| Variant | Description | When to use |
|---------|-------------|-------------|
| **Batch GD** | Gradient over entire dataset | Too slow for large datasets |
| **SGD** | Gradient over single example | Noisy but fast |
| **Mini-batch SGD** | Gradient over small batch (32–512) | Standard in practice |
| **Momentum** | Accumulate gradient history | Helps escape local minima |
| **Adam** | Adaptive learning rates per parameter | Default choice |
| **AdamW** | Adam with weight decay | Best for transformers |

### Adam Optimizer

The default for most neural networks:

```
m_t = β₁ × m_{t-1} + (1-β₁) × ∇L     (1st moment: exponential avg of gradients)
v_t = β₂ × v_{t-1} + (1-β₂) × ∇L²    (2nd moment: exponential avg of squared gradients)

m̂_t = m_t / (1 - β₁^t)               (bias correction)
v̂_t = v_t / (1 - β₂^t)

θ = θ - α × m̂_t / (√v̂_t + ε)
```

Default hyperparameters: β₁=0.9, β₂=0.999, ε=1e-8

**Intuition**: divides by the running average of squared gradients → parameters with large recent gradients get smaller learning rates (adaptive). Effectively gives each parameter its own learning rate.

### Learning Rate Schedules

Don't use a constant learning rate:

| Schedule | Pattern | When |
|----------|---------|------|
| Linear decay | High → 0 over training | Common |
| Cosine annealing | Cosine curve decay | Very common |
| Warmup + decay | Low → High → Low | Standard for transformers |
| Step decay | Drop by factor every N steps | Classification |
| Cyclical | Oscillate up and down | Finding good minima |

**Linear warmup for transformers**: start with very small lr (prevents divergence), ramp up, then decay. This is the standard for training LLMs.

### Convexity and Local Minima

**Convex function**: a single global minimum exists; gradient descent is guaranteed to find it.

**Non-convex function** (all neural networks): many local minima exist.

In practice: neural network loss landscapes have many local minima, but they tend to be **approximately as good as the global minimum** (flat landscape with many near-global minima, not narrow valleys with bad minima).

Key insight from Goodfellow et al.: for high-dimensional parameter spaces, saddle points (not local minima) are the main challenge. Modern optimizers handle this.

### Jacobian and Hessian

**Jacobian**: for a vector-valued function f: Rⁿ → Rᵐ, the Jacobian is an m×n matrix of all partial derivatives. Used in computing backpropagation through layers.

**Hessian**: second derivatives matrix. Used in second-order optimization methods (rarely used in practice due to O(n²) memory requirement).

---

## 3. Probability & Statistics

ML is fundamentally about learning from uncertain data and making uncertain predictions.

### Probability Basics

**Sample space (Ω)**: all possible outcomes
**Event (A)**: subset of outcomes
**Probability P(A)**: number between 0 and 1 measuring likelihood

**Axioms:**
- 0 ≤ P(A) ≤ 1
- P(Ω) = 1
- P(A ∪ B) = P(A) + P(B) - P(A ∩ B)

**Conditional Probability:**
```
P(A|B) = P(A ∩ B) / P(B)
```
Probability of A given B occurred.

**Bayes' Theorem:**
```
P(A|B) = P(B|A) × P(A) / P(B)
```

In ML:
```
P(model | data) = P(data | model) × P(model) / P(data)
 posterior      =   likelihood   ×    prior   /  evidence
```

Fundamental to understanding regularization as prior, Bayesian inference, and why we have priors over model weights.

### Probability Distributions

**Discrete:**

| Distribution | When | PMF |
|-------------|------|-----|
| Bernoulli | Binary outcome | P(X=1) = p |
| Categorical | Multi-class | P(X=k) = pₖ |
| Binomial | n Bernoulli trials | C(n,k) × pᵏ × (1-p)^(n-k) |
| Poisson | Count events | e^(-λ) × λˣ / x! |

**Continuous:**

| Distribution | When | Parameterized by |
|-------------|------|-----------------|
| Gaussian (Normal) | Default assumption | μ (mean), σ² (variance) |
| Uniform | Equal probability | [a, b] |
| Beta | Probability of probability | α, β |
| Dirichlet | Distribution over distributions | α₁...αₖ |
| Laplace | Robust to outliers | μ, b |

**The Gaussian (Normal) Distribution:**
```
f(x) = 1/(σ√(2π)) × exp(-(x-μ)²/(2σ²))
```
- ~68% of data within ±1σ
- ~95% within ±2σ
- ~99.7% within ±3σ

Why it's ubiquitous: Central Limit Theorem — sum of independent random variables approaches Gaussian regardless of original distribution.

### Maximum Likelihood Estimation (MLE)

Given data X = {x₁, ..., xₙ}, find parameters θ that maximize the probability of observing this data:

```
θ* = argmax P(X | θ) = argmax Π P(xᵢ | θ)
```

Taking log (for numerical stability, turns products into sums):
```
θ* = argmax Σ log P(xᵢ | θ)
```

This is the foundation of training neural networks:
- **Binary cross-entropy**: MLE under Bernoulli assumption
- **Categorical cross-entropy**: MLE under Categorical assumption
- **Mean squared error**: MLE under Gaussian noise assumption

### Expectation and Variance

**Expectation (mean):**
```
E[X] = Σ x × P(X=x)   (discrete)
E[X] = ∫ x × f(x) dx  (continuous)
```
Properties: E[aX + b] = a×E[X] + b, E[X+Y] = E[X] + E[Y]

**Variance:**
```
Var(X) = E[(X - E[X])²] = E[X²] - (E[X])²
```
Measures spread around the mean. Standard deviation = √Var(X).

**Bias-Variance Tradeoff:**
```
Expected error = Bias² + Variance + Noise
```
- High bias (underfitting): model too simple, wrong on training data
- High variance (overfitting): model memorizes training data, fails on new data
- Goal: minimize both

### Covariance and Correlation

**Covariance**: how two variables vary together
```
Cov(X,Y) = E[(X - E[X])(Y - E[Y])]
```

**Correlation**: normalized covariance
```
ρ(X,Y) = Cov(X,Y) / (σ_X × σ_Y)   ∈ [-1, 1]
```

**Covariance matrix**: for n-dimensional data, n×n matrix where entry (i,j) is Cov(Xᵢ, Xⱼ). Diagonal entries are variances. Used in PCA.

### Statistical Hypothesis Testing

**Null hypothesis (H₀)**: no effect, no difference.
**Alternative hypothesis (H₁)**: there is an effect.

**p-value**: probability of observing data at least as extreme as observed, assuming H₀ is true.
- p < 0.05 → "statistically significant" (reject H₀)
- This threshold is somewhat arbitrary

**In ML context:**
- When comparing model A vs model B on a benchmark, random variation in the test set matters
- A/B testing of model deployments requires statistical testing
- Bootstrap methods for confidence intervals on metrics

---

## 4. Information Theory

The mathematical theory of communication (Shannon, 1948). Directly connected to loss functions and compression.

### Entropy

Entropy measures uncertainty / information content:

```
H(X) = -Σ P(x) × log₂ P(x)   (bits)
H(X) = -Σ P(x) × ln P(x)     (nats, used in ML)
```

- High entropy: near-uniform distribution, high uncertainty
- Low entropy: concentrated distribution, low uncertainty
- Maximum entropy for n outcomes = log(n) (uniform distribution)
- Minimum entropy = 0 (deterministic outcome)

**Example:**
```
Fair coin:    H = -(0.5 × log₂(0.5) + 0.5 × log₂(0.5)) = 1 bit
Biased coin (0.9/0.1): H = -(0.9 × log₂(0.9) + 0.1 × log₂(0.1)) ≈ 0.47 bits
```

### Cross-Entropy

Measures the average bits needed to encode data from distribution P using code optimized for distribution Q:

```
H(P, Q) = -Σ P(x) × log Q(x)
```

**This is the loss function for classification!**
```
Cross-entropy loss = -Σᵢ yᵢ × log(ŷᵢ)
```
where y is the true distribution (one-hot) and ŷ is the predicted probability distribution.

Minimizing cross-entropy = maximum likelihood estimation under categorical distribution.

### KL Divergence

Measures how different distribution Q is from distribution P:

```
KL(P || Q) = Σ P(x) × log(P(x) / Q(x)) = H(P, Q) - H(P)
```

Properties:
- KL(P||Q) ≥ 0 always
- KL(P||Q) = 0 iff P = Q
- **Not symmetric**: KL(P||Q) ≠ KL(Q||P)

Used in:
- Variational Autoencoders (VAE): KL divergence between latent distribution and prior
- RLHF: KL penalty between RLHF-tuned policy and original policy (prevents model from drifting too far)
- Knowledge distillation

### Mutual Information

How much knowing X tells you about Y:
```
I(X;Y) = H(X) - H(X|Y) = KL(P(X,Y) || P(X)P(Y))
```
- I(X;Y) = 0 means X and Y are independent
- Used in feature selection: keep features with high mutual information with the target

---

## 5. Numerical Considerations

Real-world ML code must handle numerical precision carefully.

### Floating-Point Arithmetic

Most ML uses 32-bit float (float32):
- Range: ~10⁻³⁸ to ~10³⁸
- Precision: ~7 decimal digits
- **Key problem**: catastrophic cancellation (subtracting two nearly equal numbers loses precision)

Modern LLM training often uses:
- **float16 / bfloat16**: 16-bit for most computations (2x memory, 2x speed)
- **float32**: for gradients and optimizer states
- **Mixed precision training**: float16 forward/backward, float32 accumulation

bfloat16 vs float16:
- float16: 5 exponent bits, 10 mantissa bits → smaller max value
- bfloat16: 8 exponent bits, 7 mantissa bits → same range as float32, less precision

bfloat16 is preferred for training because it handles the wide range of gradient magnitudes better.

### The Log-Sum-Exp Trick

Computing softmax directly is numerically unstable:
```python
# Unstable version
softmax = exp(x) / sum(exp(x))
# If x contains large values (e.g., 1000), exp(1000) = inf
```

Stable version (log-sum-exp):
```python
# Subtract max before exponentiating
m = max(x)
softmax = exp(x - m) / sum(exp(x - m))
# exp(x - m) ≤ 1, so no overflow
```

### Gradient Clipping

During training, gradients can explode (go to infinity):
```python
# Clip gradients to max norm
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Standard practice for all LLM training. The threshold is usually 1.0.

### Numerical Stability in Softmax and Cross-Entropy

Always compute log-softmax and then use negative log-likelihood loss, rather than computing softmax then log:
```python
# Unstable
loss = -log(softmax(logits)[true_class])

# Stable (PyTorch's CrossEntropyLoss does this internally)
loss = F.cross_entropy(logits, targets)
# Equivalent to: F.nll_loss(F.log_softmax(logits), targets)
```

---

## 6. How the Math Shows Up in Practice

### In Data Preprocessing

- **Normalization**: make features have mean 0, std 1 using mean/std statistics
- **Min-max scaling**: map to [0, 1]
- **PCA**: use eigendecomposition of covariance matrix to reduce dimensions

### In Model Architecture

| Concept | Where in neural network |
|---------|----------------------|
| Matrix multiply | Every linear layer: y = Wx + b |
| Dot product | Attention: Q·K^T |
| Softmax | Classification output, attention weights |
| Cross-entropy | Classification loss |
| MSE | Regression loss, diffusion models |
| KL divergence | VAE, RLHF penalty |
| L2 norm | Weight decay regularization |

### In Training

| Concept | Where |
|---------|-------|
| Gradient descent | Parameter update rule |
| Chain rule | Backpropagation |
| Adam | Most common optimizer |
| Learning rate schedule | Warmup + cosine decay |
| Gradient clipping | Prevent exploding gradients |

### In Evaluation

| Concept | Where |
|---------|-------|
| Cross-entropy / perplexity | Language model evaluation |
| Cosine similarity | Embedding similarity, RAG retrieval |
| Hypothesis testing | A/B testing model deployments |
| Expected value | Expected reward in RL |

---

## Key Points for Exams

1. **Dot product** measures similarity; it equals |a||b|cos(θ)
2. **Matrix multiply**: (m,k) × (k,n) → (m,n); inner dimensions must match
3. **Gradient**: vector pointing in direction of steepest ascent; we descend to minimize
4. **Chain rule**: foundation of backpropagation; multiply local derivatives along the path
5. **Adam optimizer**: adaptive learning rates via first and second moment estimates
6. **Cross-entropy loss = negative log-likelihood = MLE** under categorical distribution
7. **KL divergence**: asymmetric, non-negative, measures distribution difference
8. **Entropy**: uncertainty in a distribution; maximized by uniform distribution
9. **Bias-variance tradeoff**: underfitting vs overfitting
10. **Log-sum-exp trick**: numerical stability when computing softmax

---

## Practice Questions

1. What is the dot product and what does it measure geometrically?
2. If matrix A has shape (3,4) and matrix B has shape (4,5), what is the shape of AB?
3. Explain gradient descent in one sentence. What is the learning rate?
4. What is the chain rule, and why is it essential for training neural networks?
5. What is cross-entropy loss? Why is it used for classification?
6. Explain KL divergence. Where is it used in modern AI systems?
7. What does entropy measure? What distribution has maximum entropy?
8. What is the bias-variance tradeoff? How does it relate to overfitting/underfitting?
9. What is the log-sum-exp trick and why is it needed?
10. What is the difference between float16 and bfloat16, and which is preferred for LLM training?
