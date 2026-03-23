# 00b — Classical Machine Learning

> **ML before neural networks (and why it still matters).**
> Classical ML dominates tabular/structured data, provides interpretable models,
> and forms the conceptual foundation that deep learning builds on.
> Every AI engineer should understand these algorithms.

---

## Table of Contents

1. [The ML Paradigm](#1-the-ml-paradigm)
2. [Supervised Learning](#2-supervised-learning)
3. [Key Supervised Algorithms](#3-key-supervised-algorithms)
4. [Unsupervised Learning](#4-unsupervised-learning)
5. [Key Unsupervised Algorithms](#5-key-unsupervised-algorithms)
6. [Reinforcement Learning (Intro)](#6-reinforcement-learning)
7. [Model Evaluation & Selection](#7-model-evaluation--selection)
8. [Feature Engineering](#8-feature-engineering)
9. [Classical ML vs Deep Learning](#9-classical-ml-vs-deep-learning)

---

## 1. The ML Paradigm

### What is Machine Learning?

Tom Mitchell's definition (1997):
> "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."

**Three components:**
- **Task (T)**: What we want the system to do (classify, predict, translate)
- **Experience (E)**: Data the system learns from
- **Performance (P)**: How we measure success

### The Core Assumption

Machine learning assumes the **i.i.d. assumption**: training data and test data are independently and identically distributed — drawn from the same distribution. If this breaks (distribution shift), models fail.

### The Machine Learning Workflow

```
1. Define the task
2. Collect and explore data (EDA)
3. Preprocess and clean data
4. Feature engineering / selection
5. Choose and train a model
6. Evaluate on held-out data
7. Tune hyperparameters
8. Final evaluation on test set
9. Deploy and monitor
```

### Types of ML

| Type | Training Signal | Goal |
|------|----------------|------|
| **Supervised** | Labeled examples (X, y) | Predict y for new X |
| **Unsupervised** | Unlabeled examples (X only) | Find structure in X |
| **Semi-supervised** | Few labels + many unlabeled | Leverage unlabeled data |
| **Reinforcement** | Reward signal from environment | Learn optimal policy |
| **Self-supervised** | Synthetic labels from data itself | Learn representations |

---

## 2. Supervised Learning

### Regression vs Classification

| | Regression | Classification |
|-|-----------|---------------|
| Output | Continuous value | Discrete class |
| Loss | MSE, MAE, Huber | Cross-entropy, hinge |
| Example | House price prediction | Spam detection |
| Metrics | RMSE, MAE, R² | Accuracy, F1, AUC |

### The Training-Validation-Test Split

Never evaluate on training data. Standard splits:

```
Full dataset
├── Training set (60-80%)   → fit model parameters
├── Validation set (10-20%) → tune hyperparameters, select model
└── Test set (10-20%)       → final evaluation (touch ONCE)
```

**K-Fold Cross-Validation:**
```
Split data into K folds (typically K=5 or K=10)
For each fold k:
  Train on all folds except k
  Validate on fold k
Average performance across all K folds
```

Reduces variance in performance estimate. Use when data is limited.

### Overfitting and Underfitting

```
Training Error vs Validation Error:

Underfitting: both high (model too simple)
Just right:   both low
Overfitting:  training low, validation high (model memorized training data)
```

**Signs of overfitting:**
- Perfect training accuracy, poor validation accuracy
- Performance degrades as model complexity increases (beyond a point)

**Remedies:**
- More training data
- Regularization (L1, L2, dropout)
- Reduce model complexity
- Early stopping

---

## 3. Key Supervised Algorithms

### Linear Regression

Fit a hyperplane to minimize sum of squared errors:

```
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

Loss: MSE = (1/n) × Σ(yᵢ - ŷᵢ)²
```

**Closed-form solution**: β = (X^T X)⁻¹ X^T y (when n is small)
**Gradient descent**: for large datasets

**Regularization variants:**
- **Ridge (L2)**: add λΣβᵢ² to loss → shrinks coefficients, keeps all features
- **Lasso (L1)**: add λΣ|βᵢ| → produces sparse models, feature selection
- **ElasticNet**: combination of L1 and L2

**Key assumptions:**
1. Linearity (relationship between X and y is linear)
2. Independence of errors
3. Homoscedasticity (constant variance in errors)
4. No multicollinearity

### Logistic Regression

Despite the name, this is a **classification** algorithm.

Applies sigmoid to linear combination:
```
P(y=1|x) = σ(β₀ + β₁x₁ + ... + βₙxₙ)
         = 1 / (1 + e^-(βᵀx))
```

Decision boundary: predict class 1 if P(y=1|x) > 0.5

Loss: Binary cross-entropy (log-loss):
```
L = -(y × log(p) + (1-y) × log(1-p))
```

Multi-class: use softmax instead of sigmoid (Multinomial Logistic Regression).

**Strengths**: interpretable coefficients, outputs calibrated probabilities, fast.

### Decision Trees

Recursively split the feature space to reduce impurity:

```
Tree structure:
  Node: feature and threshold (e.g., "Age > 30?")
  Branch: outcome of split
  Leaf: prediction (class or value)
```

**Splitting criteria:**
- Classification: Gini impurity or Information Gain (entropy-based)
- Regression: Variance reduction

```
Gini impurity = 1 - Σ pᵢ²   (lower is better)
Information Gain = H(parent) - weighted average H(children)
```

**Advantages**: interpretable, handles mixed data types, no normalization needed
**Disadvantages**: tends to overfit, unstable (small data changes → different tree)

### Random Forests

Ensemble of decision trees with two sources of randomness:

1. **Bootstrap aggregating (Bagging)**: each tree trained on random sample with replacement
2. **Feature randomness**: at each split, only consider random subset of features

Prediction: majority vote (classification) or average (regression)

**Why it works**: uncorrelated errors. Individual trees may overfit, but their errors are different, and averaging reduces variance.

**Hyperparameters:**
- n_estimators: number of trees (more = better, diminishing returns)
- max_depth: tree depth (controls overfitting)
- max_features: how many features to consider at each split (√n for classification, n/3 for regression)

**Feature importance**: can rank which features contribute most to predictions.

### Gradient Boosting

Build trees sequentially, each correcting the errors of the previous:

```
Step 1: Start with a simple prediction (mean of y)
Step 2: Fit a tree to the residuals (errors)
Step 3: Add this tree's predictions to the ensemble (weighted by learning rate)
Step 4: Compute new residuals, repeat
```

More formally: gradient descent in function space, where each tree is a gradient step.

**Implementations:**
- **Scikit-learn GBM**: reference implementation, slow
- **XGBoost (2014)**: optimized with regularization, fast, handles missing values
- **LightGBM (2017)**: Microsoft, leaf-wise growth, faster than XGBoost
- **CatBoost (2017)**: Yandex, handles categorical features natively

**XGBoost has dominated ML competition leaderboards for 10 years for tabular data.**

Hyperparameters:
- n_estimators, learning_rate: more trees + lower lr = better (until overfitting)
- max_depth: tree depth
- subsample: fraction of data per tree (0.8 typical)
- colsample_bytree: fraction of features per tree

### Support Vector Machines (SVM)

Find the hyperplane that maximizes the margin between classes:

```
Decision boundary: wᵀx + b = 0
Margin: 2 / ||w||

Maximize margin = Minimize ||w||²
Subject to: yᵢ(wᵀxᵢ + b) ≥ 1
```

**Kernel trick**: implicitly map to higher-dimensional space where data is linearly separable:
```
K(x, x') = φ(x) · φ(x')

Common kernels:
- Linear: K(x, x') = xᵀx'
- Polynomial: K(x, x') = (γxᵀx' + r)^d
- RBF (Gaussian): K(x, x') = exp(-γ||x - x'||²)   ← most common
- Sigmoid: K(x, x') = tanh(γxᵀx' + r)
```

**Support vectors**: the training examples that lie closest to the decision boundary — the only ones that matter.

**Strengths**: works well in high dimensions, effective when n_features > n_samples
**Weaknesses**: slow on large datasets (O(n²) to O(n³)), sensitive to feature scaling

**Practical note**: for large datasets (>100K samples), SVMs are largely replaced by gradient boosting or neural networks.

### K-Nearest Neighbors (KNN)

Non-parametric: no "training." At prediction time:
1. Find K closest training examples by distance
2. Return majority class (classification) or average value (regression)

```python
Distance metrics:
- Euclidean: √(Σ(xᵢ - yᵢ)²)
- Manhattan: Σ|xᵢ - yᵢ|
- Cosine similarity: (x·y) / (||x|| ||y||)
```

**K selection**: small K → complex boundary (overfitting), large K → smooth boundary (underfitting). Tune via cross-validation.

**Strengths**: simple, no training time, naturally handles multi-class
**Weaknesses**: O(n) prediction time (slow for large datasets), sensitive to irrelevant features, requires scaling

**Modern relevance**: the concept of "find nearest neighbors" is core to RAG systems and vector databases.

### Naive Bayes

Applies Bayes' theorem with the "naive" assumption that features are conditionally independent:

```
P(y|x) ∝ P(y) × Π P(xᵢ|y)
```

Variants:
- **Gaussian NB**: continuous features, assumed Gaussian
- **Multinomial NB**: count features (word counts in text)
- **Bernoulli NB**: binary features

**Strengths**: very fast, works well for text classification, probabilistic output
**Weaknesses**: independence assumption often violated, numerical underflow for long texts

Despite the "naive" assumption, works surprisingly well in practice — was the dominant spam filter algorithm for years.

---

## 4. Unsupervised Learning

### What is Unsupervised Learning?

No labels. The goal is to discover hidden structure in data:
- Cluster similar examples together
- Find compressed representations
- Detect anomalies
- Generate new examples

### Why It Matters

- Most data in the world is unlabeled (cheap to collect, expensive to label)
- **Self-supervised learning** (the foundation of modern LLMs) is a form of unsupervised learning where labels are automatically derived from the data itself
- Feature learning: learn useful representations that can then be used for downstream supervised tasks

---

## 5. Key Unsupervised Algorithms

### K-Means Clustering

Partition n points into K clusters by minimizing within-cluster variance:

```
Algorithm:
1. Initialize K cluster centroids randomly
2. Assign each point to nearest centroid
3. Recompute centroids as mean of assigned points
4. Repeat 2-3 until convergence

Objective: minimize Σ Σ ||xᵢ - μₖ||²
              k  xᵢ∈Cₖ
```

**K selection**: Elbow method (plot inertia vs K, pick the "elbow"), silhouette score.

**Weaknesses**: assumes spherical clusters, sensitive to initialization (use K-means++ initialization), must specify K in advance.

### DBSCAN

Density-Based Spatial Clustering of Applications with Noise:

```
Core point: has ≥ minPts neighbors within radius ε
Border point: within ε of a core point, but fewer than minPts neighbors
Noise: neither core nor border

Algorithm:
For each unvisited point:
  If core point: expand cluster (recursively find all density-reachable points)
  Else: mark as noise (may later be claimed as border point)
```

**Advantages over K-means**:
- No need to specify K
- Finds arbitrarily shaped clusters
- Handles noise/outliers explicitly

Used in anomaly detection and geospatial clustering.

### Hierarchical Clustering

Build a hierarchy of clusters:

**Agglomerative (bottom-up)**: start with each point as its own cluster, merge closest pairs.
**Divisive (top-down)**: start with all points in one cluster, split recursively.

**Linkage criteria** (how to measure cluster-to-cluster distance):
- Single linkage: min distance between any two points
- Complete linkage: max distance between any two points
- Average linkage: average distance
- Ward: minimize within-cluster variance (most common)

Result: dendrogram — a tree of merges. Cut at any level to get K clusters.

### Principal Component Analysis (PCA)

Linear dimensionality reduction: find orthogonal directions of maximum variance.

```
Algorithm:
1. Center the data (subtract mean)
2. Compute covariance matrix: Σ = XᵀX / (n-1)
3. Eigendecomposition: Σ = QΛQᵀ
4. Sort eigenvectors by eigenvalue (descending)
5. Project data onto top K eigenvectors

Explained variance ratio = λₖ / Σλᵢ
```

**Applications:**
- Visualization (reduce to 2D for plotting)
- Remove multicollinearity before regression
- Compression: keep 95% of variance with far fewer dimensions
- Noise reduction

**PCA vs t-SNE:**
- PCA: linear, preserves global structure, fast, invertible
- t-SNE: non-linear, preserves local structure (clusters), slow, not invertible

### t-SNE (t-distributed Stochastic Neighbor Embedding)

2D/3D visualization of high-dimensional data:

1. Compute pairwise similarities in high-dimensional space (Gaussian)
2. Compute pairwise similarities in low-dimensional space (t-distribution)
3. Minimize KL divergence between the two similarity distributions

Key properties:
- Reveals cluster structure clearly
- **Not deterministic**: different runs give different results
- t-distribution in low-dimensional space: heavy tails prevent crowding
- Hyperparameter: **perplexity** (roughly: number of effective nearest neighbors) — typical range 5–50

**UMAP** (2018): faster than t-SNE, better global structure preservation — now preferred.

### Autoencoders

Neural network approach to unsupervised representation learning:

```
Encoder: input → compressed latent vector z
Decoder: latent vector z → reconstructed input

Loss: ||x - x_reconstructed||² (reconstruction loss)
```

The bottleneck forces the encoder to learn a compressed representation.

**Variants:**
- **Vanilla autoencoder**: deterministic latent space
- **Denoising autoencoder**: add noise to input, train to reconstruct clean version → more robust representations
- **Variational Autoencoder (VAE)**: latent space is a Gaussian distribution, enables generation of new samples
- **Sparse autoencoder**: adds sparsity constraint → features are interpretable (used in LLM interpretability research)

### Gaussian Mixture Models (GMM)

Soft clustering: each data point has probability of belonging to each cluster.

Models data as a mixture of Gaussians:
```
p(x) = Σₖ πₖ × N(x | μₖ, Σₖ)
```

Fit using **Expectation-Maximization (EM) algorithm**:
- E-step: compute probability each point belongs to each cluster
- M-step: recompute cluster parameters using these probabilities

**Advantage over K-means**: soft assignments, handles elliptical clusters.

---

## 6. Reinforcement Learning

### The RL Framework

An **agent** interacts with an **environment**:

```
State (s) → Agent → Action (a)
                ↓
Environment → Reward (r) + Next State (s')
```

**Goal**: learn a **policy** π(a|s) that maximizes expected cumulative reward.

```
Return: Gₜ = rₜ + γ × rₜ₊₁ + γ² × rₜ₊₂ + ...
           = Σ γᵏ rₜ₊ₖ₊₁
```

γ (gamma) = discount factor (0 to 1): how much to value future rewards.

### Key Concepts

**Value functions:**
- V(s): expected return from state s following policy π
- Q(s,a): expected return from taking action a in state s, then following π

**Bellman equation:**
```
Q(s,a) = r + γ × max_a' Q(s', a')
```
The value of a state-action is the immediate reward plus the discounted best future value.

### Classic Algorithms

**Q-Learning:**
```
Q(s,a) ← Q(s,a) + α × [r + γ × max_a' Q(s',a') - Q(s,a)]
```
Updates Q-table based on observed transitions. Converges to optimal Q-function.

**SARSA (On-Policy TD)**: similar to Q-learning but updates based on actual next action taken.

**Policy Gradient (REINFORCE):**
```
∇J(θ) ∝ Σ Q(s,a) × ∇ log π(a|s; θ)
```
Directly optimize policy parameters. Foundational to PPO used in RLHF.

**Actor-Critic:**
- Actor: learns policy π(a|s)
- Critic: learns value function V(s) to reduce variance in policy updates

**PPO (Proximal Policy Optimization, 2017):**
Clips policy updates to prevent too-large changes:
```
L(θ) = min(r(θ) × Â, clip(r(θ), 1-ε, 1+ε) × Â)
where r(θ) = π_θ(a|s) / π_θ_old(a|s)
```

**PPO is the algorithm used in RLHF for LLM alignment** (InstructGPT, ChatGPT).

### RL vs Other ML

| Aspect | Supervised ML | Reinforcement Learning |
|--------|--------------|----------------------|
| Training signal | Labeled examples | Reward from environment |
| Feedback timing | Immediate | Delayed (credit assignment) |
| Data generation | Static dataset | Agent-generated through interaction |
| Goal | Minimize prediction error | Maximize cumulative reward |

### RL in Modern AI

- **AlphaGo/AlphaZero** (2016): RL from self-play → superhuman Go
- **OpenAI Five** (2018): RL for Dota 2 → beat world champions
- **RLHF**: RL to align LLMs with human preferences
- **Robotics**: learning manipulation policies
- **Game playing, resource management, recommendation**

---

## 7. Model Evaluation & Selection

### Classification Metrics

**Confusion Matrix:**
```
                Predicted Positive  Predicted Negative
Actual Positive    TP (True +)         FN (False -)
Actual Negative    FP (False +)        TN (True -)
```

**Derived metrics:**
```
Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)     ← of all predicted positives, how many are correct?
Recall    = TP / (TP + FN)     ← of all actual positives, how many did we catch?
F1 Score  = 2 × (Precision × Recall) / (Precision + Recall)
```

**When to use what:**
- **Accuracy**: balanced classes, equal cost of errors
- **Precision**: costly false positives (spam filter, fraud detection)
- **Recall**: costly false negatives (disease detection, safety systems)
- **F1**: imbalanced classes, when both matter

**ROC Curve and AUC:**
- Plot True Positive Rate vs False Positive Rate at different thresholds
- AUC (Area Under Curve): 0.5 = random classifier, 1.0 = perfect
- AUC measures ranking ability, robust to class imbalance

**Precision-Recall Curve:**
- Better than ROC for highly imbalanced datasets
- Use when positive class is rare

### Regression Metrics

```
MAE  = (1/n) Σ |yᵢ - ŷᵢ|           (mean absolute error)
MSE  = (1/n) Σ (yᵢ - ŷᵢ)²          (mean squared error)
RMSE = √MSE                          (root mean squared error, same units as y)
R²   = 1 - Σ(yᵢ-ŷᵢ)² / Σ(yᵢ-ȳ)²   (coefficient of determination, % variance explained)
```

- MAE: robust to outliers
- MSE/RMSE: penalizes large errors heavily
- R²: proportion of variance explained (1.0 = perfect, 0 = predicts mean)

### Hyperparameter Tuning

**Grid Search**: exhaustively try all combinations
```python
param_grid = {'max_depth': [3, 5, 7], 'n_estimators': [100, 200, 500]}
# Tries 3 × 3 = 9 combinations
```

**Random Search**: sample random combinations (often as good as grid search, faster)

**Bayesian Optimization**: build a surrogate model of how hyperparameters affect performance, search intelligently. Tools: Optuna, Ray Tune, W&B Sweeps.

### Class Imbalance Handling

When 99% of examples are class 0, a model that always predicts 0 gets 99% accuracy.

**Techniques:**
1. **Resampling**: oversample minority class (SMOTE) or undersample majority
2. **Class weights**: penalize misclassifying minority class more heavily
3. **Threshold adjustment**: move decision threshold from 0.5
4. **Anomaly detection framing**: treat rare class as anomalies

---

## 8. Feature Engineering

### Why It Matters

For classical ML, feature engineering often matters more than algorithm choice.

"Applied machine learning is basically feature engineering." — Andrew Ng

### Common Techniques

**Handling Missing Values:**
- Delete rows/columns with many missing values
- Impute: mean/median (numerical), mode (categorical), or model-based
- Add indicator variable: "was this field missing?"

**Encoding Categorical Variables:**
- One-hot encoding: binary column per category (for non-ordinal)
- Ordinal encoding: integer per category (for ordinal: small=1, medium=2, large=3)
- Target encoding: replace with mean target value per category
- Embedding: learn a vector per category (neural networks)

**Numerical Transformations:**
- Standardization: (x - mean) / std → zero mean, unit variance
- Min-max normalization: (x - min) / (max - min) → [0, 1]
- Log transformation: log(1 + x) → compresses skewed distributions
- Binning: convert continuous to categorical ranges

**Feature Crosses:**
```
If "city" and "product_type" both matter, create "city × product_type"
Captures interactions that individual features can't
```

**Time-based features (for time series):**
- Hour, day of week, month, year
- Time since last event
- Rolling averages (window functions)
- Is holiday, is weekend

### Feature Selection

Too many features → overfitting, slower training, noisy model.

**Filter methods**: rank features by correlation with target, mutual information, etc. Fast but ignores feature interactions.

**Wrapper methods**: train model with different feature subsets, select best. Expensive but accurate.

**Embedded methods**: regularization (L1/Lasso) automatically sets some weights to zero.

**Tree-based importance**: random forests and gradient boosting can rank feature importance.

---

## 9. Classical ML vs Deep Learning

### When to Use Classical ML

| Scenario | Classical ML | Deep Learning |
|----------|-------------|--------------|
| Tabular/structured data | Usually better | Usually worse |
| Small datasets (<10K rows) | Often better | Often worse |
| Need interpretability | Strong choice | Difficult |
| Limited compute | Fine | May be required |
| Text classification (simple) | Competitive | Usually better |
| Images, audio, video | Poor | State of the art |
| NLP (complex) | Poor | State of the art |

**The rule**: for structured tabular data with thousands to millions of rows, XGBoost/LightGBM is hard to beat. Deep learning wins on unstructured data (images, text, audio).

### Why Gradient Boosting Dominates Tabular Data

1. Handles heterogeneous features natively (no normalization needed)
2. Robust to outliers (tree-based)
3. Feature importance built in
4. Handles missing values
5. Works well without extensive tuning
6. Interpretable (can trace decisions)

Many Kaggle competitions: "if you're not using XGBoost or LightGBM, you're leaving points on the table."

### How Classical ML Concepts Live in Deep Learning

| Classical Concept | Deep Learning Equivalent |
|-------------------|------------------------|
| Linear regression | Fully connected layer with linear activation |
| Logistic regression | Fully connected layer with softmax |
| Feature engineering | Representation learning (automatic) |
| Regularization (L2) | Weight decay |
| Regularization (L1) | Sparse penalties, pruning |
| Ensemble methods | Model ensembling, dropout |
| Dimensionality reduction (PCA) | Autoencoders, learned embeddings |
| Nearest neighbors | Retrieval augmentation, vector search |

---

## Key Points for Exams

1. **Three types of ML**: supervised (labeled data), unsupervised (no labels), reinforcement (reward signal)
2. **Bias-variance tradeoff**: underfitting (high bias) vs overfitting (high variance)
3. **K-fold cross-validation**: reduces variance in performance estimate
4. **Random Forest = bagging + random feature subsets**: uncorrelated errors → lower variance
5. **Gradient Boosting = sequential trees correcting residuals**: usually best for tabular
6. **SVM kernel trick**: implicitly maps to higher dimensions for linear separability
7. **PCA**: eigendecomposition of covariance matrix → directions of maximum variance
8. **K-means**: minimize within-cluster variance; K-means++ initialization
9. **Precision vs Recall**: precision = few false alarms; recall = few missed detections
10. **F1 score**: harmonic mean of precision and recall; use for imbalanced classes
11. **PPO** is the RL algorithm used in RLHF
12. **Feature engineering often matters more than algorithm choice** for tabular data

---

## Practice Questions

1. What is the difference between supervised, unsupervised, and reinforcement learning?
2. Why is the F1 score preferred over accuracy for imbalanced datasets?
3. What is the difference between precision and recall? Give an example where you'd optimize each.
4. Explain the Random Forest algorithm. Why does it have lower variance than a single decision tree?
5. What is gradient boosting and how does it differ from random forests?
6. What is the kernel trick in SVMs and what problem does it solve?
7. Explain PCA. What is it used for?
8. What is K-means clustering? What are its limitations?
9. What is the bias-variance tradeoff? How does regularization help?
10. When should you use classical ML vs deep learning?
11. What is cross-validation and why is it needed?
12. What is the RL framework (state, action, reward, policy)? Where is RL used in modern LLMs?
