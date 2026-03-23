# 00d — Deep Learning & Convolutional Networks

> **The computer vision revolution that proved deep learning worked.**
> CNNs went from academic curiosity to the dominant paradigm in 3 years (2012–2015).
> Understanding them explains why deep networks generalize well and why architectures matter.

---

## Table of Contents

1. [What Makes it "Deep"?](#1-what-makes-it-deep)
2. [The Convolution Operation](#2-the-convolution-operation)
3. [CNN Components](#3-cnn-components)
4. [Historic CNN Architectures](#4-historic-cnn-architectures)
5. [Residual Networks & Skip Connections](#5-residual-networks--skip-connections)
6. [Modern Architecture Patterns](#6-modern-architecture-patterns)
7. [Transfer Learning with CNNs](#7-transfer-learning-with-cnns)
8. [Object Detection & Segmentation](#8-object-detection--segmentation)
9. [Generative Models](#9-generative-models)
10. [Key Insights from the Deep Learning Era](#10-key-insights)

---

## 1. What Makes it "Deep"?

### Shallow vs Deep Networks

**Shallow network**: typically 1-2 hidden layers. Sufficient in theory (universal approximation) but requires exponentially many neurons for certain functions.

**Deep network**: many hidden layers (5, 20, 100+). Learns hierarchical representations.

### Hierarchical Feature Learning

This is the key insight of deep learning in computer vision:

```
CNN for image classification learns:
  Layer 1: Edges, corners, color gradients
  Layer 2: Textures, simple shapes (arcs, lines)
  Layer 3: Parts (eyes, wheels, windows)
  Layer 4+: Objects (faces, cars, buildings)
```

Lower layers are nearly universal (transferable across tasks). Higher layers are task-specific.

This hierarchy mirrors how the visual cortex is organized (V1 → V2 → V4 → IT), though the analogy should not be taken too literally.

### Why Depth > Width for Vision

Consider recognizing a face:
- A face has eyes, a nose, a mouth
- An eye has eyelid, iris, pupil
- Each part has edges and textures

This compositional structure maps naturally to depth. A single wide layer would need to capture all these compositions simultaneously, requiring exponentially more capacity.

---

## 2. The Convolution Operation

### Core Idea

Instead of each neuron connecting to every pixel (MLP), **convolutional neurons connect to small local patches** of the input.

A **filter** (kernel) slides over the input and computes a dot product at each position.

```
Input image: 6×6
Filter:      3×3

Output: (6-3+1)×(6-3+1) = 4×4

At each position (i,j):
  output[i,j] = Σₘ Σₙ input[i+m, j+n] × filter[m, n]
```

### What Filters Learn

An edge detector filter:
```
Horizontal edge filter:
[[-1, -1, -1],
 [ 0,  0,  0],
 [ 1,  1,  1]]

Responds strongly where pixel values change vertically (horizontal edge)
```

In early CNN layers, filters spontaneously learn to detect edges, orientations, and colors — without being told to.

### Key Properties of Convolution

**Local connectivity**: each filter sees only a small patch → detects local patterns.

**Weight sharing**: the same filter weights are used at every spatial position → the filter looks for the same pattern everywhere in the image.

This is the key efficiency gain. A 3×3 filter has only 9 parameters but scans the entire image.

**Translation equivariance**: if an edge moves one pixel to the right, the activation moves one pixel to the right. CNNs naturally handle object position variation.

### Padding

Without padding, output shrinks with each layer:
```
Input 6×6 + filter 3×3 → output 4×4 → 2×2 after next conv...
```

**Zero padding**: add zeros around the input so output is same size as input (same padding).

### Stride

Move the filter by s pixels instead of 1:
```
Stride 2: output size = ⌊(input_size - filter_size) / stride⌋ + 1
         = ⌊(6-3)/2⌋ + 1 = 2 (for 6×6 input, 3×3 filter)
```

Stride > 1 downsamples the feature map.

### Multiple Channels

Real images have 3 channels (RGB). A filter must cover all channels:

```
Input:  H × W × 3  (RGB image)
Filter: 3 × 3 × 3  (3×3 spatial, depth=3 for all channels)
One filter → one feature map: H × W × 1

With K filters → K feature maps: H × W × K
```

The number of filters K is the number of output channels (a hyperparameter, called "depth" or "channels").

### Convolution as Matrix Multiplication

Under the hood, convolution is implemented as matrix multiplication (using im2col or Winograd transforms) to leverage GPU's matrix multiply hardware.

---

## 3. CNN Components

### Convolutional Layer

```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
```

- Parameters: `kernel_size × kernel_size × in_channels × out_channels` + biases
- Output: `(H_out, W_out, out_channels)`

### Activation (after every conv)

ReLU is standard:
```python
nn.ReLU(inplace=True)
```

### Pooling Layers

Downsample the spatial dimensions, creating spatial invariance:

**Max Pooling:**
```
Take the maximum value in each window
2×2 pooling with stride 2 → halves H and W

Feature map: [[1,3,2,4],   → max pool 2×2 → [[3,4],
              [5,2,1,3],                        [5,3]]
              [3,1,4,2],
              [2,4,1,3]]
```

**Average Pooling:**
Take the average. Used in final layers (Global Average Pooling).

**Global Average Pooling (GAP):**
Average over the entire spatial dimension → (H, W, C) → (1, 1, C) → flatten to (C,).
Replaces fully connected layers in modern architectures. No spatial parameters.

### Flattening and Fully Connected Layers

After convolutional layers, flatten to 1D vector → feed into standard fully connected layers.

```python
# After conv layers: feature map is (batch, channels, H, W)
x = x.view(x.size(0), -1)  # Flatten: (batch, channels*H*W)
x = nn.Linear(channels*H*W, 1024)(x)
```

### A Complete Simple CNN

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 32 filters
            nn.ReLU(),
            nn.MaxPool2d(2),                              # halve spatial dims
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),  # for 32×32 input
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

---

## 4. Historic CNN Architectures

### LeNet-5 (LeCun, 1989)

The first successful CNN. Used to read handwritten ZIP codes and checks.

```
Architecture: Conv → Pool → Conv → Pool → FC → FC → Output
Input: 32×32 grayscale
Parameters: ~60K
```

Limited by compute power — required specialized hardware even at the time.

### AlexNet (Krizhevsky, Sutskever, Hinton, 2012)

The breakthrough model. Won ImageNet 2012 by 10 percentage points.

```
Architecture (5 conv + 3 FC layers):
  Conv(11×11, 96) → MaxPool
  Conv(5×5, 256)  → MaxPool
  Conv(3×3, 384)
  Conv(3×3, 384)
  Conv(3×3, 256)  → MaxPool
  FC(4096) → Dropout
  FC(4096) → Dropout
  FC(1000) → Softmax
```

Key innovations:
- **ReLU** (first mainstream use in a CNN)
- **Dropout** for regularization
- Data augmentation
- Two-GPU training (model split across GPUs)
- Local Response Normalization (later abandoned)

### VGGNet (Simonyan & Zisserman, Oxford, 2014)

Principle: use **very deep networks with small (3×3) filters**.

```
Architecture example (VGG-16):
  [3×3, 64] × 2  → MaxPool
  [3×3, 128] × 2 → MaxPool
  [3×3, 256] × 3 → MaxPool
  [3×3, 512] × 3 → MaxPool
  [3×3, 512] × 3 → MaxPool
  FC(4096) → FC(4096) → FC(1000)
  Total: 138M parameters
```

**Why small filters?**
Two 3×3 conv layers have the same **receptive field** as one 5×5 layer, but:
- Fewer parameters: 2 × (3×3×C×C) = 18C² vs 1 × (5×5×C×C) = 25C²
- More nonlinearity (two ReLUs instead of one)

**VGGNet's lesson**: depth matters more than filter size. Standard 3×3 convolutions remain dominant.

### GoogLeNet / Inception v1 (Google, 2014)

Introduced the **Inception module**: apply multiple filter sizes in parallel, concatenate results.

```
Inception module:
         Input
           │
    ┌──────┼──────┬──────┐
    │      │      │      │
  1×1    1×1    1×1    3×3 pool
   ↓      ↓      ↓       ↓
  1×1   3×3    5×5     1×1
   ↓      ↓      ↓       ↓
         └──────────────┘
               Concat
```

1×1 convolutions are used for dimensionality reduction before expensive 3×3 and 5×5 convolutions.

Key ideas:
- **1×1 convolutions**: reshape channel dimensions cheaply
- **Auxiliary classifiers**: inject gradient in the middle of the network (addressed vanishing gradient before ResNets)
- 22 layers deep with only 6.8M parameters (vs VGG's 138M)

### ResNet (He et al., Microsoft, 2015)

**The most influential neural network architecture**, still widely used.

**Problem**: adding more layers made networks *worse*, even on training data — degradation problem (not overfitting, the deeper model was harder to optimize).

**Solution**: **Residual connections** (skip connections):

```
Standard layer:      H(x) = F(x)
Residual block:      H(x) = F(x) + x

Where F(x) is the "residual" to be learned:
F(x) = H(x) - x  (what needs to be added to input to get desired output)
```

If the optimal function is close to identity, the network only needs to learn small F(x) ≈ 0, which is easier than learning H(x) directly.

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual  # Skip connection!
        return F.relu(out)
```

**Why skip connections work (gradient perspective):**
```
∂L/∂x = ∂L/∂H × ∂H/∂x = ∂L/∂H × (∂F/∂x + 1)
```
The "+1" provides a direct gradient path that doesn't vanish even if ∂F/∂x ≈ 0.

**Results:**
- ResNet-152: 152 layers, 3.57% top-5 error on ImageNet (superhuman)
- Could train networks of hundreds or thousands of layers
- Enabled training arbitrarily deep networks

---

## 5. Residual Networks & Skip Connections

### Why They Revolutionized Everything

Skip connections solved two problems:
1. **Vanishing gradient**: gradient flows directly through skip connections
2. **Optimization**: easier to learn residuals than full transformations

And they enabled:
- Training 1000+ layer networks
- The **Transformer** (which uses skip connections around every sublayer)
- Dense connections (DenseNet)
- U-Nets for segmentation

### Bottleneck Residual Block

For very deep networks, use 1×1 convolutions to reduce channels:

```
Input (256 channels)
  → 1×1 conv (64 channels)   [reduce]
  → 3×3 conv (64 channels)   [process]
  → 1×1 conv (256 channels)  [expand]
+ Input
= Output (256 channels)
```

Much cheaper than using 3×3 on full 256 channels.

### DenseNet (Huang et al., 2017)

Takes skip connections further: each layer receives input from ALL previous layers:

```
x₄ = f₄(x₀, x₁, x₂, x₃)   (concatenation of all previous)
```

**Advantages**: feature reuse, very efficient, strong gradient flow.
**Used in**: medical imaging, cases where labeled data is limited.

### U-Net (Ronneberger et al., 2015)

Encoder-decoder with skip connections between encoder and decoder:

```
Encoder: 256 → 128 → 64 → 32
          ↓      ↓     ↓    ↓  ← bottleneck
Decoder: 32 → 64 → 128 → 256

Skip connections:
  Decoder layer at resolution r receives feature maps from encoder layer at same r
```

Skip connections preserve spatial information lost during downsampling.

**Used in**: image segmentation, medical imaging, image generation (UNet-based diffusion).

---

## 6. Modern Architecture Patterns

### Depthwise Separable Convolutions (MobileNet, 2017)

Standard convolution: filter covers all input channels simultaneously.

Depthwise separable: split into two steps:
1. **Depthwise**: one filter per input channel (spatial convolution only)
2. **Pointwise**: 1×1 conv to combine channels

```
Standard 3×3: D_K × D_K × M × N   parameters
Depthwise sep: D_K × D_K × M + M × N parameters

For typical values: 8-9x fewer parameters and computation
```

Used in: MobileNet, EfficientNet, lightweight models for edge deployment.

### EfficientNet (Tan & Le, Google, 2019)

**Compound scaling**: scale all three dimensions — depth, width, resolution — simultaneously, with a principled scaling coefficient.

Formula: if we want to use 2ᴺ more compute:
```
depth    = α^N
width    = β^N
resolution = γ^N
where αβ²γ² ≈ 2 (constrained to double FLOPs per unit N)
```

EfficientNet B7 achieved 84.3% on ImageNet with far fewer parameters than competing models.

### Vision Transformers (ViT, 2020)

**What if you apply a transformer directly to images?**

Patchify: divide image into 16×16 patches → embed each patch → process with transformer.

```
224×224 image → 14×14 grid of 16×16 patches → 196 "tokens"
Each patch is flattened and embedded: 16×16×3 = 768 → learned embedding
Add positional encoding
Process with standard transformer encoder
[CLS] token output → classification head
```

Results: ViT-L matches or exceeds CNNs on ImageNet *but only at large scale* (trained on JFT-300M). On small datasets, CNN inductive biases (local connectivity, translation equivariance) are helpful.

**Current status**: many state-of-the-art models combine convolutions and attention (ConvNeXt, DeiT, Swin Transformer).

---

## 7. Transfer Learning with CNNs

### The ImageNet Revolution in Transfer Learning

AlexNet/VGG/ResNet pretrained on ImageNet → fine-tune on your task.

```
# PyTorch example
import torchvision.models as models

model = models.resnet50(pretrained=True)

# Option 1: Fine-tune all layers
for param in model.parameters():
    param.requires_grad = True

# Option 2: Freeze backbone, train only classifier
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(2048, num_classes)  # Replace final layer
```

### When to Fine-Tune

| Scenario | Strategy |
|----------|----------|
| Small dataset + similar domain | Freeze all → train only classifier |
| Small dataset + different domain | Freeze early layers → train later layers + classifier |
| Large dataset + any domain | Fine-tune all layers |

### What Each Layer Learns

When you fine-tune, **which layers to freeze**:
- **Early layers**: generic features (edges, textures) — rarely need to retrain
- **Middle layers**: more specific features — sometimes retrain
- **Late layers**: task-specific features — usually retrain
- **Final layer**: always retrain (new number of classes)

### Feature Extraction

Use CNN as a fixed feature extractor:
```python
# Get features from second-to-last layer
model = models.resnet50(pretrained=True)
model.fc = nn.Identity()  # Remove classifier

features = model(image)  # (batch, 2048) feature vectors
# Use these features with a classical ML classifier
```

This works well even without GPU fine-tuning.

---

## 8. Object Detection & Segmentation

### Beyond Classification: Localization

**Classification**: "What is in the image?" → single label
**Detection**: "What is in the image and where?" → bounding boxes + labels
**Segmentation**: "Which pixels belong to which object?" → pixel masks

### YOLO Family (You Only Look Once)

Reformulate detection as regression:
- Divide image into grid
- Each cell predicts bounding boxes + class probabilities
- Single forward pass → real-time detection

YOLO → YOLOv2 → YOLOv3 → YOLOv5 → YOLOv8 (current)

**Used in**: surveillance, autonomous driving, robotics.

### Mask R-CNN (He et al., 2017)

Two-stage detector:
1. Region Proposal Network → candidate bounding boxes
2. For each box: classify + refine + predict pixel mask

High accuracy, slower than YOLO. State-of-the-art for instance segmentation.

### Semantic Segmentation

Assign a class label to every pixel. Architectures:
- **FCN** (Fully Convolutional Network): first end-to-end approach
- **DeepLab**: atrous/dilated convolutions to increase receptive field without downsampling
- **U-Net**: encoder-decoder with skip connections (described above)
- **SegFormer**: transformer-based, state of the art

---

## 9. Generative Models

### GANs (Generative Adversarial Networks, Goodfellow 2014)

Two networks in competition:

```
Generator G: random noise z → fake image
Discriminator D: image → P(real)

Training:
  D tries to distinguish real from fake
  G tries to fool D

Objective (minimax):
  min_G max_D E[log D(x)] + E[log(1 - D(G(z)))]
```

**Mode collapse**: G learns to generate only a few good images, ignoring diversity.
**Training instability**: notoriously hard to train.

Major variants:
- **DCGAN**: deep convolutional GAN — first stable GAN
- **Progressive GAN**: grow resolution gradually during training
- **StyleGAN**: high-quality facial images, disentangled style control
- **BigGAN**: large-scale class-conditional generation
- **CycleGAN**: unpaired image-to-image translation (horse↔zebra)
- **Pix2Pix**: paired image translation (sketch→photo)

### VAEs (Variational Autoencoders, Kingma & Welling, 2013)

Encoder learns to map to a **distribution** (not a point) in latent space.

```
Encoder: x → μ, σ (mean and std of Gaussian)
Sample: z ~ N(μ, σ²)
Decoder: z → x̂

Loss: Reconstruction loss + KL(q(z|x) || p(z))
              ↓                         ↓
      How well can we decode?   Stay close to N(0,1)
```

The KL regularization forces the latent space to be smooth and continuous → can sample z ~ N(0,1) to generate new examples.

### Diffusion Models (2020–2022)

The currently dominant approach for image generation:

**Noising process** (forward): gradually add Gaussian noise over T steps until pure noise.
**Denoising process** (backward): learn to reverse each noising step.

```
Forward: x₀ → x₁ → x₂ → ... → xₜ → noise
         (add small Gaussian noise at each step)

Backward (learned): noise → xₜ → ... → x₁ → x₀
                    (denoise one step at a time)
```

The network learns to predict the noise at each step (or equivalently, predict x₀ directly).

**Why they dominate GANs:**
- More stable training (no adversarial dynamics)
- Better coverage of data distribution
- Easier to condition on text (for text-to-image)

**Models**: DDPM → DDIM → Stable Diffusion (latent diffusion), DALL-E 2, Midjourney, Imagen

**Latent diffusion** (Stable Diffusion): diffuse in latent space (via VAE encoder/decoder), not pixel space. Much more efficient.

---

## 10. Key Insights from the Deep Learning Era

### Insight 1: Scale Works

AlexNet had 8 layers and 60M parameters. ResNets had 1000 layers. Modern vision transformers have billions of parameters. Performance kept improving with scale.

### Insight 2: Architecture Matters More Than Expected

AlexNet's ReLU activation was critical. ResNet's skip connections unlocked depth. The difference between 26% and 3.57% ImageNet error isn't just scale — architectural decisions mattered enormously.

### Insight 3: Representation Quality Transfers

Features learned on ImageNet transfer to medical imaging, satellite imagery, style transfer, object detection. The hierarchy of representations is genuinely useful.

### Insight 4: Data Augmentation Is Regularization

Rotations, flips, color jitter — training on transformed images prevents overfitting and teaches invariances that generalize. This was eventually superseded by contrastive learning and self-supervised approaches.

### Insight 5: Inductive Biases Are a Double-Edged Sword

CNNs' inductive biases (locality, translation equivariance) are helpful when data is limited but constrain what can be learned. Vision Transformers, with weaker inductive biases, need more data but ultimately achieve better performance.

### Insight 6: The Best Architecture for Images Uses Convolutions (Still)

Despite the ViT excitement, ConvNeXt (2022) showed that pure CNNs, updated with modern training practices, remain competitive with vision transformers. The future is likely a mixture.

---

## Key Points for Exams

1. **CNNs use local filters + weight sharing** → detects local patterns, far fewer parameters than fully connected
2. **AlexNet (2012)** used ReLU, dropout, GPUs → won ImageNet by 10pp → deep learning revolution
3. **VGGNet**: deeper networks with 3×3 filters; two 3×3 = same receptive field as one 5×5 but fewer params
4. **ResNets (2015)**: skip connections allow gradient to bypass layers → enabled 100+ layer networks
5. **Skip connections**: gradient path that doesn't vanish; learned residual is easier than full function
6. **Batch normalization**: normalize activations → allows higher learning rates, acts as regularization
7. **Transfer learning**: pretrain on ImageNet, freeze early layers, fine-tune late layers on target task
8. **GANs**: generator + discriminator in minimax game; mode collapse is the main failure mode
9. **Diffusion models** now dominate image generation (Stable Diffusion, DALL-E); more stable than GANs
10. **ViT**: apply transformer to image patches; needs more data than CNNs to outperform

---

## Practice Questions

1. What is the key innovation of a convolutional layer over a fully connected layer for images?
2. Explain weight sharing in CNNs. Why is it beneficial?
3. What is max pooling and what does it achieve?
4. What was AlexNet and why was it significant?
5. What problem did ResNets solve? How do skip connections address it?
6. Explain the vanishing gradient problem in the context of deep CNNs.
7. What is transfer learning and how is it applied with pretrained CNNs?
8. Compare GANs and diffusion models as generative approaches. Which is currently dominant?
9. What are 1×1 convolutions and what are they used for?
10. What is the difference between semantic and instance segmentation?
