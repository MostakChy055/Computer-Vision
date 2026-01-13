# Background Problem

First, we need to start with the fundamental problem Generative Adversarial Networks (GANs) try to solve: **comparing two probability distributions.**

In the context of computer vision, imagine every image is a point in a high-dimensional space. "Real" images (like photos of cats) cluster together in a specific shape or "distribution." The goal of a GAN is to make the Generator's distribution of "fake" images overlap perfectly with the "real" distribution. To do this, we need a mathematical "ruler" to measure the distance between these distributions so we can minimize it.


## 1. The Basics: What are we measuring?

Before diving into formulas, let's establish what \( P \) and \( Q \) represent:

- \( P \): The **True Distribution** (e.g., the actual pixel patterns found in real human faces).
- \( Q \): The **Model Distribution** (e.g., the pixel patterns your Generator is currently producing).

If \( P \) and \( Q \) are identical, the distance between them should be zero. If they are very different, the distance should be high.

## 2. KL Divergence (Kullback–Leibler)

KL Divergence, often called "Relative Entropy," measures how much information is lost when we use \( Q \) to approximate \( P \).

### The Formula (Discrete)

```math
D_{KL}(P \,\|\, Q) = \sum_x P(x)\,\log\frac{P(x)}{Q(x)}
```

# KL Divergence, Blobs, and Mode Behavior in Vision Models


## The Logic and "Flow"

### The Ratio

```math
\frac{P(x)}{Q(x)}
```

We divide the probability of an event in the real world (`P`) by the probability our model assigned to it (`Q`).

If `Q(x)` is very small (the model thinks an image is impossible) but `P(x)` is large (the image is actually real), this ratio becomes huge.


### The Logarithm

```math
\log(\cdot)
```

We use the log because in information theory, "bits" of information are logarithmic.  
It also turns division into subtraction:

```math
\log P - \log Q
```

making it easier to calculate gradients for optimization.


### The Weighting

```math
P(x)
```

We multiply by `P(x)` because we care most about the points where the real data actually exists.

If `P(x)` is zero (an impossible image), we don't care what the model `Q` thinks about it.


## The Problem: Asymmetry

KL Divergence is asymmetric, meaning:

```math
D_{KL}(P \parallel Q) \neq D_{KL}(Q \parallel P)
```

### Mode Seeking

If we minimize:

```math
D_{KL}(P \parallel Q)
```

the model tries to cover all regions where `P` has high probability.


### Mode Collapsing

If we minimize:

```math
D_{KL}(Q \parallel P)
```

the model prefers to stick to a single "safe" mode where it knows the real data exists, potentially ignoring other variations.


## What is a "Blob" of data?

In statistics and machine learning, a "blob" refers to a **Probability Distribution**, specifically a **Unimodal** one (having one peak).


### The Basics: The Gaussian (Normal) Distribution

Imagine you are measuring the heights of people. Most people are average height, a few are very tall, and a few are very short. When you plot this, it looks like a bell-shaped curve.

- **Therefore, the term "Blob":** We call this a "blob" because the probability "mass" is concentrated in one central area.
- **The Constraint:** Simple models (like a single Gaussian) only have one mean () and one variance (). They are mathematically incapable of having two peaks.



### The Vision Context: The "Cat-Dog" Problem

Imagine "Real Data" consists of two distinct types of images: **Cats** and **Dogs**.

In the high-dimensional space of pixels, "Cats" are one blob and "Dogs" are another.

This is called a **Multimodal Distribution** (it has multiple "modes" or peaks).

If a Model is a "simple blob" (unimodal), it faces a crisis: It physically cannot be in two places at once.

It has to choose:

1. Stretch itself to cover both peaks (and the empty space in between).
2. Sit on top of just one peak and ignore the other.



## 2. : The "Covering" (Mean-Seeking) Intuition

When we write:

```math
D_{KL}(P \parallel Q)
```

the **Real Data (`P`)** is the "weight" that determines how much we care about the error.

---

### The Logic and Flow

The formula is:

```math
\sum_x P(x)\log\frac{P(x)}{Q(x)}
```

1. **The Role of `P(x)` as a Multiplier:** Think of `P(x)` as the "Importance Score." If `P(x)` is high, the math says: *"Pay close attention to what happens here!"*
2. **The "Under-coverage" Penalty:** If there is a real dog at position `x`, then `P(x)` is a high positive number. If your model `Q(x)` ignores dogs, then `Q(x)` is near `0`.
3. **The Explosion:**

```math
\log\frac{P(x)}{Q(x)} \rightarrow \infty
```

4. **Therefore, the "Terror":** Because the infinity is multiplied by a high `P(x)`, the total Loss becomes astronomical.
5. **The Purpose:** To make the Loss smaller, the model `Q` is **forced** to put some probability mass everywhere that `P` exists.

---

### The Visual Result

To cover two separate "Real" blobs (Cats and Dogs) with only one "Model" blob, the model must stretch itself wide.

- **The Side Effect:** It ends up putting probability in the empty space between cats and dogs.
- **Vision Outcome:** This "empty space" represents images that are half-cat, half-dog.

This is why models trained with this loss produce **blurry images**.


## 3. : The "Picking" (Mode-Seeking) Intuition

Now we swap them:

```math
D_{KL}(Q \parallel P)
```

Now, the **Model (`Q`)** is the "weight."

---

### The Logic and Flow

1. **The Role of `Q(x)` as a Multiplier:** Now, the math only cares about the places where the **model** is currently putting its effort.
2. **The "Hallucination" Penalty:** Suppose the model `Q(x)` stretches into the empty space between Cats and Dogs. In that empty space, `Q(x)` is high, but `P(x)` is `0`.
3. **The Explosion:**

```math
\log\frac{Q(x)}{P(x)} \rightarrow \infty
```

4. **Therefore, the "Terror":** Because this infinity is multiplied by the model's own weight `Q(x)`, the Loss explodes.
5. **The Purpose:** To keep the Loss low, the model learns to **only** exist where `P(x)` is definitely greater than zero.

---

### The Visual Result

Instead of stretching, the model decides to collapse onto a single peak.

- **The Decision:** "I'll just stay on the 'Cat' peak."
- **Vision Outcome:** You get a **sharp, perfect image** of a cat.

The model completely forgets that dogs exist.

This is **Mode Collapse**.

---

## Summary of the "Terror"

- **In `D_{KL}(P \parallel Q)`:** The model is terrified of **missing** something real.
- **In `D_{KL}(Q \parallel P)`:** The model is terrified of **inventing** something fake.

---

In GANs, we use the Jensen-Shannon Divergence because it behaves like a "fair judge."

It averages these two fears so the model tries to cover the data (to be diverse) but refuses to stay in the "fake" empty spaces (to be sharp).

# Jensen–Shannon Divergence: The Diplomat of Probability Measures

To help you master vision models, we have to look at the "diplomat" of probability measures. If KL Divergence is a stubborn judge who only sees one side of the story, **Jensen–Shannon (JS) Divergence** is the mediator that finds the middle ground.

**The Question:** What is Jensen–Shannon (JS) Divergence, and how does it solve the fundamental flaws of KL Divergence to make GANs actually work?

---

## 1. The Ground Up: The "Midpoint" Concept

Before looking at the math, imagine two piles of sand:  (the real data) and  (the model's fake data).

In KL Divergence, we compare  directly to . As we learned, if  misses even one grain of , the math "explodes" to infinity. To fix this, JS Divergence introduces a **third distribution**, .

---

### The Logic of  (The Mixture)

We define  as the average of  and :

```math
M(x) = \frac{1}{2}\left(P(x) + Q(x)\right)
```

* **The Reason/Purpose:** By creating , we ensure that any point  that exists in **either** the real data () or the fake data () will also exist in .
* **The Benefit:** Since  contains half of  and half of , the denominator in our comparison can **never be zero** as long as there is data in at least one of the two piles. This effectively "smooths" the landscape so the model doesn't hit infinite penalties.

---

## 2. The Formula: The Flow of Logic

The JS Divergence is defined as the average of two KL Divergences, both using the "mediator"  as the reference.

```math
D_{JS}(P \parallel Q)
=
\frac{1}{2} D_{KL}(P \parallel M)
+
\frac{1}{2} D_{KL}(Q \parallel M)
```

---

### Why this specific structure?

1. **:** This part of the formula punishes the model if it **misses** any part of the real data (the "Covering" fear).
2. **:** This part punishes the model if it **hallucinates** fake data where nothing real exists (the "Picking" fear).
3. **The Resulting Symmetry:** Because both  and  are being compared to the same , it doesn't matter if you swap  and .
* **Therefore:**

```math
D_{JS}(P \parallel Q) = D_{JS}(Q \parallel P)
```

---

## 3. How JS Solves the KL Divergence Problems

### Problem 1: The "Infinity" Explosion

In KL Divergence, if the model  assigned 0% probability to a region where real data  existed, the loss became .

```math
\log \frac{P(x)}{Q(x)} \rightarrow \infty
```

This makes the computer crash or the math break.

* **The JS Solution:** In JS, the denominator is .
* **If:**  

```math
Q(x) = 0 \quad \text{and} \quad P(x) > 0
```

then:

```math
M(x) = \frac{1}{2} P(x)
```

* **Intuition:** The ratio becomes:

```math
\log \frac{P(x)}{M(x)} = \log 2
```

The  is a finite number. The model gets a "firm warning" instead of an "infinite death sentence."

---

### Problem 2: The Asymmetry (Blur vs. Collapse)

As we discussed, KL Divergence forces you to choose between blurry images (Mean Seeking) or repetitive images (Mode Collapse).

* **The JS Solution:** By averaging the two perspectives, JS Divergence acts as a **balanced regularizer**. It pushes the Generator to cover the whole range of data while still keeping the images sharp enough to look realistic.

---

## 4. The "Master Insight": Why GANs use JS Divergence

When Ian Goodfellow invented GANs, he didn't just pick JS Divergence out of a hat. He proved that the **Discriminator's job** is actually a way to calculate JS Divergence.

---

### The Flow of a GAN's Math:

1. The Discriminator  tries to maximize its ability to tell real from fake.
2. **Therefore:** The "Optimal Discriminator" () ends up being exactly the ratio of real data to the total data:

```math
D^*(x) = \frac{P(x)}{P(x) + Q(x)}
```

3. When you plug this "Optimal " back into the GAN's loss function, the math simplifies perfectly into:

```math
\min_G \max_D V(D, G)
=
- \log 4
+
2 \cdot D_{JS}(P \parallel Q)
```

---

**The Takeaway:** When you train a GAN, the Generator is literally trying to "minimize the Jensen-Shannon Divergence" between its fakes and the real world.

---

## Summary Table

| Feature | KL Divergence | JS Divergence |
| --- | --- | --- |
| **Philosophy** | "How much info is lost?" | "How far are we from the average?" |
| **Symmetry** | No (Order matters) | Yes (Order doesn't matter) |
| **Limits** | can be  | Maxes out at  |
| **Visual Result** | Blurry OR Collapsed | Balanced diversity and sharpness |

---

# GANs, JS Divergence, and Wasserstein Distance  
*A Mathematical Proof Connecting Game Theory to Probability*

To help you master vision models, we have to look at the **Mathematical Proof** that connects game theory to probability. It is one of the most elegant parts of deep learning: the discovery that a simple competition between two networks is actually a hidden machine for calculating the similarity between two worlds.

**The Question:** How does the GAN loss function mathematically simplify into Jensen–Shannon (JS) Divergence, what are the catastrophic "blind spots" of JS Divergence, and how does Wasserstein Distance (WGAN) fix them?

---

## 1. Part I: How the GAN Loss becomes JS Divergence

In 2014, Ian Goodfellow proposed the Minimax game. The math follows a specific flow.

---

### Step A: The Objective Function

The "Value Function" represents the game. The Discriminator () wants to maximize it, and the Generator () wants to minimize it.

```math
\min_G \max_D V(D, G)
=
\mathbb{E}_{x \sim P_{\text{data}}}[\log D(x)]
+
\mathbb{E}_{z \sim P_z}[\log(1 - D(G(z)))]
```

* **The Reason/Purpose:**  
  The first term rewards  for correctly identifying real data ().  
  The second term rewards  for correctly identifying fake data ().

---

### Step B: The "Optimal Judge" ()

Before we can train the Generator, we ask:

> If the Generator is fixed, what is the best possible version of the Discriminator?

By taking the derivative of the value function with respect to  and setting it to zero, we obtain:

```math
D^*(x) = \frac{P_{\text{data}}(x)}{P_{\text{data}}(x) + P_G(x)}
```

* **Intuition:**  
  If a specific pixel pattern  is twice as likely to be real as fake, the optimal Discriminator outputs .
  This is a pure probability ratio.

---

### Step C: The Substitution

Substitute  back into the original objective function:

```math
V(D^*, G)
=
\mathbb{E}_{x \sim P_{\text{data}}}
\left[
\log \frac{P_{\text{data}}(x)}{P_{\text{data}}(x) + P_G(x)}
\right]
+
\mathbb{E}_{x \sim P_G}
\left[
\log \frac{P_G(x)}{P_{\text{data}}(x) + P_G(x)}
\right]
```

After algebraic rearrangement:

```math
V(D^*, G)
=
- \log 4
+
2 \cdot D_{JS}(P_{\text{data}} \parallel P_G)
```

* **Therefore, the line:** This is exactly the definition of JS Divergence.
* **The Conclusion:**

```math
\min_G \max_D V(D, G)
\quad \Longleftrightarrow \quad
\min_G D_{JS}(P_{\text{data}} \parallel P_G)
```

Training a GAN is mathematically identical to minimizing JS Divergence.

---

## 2. Part II: The Problem with JS Divergence

JS Divergence has a fatal flaw in high-dimensional spaces: **Disjoint Support**.

---

### The "Zero Gradient" Wall

Assume:

```math
P_{\text{data}}(x) = \delta(x - a)
```

```math
P_G(x) = \delta(x - b)
```

where .

1. If the supports do not overlap:

```math
\text{supp}(P_{\text{data}}) \cap \text{supp}(P_G) = \emptyset
```

2. Then:

```math
D_{JS}(P_{\text{data}} \parallel P_G) = \log 2
```

3. The gradient:

```math
\nabla_\theta D_{JS}(P_{\text{data}} \parallel P_G) = 0
```

* **The Vision Outcome:**  
  The Discriminator is perfect.  
  The Generator receives **zero gradient**.  
  Training stalls completely.

---

### Intuition

The Generator is told it is wrong, but receives no directional signal.

```text
No slope → No movement → No learning
```

---

## 3. Part III: How WGAN Solves It (Earth Mover’s Distance)

In 2017, the Wasserstein GAN (WGAN) was introduced to replace the "ruler" (JS Divergence) with a new one: the Earth Mover's Distance (EMD).
### The Intuition: 
Moving DirtThink of the real distribution ($P$) as a pile of dirt and the fake distribution ($Q$) as a hole in the ground of the same shape.
- The Metric: The distance is the "minimum work" required to move the dirt from **P** into the hole at **Q**.
- Work = (Amount of Dirt) x (Distance it travels).
---
### Why is this better?
Even if the pile of dirt and the hole are 10 miles apart and don't overlap at all, the "work" required to move the dirt is still a clear number 10 times mass.
- If the distance increases to 11 miles: The work increases.
- If the distance decreases to 9 miles: The work decreases.
  
### The Crucial Difference
Unlike JS Divergence (which stays flat), Wasserstein Distance provides a smooth, constant slope (gradient) even when the distributions have zero overlap.8 The Generator always knows which way to move to reduce the "work."The "Weight Clipping" or "Gradient Penalty"The Reason/Purpose: To calculate this "Work," WGAN uses a "Critic" instead of a "Discriminator."9 For the math to hold up, the Critic must be "smooth" (technically, it must be 1-Lipschitz continuous).Therefore, the line: We must constrain the weights of the Critic (using clip_weights or Gradient Penalty) to ensure it doesn't grow too fast. This keeps the "ruler" stable.

WGAN replaces JS Divergence with **Wasserstein Distance**.

---

### The Definition

```math
W(P, Q)
=
\inf_{\gamma \in \Pi(P, Q)}
\mathbb{E}_{(x, y) \sim \gamma} [\|x - y\|]
```

Where:
-  is a joint distribution with marginals  and 
- The cost is physical distance

---

### Intuition: Moving Dirt

*  → pile of dirt  
*  → hole of same volume  

```math
\text{Work} = \text{mass} \times \text{distance}
```

Even if the supports are disjoint:

```math
W(P, Q) > 0
```

and critically:

```math
\nabla_\theta W(P, Q) \neq 0
```

---

### Why This Fixes GANs

| Property | JS Divergence | Wasserstein Distance |
|--------|---------------|---------------------|
| Requires overlap | Yes | No |
| Gradient when disjoint | 0 | Non-zero |
| Training signal | Unstable | Smooth |
| Interpretability | Poor | Excellent |

---

### The WGAN Objective

```math
\min_G \max_{D \in \mathcal{L}}
\mathbb{E}_{x \sim P_{\text{data}}}[D(x)]
-
\mathbb{E}_{x \sim P_G}[D(x)]
```

Where  is the set of **1-Lipschitz functions**.

---

### Enforcing 1-Lipschitz Continuity

#### Weight Clipping

```python
for p in critic.parameters():
    p.data.clamp_(-c, c)
```

#### Gradient Penalty (WGAN-GP)

```math
\lambda \mathbb{E}_{\hat{x}}
\left[
(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2
\right]
```

```math
\hat{x} = \epsilon x + (1 - \epsilon) \tilde{x}
```

---

## Comparison Summary

| Feature | JS Divergence (GAN) | Wasserstein Distance (WGAN) |
|-------|--------------------|-----------------------------|
| Intuition | Curve overlap | Earth moving |
| Failure Mode | Zero gradient | None |
| Loss Meaning | Noise | Distance |
| Stability | Poor | Strong |
| Mode Collapse | Common | Rare |


## 1. The Starting Point: The Primal Form (EMD)

The mathematical foundation of WGAN is the **Wasserstein-1 distance**, also known as the **Earth Mover's Distance (EMD)**.

### The Definition

Imagine the real data distribution  is a pile of earth and the generated distribution  is a hole.  
The EMD is the minimum cost of moving the earth to fill the hole.

### The Logic and Flow

- **(The Transport Plan):** This represents "how much dirt" to move from location  to location .
- **(The Set of all Plans):** This is the collection of every possible way to move the dirt.
- **(The Cost):** This is the distance the dirt travels.
- **The Infimum ():** This means we want the **cheapest** possible plan.
- **The Problem:** Calculating this is computationally impossible for high-dimensional images because you would have to check every possible way to move every pixel to every other pixel.

---

## 2. The Transformation: Kantorovich–Rubinstein Duality

To make this solvable by a neural network, we use a mathematical shortcut called **Duality**.  
This flips the problem from **"finding a plan"** to **"finding a function."**

### The Dual Formula

Through the Kantorovich–Rubinstein duality theorem, the formula becomes:

```math
W(P_r, P_g)
=
\sup_{\|f\|_L \le 1}
\left(
\mathbb{E}_{x \sim P_r}[f(x)]
-
\mathbb{E}_{x \sim P_g}[f(x)]
\right)
```
## The Reason / Purpose of Each Term

- **The Supremum (`\sup`)**  
  Instead of the lowest cost (infimum), we now look for the **maximum difference**.

- **The Function `f` (The Critic)**  
  We replace the *Discriminator* with a **Critic** `f`.  
  Its job is to output a high score for real images and a low score for fake images.

- **The Constraint (`\|f\|_L \le 1`)**  
  This is the **1-Lipschitz constraint**.  
  It is the most critical part of WGAN.

---

## 3. Understanding the 1-Lipschitz Constraint

### Why do we need `\|f\|_L \le 1`?

### The Intuition

The 1-Lipschitz constraint means the *slope* of the function `f` cannot be steeper than 1.

```math
|f(x_1) - f(x_2)| \le |x_1 - x_2|
```
## The Problem without the Constraint

**The Problem:**  
If we don't limit the slope, the Critic \( f \) will try to output \( +\infty \) for every real image and \( -\infty \) for every fake image to make the Wasserstein distance as large as possible.

**The Solution:**  
By forcing the slope to be \( \le 1 \), we ensure the Critic is **smooth**. This guarantees that even if the real and fake images are far apart, there is a continuous slope (gradient) connecting them.

---

## 4. From Math to Code: Enforcing the Constraint

Since we use a Neural Network for \( f \), we need a way to force it to be **1-Lipschitz**. There are two primary ways:

---

### Method A: Weight Clipping (Original WGAN)

**The Action:**  
After every update to the Critic's weights \( w \), we force the weights to stay within a small range, such as: [-0.01, 0.01]


**The Reason:**  
If weights are small, the output cannot change too rapidly relative to the input, effectively limiting the slope.

**The Failure:**  
This is a *brute-force* method and often leads to very simple, poorly trained Critics.

---

### Method B: Gradient Penalty (WGAN-GP)

**The Action:**  
We add a penalty term to the loss function that punishes the Critic if its gradient norm is not close to 1.

\[
L = \text{Original Loss} + \lambda \mathbb{E}_{\hat{x}}
\left[
\left(\|\nabla_{\hat{x}} f(\hat{x})\|_2 - 1\right)^2
\right]
\]

**The Purpose:**  
This *encourages* the Critic to have a slope of 1 everywhere between the real and fake data, providing the best possible gradients for the Generator.

---

## 5. The Step-by-Step WGAN Algorithm

To implement this, the training flow changes significantly from a standard GAN.

---

### Step 1: The Critic Loop

In standard GANs, we update \( D \) and \( G \) once.  
In WGAN, we update the Critic \( f \) **\( n_{\text{critic}} \) times** (usually 5) for every **1 update** of the Generator.

**The Reason:**  
The theory requires an *optimal* Critic to accurately measure the Wasserstein distance. We need \( f \) to be as close to the **supremum** as possible before moving the Generator.

---

### Step 2: The Critic Loss

The Critic maximizes the distance:
\[
L_{\text{critic}} = \mathbb{E}[f(G(z))] - \mathbb{E}[f(x)]
\]

> **Note:** In practice, we minimize the negative of this loss in code.

---

### Step 3: The Generator Loss

The Generator tries to make the Critic's score for its images as high as possible:
\[
L_{\text{gen}} = -\mathbb{E}[f(G(z))]
\]

---

## 6. Summary of the WGAN “Master Logic”

- **Start with EMD:**  
  Provides a distance that never *plateaus* (no zero gradients).

- **Use Duality:**  
  Turns a *moving dirt* problem into a *scoring images* problem via \( f \).

- **Apply Lipschitz Constraint:**  
  Ensures the scoring function \( f \) is smooth so the Generator can follow the slope.

- **Train the Critic More:**  
  Ensures we are measuring the true distance before the Generator moves.

- **Visual Result:**  
  Unlike JS Divergence (which can look flat), the Wasserstein distance looks like a **steady hill**.  
  No matter how far the Generator is, it can always *see* which way is uphill toward real data.

<img width="889" height="406" alt="image" src="https://github.com/user-attachments/assets/45933fee-37ca-4755-9ede-c28ca9a71c4e" />
<img width="878" height="524" alt="image" src="https://github.com/user-attachments/assets/3fa13ab3-0e0f-4861-8f92-d89667d79dc4" />
<img width="872" height="545" alt="image" src="https://github.com/user-attachments/assets/94a99383-856f-4875-a522-d02ff93df314" />
<img width="886" height="544" alt="image" src="https://github.com/user-attachments/assets/ea92ca75-6674-482a-a9e0-52496fe424bd" />
<img width="881" height="499" alt="image" src="https://github.com/user-attachments/assets/f85f4e8a-9381-4263-89e4-8b7239c47ab6" />
<img width="982" height="474" alt="image" src="https://github.com/user-attachments/assets/c6e7a84e-28d0-4aef-b14c-5c8306afbb8a" />
<img width="954" height="423" alt="image" src="https://github.com/user-attachments/assets/e5249e3e-7d25-45c9-a69e-a4a1ef8523be" />
<img width="930" height="322" alt="image" src="https://github.com/user-attachments/assets/f729a175-3503-4730-8eeb-eeecce34e4b8" />
<img width="954" height="423" alt="image" src="https://github.com/user-attachments/assets/76ce2ca6-7bb8-45b9-98d2-f786ad56c12a" />
<img width="930" height="636" alt="image" src="https://github.com/user-attachments/assets/69b27e5b-782f-4d23-827b-6ba560ab0ad5" />

## Conditional GAN
To master vision models, one must understand that a standard GAN is "blindly" creative. If one ask it to generate an image from noise z, you have no idea if it will produce a dog, a cat, or a car. Conditional GANs (cGANs) are the solution to this lack of control—they turn the generator into a "controllable" artist by giving it a specific prompt.

<img width="1009" height="428" alt="image" src="https://github.com/user-attachments/assets/0310598a-0cf6-4f52-8049-5b495f860d90" />
<img width="1001" height="525" alt="image" src="https://github.com/user-attachments/assets/574d2923-8d52-491c-a55e-199012d01f2c" />
<img width="999" height="610" alt="image" src="https://github.com/user-attachments/assets/e983d610-7644-486d-a62a-b73b67c709d8" />
<img width="999" height="610" alt="image" src="https://github.com/user-attachments/assets/da12c20a-b668-4b64-9e7a-fbacbf4afd8d" />
<img width="978" height="226" alt="image" src="https://github.com/user-attachments/assets/91ed1680-2ab4-4cc7-9f8c-77957ac0a17c" />
<img width="989" height="620" alt="image" src="https://github.com/user-attachments/assets/7b65b6c9-5378-4ff9-b314-c8ef67e24a1d" />
<img width="993" height="295" alt="image" src="https://github.com/user-attachments/assets/e12297ed-5d25-4a9c-b6d3-b68b47f99f6b" />
<img width="993" height="295" alt="image" src="https://github.com/user-attachments/assets/109f3db3-0697-4a70-951e-c55e4590e851" />
<img width="1049" height="582" alt="image" src="https://github.com/user-attachments/assets/57ba8703-5c90-42aa-ba31-2d1fd6cc2caa" />
<img width="966" height="686" alt="image" src="https://github.com/user-attachments/assets/443930c0-4050-4da0-9db9-660bbd858994" />
<img width="983" height="630" alt="image" src="https://github.com/user-attachments/assets/b97f768b-522b-4400-aeb1-24b13585ce5b" />
<img width="939" height="646" alt="image" src="https://github.com/user-attachments/assets/b6fc1355-190a-4be1-b3e6-c7e824c92791" />
<img width="922" height="162" alt="image" src="https://github.com/user-attachments/assets/326158b6-8f70-40b2-883a-f81488e980a5" />
<img width="950" height="556" alt="image" src="https://github.com/user-attachments/assets/235d3ae4-962a-454d-b6b2-c4ed0736b8b3" />
<img width="1024" height="458" alt="image" src="https://github.com/user-attachments/assets/18cd2783-3d15-4d54-8e12-c8d599adbd2d" />
<img width="991" height="355" alt="image" src="https://github.com/user-attachments/assets/19030355-4c2c-4e16-a5ab-98ced54702e0" />
<img width="927" height="402" alt="image" src="https://github.com/user-attachments/assets/2eb8ccff-24a7-4a60-91de-645b5bdb2a33" />

To master vision models, you must stop looking at the pixels and start looking at the Latent Space Z. Think of the latent space as a high-dimensional "hidden map" where every point represents a possible image. If the model is well-trained, this map isn't just a random pile of photos; it is a structured universe where similar concepts live near each other.

<img width="965" height="592" alt="image" src="https://github.com/user-attachments/assets/ab218119-d602-4d48-adac-49783c7af5ff" />
<img width="1014" height="419" alt="image" src="https://github.com/user-attachments/assets/ec90318d-10ee-41fe-a972-cbcddfe4b0ff" />
<img width="949" height="642" alt="image" src="https://github.com/user-attachments/assets/1cda6fe0-45ee-425b-9a68-851b5461f563" />
<img width="943" height="691" alt="image" src="https://github.com/user-attachments/assets/86963f24-4b82-4014-a8db-5d72502349fd" />
<img width="922" height="541" alt="image" src="https://github.com/user-attachments/assets/36953ec6-e34b-41bc-a9ad-df865a9a27fd" />
<img width="1002" height="434" alt="image" src="https://github.com/user-attachments/assets/3b864441-10aa-4afa-b2e5-7244473a833e" />
