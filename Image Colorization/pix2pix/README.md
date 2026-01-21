## Table of Contents
- [Fundamentals](#fundamentals)
    - [Image Colorization: Regression vs. Classification](#imagecolorization:regressionvs.classification)
    - [Color Diversity: Class Rebalancing](#colordiversity:classrebalancing)
    - [K-Means: Designing the "Artist's Palette](#K-Means:Designingthe"Artist'sPalette)
    - [Diplomacy between Mean and Mode: Temperature and Annealed Mean](#DiplomacybetweenMeanandMode:TemperatureandAnnealedMean)
- [Generator](#Generator)
- [Problem Formulation](#problem-formulation)
- [Methodology](#methodology)
  - [Model Architecture](#model-architecture)
  - [Training Strategy](#training-strategy)
  - [Loss Functions](#loss-functions)
- [Experiments](#experiments)
  - [Datasets](#datasets)
  - [Evaluation Metrics](#evaluation-metrics)
- [Results and Analysis](#results-and-analysis)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [References](#references)

# Fundamentals
## Image Colorization: Regression vs. Classification
If we ask a model to predict the exact $a$ and $b$ values (regression), the model often gets "scared" of making a mistake. To minimize the error, it predicts the average of all possible colors, which is usually a dull gray.

To solve this, researchers like Zhang et al. proposed treating colorization as a classification problem. We divide the 2D AB color space into discrete "buckets" or bins. The model then predicts the probability that a pixel belongs to a specific color bucket. This allows the model to be "bold" and pick vibrant colors.

---

## Color Diversity: Class Rebalancing (Color Rarity)
1. **The Basics: The "Gray World" Distribution**
If you look at every pixel in 100,000 random photos, you will find that the vast majority of pixels are desaturated (near-zero AB values). Think of asphalt, clouds, beige walls, and shadows.
- **The Data Bias:** Bright, vibrant colors (like a neon orange flower or a turquoise sea) are mathematically "rare" in a dataset.
- **The Model's "Laziness":** During training, the model tries to minimize its Loss. If 90% of the pixels in your dataset are "mostly gray," the model discovers a "cheat code": if it predicts gray for everything, it will be right 90% of the time. This is why many colorization models produce boring, brownish-gray results.


2. **The Intuition: Prior-Weighted Loss**
Imagine you are studying for a test. 90% of the questions are "True/False" and 10% are "Complex Calculus." If you get a "True/False" wrong, you lose 1 point. If you get a "Calculus" question wrong, you lose 100 points. You will suddenly start paying much more attention to the Calculus.

In your code, your ab_class_head outputs 313 bins. Without rebalancing, the Cross-Entropy (CE) loss treats a mistake on "Gray" the same as a mistake on "Vibrant Purple."


3. **The Mathematical Solution: Weighted Cross-Entropy**
We calculate a weight w for every color bin q based on its frequency in the dataset P(q). The weight is usually defined as:

<img width="367" height="99" alt="image" src="https://github.com/user-attachments/assets/50923f28-11d1-422f-98d7-1e1462fcc4fe" />

- P(q): The probability of color q appearing in your training data.
- A: A smoothing factor (usually 0.5) to prevent rare colors from having "infinite" weight.
- **Purpose:** This formula ensures that when the model fails to predict a rare, vibrant color, the "penalty" (the Loss) is much higher than when it misses a common gray color.

4. **Alternative: Focal Loss**

You also mentioned Focal Loss. Instead of weighting by color frequency, Focal Loss weights by difficulty. The formula is:
<img width="369" height="65" alt="image" src="https://github.com/user-attachments/assets/07e893c3-b303-4629-9c88-9c8f2c7509fe" />

- Intuition: If the model is already confident (pt is high) that a pixel is gray, the term (1 - Pt) becomes very small, essentially "turning off" the loss for that pixel. If the model is
struggling with a "hard" vibrant pixel, the loss remains high, forcing the model to focus its learning energy there.


## K-Means: Designing the "Artist's Palette"

<img width="687" height="708" alt="image" src="https://github.com/user-attachments/assets/869eba80-b90e-4820-993f-15d183223fa7" />


Imagine you have a thousand marbles of different shades of blue and green scattered on a floor. If I tell you to group them into exactly 5 piles, you would naturally put the darkest blues together, the sea-foams together, etc. K-Means is the mathematical version of this. It finds "centroids" (center points) that represent the average of a cluster of data points.

### The Purpose: Why do we need it for Colorization? 
In the L*a*b* color space, the a and b channels can technically take on thousands of combinations. If we treat colorization as a classification problem (picking a category), we can't
have 10,000 categories-the model would be too slow and would never have enough data to learn them all.

We need to "quantize" (limit) the colors to a manageable number, like 313 bins.

### The Logic Flow & Reasoning

1. ** Uniform Grids are Wasteful:** If we just drew a grid over the ab space, we would create "bins" for colors that don't exist in the real world (like a color that is "ultra-neon purple-
green").
2. **Real-World Data is Clumpy:** Real photos have lots of "flesh tones," "sky blues," and "forest greens."
3. **K-Means to the Rescue:** Therefore, we run K-Means on a dataset of millions of real pixels. K-Means looks at where the data actually "clumps" and places our 313 bins exactly in those spots.

- **Result:** Our 313 categories are "smart." We have many bins in the common color regions and very few in the impossible color regions.

**Intuition:** Imagine you are an artist who can only carry 313 tubes of paint. You wouldn't pick 313 random colors. You would look at the world and realize you need 50 different browns and
greens for trees, but maybe only 1 or 2 weird neon pinks. K-Means is the process of choosing those 313 tubes based on what the world actually looks like.

---

### Diplomacy between Mean and Mode: Temperature and Annealed Mean
After the model is trained, for every pixel, it outputs a list of 313 probabilities (e.g., "I'm 40% sure this is Bin #5, 30% sure it's Bin #6 ... "). We now have to turn those 313 numbers back into
one single ab color.

**The Basics: The "Mean" vs. The "Mode"**
To get one color from many probabilities, we have two "basic" options:

1. **The Mode (Argmax):** Take the color of the bin with the highest probability.
- **Problem:** If one pixel is Bin #5 and the pixel next to it is Bin #10, the color might jump instantly. This creates "spatial artifacts" (ugly, jagged edges of color).

2. **The Mean (Expectation):** Take a weighted average of all bins.
- Problem: If the model thinks a pixel is either Red or Blue, the average is Gray. This leads to the "muddy" desaturated look we want to avoid.


### The Reasoning: Why use "Annealed Mean"?

We want a compromise: the vibrancy of the Mode but the smoothness of the Mean. We achieve this using a Temperature (T) inside a Softmax function.

The formula for the probability Pi of a bin i with a logit (raw score) zi is:
<img width="313" height="116" alt="image" src="https://github.com/user-attachments/assets/5f11c32b-2632-4e48-88c1-8b78bf747d41" />

**How Temperature (T) Changes the Flow:**
- **High Temperature (T > 1):** It makes the distribution "flatter." The differences between "likely" and "unlikely" become smaller. This results in a very blurry, grayish image.
- **Low Temperature (T -> 0):** It "sharpens" the distribution. It makes the highest probability much, much higher than the others, effectively ignoring the weak guesses.
- **The "Sweet Spot" (T = 0.38):** Research (Zhang et al.) found that 0.38 is the magic number. It is low enough to keep colors vibrant (pushing the average toward the strongest guess) but high enough to keep

  

### Why not use resnet's batch-normalization?

BatchNorm was designed to solve "Internal Covariate Shift." As data flows through a network, the distribution of values can shift wildly, making it hard for the next layer to learn. BN stabilizes
this by:

- 1. Calculating the Mean and Variance of the features in a batch.
- 2. Normalizing the features to have a mean of 0 and a variance of 1.
- 3. Scaling and shifting them using two learnable parameters: y (Gamma) and ß (Beta).

### The Problem: The Statistics of Lightness vs. Color
When you use a pretrained ResNet, the bn1 layer has "Running Mean" and "Running Variance" values saved from training on millions of RGB images.

- **RGB Statistics:** In RGB, the three channels are highly correlated. The mean value of a "Red" channel in ImageNet is very different from the mean value of an "L" (Lightness)
channel in LAB space.
- **The Mismatch:** Your new initial[0] convolution now produces features based on a resnet.bn1 "correct" them using the old RGB-based statistics it remembers.

### The Effect on your Model

- **Information Washout:** The BN might shift your Lightness features so aggressively that subtle textures (which are vital for deciding color) are flattened out.
- **Desaturation:** If the BN layer "thinks" the input is darker or lighter than it actually is, the rest of the network receives skewed data, leading to the "muted" or gray colors you mentioned.

### 4. The Fix: Why GroupNorm (GN)?
In your code, you already usednn.GroupNorm(32, channels) in your ResBlocks. 
- **The Reason:** Unlike BatchNorm, GroupNorm does not care about the batch size or "running statistics" from the past. It calculates the mean and variance for groups of channels within a
single image.
- **Purpose of the fix:** By replacing resnet.bn1 with a GroupNorm , the normalization becomes independent of the ImageNet RGB history. It adapts purely to the L-channel features of the current image, ensuring the "Lightness" information remains stable and accurate.


transitions smooth.

**Intuition: The "Election" Analogy**
Imagine 313 people voting on what color a pixel should be.
- **Without Temperature (T = 1):** You take everyone's vote equally. If 50 people vote "Red" and 50 vote "Blue," you paint it Purple (grayish/muddy).
- **With Low Temperature (T = 0.38):** You give the people with the most confidence a "megaphone." The person who is most sure it's Red gets their vote multiplied. Now, instead of a muddy purple, the "Red" influence wins out, but the "Blue" voters still provide a tiny bit of influence to keep the edges from being too sharp.

"Annealing" is a term from metallurgy (cooling metal). In this context, it refers to "cooling" the probability distribution to make it more rigid and certain, rather than fluid and vague.

# Generator
