## Regression vs. Classification
If we ask a model to predict the exact $a$ and $b$ values (regression), the model often gets "scared" of making a mistake. To minimize the error, it predicts the average of all possible colors, which is usually a dull gray.

To solve this, researchers like Zhang et al. proposed treating colorization as a classification problem. We divide the 2D AB color space into discrete "buckets" or bins. The model then predicts the probability that a pixel belongs to a specific color bucket. This allows the model to be "bold" and pick vibrant colors.

---

We used our own didn't use the batch-normalization layer of the resnet because?

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

---

### The Effect on your Model

- **Information Washout:** The BN might shift your Lightness features so aggressively that subtle textures (which are vital for deciding color) are flattened out.
- **Desaturation:** If the BN layer "thinks" the input is darker or lighter than it actually is, the rest of the network receives skewed data, leading to the "muted" or gray colors you mentioned.

---

### 4. The Fix: Why GroupNorm (GN)?
In your code, you already usednn.GroupNorm(32, channels) in your ResBlocks. 
- **The Reason:** Unlike BatchNorm, GroupNorm does not care about the batch size or "running statistics" from the past. It calculates the mean and variance for groups of channels within a
single image.
- **Purpose of the fix:** By replacing resnet.bn1 with a GroupNorm , the normalization becomes independent of the ImageNet RGB history. It adapts purely to the L-channel features of the current image, ensuring the "Lightness" information remains stable and accurate.

---

## Color
