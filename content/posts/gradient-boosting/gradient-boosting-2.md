---
title: "Gradient Boosting — 2"
date: "2025-04-18"
summary: " "
description: " "
toc: false
readTime: false
autonumber: false
math: true
tags: ["Machine Learning"]
showTags: false
hideBackToTop: false
---

In Part 1, we derived gradient boosting by computing gradients $g_i$ and hessians $h_i$ of the loss function, then using them to build trees. The word "gradient" is right there in "gradient boosting." So here's a natural question:

**Is gradient boosting just gradient descent?**

At first glance, it seems like it should be. We're computing gradients. We're iteratively improving a model. We're minimizing a loss function. That's exactly what gradient descent does. But look more carefully at what Part 1 actually showed us, and a puzzle emerges.

## The Puzzle: What Are We Descending?

Let's recall gradient descent in its most familiar form. Suppose we have parameters $\theta \in \mathbb{R}^p$ (say, the weights in a neural network) and a loss function $L(\theta)$. Gradient descent updates the parameters by moving opposite to the gradient:

$$
\theta_{t+1} = \theta_t - \alpha \nabla_\theta L(\theta_t)
$$

The gradient $\nabla_\theta L$ is a vector in $\mathbb{R}^p$ pointing in the direction of steepest increase. We move in the opposite direction to decrease the loss. Simple. Effective. Well-understood.

Now look at what we did in Part 1. At each iteration $t$, we:
1. Computed gradients $g_i = \frac{\partial L(y_i, F_{t-1}(x_i))}{\partial F_{t-1}(x_i)}$ for each training point
2. Built a tree $f_t$ using these gradients
3. Updated our model: $F_t(x) = F_{t-1}(x) + f_t(x)$

We computed gradients, yes. But gradients with respect to what? Not parameters $\theta$. We computed:

$$
g_i = \frac{\partial L(y_i, \hat{y})}{\partial \hat{y}}\bigg|_{\hat{y}=F_{t-1}(x_i)}
$$

These are derivatives with respect to the **predictions** $F(x_i)$ at each point $x_i$, not with respect to any parameters.

Here's the puzzle: **In Part 1, we computed gradients with respect to predictions, but we didn't use them the way gradient descent uses gradients.** We didn't update $F(x_i) \leftarrow F(x_i) - \alpha g_i$. Instead, we fit a tree to approximate some relationship involving $g_i$ and $h_i$.

So what's really going on? If this is gradient descent, what are we taking a gradient descent step in?

## The Answer: We're Optimizing Over Functions

The resolution to this puzzle requires a shift in perspective. In gradient boosting, we are not optimizing parameters. We are optimizing the prediction function itself.

Let's make this concrete. Our goal is to find a function $F: \mathcal{X} \to \mathbb{R}$ that minimizes:

$$
\mathcal{L}[F] = \sum_{i=1}^n L(y_i, F(x_i))
$$

Notice the notation: $\mathcal{L}[F]$ uses square brackets because it's a **functional**—it takes a function $F$ as input and returns a number. We want to minimize this functional over all possible prediction functions.

But here's the key question: **What does it mean to do gradient descent when the thing we're optimizing is not a vector $\theta \in \mathbb{R}^p$, but a function $F$ from some infinite-dimensional space?**

This is not a rhetorical question. The entire conceptual foundation of gradient boosting rests on answering it carefully.

## What Would Gradient Descent Over Functions Look Like?

To build intuition, let's think about what gradient descent would mean in function space.

**In parameter space:** We have $\theta \in \mathbb{R}^p$, and the gradient $\nabla_\theta L$ tells us how to change each component $\theta_j$ to decrease the loss. The update is:

$$
\theta_{t+1} = \theta_t - \alpha \nabla_\theta L(\theta_t)
$$

**In function space:** We have $F \in \mathcal{F}$ (some space of functions), and we need something analogous to the gradient that tells us how to change the function $F$ to decrease the loss.

But what does it mean to "change a function"? A function $F: \mathcal{X} \to \mathbb{R}$ assigns a real number to every point in the input space $\mathcal{X}$. To change $F$, we could change its output at any point $x$. That's an infinite-dimensional object!

Let's simplify by focusing on what we can actually observe: the training data. At each training point $x_i$, the function outputs some value $F(x_i)$. The loss depends on these outputs:

$$
\mathcal{L}[F] = \sum_{i=1}^n L(y_i, F(x_i))
$$

To decrease the loss, we want to change the values $F(x_i)$ in helpful directions. How do we figure out which direction is helpful? By computing how the loss changes when we change each $F(x_i)$:

$$
\frac{\partial \mathcal{L}[F]}{\partial F(x_i)} = \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}
$$

This is exactly the gradient $g_i$ from Part 1!

So the gradient with respect to the function $F$, evaluated at the training points, is the vector:

$$
\nabla_F \mathcal{L}[F] = \begin{bmatrix} g_1 \\ g_2 \\ \vdots \\ g_n \end{bmatrix} = \begin{bmatrix} \frac{\partial L(y_1, F(x_1))}{\partial F(x_1)} \\ \frac{\partial L(y_2, F(x_2))}{\partial F(x_2)} \\ \vdots \\ \frac{\partial L(y_n, F(x_n))}{\partial F(x_n)} \end{bmatrix}
$$

This is the **functional gradient**—the gradient of a functional with respect to a function.

## Why Part 1 Is Not (Quite) Gradient Descent

Now we can see why Part 1's approach is not straightforward gradient descent, and why that matters.

If we were doing gradient descent in function space, we would update the function by moving in the direction opposite to the functional gradient:

$$
F_{t+1}(x) = F_t(x) - \alpha \nabla_F \mathcal{L}[F_t]
$$

But wait—there's an immediate problem. The functional gradient $\nabla_F \mathcal{L}[F_t]$ is a vector in $\mathbb{R}^n$ (one component for each training point). It tells us how to change $F(x_i)$ at each training point. But $F(x)$ needs to be defined for all $x \in \mathcal{X}$, not just at the training points!

This is the fundamental challenge: **The functional gradient only tells us how to decrease the loss at the observed training points. It says nothing about what to do at new points $x \notin \{x_1, \ldots, x_n\}$.**

In other words, if we literally followed gradient descent and updated:
- $F_t(x_i) \leftarrow F_{t-1}(x_i) - \alpha g_i$ for each training point $x_i$

we would have no idea what value $F_t(x)$ should take at a new test point $x$.

**This is why Part 1 is not just gradient descent, and this is why we need trees.**

## The Role of Weak Learners: Generalizing the Gradient

The solution to this challenge is both elegant and practical. Instead of directly updating $F$ by subtracting the gradient at the training points, we:

1. **View the negative gradient as a target:** The values $-g_i$ tell us how we'd like to change $F(x_i)$ at each training point
2. **Fit a weak learner to approximate this target:** Train a tree $f_t$ to predict $-g_i$ from $x_i$
3. **Add the weak learner to our ensemble:** Update $F_t(x) = F_{t-1}(x) + \alpha f_t(x)$

The weak learner $f_t$ serves a critical purpose: **it generalizes the gradient from the training points to the entire input space.**

At the training points, $f_t(x_i) \approx -g_i$, so we're approximately moving in the direction of steepest descent (as gradient descent would). But because $f_t$ is a proper function (a tree), it's also defined at new points $x \notin \{x_1, \ldots, x_n\}$, giving us a principled way to make predictions on test data.

This is why fitting trees to gradients works, and why the method is called "gradient boosting" even though it's not literally gradient descent.

**Gradient boosting is gradient descent in function space, implemented via weak learners that generalize the functional gradient from training points to the entire input space.**

## Formalizing the Functional Gradient Descent Algorithm

Now that we understand why weak learners are necessary, let's formalize the gradient boosting algorithm from the functional gradient perspective.

At iteration $t$, we have a current model $F_{t-1}$ and want to improve it. The functional gradient descent procedure is:

**Step 1 - Compute the functional gradient:**
$$
g_i = \frac{\partial L(y_i, F_{t-1}(x_i))}{\partial F_{t-1}(x_i)} \quad \text{for } i=1,\ldots,n
$$

This tells us the direction of steepest ascent in the loss at each training point. We want to move in the opposite direction.

**Step 2 - Fit a weak learner to the negative gradient:**
$$
f_t = \arg\min_{f \in \mathcal{H}} \sum_{i=1}^n \left(f(x_i) + g_i\right)^2
$$

where $\mathcal{H}$ is our hypothesis class (typically regression trees of limited depth). The weak learner $f_t$ approximates $-g_i$ at training points and generalizes to new points.

**Step 3 - Update the model:**
$$
F_t(x) = F_{t-1}(x) + \alpha f_t(x)
$$

where $\alpha > 0$ is a learning rate (step size).

This is gradient descent in function space. The functional gradient $g_i$ provides the direction, the weak learner $f_t$ generalizes this direction to the entire input space, and the update moves our function in this direction.

## Why "Fitting Residuals" Works (And Why It's Not The Whole Story)

Many introductions to gradient boosting describe it as "iteratively fitting trees to residuals." If you've seen this before, you might wonder how it connects to the functional gradient view we've developed.

The answer reveals something important: residual fitting is a special case of functional gradient descent that works for exactly one loss function.

Consider squared loss: $L(y, \hat{y}) = \frac{1}{2}(y - \hat{y})^2$.

The functional gradient is:

$$
g_i = \frac{\partial}{\partial F(x_i)} \frac{1}{2}\left(y_i - F(x_i)\right)^2 = F(x_i) - y_i
$$

So the negative functional gradient is:

$$
-g_i = -(F(x_i) - y_i) = y_i - F(x_i) = r_i
$$

where $r_i = y_i - F(x_i)$ is the residual—the prediction error at point $i$.

**For squared loss, and only for squared loss, the negative functional gradient equals the residual.**

This is why the intuitive explanation "fit trees to residuals" works for regression problems with squared loss. When you fit a tree to predict $-g_i$, you're fitting it to predict the residuals. The tree learns to correct the mistakes of the current model.

But here's the crucial insight: **residual fitting is not a fundamental principle of gradient boosting. It's an accident of squared loss having a particularly simple gradient.**

For any other loss function, the functional gradient $g_i$ will not equal the negative residual $-r_i$, and fitting trees to residuals will not give you gradient descent in function space. The functional gradient view is the fundamental principle that works for all differentiable losses.

## The Power of Functional Gradients: Beyond Squared Loss

Once we understand that gradient boosting is fundamentally about following functional gradients, we can apply it to any differentiable loss function. This is the true power of the functional gradient view: **it's a universal recipe that works across all problem types.**

To see this, let's work through several important loss functions and see what their functional gradients look like. In each case, we'll see that:
1. The functional gradient captures something meaningful about the prediction errors
2. Fitting trees to the negative gradient gives us a sensible algorithm
3. The gradient is generally not the residual (except for squared loss)

### Absolute Loss (Robust Regression)

Suppose we want to be robust to outliers. Squared loss penalizes large errors heavily (quadratically), so outliers dominate the optimization. Absolute loss treats all errors equally:

$$
L(y, \hat{y}) = |y - \hat{y}|
$$

The functional gradient is:

$$
g_i = \frac{\partial}{\partial F(x_i)} |y_i - F(x_i)| = -\text{sign}(y_i - F(x_i)) = \begin{cases}
-1 & \text{if } F(x_i) < y_i \\
+1 & \text{if } F(x_i) > y_i
\end{cases}
$$

Notice: we fit trees to $-g_i = \text{sign}(r_i)$, the sign of the residual. Large and small errors contribute equally to the gradient. This makes the algorithm robust to outliers—no single large error can dominate the gradient.

### Huber Loss (Adaptive Robustness)

Huber loss combines the best of both worlds: squared loss for small errors (efficient) and absolute loss for large errors (robust). For a parameter $\delta > 0$:

$$
L_\delta(y, \hat{y}) = \begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
\delta|y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise}
\end{cases}
$$

The functional gradient adapts to the error magnitude:

$$
g_i = \begin{cases}
F(x_i) - y_i & \text{if } |y_i - F(x_i)| \leq \delta \quad \text{(like squared loss)} \\
\delta \cdot \text{sign}(F(x_i) - y_i) & \text{otherwise} \quad \text{(like absolute loss)}
\end{cases}
$$

For small residuals, we get the full gradient (residual). For large residuals, the gradient saturates at $\pm\delta$, limiting the influence of outliers.

### Logistic Loss (Binary Classification)

For binary classification with $y_i \in \{-1, +1\}$, we use logistic loss:

$$
L(y, \hat{y}) = \log(1 + e^{-y\hat{y}})
$$

The functional gradient is:

$$
g_i = \frac{\partial}{\partial F(x_i)} \log(1 + e^{-y_i F(x_i)}) = -\frac{y_i}{1 + e^{y_i F(x_i)}}
$$

We can rewrite this in terms of the predicted probability. If $p_i = \frac{1}{1 + e^{-y_i F(x_i)}}$ is the probability we assign to the correct class, then:

$$
g_i = -y_i(1 - p_i)
$$

The gradient magnitude is $(1-p_i)$—large when we're uncertain or wrong (small $p_i$), small when we're confident and correct (large $p_i$). This makes sense: confident correct predictions don't need much adjustment.

### Poisson Deviance (Count Data)

For count data $y_i \in \{0, 1, 2, \ldots\}$, we often model $y_i \sim \text{Poisson}(\lambda_i)$ where $\lambda_i = e^{F(x_i)}$. The negative log-likelihood (deviance) is:

$$
L(y, \hat{y}) = e^{\hat{y}} - y\hat{y} \quad \text{(ignoring constant terms)}
$$

The functional gradient is:

$$
g_i = \frac{\partial}{\partial F(x_i)} \left(e^{F(x_i)} - y_i F(x_i)\right) = e^{F(x_i)} - y_i
$$

This is the difference between our predicted count $\lambda_i = e^{F(x_i)}$ and the observed count $y_i$. When we overpredict ($\lambda_i > y_i$), the gradient is positive, pushing $F(x_i)$ down. When we underpredict, the gradient is negative, pushing $F(x_i)$ up.

### The Pattern

In each case, the functional gradient:
- Captures the direction in which changing $F(x_i)$ would decrease the loss
- Incorporates the structure of the loss function (robustness, probability calibration, count modeling, etc.)
- Generally does not equal the residual $y_i - F(x_i)$
- Provides exactly what we need to fit a tree and take a gradient descent step in function space

## From First-Order to Second-Order: Incorporating Curvature

The functional gradient tells us the direction of steepest descent. But it doesn't tell us how far to move in that direction. In standard gradient descent, we use a fixed learning rate $\alpha$. But different regions of the loss surface may have different curvature—in some regions, we can take large steps safely, while in others, large steps might overshoot the minimum.

This is where second derivatives come in. Let's see why they matter and how they connect to Part 1.

### The Limitation of First-Order Methods

When we fit a tree to the negative gradient $-g_i$, we're implicitly assuming the loss surface has similar curvature everywhere. But this isn't true. Consider two points:

- Point $A$ where the loss is changing rapidly ($g_A$ large) but the curvature is gentle (second derivative $h_A$ small)
- Point $B$ where the loss is changing slowly ($g_B$ small) but the curvature is steep (second derivative $h_B$ large)

First-order gradient descent treats these the same if $|g_A| = |g_B|$. But we should probably take a larger step at point $A$ (gentle curvature allows large steps) and a smaller step at point $B$ (steep curvature requires caution).

### The Functional Hessian

Just as we defined the functional gradient, we can define the functional Hessian—the second derivative of the loss with respect to the function values:

$$
h_i = \frac{\partial^2 L(y_i, F(x_i))}{\partial F(x_i)^2}
$$

This measures the curvature of the loss at each point. When $h_i$ is large, the loss is highly curved at point $i$—small changes in $F(x_i)$ lead to large changes in loss. When $h_i$ is small, the loss surface is flatter.

In standard Newton's method with parameters $\theta \in \mathbb{R}^p$, we update:

$$
\theta_{t+1} = \theta_t - H^{-1} \nabla_\theta L(\theta_t)
$$

where $H$ is the Hessian matrix. This adjusts the step size in each direction based on the local curvature.

In function space, the Newton direction at each point would be $-g_i/h_i$. We could fit our tree to this Newton direction:

$$
f_t = \arg\min_{f \in \mathcal{H}} \sum_{i=1}^n \left(f(x_i) + \frac{g_i}{h_i}\right)^2
$$

This is one way to incorporate second-order information. But as we'll see, there's a more elegant approach that leads directly to Part 1's framework.

## Connecting to Part 1: The Taylor Expansion Emerges

Now we can see how the functional gradient view leads naturally to the Taylor expansion framework from Part 1. This connection reveals why Part 1's approach works and what it's really doing.

### The Question of How to Use Second-Order Information

We've established that we want to incorporate both gradients $g_i$ and Hessians $h_i$. But how exactly should we use them when fitting trees?

One approach: fit trees to the Newton direction $-g_i/h_i$. But there's another approach that's more principled and leads to Part 1's framework: **approximate the loss functional itself using Taylor expansion, then minimize this approximation.**

### Building a Quadratic Approximation

At iteration $t$, we want to choose $f_t$ to minimize:

$$
\mathcal{L}[F_{t-1} + f_t] = \sum_{i=1}^n L(y_i, F_{t-1}(x_i) + f_t(x_i))
$$

This is hard to optimize directly—we don't know what form $f_t$ should take, and the loss $L$ might be complicated.

But we can approximate it. Around the point $F_{t-1}(x_i)$, we can expand each loss term using a second-order Taylor approximation:

$$
L(y_i, F_{t-1}(x_i) + f_t(x_i)) \approx L(y_i, F_{t-1}(x_i)) + \underbrace{\frac{\partial L(y_i, F_{t-1}(x_i))}{\partial F_{t-1}(x_i)}}_{g_i} f_t(x_i) + \frac{1}{2} \underbrace{\frac{\partial^2 L(y_i, F_{t-1}(x_i))}{\partial F_{t-1}(x_i)^2}}_{h_i} f_t(x_i)^2
$$

The first term $L(y_i, F_{t-1}(x_i))$ is constant (doesn't depend on $f_t$), so we can drop it. This gives us an approximate objective:

$$
\tilde{\mathcal{L}}[f_t] = \sum_{i=1}^n \left(g_i f_t(x_i) + \frac{1}{2} h_i f_t(x_i)^2\right)
$$

**This is exactly the objective from Part 1!**

The functional gradient $g_i$ and functional Hessian $h_i$ appear naturally as the coefficients in the Taylor expansion. We're approximating the true loss functional with a quadratic functional that's easier to work with.

### Why This Matters

This Taylor approximation perspective reveals several things:

1. **The gradients and Hessians in Part 1 are functional gradients and Hessians**—derivatives with respect to the function values $F(x_i)$, not with respect to parameters.

2. **Part 1's approach is second-order functional gradient descent**—we're not just following the gradient direction, we're using curvature information to build a better approximation of the loss.

3. **The tree-fitting procedure from Part 1 is minimizing this quadratic approximation**—when we find optimal leaf values using $f_j^* = -\sum g_i / \sum h_i$, we're minimizing $\tilde{\mathcal{L}}[f_t]$, not directly minimizing the true loss.

4. **This connects everything**: The functional gradient view explains *why* we compute gradients with respect to predictions. The Taylor expansion explains *how* we use both gradients and Hessians together efficiently.

### From Function Space Back to Practice

Let's trace the full path from theory to implementation:

**Conceptual level:** We want to minimize a loss functional over the space of functions.

**First insight (Part 2):** Use functional gradient descent—compute how the loss changes with each function value.

**Second insight (Part 2):** We can't just update at training points; we need weak learners to generalize the gradient.

**Third insight (Part 2 + Part 1):** Use second-order Taylor approximation to build a quadratic surrogate loss that accounts for curvature.

**Implementation (Part 1):** Build trees by optimizing this surrogate loss, using gradients and Hessians to find optimal splits and leaf values.

The functional gradient view is the conceptual foundation. The Taylor expansion is the computational technique that makes it practical. Together, they give us the complete picture of gradient boosting.

## A Complete Example: Squared Loss from Both Perspectives

Let's work through a complete iteration to see how the functional gradient view and the Taylor expansion view give us the same algorithm.

**Setup:** We have data $(x_i, y_i)$ and current model $F_0(x) = \bar{y}$ (the mean of all $y_i$).

### From the Functional Gradient Perspective

**Step 1 - Compute functional gradient:**

For squared loss $L(y, \hat{y}) = \frac{1}{2}(y - \hat{y})^2$:

$$
g_i = \frac{\partial}{\partial F_0(x_i)} \frac{1}{2}(y_i - F_0(x_i))^2 = F_0(x_i) - y_i = \bar{y} - y_i
$$

The negative gradient is: $-g_i = y_i - \bar{y} = r_i$ (the residual).

**Step 2 - Fit tree to approximate negative gradient:**

Fit a regression tree $f_1$ to predict $-g_i = r_i$ from $x_i$:

$$
f_1 = \arg\min_{f \in \mathcal{H}} \sum_{i=1}^n (f(x_i) - r_i)^2
$$

**Step 3 - Update model:**

$$
F_1(x) = F_0(x) + \alpha f_1(x)
$$

### From the Taylor Expansion Perspective (Part 1)

**Step 1 - Compute gradients and Hessians:**

$$
g_i = F_0(x_i) - y_i = \bar{y} - y_i, \quad h_i = \frac{\partial^2 L}{\partial F^2} = 1
$$

**Step 2 - Build quadratic approximation:**

$$
\tilde{\mathcal{L}}[f_1] = \sum_{i=1}^n \left(g_i f_1(x_i) + \frac{1}{2} h_i f_1(x_i)^2\right) = \sum_{i=1}^n \left(g_i f_1(x_i) + \frac{1}{2} f_1(x_i)^2\right)
$$

**Step 3 - Minimize approximation to find optimal leaf values:**

For a tree partition $\{R_j\}$, the optimal value in leaf $j$ is:

$$
f_{1,j}^* = -\frac{\sum_{i \in R_j} g_i}{\sum_{i \in R_j} h_i} = -\frac{\sum_{i \in R_j} (\bar{y} - y_i)}{|R_j|} = \frac{1}{|R_j|}\sum_{i \in R_j} (y_i - \bar{y}) = \frac{1}{|R_j|}\sum_{i \in R_j} r_i
$$

This is the mean residual in each leaf!

**Step 4 - Update model:**

$$
F_1(x) = F_0(x) + \alpha f_1(x)
$$

### The Two Views Agree

Both perspectives give us the same algorithm:
- **Functional gradient view**: Fit a tree to the residuals (negative gradients)
- **Taylor expansion view**: Minimize the quadratic approximation, which leads to fitting a tree with leaf values equal to mean residuals

For squared loss, both views are mathematically equivalent. But conceptually, they tell us different things:
- The **functional gradient** view tells us *why* we're fitting residuals (we're doing gradient descent in function space)
- The **Taylor expansion** view tells us *how* to incorporate second-order information efficiently (build a quadratic surrogate)

Together, they provide complete understanding: gradient boosting is second-order functional gradient descent, implemented via trees that minimize a quadratic approximation of the loss functional.

## Summary: Two Views, One Algorithm

We started with a puzzle: Part 1 computed gradients, so why isn't gradient boosting just gradient descent? The answer required understanding function space optimization.

**What Part 1 Actually Does:**
- Computes gradients with respect to predictions, not parameters
- Uses these gradients (and Hessians) to build trees
- But doesn't directly perform gradient descent updates

**What Part 2 Reveals:**
- We're doing gradient descent in an infinite-dimensional space of functions
- Gradients with respect to function values (functional gradients) tell us how to improve predictions
- Weak learners generalize these gradients from training points to the entire input space
- Second-order Taylor approximation efficiently incorporates curvature information

**The Complete Picture:**

Gradient boosting is **functional gradient descent** implemented through **tree-based approximation**:

1. **Compute the functional gradient** $g_i = \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}$ at each training point

2. **Generalize via weak learners**: Fit a tree to approximate the negative gradient direction

3. **Use second-order information**: Build a quadratic approximation of the loss that accounts for curvature ($h_i$)

4. **Optimize efficiently**: Use gradients and Hessians to find optimal tree structures and leaf values

**Why This Matters:**

The functional gradient view explains why gradient boosting is so flexible:
- **Any differentiable loss**: Just compute its functional gradient
- **Any weak learner class**: Trees, linear models, neural networks—anything that can approximate the gradient
- **Principled optimization**: We're following the geometry of the loss surface in function space

For squared loss, this reduces to "fitting residuals"—a familiar intuition. But the functional gradient view reveals this as a special case of a much more general principle.

**Connecting the Parts:**

Part 1 showed us *how* to build gradient boosting trees—the mechanics of splits, gains, and leaf values. Part 2 showed us *why* this works—we're doing gradient descent in function space, using Taylor approximation to make it practical.

Together, they give us complete understanding of one of machine learning's most powerful and widely-used algorithms. The algorithm is elegant precisely because it unifies a deep mathematical principle (functional gradient descent) with a practical implementation (tree-based approximation) in a way that works across virtually any loss function and problem type.
