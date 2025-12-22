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

In Part 1, we saw how gradient boosting builds an ensemble model by iteratively adding trees that optimize a second-order Taylor approximation of the loss. But where does this approach come from? Why are we computing gradients and hessians of the loss function?

To understand gradient boosting at a deeper level, we need to shift our perspective from optimizing parameters to optimizing functions. This functional view reveals gradient boosting as gradient descent in function space, where each iteration takes a step in the direction of steepest descent—not in a finite-dimensional parameter space, but in an infinite-dimensional space of functions.

## From Parameters to Functions

In standard gradient descent, we optimize a loss function over parameters $\theta \in \mathbb{R}^p$:

$$
\min_{\theta} \sum_{i=1}^n L(y_i, f_\theta(x_i))
$$

We iteratively update parameters by moving in the direction opposite to the gradient:

$$
\theta_{t+1} = \theta_t - \alpha \nabla_\theta \sum_{i=1}^n L(y_i, f_{\theta_t}(x_i))
$$

But what if we want to optimize directly over the space of all possible prediction functions $F: \mathcal{X} \to \mathbb{R}$, without committing to a particular parametric form? This is the fundamental idea behind functional gradient descent:

$$
\min_{F \in \mathcal{F}} \sum_{i=1}^n L(y_i, F(x_i))
$$

where $\mathcal{F}$ is some space of functions (potentially infinite-dimensional).

The challenge: if $F$ is not parameterized, how do we take a gradient step? What does it even mean to compute a gradient with respect to a function?

## Functional Gradients

Consider the loss functional that maps each function $F$ to a real number:

$$
\mathcal{L}[F] = \sum_{i=1}^n L(y_i, F(x_i))
$$

We want to find the direction in function space that most rapidly decreases this functional. In finite dimensions, the gradient points in this direction. In function space, we need the **functional gradient**.

The functional gradient of $\mathcal{L}$ with respect to $F$ at the training points is the vector:

$$
\nabla_F \mathcal{L}[F] = \left[\frac{\partial L(y_1, F(x_1))}{\partial F(x_1)}, \ldots, \frac{\partial L(y_n, F(x_n))}{\partial F(x_n)}\right]^\top
$$

This is simply the vector of partial derivatives of the loss with respect to the function values at each training point. Note that this is exactly the gradient vector $g$ we saw in Part 1:

$$
g_i = \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}
$$

## Gradient Descent in Function Space

Following the standard gradient descent update rule, we would like to update our function as:

$$
F_{t+1}(x) = F_t(x) - \alpha \nabla_F \mathcal{L}[F_t]
$$

But there's a problem: $\nabla_F \mathcal{L}[F_t]$ is only defined at the training points $\{x_1, \ldots, x_n\}$. We need a function that:
1. Equals $-g_i$ at each training point $x_i$ (to move in the direction of steepest descent)
2. Can make predictions at new points (generalizes beyond the training set)

This is exactly where weak learners come in. We fit a function $f_t$ (typically a regression tree) to approximate the negative gradient:

$$
f_t = \arg\min_{f \in \mathcal{H}} \sum_{i=1}^n \left(f(x_i) + g_i\right)^2
$$

where $\mathcal{H}$ is our hypothesis class (e.g., regression trees of a certain depth).

The update becomes:

$$
F_t(x) = F_{t-1}(x) + \alpha f_t(x)
$$

This is **gradient boosting**: we're doing gradient descent in function space, where each "step" is a weak learner fitted to approximate the negative functional gradient.

## Connection to Residual Fitting

For many practitioners, gradient boosting is first encountered through the lens of "fitting residuals." Let's see how this emerges naturally from the functional gradient view.

Consider squared loss: $L(y, \hat{y}) = \frac{1}{2}(y - \hat{y})^2$.

The functional gradient is:

$$
g_i = \frac{\partial}{\partial F(x_i)} \frac{1}{2}\left(y_i - F(x_i)\right)^2 = -(y_i - F(x_i)) = -r_i
$$

where $r_i = y_i - F(x_i)$ is the residual.

So for squared loss, the negative functional gradient is exactly the residual! When we fit a tree to approximate $-g_i$, we're fitting it to the residuals $r_i$.

This is why residual fitting works for squared loss. But the functional gradient view reveals something crucial: residual fitting is just a special case. For other loss functions, the functional gradient takes different forms.

## Generalization to Arbitrary Losses

The power of the functional gradient perspective is that it works for any differentiable loss function. Let's examine several important cases.

**Absolute Loss:** $L(y, \hat{y}) = |y - \hat{y}|$

The functional gradient is:

$$
g_i = \frac{\partial}{\partial F(x_i)} |y_i - F(x_i)| = -\text{sign}(y_i - F(x_i))
$$

Here we fit trees to the sign of the residuals, giving equal weight to all errors regardless of magnitude. This makes the method robust to outliers.

**Huber Loss:** For a parameter $\delta > 0$,

$$
L_\delta(y, \hat{y}) = \begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
\delta|y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise}
\end{cases}
$$

The functional gradient is:

$$
g_i = \begin{cases}
-(y_i - F(x_i)) & \text{if } |y_i - F(x_i)| \leq \delta \\
-\delta \cdot \text{sign}(y_i - F(x_i)) & \text{otherwise}
\end{cases}
$$

This combines squared loss behavior for small residuals with absolute loss for large residuals, balancing efficiency and robustness.

**Logistic Loss (Binary Classification):** For $y_i \in \{-1, +1\}$ and $L(y, \hat{y}) = \log(1 + e^{-y\hat{y}})$,

The functional gradient is:

$$
g_i = \frac{\partial}{\partial F(x_i)} \log(1 + e^{-y_i F(x_i)}) = -\frac{y_i}{1 + e^{y_i F(x_i)}}
$$

This can be rewritten as:

$$
g_i = -y_i \cdot (1 - p_i)
$$

where $p_i = \frac{1}{1 + e^{-y_i F(x_i)}}$ is the predicted probability of correct classification. The gradient is larger when we're more confident but wrong, and smaller when we're already correct.

**Poisson Deviance (Count Data):** For $y_i \geq 0$ (counts) and $L(y, \hat{y}) = -y\hat{y} + e^{\hat{y}}$,

The functional gradient is:

$$
g_i = \frac{\partial}{\partial F(x_i)} \left(-y_i F(x_i) + e^{F(x_i)}\right) = e^{F(x_i)} - y_i
$$

The gradient is the difference between the predicted count $e^{F(x_i)}$ and the actual count $y_i$.

## From Functional Gradients to Second-Order Methods

The functional gradient view gives us first-order gradient descent in function space. But we can do better by incorporating curvature information through second-order methods.

Recall that in standard optimization, Newton's method uses both the gradient and the Hessian:

$$
\theta_{t+1} = \theta_t - H^{-1} \nabla_\theta L(\theta_t)
$$

where $H$ is the Hessian matrix. This accounts for the curvature of the loss surface and can converge faster than gradient descent.

In function space, the analog is to use both the functional gradient and the functional Hessian. The functional Hessian at training points is:

$$
h_i = \frac{\partial^2 L(y_i, F(x_i))}{\partial F(x_i)^2}
$$

Instead of fitting $f_t$ to approximate $-g_i$ (first-order), we can fit it to approximate the Newton direction $-g_i/h_i$ (second-order):

$$
f_t = \arg\min_{f \in \mathcal{H}} \sum_{i=1}^n \left(f(x_i) + \frac{g_i}{h_i}\right)^2
$$

But there's a more elegant approach. Instead of changing the target, we can keep the gradient as the target but weight each sample by its curvature. This leads to the Taylor expansion view from Part 1.

## Connecting to Part 1: The Taylor Expansion

Let's see how the functional gradient perspective connects to the second-order approximation we saw in Part 1.

Starting from:

$$
\mathcal{L}[F_t] = \sum_{i=1}^n L(y_i, F_{t-1}(x_i) + f_t(x_i))
$$

We expand $L(y_i, F_{t-1}(x_i) + f_t(x_i))$ around $F_{t-1}(x_i)$ using Taylor's theorem:

$$
L(y_i, F_{t-1}(x_i) + f_t(x_i)) \approx L(y_i, F_{t-1}(x_i)) + g_i f_t(x_i) + \frac{1}{2} h_i f_t(x_i)^2
$$

where $g_i$ and $h_i$ are the functional gradient and Hessian at the previous iterate.

Dropping the constant term and summing over all samples:

$$
\mathcal{L}[F_t] \approx \sum_{i=1}^n \left(g_i f_t(x_i) + \frac{1}{2} h_i f_t(x_i)^2\right)
$$

This is exactly the objective we optimized in Part 1! The Taylor expansion provides a quadratic approximation to the loss functional, which we then minimize over the space of weak learners.

The gradient $g_i$ tells us the direction to move, and the Hessian $h_i$ tells us about the local curvature, allowing us to calibrate our step size appropriately at each point.

## Concrete Example: From Function Space to Trees

Let's work through a complete example with squared loss to see how everything connects.

**Setup:** We have data $(x_i, y_i)$ and current model $F_0(x) = \bar{y}$ (constant prediction).

**Step 1 - Compute Functional Gradient:**

For squared loss, $g_i = -(y_i - F_0(x_i)) = -(y_i - \bar{y}) = -r_i$ (negative residuals).

**Step 2 - Fit Weak Learner:**

We fit a regression tree $f_1$ to the targets $-g_i = r_i$:

$$
f_1 = \arg\min_f \sum_{i=1}^n (f(x_i) - r_i)^2
$$

The tree partitions the input space into regions $R_j$ and predicts a constant $f_{1,j}$ in each region.

**Step 3 - Optimal Leaf Values (Second-Order):**

For squared loss, $h_i = 1$ for all $i$. The optimal prediction in leaf $j$ from Part 1 is:

$$
f_{1,j}^* = -\frac{\sum_{i \in R_j} g_i}{\sum_{i \in R_j} h_i} = -\frac{\sum_{i \in R_j} g_i}{|R_j|} = \frac{\sum_{i \in R_j} r_i}{|R_j|}
$$

This is just the mean residual in the leaf—exactly what a standard regression tree would compute!

**Step 4 - Update:**

$$
F_1(x) = F_0(x) + \alpha f_1(x) = \bar{y} + \alpha f_1(x)
$$

For squared loss, the functional gradient view, residual fitting, and the second-order Taylor approximation all give the same algorithm. But only the functional gradient view generalizes seamlessly to arbitrary differentiable losses.

## The Elegant Unity

The functional gradient perspective reveals the elegant unity underlying gradient boosting:

1. **Function space optimization:** We're doing gradient descent over functions, not parameters
2. **Weak learners as gradient approximators:** Trees approximate the direction of steepest descent
3. **Loss function flexibility:** Any differentiable loss works; just compute its gradient
4. **Second-order refinement:** Hessians provide curvature information for better steps
5. **Residual fitting as special case:** For squared loss, gradients happen to be residuals

This perspective also clarifies why gradient boosting is so powerful: we're directly optimizing the predictive function in the space of all possible functions (constrained by our weak learner class), guided by the geometry of the loss surface at each iteration.

In Part 1, we saw the practical machinery of gradient boosting through Taylor expansions and gain calculations. Here, we've seen the conceptual foundation: gradient boosting is simply gradient descent in function space, with weak learners playing the role of gradient approximators. Together, these views provide both the intuition and the implementation recipe for one of machine learning's most effective algorithms.
