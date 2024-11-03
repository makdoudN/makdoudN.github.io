---
title: "Bayesian Ordinal Regression - WIP"
date: "2024-10-01"
summary: "A gentle introduction to Bayesian Ordinal Regression"
description: "A gentle introduction to Bayesian Ordinal Regression"
toc: false
readTime: true
autonumber: true
math: true
tags: ["Machine Learning", "Bayesian Inference"]
showTags: false
hideBackToTop: false
---

**What is Ordinal Regression.**  **Ordinal regression** is a type of regression analysis used when the dependent variable is ordinal, meaning the categories have a natural order, but the intervals between them are not necessarily equal. 
The goal is to predict the ordinal outcomes while considering both the order and the unequal spacing between categories. 
For example, in a rating scale ("poor," "fair," "good," "excellent"), the difference between "good" and "excellent" might not be the same as between "poor" and "fair." 
Additionally, this variability in the differences can be subject to heterogeneity, meaning that different factors or groups may influence how the distances between categories vary, and this can be modeled explicitly using ordinal regression techniques.

**Why it is different from Classical Regression.** Classification treats all categories as independent and does not consider the natural order in ordinal data. For example, "poor" and "excellent" would be treated as equally different from "fair," which ignores the ordinal structure.

**Why is Ordinal Regression Important?**

1. **Preserving Ordinal Structure**. It respects the order of categories, unlike classification, which treats categories as unrelated. This leads to **more accurate models** for ordinal data by avoiding **incorrect assumptions about the relationships between outcomes**.
2. **Handling Unequal Intervals**. It acknowledges that the difference between adjacent categories may not be the same. This is crucial in many real-world situations (e.g., satisfaction scales), where these differences are not uniform. Or 
3. **Capturing Heterogeneity**. Ordinal regression allows for modeling heterogeneity between groups or categories. For instance, different population segments may perceive the distance between "good" and "excellent" differently, and this variability can be accounted for in the model.
4. **Better Interpretability**: Since the model respects the ordinal nature of the data, the results are more interpretable and meaningful when analyzing ordinal outcomes, compared to treating them as continuous or nominal categories.

## A Start with Binary Classification
---

Let's start by assuming we want to predict the binary variable $y$ from a number $N$ features $X \in \mathbb{R}^N$. 
A common approach in statistic is to use an inverse link function $f$ to (un)surprinsingly links the feature $X$ to the $y$. 
To be more precise, we will assume some probabilistic structure attached to the binary variable $y$ by letting $y \sim \operatorname{Ber}(y \mid p)$ and the inverse link function links the feature to the mean of the function $\mathbb{E}(y)=p=f(X ; p)$.

In the case of a linear binary logistic regression, 

$$ 
\mathbb{E}(y)= p =\frac{1}{1+\exp (-\eta)} \quad \text{with} \quad \eta = \sum X_i \omega_i + b
$$

We can then leverage data to infer $p$ using MLE, MAP, bayesian inference or any approach that fits you.

As an example: here a code to infer the parameters using MLE approaches: 
```python
import numpy as np
from scipy.optimize import minimize

# YOUR DATA GOES HERE
# X = ...      (shape: [n_samples, n_features])
# y = ...      (shape: [n_samples,], integer from 0 to K-1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def nll(X, y, theta):
    """
    Compute the negative log-likelihood for logistic regression.

    Parameters:
    X : ndarray
        Feature matrix where each row represents a sample and each column represents a feature.
    y : ndarray
        Target vector where each element is the target value for the corresponding sample.
    theta : ndarray
        Parameter vector for logistic regression.

    Returns:
    float
        The negative log-likelihood value.
    """
    predictions = sigmoid(X @ theta)
    # Ensure numerical stability by clipping predictions
    predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
    
    log_likelihood = y * np.log(predictions) + (1 - y) * np.log(1 - predictions)
    return -np.sum(log_likelihood)

# Initialization 
theta_init = np.ones(X_noisy.shape[1])

# Use scipy's minimize function to find the MLE estimate
result = minimize(
    lambda theta, X, y: nll(X, y, theta),
    theta_init,
    args=(X_noisy, y),
    method='BFGS'
)

print("Optimization Result:")
print("Success:", result.success)
print("Message:", result.message)
print("Estimated Parameters:", result.x)
print("Function Value (Negative Log-Likelihood):", result.fun)
```



## Bayesian Binary Classification with a twist
---

**Data Generative Process —** There is a latent continuous variable which censored yield to the ordinal probabilities. Features (or Covariate) influenced the latent variable and as a result influences the final ordinal probabilities.

1. **Latent Variable Model** $z$. The latent variable models a kind of continuous affinity from the first ordinal category to the latest one.

$$
z_i =\mathbf{x}_i^{\top} \boldsymbol{\beta}+\varepsilon_i
$$

2. **Threshold Mechanism**.
The observed ordinal outcome $y_i$ is determined by where $z_i$ falls relative to the cutpoints $\alpha_k$ :

$$
y_i = \begin{cases}
1 & \text{if } z_i \leq \alpha_1 \\
2 & \text{if } \alpha_1 < z_i \leq \alpha_2 \\
\vdots & \\
k & \text{if } \alpha_{k-1} < z_i \leq \alpha_k \\
\vdots & \\
K & \text{if } z_i > \alpha_{K-1}
\end{cases}
$$

So, we will need:

$$
P(y_i=k \mid \mathbf{x}_i) = P(\alpha_{k-1} < z_i \leq \alpha_k \mid \mathbf{x}_i)
$$

Substitute the Latent Variable Model:

$$
P\left(y_i=k \mid \mathbf{x}_i\right)=P\left(\alpha_{k-1}<\mathbf{x}_i^{\top} \boldsymbol{\beta}+\varepsilon_i \leq \alpha_k \mid \mathbf{x}_i\right)
$$

Isolate the Error Term $\varepsilon_i$ :

$$P\left(y_i=k \mid \mathbf{x}_i\right)=P\left(\alpha_{k-1}-\mathbf{x}_i^{\top} \boldsymbol{\beta}<\varepsilon_i \leq \alpha_k-\mathbf{x}_i^{\top} \boldsymbol{\beta}\right)$$

Express in Terms of the CDF $G$ — *TODO Explain More* :

Since $\varepsilon_i$ has $\operatorname{CDF} G$ :

$$
P\left(y_i=k \mid \mathbf{x}_i\right)=G\left(\alpha_k-\mathbf{x}_i^{\top} \boldsymbol{\beta}\right)-G\left(\alpha_{k-1}-\mathbf{x}_i^{\top} \boldsymbol{\beta}\right)
$$

In the special case of a **binary ordinal classification**, there are only two categories:
- Categories: $k=1$ and $k=2$
- Thresholds: $\alpha_0=-\infty, \alpha_1, \alpha_2=\infty$

Compute Probabilities
1. For $k=1$ :

$$
P\left(y_i=1 \mid \mathbf{x}_i\right)=G\left(\alpha_1-\mathbf{x}_i^{\top} \boldsymbol{\beta}\right)-G\left(-\infty-\mathbf{x}_i^{\top} \boldsymbol{\beta}\right)
$$


Since $G(-\infty)=0$ :

$$
P\left(y_i=1 \mid \mathbf{x}_i\right)=G\left(\alpha_1-\mathbf{x}_i^{\top} \boldsymbol{\beta}\right)
$$

2. For $k=2$ :

$$
P\left(y_i=2 \mid \mathbf{x}_i\right)=G\left(\infty-\mathbf{x}_i^{\top} \boldsymbol{\beta}\right)-G\left(\alpha_1-\mathbf{x}_i^{\top} \boldsymbol{\beta}\right)
$$


Since $G(\infty)=1$ :

$$
P\left(y_i=2 \mid \mathbf{x}_i\right)=1-G\left(\alpha_1-\mathbf{x}_i^{\top} \boldsymbol{\beta}\right)
$$

We can adapt the previous code with the following likelihood (with some tiny change at the level of the optimizer)

```python
def binary_ordinal_nll(theta, X, y):
    """
    Negative log-likelihood for binary ordinal regression.

    Parameters:
    theta : ndarray
        Parameter vector including both beta coefficients and the threshold alpha_1.
        Length is n_features + 1.
    X : ndarray
        Feature matrix.
    y : ndarray
        Target vector with binary outcomes (0 or 1).

    Returns:
    float
        The negative log-likelihood value.
    """
    _, n_features = X.shape
    beta = theta[:n_features]
    alpha_1 = theta[n_features]  # Threshold parameter

    # Linear predictor
    eta = X @ beta

    # Compute probability using logistic CDF
    # P(y_i = 1 | x_i) = G(alpha_1 - eta)
    G = lambda z: 1 / (1 + np.exp(-z))
    prob = G(alpha_1 - eta)

    # For y_i = 1, use prob; for y_i = 0, use 1 - prob
    # Ensure numerical stability
    prob = np.clip(prob, 1e-15, 1 - 1e-15)
    nll = -np.sum(y * np.log(prob) + (1 - y) * np.log(1 - prob))
    return nll
```







