---
title: "Bayesian Ordinal Regression - WIP"
date: "2024-10-01"
summary: "A gentle introduction to Bayesian Ordinal Regression"
description: "A gentle introduction to Bayesian Ordinal Regression"
toc: false
readTime: true
autonumber: false
math: true
tags: ["Machine Learning", "Bayesian Inference"]
showTags: false
hideBackToTop: false
---

**What is Ordinal Regression.**  Ordinal regression is a type of regression analysis used when the dependent variable is ordinal, meaning the categories have a natural order, but the intervals between them are not necessarily equal. 
The ordering may be subject to heterogeneity, meaning that different factors or groups may influence how the distances between categories vary, and this can be modeled explicitly using ordinal regression techniques.
The goal is to predict the ordinal outcomes while considering both the order and the unequal spacing between categories. 

For example, the temperature feeling is somewhat subjective and can be categorized as "cold", "cool", "neutral", "warm", "hot". 
Such a categorical variable is ordinal, we know that "cold" is colder than "cool", "cool" is colder than "neutral", etc.
Still, for one person, the difference between "cool" and "neutral" might not be the same as between "warm" and "hot".
Different people might have different perceptions of what is cold or hot.

This is where ordinal regression comes in.

**Why it is different from Classical Regression.** Classification treats all categories as independent and does not consider the natural order in ordinal data. For example, "poor" and "excellent" would be treated as equally different from "fair," which ignores the ordinal structure.

**Why is Ordinal Regression Important?**

1. **Preserving Ordinal Structure**. It respects the order of categories, unlike classification, which treats categories as unrelated. This leads to **more accurate models** for ordinal data by avoiding **incorrect assumptions about the relationships between outcomes**.
2. **Handling Unequal Intervals**. It acknowledges that the difference between adjacent categories may not be the same. This is crucial in many real-world situations (e.g., satisfaction scales), where these differences are not uniform. Or 
3. **Capturing Heterogeneity**. Ordinal regression allows for modeling heterogeneity between groups or categories. For instance, different population segments may perceive the distance between "good" and "excellent" differently, and this variability can be accounted for in the model.
4. **Better Interpretability**: Since the model respects the ordinal nature of the data, the results are more interpretable and meaningful when analyzing ordinal outcomes, compared to treating them as continuous or nominal categories.

How to derive from first principles a Bayesian Ordinal Regression?

---
### A detour with Binary Classification

Let's start by assuming we want to predict the binary variable $y$ from a number $N$ features $X \in \mathbb{R}^N$. 
A common approach in statistic is to use an inverse link function $f$ to (un)surprinsingly links the feature $X$ to the $y$. 
To be more precise, we will assume some probabilistic structure attached to the binary variable $y$ by letting $y \sim \operatorname{Ber}(y \mid p)$ and the inverse link function links the feature to the mean of the function $\mathbb{E}(y)=p=f(X ; p)$.

In the case of a linear binary logistic regression, 

$$ 
\mathbb{E}(y)= p =\frac{1}{1+\exp (-\eta)} \quad \text{with} \quad \eta = \sum X_i \omega_i + b
$$

**Why this form?**
We know that $p \in [0,1]$ so we need a function that outputs a value in this range.
The logistic function is a sigmoid function, that is, it is strictly increasing with values in $(0,1)$.
So basically, our program will infer $\omega$ that fits the data.
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

Here, we have derive one model for binary classification. 
In the next section, we will propose an alternative model for binary classification that is more suitable for ordinal data.
In practice, this model is never used for binary classification but is will be a simple use-case to understand the modeling process of ordinal regression.

---
### Bayesian Binary Classification with a twist


**Data Generative Process —** 
We assume that there is a latent continuous variable which censored yield to the ordinal probabilities. Features  influenced the latent variable and as a result influences the final ordinal probabilities.

In contrast to previous approach, we introduce a latent variable $z$ and also a cutpoint $\alpha$ to model the ordinal probabilities.

0. **Prior Distribution.** 
We start by defining the prior distribution of the unknown parameters.
It reflects our beliefs about the likely values of the parameters before we have seen any data.
In this section, we will assume that the ==cutpoints== are fixed and known.

$$\boldsymbol{\beta} \sim \mathcal{N}\left(\mathbf{0}, \sigma_\beta^2\right)$$


1. **Latent Variable Model** $z$. The latent variable is the result of a linear combination of the features and the parameters.
The model will infer the parameters $\boldsymbol{\beta}$ and $\alpha$ from the data. 
We hope that the model will be able to identify the latent variable $z$ that will allow us to predict the ordinal outcome $y$.
Intuitively, for the example of the temperature, for a person, the latent variable $z$ will represent a continuous affinity from the coldest to the hottest temperature.

$$
z = \mathbf{x}_i^{\top} \boldsymbol{\beta}
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

A simple code in numpyro to fit the model:

```python
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

fixed_thresholds = np.array([-1.0, 0, 1])  # Fixed cutpoints

def ordinal_probit_model(X, y, alpha_cutpoints, sigma_beta=2.0, sigma_z=1.0):
    """
    Bayesian Ordinal Probit Model with fixed cutpoints and latent variables.
    For identification, we fix sigma_z=1.0 (standard probit model assumption).
    
    Args:
        X (array): Input features of shape (n_samples, n_features)
        y (array): Target ordinal labels of shape (n_samples,)
        alpha_cutpoints (array-like): Fixed cutoff points for ordinal categories
        sigma_beta (float): Standard deviation for beta prior
    """
    # Get dimensions
    n_samples, num_features = X.shape
    num_categories = len(alpha_cutpoints) + 1
        
    # Sample regression coefficients from prior with larger variance
    beta = numpyro.sample(
        "beta",
        dist.Normal(
            loc=jnp.zeros(num_features),
            scale=jnp.ones(num_features) * sigma_beta
        )
    )
    
    z = jnp.dot(X, beta)
    
    # Compute probabilities using the standard normal CDF
    probs = jnp.zeros((n_samples, num_categories))
    
    # Binary classification case
    # P(y=1) = P(z ≤ α)
    p1 = numpyro.distributions.Normal(0, 1).cdf((alpha_cutpoints[0] - z))
    probs = probs.at[:, 0].set(p1)
    probs = probs.at[:, 1].set(1 - p1)

    # Ensure probabilities sum to 1 and are positive
    probs = jnp.clip(probs, 1e-8, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    
    # Sample observations
    with numpyro.plate("obs", n_samples):
        numpyro.sample("y", dist.Categorical(probs=probs), obs=y)

nuts_kernel = NUTS(ordinal_probit_model)
mcmc = MCMC(nuts_kernel, num_warmup=1500, num_samples=2000, num_chains=1)
mcmc.run(jax.random.PRNGKey(0), X=X, y=y, alpha_cutpoints=fixed_thresholds)

```

TODO: Improve the code that could leverage numpyro better.

TODO: See that it fits well with the data. 

TODO: Speak about the cutoffs points that are fixed 

TODO: Speak about changing the cutoffs points if unknown 

TODO: Should we care about cutoffs points ? 

TODO: Speak about non identificability of cutoffs points.


---
Full Bayesian Approach to Ordinal Regression

TODO

---
# Archive

**What is the probability of a given ordinal outcome $P\left(y_i \mid \mathbf{x}_i\right)$?**

Given the threshold mechanism, the probability of a given ordinal outcome $P\left(y_i \mid \mathbf{x}_i\right)$ is the probability that the latent variable $z_i$ falls within the interval defined by the threshold $\alpha_{k-1}$ and $\alpha_k$.

$$ P(y=k | \mathbf{x})= P(\alpha_{k-1} < z \leq \alpha_k \mid \mathbf{x})= \Phi\left(\frac{\alpha_k-\mathbf{x}^{\top} \boldsymbol{\beta}}{\sigma}\right)-\Phi\left(\frac{\alpha_{k-1}-\mathbf{x}^{\top} \boldsymbol{\beta}}{\sigma}\right)$$


{{< details  title="Full Derivation" >}} 

Since $z \sim \mathcal{N}\left(\mathbf{x}^{\top} \boldsymbol{\beta}, \sigma^2\right)$, we standardize $z$ to the standard normal distribution $(\mathcal{N}(0,1))$ using the transformation:

$$
z^{\prime}=\frac{z-\mathbf{x}^{\top} \boldsymbol{\beta}}{\sigma} \quad \text { so that } z^{\prime} \sim \mathcal{N}(0,1)
$$

Rewriting the thresholds in terms of $z^{\prime}$ :

$$
P(y=k \mid \mathbf{x})= P\left(\frac{\alpha_{k-1}-\mathbf{x}^{\top} \boldsymbol{\beta}}{\sigma} < z' \leq \frac{\alpha_k-\mathbf{x}^{\top} \boldsymbol{\beta}}{\sigma}\right)
$$

For $z^{\prime} \sim \mathcal{N}(0,1)$, the cumulative distribution function (CDF) of $z^{\prime}$, denoted as $\Phi(\cdot)$, gives:

$$
P(y=k \mid \mathbf{x})=\Phi\left(\frac{\alpha_k-\mathbf{x}^{\top} \boldsymbol{\beta}}{\sigma}\right)-\Phi\left(\frac{\alpha_{k-1}-\mathbf{x}^{\top} \boldsymbol{\beta}}{\sigma}\right)
$$

{{< /details >}}

It is relatively easy to adapt this derivation to the MLE case and to derive a code to fit the model:

```python
def ordinal_nll(X, y, theta, thresholds, sigma=1.0):
    """
    Compute the negative log-likelihood for ordinal regression (vectorized).

    Parameters:
    X : ndarray
        Feature matrix where each row represents a sample and each column represents a feature.
    y : ndarray
        Target vector where each element is the target ordinal value for the corresponding sample.
    theta : ndarray
        Parameter vector for features (weights).
    thresholds : ndarray
        Thresholds for ordinal categories (K-1 cutpoints).
    sigma : float
        Standard deviation of the latent variable.

    Returns:
    float
        The negative log-likelihood value.
    """
    # Compute the linear predictor
    linear_pred = X @ theta  # shape: [n_samples]

    # Prepend -inf and append +inf to thresholds for boundary conditions
    thresholds = np.concatenate([[-np.inf], thresholds, [np.inf]])
    
    # Compute probabilities for each category (vectorized)
    cdf_upper = norm.cdf((thresholds[y + 1] - linear_pred) / sigma)
    cdf_lower = norm.cdf((thresholds[y] - linear_pred) / sigma)
    prob_y = cdf_upper - cdf_lower

    # Avoid log(0) with numerical stability
    prob_y = np.clip(prob_y, 1e-15, 1 - 1e-15)
    
    # Compute negative log-likelihood
    return -np.sum(np.log(prob_y))
```








[. . .]