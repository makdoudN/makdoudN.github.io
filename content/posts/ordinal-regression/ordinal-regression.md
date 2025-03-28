---
title: "An introduction to Ordinal Regression"
date: "2024-10-01"
summary: " "
description: " "
toc: false
readTime: true
autonumber: false
math: true
tags: ["Machine Learning"]
showTags: false
hideBackToTop: false
---

### **What is Ordinal Regression.**

Ordinal regression is a type of regression analysis used when the dependent variable is ordinal, meaning the categories have a natural order, but the intervals between them are not necessarily equal. 
The ordering may be subject to heterogeneity, meaning that different factors or groups may influence how the distances between categories vary, and this can be modeled explicitly using ordinal regression techniques.
The goal is to predict the ordinal outcomes while considering both the order and the unequal spacing between categories. 

For example, the temperature feeling is somewhat subjective and can be categorized as "cold", "cool", "neutral", "warm", "hot". 
Such a categorical variable is ordinal, we know that "cold" is colder than "cool", "cool" is colder than "neutral", etc.
Still, for one person, the difference between "cool" and "neutral" might not be the same as between "warm" and "hot".
Different people might have different perceptions of what is cold or hot.

**Why it is different from Classical Regression.** Classification treats all categories as independent and does not consider the natural order in ordinal data. For example, "poor" and "excellent" would be treated as equally different from "fair," which ignores the ordinal structure.

### **Why is Ordinal Regression Important?**

**Preserving Ordinal Structure**. It respects the order of categories, unlike classification, which treats categories as unrelated. This leads to **more accurate models** for ordinal data by avoiding **incorrect assumptions about the relationships between outcomes**.

**Capturing Heterogeneity**. Ordinal regression allows for modeling heterogeneity between groups or categories. For instance, different population segments may perceive the distance between "good" and "excellent" differently, and this variability can be accounted for in the model.

**Better Interpretability**: Since the model respects the ordinal nature of the data, the results are more interpretable and meaningful when analyzing ordinal outcomes, compared to treating them as continuous or nominal categories.

How to derive from first principles a Bayesian Ordinal Regression?

---
### A detour with Binary Classification
Let's start with observations $y \in\{0,1\}$. 
The simplest probabilistic model for binary data is the Bernoulli distribution:

$$
y \sim \operatorname{Bernoulli}(p), \quad \text { with } \quad \mathbb{E}[y]=p
$$

where $p$ is the probability of observing $y = 1$

We will assume that this probability $p$ is linked to a set of feature $X \in \mathbb{R}^N$.
We also introduce a set of parameters $\omega \in \mathbb{R}^N$ and an intercept $b \in \mathbb{R}$ and 
posit the linear relationship. 
$$
\eta=X^{\top} \omega+b
$$

However, directly equating this linear predictor $\eta$ to the probability $p$ isn't possible since $p \in[0,1]$ and $\eta \in(-\infty, \infty)$.
We complete the link between $\eta$ and $p$ by introduce a (inverse) link function (which is a smooth, monotonic transformation) mapping $\eta$ to p: 
$$
p=f(\eta)=\frac{1}{1+e^{-\eta}}
$$
Above, we use logistic function as inverse link function ensuring that $p \in(0,1)$

Given data $\left\{\left(X_i, y_i\right)\right\}_{i=1}^M$, the likelihood function is:

$$
L(\omega, b)=\prod_{i=1}^M \operatorname{Bernoulli}\left(y_i \mid p_i\right)=\prod_{i=1}^M p_i^{y_i}\left(1-p_i\right)^{1-y_i}
$$


To simplify calculations, we maximize the log-likelihood instead:

$$
(\hat{\omega}, \hat{b})=\arg \max _{\omega, b}\left[\sum_{i=1}^M \log \text { Bernoulli }\left(y_i \mid p_i\right)\right]
$$


Expanding explicitly:

$$
\ell(\omega, b)=\sum_{i=1}^M\left[y_i \log \left(p_i\right)+\left(1-y_i\right) \log \left(1-p_i\right)\right]
$$

where

$$
p_i=\frac{1}{1+e^{-\left(X_i^{\top} \omega+b\right)}}
$$

```python
import numpy as np 
from scipy.stats import norm
from scipy.optimize import minimize

def logistic_fn(x):
    return 1 / ( 1 + np.exp(-x))

# Numerically stable sigmoid
def logistic_fn_stable(z):
    out = np.empty_like(z)
    positive = z >= 0
    negative = ~positive
    out[positive] = 1 / (1 + np.exp(-z[positive]))
    exp_z = np.exp(z[negative])
    out[negative] = exp_z / (1 + exp_z)
    return out

def nll(params, X, y):
    omega, b = params[:-1], params[-1]
    eta = np.einsum('i,bi->b', omega, X) + b
    p = logistic_fn_stable(eta) 
    p = np.clip(p, 1e-8, 1 - 1e-8)
    ll = y * np.log(p) + (1 - y) * np.log(1 - p)
    return -ll.sum()

X_train = ...  # ndarray of shape (N, d)
y_train = ...  # ndarray of shape (d)

initial_params = np.zeros(X_train.shape[1] + 1)
result = minimize(nll, initial_params, args=(X_train, y_train), method='BFGS')
```
We can use the following utilities functions to score observations.
```python
def predict(params, X, threshold: float = 0.5):
    omega, b = params[:-1], params[-1]
    eta = np.einsum('i,bi->b', omega, X) + b
    probas = logistic_fn_stable(eta)
    return (probas >= threshold).astype(int)


def predict_proba(params, X, threshold: float = 0.5):
    omega, b = params[:-1], params[-1]
    eta = np.einsum('i,bi->b', omega, X) + b
    probas = logistic_fn_stable(eta)
    return probas

# Make predictions
probas = predict_probas(result.x, X_train, K)
predicted_classes = predict(result.x, X_test, K)

```

### Binary "Ordinal" Modeling

The Probit Model with a latent variable threshold mechanism intuitively captures binary classification by assuming an underlying, unobserved (latent) continuous variable that reflects a propensity or inclination toward a particular outcome.

**Latent Propensity.** Imagine each observation has a hidden score (z), influenced by features $X$. 
Higher scores represent greater inclination toward a positive outcome.

**Thresholding.** We introduce a cutoff (typically at zero) to separate outcomes. If $z$ exceeds this threshold, we observe $y=1$, otherwise, we observe $y=0$.

$$y= \begin{cases}0 & \text { if } z \leq \alpha \\ 1 & \text { if } z>\alpha\end{cases}$$

This process turns a continuous latent score into a binary decision.
Then, the probability of observing $y=1$ given predictors $X$ is:

$$
P(y=1 \mid X)=P(z> \alpha \mid X)
$$


**Probit Link Function  (Normal CDF).**

We will suppose the latent propensity variable is a linear function of the features with the addition of a noise $\epsilon \sim \mathcal{N}(0,1)$.
$$
z=X^{\top} \beta+\epsilon
$$
This result in the latent variable being Normally distributed $z\ |\ X\sim \mathcal{N}(X^{\top} \omega, 1)$.

Using the standard normal cumulative distribution function (CDF) $\Phi(\cdot)$ :

$$
P(y=1 \mid X)=1-\Phi(-X^T \beta) = \Phi(X^T \beta)
$$

The MLE objective becomes 

$$
(\hat{\beta}, \hat{b})=\arg \max _{\beta, b}\left[\sum_{i=1}^M y_i \log \Phi\left(X_i^{\top} \beta+b\right)+\left(1-y_i\right) \log \left(1-\Phi\left(X_i^{\top} \beta+b\right)\right)\right]
$$

Without the assumption of $\alpha = 0$:

$$ P(y=1 \mid X, \beta, b, \alpha)=P(z>\alpha)=1-\Phi\left(\alpha-X^{\top} \beta-b\right) $$

In this case, we would like to also optimze the threshold leading to more difficult loss function

$$
(\hat{\beta}, \hat{b}, \hat{\alpha})=\arg \max _{\beta, b, \alpha}\left[\sum_{i=1}^M y_i \log \Phi\left(X_i^{\top} \beta+b-\alpha\right)+\left(1-y_i\right) \log \left(1-\Phi\left(X_i^{\top} \beta+b\right)\right)\right]
$$

The optimization is not more complicated than in the classical binary logistic regression

```python
def probit_nll(params, X, y):
    beta, b = params[:-1], params[-1]
    eta = np.einsum('i,bi->b', beta, X) + b
    p = norm.cdf(eta)
    p = np.clip(p, 1e-8, 1 - 1e-8)
    ll = y * np.log(p) + (1 - y) * np.log(1 - p)
    return -ll.sum()

def predict_proba(params, X):
    beta, b = params[:-1], params[-1]
    eta = np.einsum('i,bi->b', beta, X) + b
    return norm.cdf(eta)

def predict(params, X, threshold=0.5):
    return (predict_proba(params, X) >= threshold).astype(int)

initial_params = np.zeros(X_train.shape[1] + 2)
result = minimize(
    probit_nll, 
    initial_params, 
    args=(X_train, y_train), method='BFGS'
)
beta_hat, b_hat, alpha_hat = result.x[:-2], result.x[-2], result.x[-1]
probas = predict_proba(result.x, X_test)
```

We are now ready to derive multi-class ordinal regression which is a simple adaptation from this binary class.

### Ordinal Probit Model with Latent Variable and Threshold Mechanism


The observed ordinal outcome $y \in\{1,2, \ldots, K\}$ is determined by comparing the latent variable $z$ to multiple ordered thresholds $\alpha_1<\alpha_2<\cdots<\alpha_{K-1}$ :

$$
y = \begin{cases}1 & \text { if } z \leq \alpha_1 \\ 2 & \text { if } \alpha_1 < z \leq \alpha_2 \\ \vdots & \\ k & \text { if } \alpha_{k-1} < z \leq \alpha_k \\ \vdots & \\ K & \text { if } z > \alpha_{K-1} \end{cases}
$$

Given the latent structure, the probability of observing category $k$ is the probability that $z$ falls between two consecutive thresholds, $\alpha_{k-1}$ and $\alpha_k$. Formally, this is expressed as:

$$
P\left(y_i=k \mid X_i, \beta, b, \alpha\right)=P\left(\alpha_{k-1} < z \leq \alpha_k\right)=\Phi\left(\alpha_k-X_i^{\top} \beta-b\right)-\Phi\left(\alpha_{k-1}-X_i^{\top} \beta-b\right)
$$


Here, $\Phi$ denotes the cumulative distribution function (CDF) of the standard normal distribution. Intuitively, this captures the idea that the latent variable must lie within the specified thresholds for the observation to belong to category $k$.

Boundary conditions are:
- $\Phi(-\infty)=0$ and $\Phi(\infty)=1$.


```python

def ordinal_probit_nll(params, X, y, K):
    n_features = X.shape[1]
    
    # Extract parameters
    beta = params[:n_features]  # Feature coefficients
    b = params[n_features]      # Intercept
    # Thresholds as cumulative sum of positive increments
    alpha_deltas = np.exp(params[n_features+1:])  # Ensure positive increments
    alphas = np.cumsum(np.concatenate(([0], alpha_deltas)))[:-1]  # K-1 thresholds
    
    # Linear predictor
    eta = np.einsum('ij,j->i', X, beta) + b
    
    # Bounds for each category
    lower_bounds = np.concatenate(([-np.inf], alphas))
    upper_bounds = np.concatenate((alphas, [np.inf]))
    
    # Probabilities
    prob = norm.cdf(upper_bounds[y] - eta) - norm.cdf(lower_bounds[y] - eta)
    prob = np.clip(prob, 1e-8, 1 - 1e-8)
    
    # Negative log-likelihood
    ll = np.sum(np.log(prob))
    return -ll

def predict_probas(params, X, K):
    n_features = X.shape[1]
    
    # Extract parameters the same way as in nll function
    beta = params[:n_features]  
    b = params[n_features]
    alphas = np.sort(params[n_features+1:])
    
    # Calculate linear predictor
    eta = np.einsum('ij,j->i', X, beta) + b
    
    # Set up bounds
    lower_bounds = np.concatenate(([-np.inf], alphas))
    upper_bounds = np.concatenate((alphas, [np.inf]))
    
    # Calculate probabilities for each category
    probas = np.zeros((X.shape[0], K))
    for k in range(K):
        probas[:, k] = norm.cdf(upper_bounds[k]) - norm.cdf(lower_bounds[k])
    
    return probas

def predict(params, X, K):
    probas = predict_probas(params, X, K)
    return np.argmax(probas, axis=1)

# Set up parameters
K = 7  # 7 wine quality categories (I use the wine quality dataset)
n_features = X_train.shape[1]  # 11 features

# Initialize parameters
# We need: n_features (beta) + 1 (intercept) + (K-1) (thresholds) = 11 + 1 + 6 = 18 parameters
initial_params = np.zeros(n_features + 1 + (K-1))

# Run optimization
result = minimize(ordinal_probit_nll, initial_params, args=(X_train, y_train, K), method='BFGS')

# Make predictions
probas = predict_probas(result.x, X_train, K)
predicted_classes = predict(result.x, X_test, K)
```