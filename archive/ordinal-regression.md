---
title: "Bayesian Ordinal Regression"
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

Let's start by assuming we want to predict the binary variable $y$ from a number $N$ features $X \in \mathbb{R}^N$. 
A common approach in statistic is to use an inverse link function $f$ to (un)surprinsingly links the feature $X$ to the $y$. 
To be more precise, we will assume some probabilistic structure attached to the binary variable $y$ by letting $y \sim \operatorname{Ber}(y \mid p)$ and the inverse link function links the feature to the mean of the function $\mathbb{E}(y)=p=f(X ; p)$.

In the case of a linear binary logistic regression, 

$$ 
\mathbb{E}(y)= p =\frac{1}{1+\exp (-\eta)} \quad \text{with} \quad \eta = \sum X_i \omega_i + b
$$

Our objective is to infer $\omega$ that fits the data.
As an example, here a tiny code to infer the parameters using MLE approaches.

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
    eta = sigmoid(X @ theta)

    # Ensure numerical stability by clipping predictions
    eta = np.clip(eta, 1e-15, 1 - 1e-15)
    
    log_likelihood = y * np.log(eta) + (1 - y) * np.log(1 - eta)
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

In the next section, we will propose an alternative model for binary classification that is more suitable for ordinal data.
In practice, this model is never used for binary classification but is will be a simple use-case to understand the modeling process of ordinal regression.

---
### Bayesian Binary Classification with a twist

**Data Generative Process —** 
We assume that there is a latent continuous variable which censored yield to the ordinal probabilities. 
Features influenced the latent variable and as a result influences the final ordinal probabilities.
In contrast to previous approach, we introduce a latent variable $z$ and also a cutpoint $\alpha$ to model the ordinal probabilities.

**Threshold Mechanism**.

The observed ordinal outcome $y_i$ is determined by where our latent variable $z_i$ falls relative to the cutpoints $\alpha_k$ :

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

**Latent Variable Model** $\boldsymbol{z}$. 

The latent variable is the result of a linear combination of the features and the parameters.
The model will infer the parameters $\boldsymbol{\beta}$ and $\alpha$ from the data. 
We hope that the model will be able to identify the latent variable $z$ that will allow us to predict the ordinal outcome $y$.
Intuitively, for the example of the temperature, for a person, the latent variable $z$ will represent a continuous affinity from the coldest to the hottest temperature.

$$
 \text{Cutoff Points:} \quad z = \mathbf{x}_i^{\top} \boldsymbol{\beta} \quad \text{with}\quad \boldsymbol{\beta} \sim \mathcal{N}\left(\mathbf{0}, \sigma_\beta^2\right)
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

In this case, we have fixed the cutpoints points.
If you do not really care about the true inference and only want prediction, it may be enough. 
If prediction is all you care about, you probably do not need Bayesian Inference. 
Now, if you care about inference, you probably want to also infer the cutpoints points from the data.

Assuming that the data generative process of ordinal regression is a good approximation of the reality, you should want to infer both the posterior distribution over the  affinity parameters $\beta$ and the cutpoints points $\alpha$. 
The main problem is that the cutpoints points are not identifiable.
As it is a common routine in Bayesian Inference, Identifiability is key for proper inference. 
Commonly, we will need to use strong priors to remove non identifiable of our model and to identify the parameters.  


---
### Full Bayesian Approach to Ordinal Regression

I will strongly used the article from [Michael Betancourt](https://betanalpha.github.io/assets/case_studies/ordinal_regression.html) to derive the model.
Let be clear, I am not an expert in Bayesian Inference. 
But I will try to do my best to understand the model. 
Recall our model:

$$ 
p_k = P(c_{k-1} < z \leq c_k) \quad \text{with} \quad  \sum_{k=1}^K p_k=1, \quad p_k \geq 0
$$

This places $\left(p_1, \ldots, p_K\right)$ on a $(K-1)$-dimensional simplex.
Given probabilities $\left(p_1, \ldots, p_K\right)$, we can compute the cumulative probabilities over the affinity $z$.

$$
q_k =P\left(z \leq c_k\right)=F\left(c_k\right)
$$

As a result, the probability mass of an ordinal category is 

$$
p_k = q_{k} - q_{k-1}
$$

**Why do we care about cumulative distribution** $\boldsymbol{q_k}$ **?** Ordinal categories correspond to cumulative partitions of a latent continuous scale. 
This construction arises naturally from our specification of the probability of an ordinal category. 
As a result of the nature of the problem the mapping of cumulative probabilities $q_k$ to cut points $c_k$ through a general inverse link function $g^{-1}$ will be crucial to remove identification issues. 

The link function $g$ and its inverse $g^{-1}$ serve as the bridge between two spaces.
First, the Cumulative Probabilities $q_k \in[0,1]$ lie in a bounded, probabilistic space.
Second, the Latent Continuous Scale $c_k \in \mathbb{R}$, which partitions the real line into intervals corresponding to ordinal categories.
The inverse link function $g^{-1}$ transforms the bounded cumulative probabilities $q_k$ into cut points $c_k$, which live on the real line while preserving their natural ordering.

$$
c_k=g^{-1}\left(q_k\right), \quad k=1, \ldots, K-1
$$

The link function $g$ defines the relationship between the latent continuous scale (cut points) and the cumulative probabilities. 
This function is monotonic, ensuring that the order of $q_k$ is preserved in the order of $c_k$.
In order to realize the link between cut points and the cumulative probabilities $q_k$, the link function needs to adhere Monotonicity property.

First properties, if $q_k \leq q_{k+1}$, then $g^{-1}\left(q_k\right) \leq g^{-1}\left(q_{k+1}\right)$. 

$$
c_1=g^{-1} \left( q_1\right) < c_2 = g^{-1} \left( q_2\right)< \cdots < c_{K-1} = g^{-1} \left( q_{K-1}\right) .
$$

The cumulative probabilities $q_k$ are constrained to $[0,1]$, which makes it difficult to directly define a model for the cut points on the unbounded real line. $g^{-1}$ maps these probabilities into the real line $\mathbb{R}$ while retaining the ordering.

The choice of $g$ depends on the specific needs of the model, as it defines how probabilities relate to the latent scale. For example:
- A logistic link $g(q)=\sigma(c)$ reflects the standard logistic distribution.
- A probit link $g(q)=\Phi(c)$ reflects the standard normal distribution.

**Regularization Induced by the Dirichlet Prior**. When our prior over $p_k$ comes from a Dirichlet prior, the mapping via $g^{-1}$ :
1. Regularizes the spacing of the cut points $c_k$ implicitly.
2. Ensures that the cut points respect the ordering constraint.

What is interesting is that the prior of $\mathbf{p}$ impacts indirectly the cut points.

```
    c_k → q_k → p_k ←  Dirichlet(α)   p_k → y
```

The Dirichlet prior regularizes $p_k$, and this regularization propagates backward to constrain $c_k$

### An example of code using `numpyro`

```python
import jax
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import norm
from numpyro.infer import MCMC, NUTS
from numpyro.distributions import constraints, CategoricalProbs
from numpyro.distributions.util import promote_shapes

class OrderedProbit(CategoricalProbs):
    """
    A categorical distribution with ordered outcomes, using a Probit link.

    :param numpy.ndarray predictor: predictions in the real domain; typically the output
        of a linear model.
    :param numpy.ndarray cutpoints: positions in the real domain to separate categories.
    """

    arg_constraints = {
        "predictor": constraints.real,
        "cutpoints": constraints.ordered_vector,
    }

    def __init__(self, predictor, cutpoints, *, validate_args=None):
        if jnp.ndim(predictor) == 0:
            (predictor,) = promote_shapes(predictor, shape=(1,))
        else:
            predictor = predictor[..., None]
        predictor, cutpoints = promote_shapes(predictor, cutpoints)
        self.predictor = predictor[..., 0]
        self.cutpoints = cutpoints

        # Compute cumulative probabilities using the probit link (normal CDF)
        cdf = norm.cdf
        probs = jnp.concatenate([
            cdf(self.cutpoints[..., 0] - self.predictor[..., None]),
            cdf(self.cutpoints[..., 1:] - self.predictor[..., None]) -
            cdf(self.cutpoints[..., :-1] - self.predictor[..., None]),
            1.0 - cdf(self.cutpoints[..., -1] - self.predictor[..., None])
        ], axis=-1)

        super(OrderedProbit, self).__init__(probs, validate_args=validate_args)

    @staticmethod
    def infer_shapes(predictor, cutpoints):
        batch_shape = jnp.broadcast_shapes(predictor.shape, cutpoints[:-1].shape)
        event_shape = ()
        return batch_shape, event_shape

    def entropy(self):
        raise NotImplementedError


def ordinal_regression_model(X, y=None, n_categories=4):
    n_features = X.shape[1]
    
    # Priors for regression coefficients
    beta = numpyro.sample("beta", 
        dist.Normal(jnp.zeros(n_features), jnp.ones(n_features))
    )
    
    # Dirichlet prior for category probabilities
    alpha = jnp.ones(n_categories)
    p = numpyro.sample("p", dist.Dirichlet(alpha))
    
    # Cumulative probabilities derived from p
    q = jnp.cumsum(p[:-1])
    
    # Cut points derived using probit link (inverse CDF of normal distribution)
    cut_points = numpyro.deterministic("cut_points", dist.Normal(0, 1).icdf(q))
    
    # Linear predictor for latent variable z
    z = jnp.dot(X, beta)
    
    # Observational model
    with numpyro.plate("data", X.shape[0]):
        numpyro.sample("y", OrderedProbit(predictor=z, cutpoints=cut_points), obs=y)****

```