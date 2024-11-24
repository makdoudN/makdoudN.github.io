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

In this case, we have fixed the cutpoints points. 
It means that it assumes that you know the cutpoints points. 
Which is unlikely in practice. 
Still, it is a good use-case to understand the model and if you do not really care about the true inference and only want prediction, it may be enough. 
Well, if prediction is all you care about, you probably do not need Bayesian statistics. 

Now, if you care about inference, you probably want to infer the cutpoints points from the data.
Assuming that the data generative process of ordinal regression, you should want to infer both the affinity parameters $\beta$ and the cutpoints points $\alpha$. 

The main problem is that the cutpoints points are not identifiable.  
You may sense it but with this data  generative story, there is a lot a random variable. 
Maintaining identifiability is a tricky task. 
These cut points are constrained by ordering ( $c_1<c_2<\cdots<c_{K-1}$ ), making regularization tricky.
Naïvely defining priors on the latent cut points without considering their ordering can lead to non-identifiable or poorly-behaved posterior distributions.
Additionally, domain expertise is challenging to incorporate directly in this latent space because it's abstract.
As it is a common routine in Bayesian Inference, Identifiability is key for proper inference. 
Commonly, we will need to use strong priors to identify the parameters.  


---
### Full Bayesian Approach to Ordinal Regression

I will strongly used the article from [Michael Betancourt](https://betanalpha.github.io/assets/case_studies/ordinal_regression.html) to derive the model.

Let be clear, I am not an expert in Bayesian Inference. 
But I will try to do my best to understand the model. 

Let's restart from our hypothesis of data generative process.
Suppose the latent variable $z$ determines the category $y \in\{1, \ldots, K\}$.
The cut points $\left(c_1, c_2, \ldots, c_{K-1}\right)$ partition the real line:

$$y = k \quad \text{if} \quad c_{k-1} < z \leq c_k$$

where $c_0=-\infty$ and $c_K=+\infty$

Our objective is to derive the probability of a category given the features $X$ and the parameters $\beta$ and $c$.
The probability of $y=k$ is expressed in terms of the latent variable and cut points:

$$ p_k = P(c_{k-1} < z \leq c_k) $$

The ordinal probabilities $\left(p_1, p_2, \ldots, p_K\right)$ are constrained:

$$
\sum_{k=1}^K p_k=1, \quad p_k \geq 0
$$

This places $\left(p_1, \ldots, p_K\right)$ on a $(K-1)$-dimensional simplex.
Given probabilities $\left(p_1, \ldots, p_K\right)$, we can compute the cut points through the cumulative probabilities:

$$
P(y \leq k)=\sum_{j=1}^k p_j
$$

Let $q_k=P(y \leq k)$ (the cumulative probabilities), then:

$$
q_k=p_1+\cdots+p_k, \quad q_0=0, \quad q_K=1 .
$$

The mapping of cumulative probabilities $q_k$ to cut points $c_k$ through a general inverse link function $g^{-1}$ is a critical step in ordinal regression. Let's delve deeply into the mechanics of this process.
The link function $g$ and its inverse $g^{-1}$ serve as the bridge between two spaces:
- Cumulative Probabilities: $q_k \in[0,1]$ lie in a bounded, probabilistic space.
- Latent Continuous Scale: $c_k \in \mathbb{R}$, which partitions the real line into intervals corresponding to ordinal categories.

The inverse link function $g^{-1}$ transforms the bounded cumulative probabilities $q_k$ into cut points $c_k$, which live on the real line while preserving their natural ordering.


$$
c_k=g^{-1}\left(q_k\right), \quad k=1, \ldots, K-1
$$


The link function $g$ defines the relationship between the latent continuous scale (cut points) and the cumulative probabilities. 
This function is monotonic, ensuring that the order of $q_k$ is preserved in the order of $c_k$.

In order to realize the link between cut points and the cumulative probabilities $q_k$, the link function needs to adhere some properties

- If $q_k \leq q_{k+1}$, then $g^{-1}\left(q_k\right) \leq g^{-1}\left(q_{k+1}\right)$. This means:

$$
c_1=g^{-1}\left(q_1\right)<c_2=g^{-1}\left(q_2\right)<\cdots<c_{K-1}=g^{-1}\left(q_{K-1}\right) .
$$

Monotonicity is a key property for ensuring that the cut points partition the real line correctly.

The cumulative probabilities $q_k$ are constrained to $[0,1]$, which makes it difficult to directly define a model for the cut points on the unbounded real line. $g^{-1}$ maps these probabilities into the real line $\mathbb{R}$ while retaining the ordering.

The choice of $g$ depends on the specific needs of the model, as it defines how probabilities relate to the latent scale. For example:
- A logistic link $g(q)=\sigma(c)$ reflects the standard logistic distribution.
- A probit link $g(q)=\Phi(c)$ reflects the standard normal distribution.

**Regularization Induced by the Dirichlet Prior**. When $q_k$ is derived from a Dirichlet prior on $\left(p_1, \ldots, p_K\right)$, the mapping via $g^{-1}$ :
1. Regularizes the spacing of the cut points $c_k$ implicitly.
2. Ensures that the cut points respect the ordering constraint.

What is interesting is that the prior of $\mathbf{p}$ impacts indirectly the cut points.

```
    c_k → p_k ← Dirichlet(α) → y
```

The Dirichlet prior regularizes $p_k$, and this regularization propagates backward to constrain $c_k$

So to summarise the new data generative process is the following:

### Observational Model

**Observational Data**. The observed data consists of ordinal outcomes $y_i \in\{1,2, \ldots, K\}$, where $y_i$ represents a category assigned based on an underlying latent process.

**Observation Model**. The following likelihood for the observations:

$$
P\left(y_i=k \mid z_i, \mathbf{c}\right)= \begin{cases}P\left( z_i \leq c_1\right), & \text { if } k=1 \\ P(c_{k-1} < z_i \leq c_k), & \text { if } 2 \leq k \leq K-1 \\ P\left(z_i>c_{K-1}\right), & \text { if } k=K\ \end{cases}
$$

where:
- $z_i$ is the latent variable for observation $i$.
- $\mathbf{c}=\left(c_1, \ldots, c_{K-1}\right)$ are the latent cut points dividing the real line into intervals corresponding to the $K$ ordinal categories.

The likelihood for $y_i$ can be expressed more compactly as:

$$
P\left(y_i=k \mid z_i, \mathbf{c}\right)= \begin{cases}\Phi\left(c_1-z_i\right), & \text { if } k=1 \\ \Phi\left(c_k-z_i\right)-\Phi\left(c_{k-1}-z_i\right), & \text { if } 2 \leq k \leq K-1 \\ 1-\Phi\left(c_{K-1}-z_i\right), & \text { if } k=K\end{cases}
$$

where $\Phi$ is the CDF of the noise distribution (e.g., standard normal or logistic).

### **Latent Structure**

#### Affinity Structure $\beta$

The regression coefficients $\boldsymbol{\beta}$ (the relationship between covariates $\mathbf{x}_i$ and the latent variable $z_i$ ) are modeled with a Gaussian prior:

$$
\boldsymbol{\beta} \sim \mathcal{N}\left(\boldsymbol{\mu}_\beta, \Sigma_\beta\right),
$$

where:
- $\boldsymbol{\mu}_\beta$ is the mean vector (prior belief about $\boldsymbol{\beta}$ ).
- $\Sigma_\beta$ is the covariance matrix (prior uncertainty in $\boldsymbol{\beta}$ ).


The latent variable $z_i$ is modeled as:

$$
z_i \sim \mathcal{N}\left(f\left(\mathbf{x}_i ; \boldsymbol{\beta}\right), \sigma_z^2\right),
$$

where:
- $f\left(\mathbf{x}_i ; \boldsymbol{\beta}\right)=\mathbf{x}_i^{\top} \boldsymbol{\beta}$ (linear predictor).
- $\sigma_z^2$ is the variance of $z_i$, capturing additional noise beyond the covariate effects.

Alternatively, we can express this as:

$$
z_i=\mathbf{x}_i^{\top} \boldsymbol{\beta}+\epsilon_i, \quad \epsilon_i \sim \mathcal{N}\left(0, \sigma_z^2\right) .
$$

#### Cut Points Modeling

The cut points $\mathbf{c}=\left(c_1, c_2, \ldots, c_{K-1}\right)$ define the boundaries between ordinal categories in the latent space.

They are implicitely regularized by imposing a prior over $p$

$$
\mathbf{p} \sim \operatorname{Dirichlet}(\boldsymbol{\alpha})
$$

where $\mathbf{p}$ represents the simplex-constrained category probabilities.

The link between $\mathbf{p}$ is the following:
- 1 **Cumulative Probabilities** are  $q_k=\sum_{j=1}^k p_j, \quad k=1, \ldots, K-1$
- 2 The link between **Cumulative Probabilities** and **Cut Points** is through the inverse link function $g^{-1}$ 

$$
c_k=g^{-1}\left(q_k\right), \quad k=1, \ldots, K-1
$$


### **Summary of the Generative Process**

1. Draw $\left(p_1, \ldots, p_K\right)$ from a Dirichlet prior to impose smoothness and simplex constraints.
2. Compute cumulative probabilities $q_k=\sum_{j=1}^k p_j$, ensuring monotonicity.
3. Map $q_k$ to cut points $c_k=g^{-1}\left(q_k\right)$ in the latent space, preserving ordering.
4. Generate latent variables $z_i$ based on a predictor function and noise.
5. Assign ordinal outcomes $y_i$ based on the intervals defined by $c_k$.

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
        numpyro.sample("y", OrderedProbit(predictor=z, cutpoints=cut_points), obs=y)

```