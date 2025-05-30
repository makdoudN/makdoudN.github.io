---
title: "Ordinal Regression in LightGBM — 2"
date: "2025-04-16"
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

In the previous post, we derived the gradient and hessian for ordinal regression. Now, let's implement these in LightGBM and see how to use them in practice.

Let's first write some basic snippet for $g$ and $h$. 
Then we will lay down the base of the Gradient Boosting Framework. 
And finally, we will apply (Simple) Ordinal Regression using LightGBM.
By the way, I want to emphasize that it may not be the last post of this project.
Ordinal Regression kind of requires from us to optimize thresholds and the mapping between observation and the latent score.
In this post, we will suppose that we know the threshold and leverage the big beast LightGBM for the mapping.

### Preliminary and Notation

Based on last post, let's just remind the core objective and variables:

First, let's start with some notations:

$$
D=\Phi_{y_i}-\Phi_{y_i-1}, \quad N = \phi_{y_i-1}-\phi_{y_i}, \quad \Phi_k=\Phi\left(u_k\right), \quad u_k=\theta_k-f_i.
$$

and 

$$
N^{\prime}=\left(\theta_{k-1}-f_i\right) \phi\left(\theta_{k-1}-f_i\right)-\left(\theta_k-f_i\right) \phi\left(\theta_k-f_i\right)
$$

Our first object is:

$$
\boxed{g_i = \frac{\partial \ell_i}{\partial f_i}  =\frac{N}{D}=\frac{\phi_{y_i-1}-\phi_{y_i}}{\Phi_{y_i}-\Phi_{y_i-1}} }
$$

Our second needed object is:

$$
\boxed{h_i = \frac{\partial^2 \ell}{\partial f_i^2}=g^{\prime}\left(f_i\right)=\frac{N^{\prime} D-N D^{\prime}}{D^2}}
$$

### Implementation

We will start with synthetic data to test our approach:

```python
import numpy as np
from scipy.stats import norm

np.random.seed(42)

n = 100      # number of samples
p = 2        # feature dimension
K = 4        # number of classes

# Random Observations.
X = np.random.randn(n, p)

# True latent score
w_true = np.array([1.5, -2.0])
f = X.dot(w_true) + 0.5 * np.random.randn(n)

# True Thresholds
theta = np.array([-np.inf, -1.0, 0.0, 1.0, np.inf])

y = np.digitize(f, bins=[-1.0, 0.0, 1.0]) + 1    # Start at 1.
```

The following code computes the gradient. 
It is nearly identical to the formula.

```python
u_k   = theta[y] - f
u_km1 = theta[y-1] - f

Phi_k   = norm.cdf(u_k)
Phi_km1 = norm.cdf(u_km1)

phi_k   = norm.pdf(u_k)
phi_km1 = norm.pdf(u_km1)

N = phi_km1 - phi_k          
D = Phi_k   - Phi_km1        

D_safe = np.clip(D, 1e-8, None)

g = N / D_safe
```

I clip `D_safe` to avoid division per 0. 

For $h$, it is also nearly identical with some exceptions:

```python
# For probit: φ'(u) = -u * φ(u) and ϕ′(u)=0 where u = infinite
# Suppress the “invalid value” warning just for this computation
with np.errstate(invalid='ignore'):
    phi_prime_k    = -u_k * phi_k
    phi_prime_k[~np.isfinite(u_k)] = 0.0

    phi_prime_km1  = -u_km1 * phi_km1
    phi_prime_km1[~np.isfinite(u_km1)] = 0.0

A = phi_prime_km1 - phi_prime_k


h = (A * D_safe - N**2) / (D_safe**2)
```

We ignore the warning due to $u$ being infinite in some case. 
But as we deal with those case due to the derivative being 0, this warning can be ignore.

At this point, we have our two objects $g$ and $h$ requires by LightGBM to fit an ordinal objective.

I was also interested in converted the latent score $f$ into a probability per class (which is suprinsingly simple to code leveraging broadcasting rules).

```python
#   f[:, None] has shape (n, 1)
#   theta[None, :] has shape (1, K+1)
U = theta[None, :] - f[:, None]   # shape (n, K+1)
C = norm.cdf(U)                   # shape (n, K+1)
P = C[:, 1:] - C[:, :-1]          # shape (n, K)
```

This code is nearly all we have to do to perform ordinal regression with LightGBM. 
So, what does it lack ?
So small measures to avoid numerical issue like Underflow or overflow in the CDF/PDF. 
Potentially, clip high absolute value in the gradient or hessian. 

Adding a bit more structure and we can use this custom loss for lightgbm 4.6

```python

from sklearn.metrics import cohen_kappa_score


class OrdinalProbitLoss:
    """
    LightGBM objective for ordinal regression with a probit link.
    We fix thresholds theta (excluding ±∞), and compute gradient & Hessian
    per-sample in O(n) time & memory.
    """
    def __init__(self, theta, eps=1e-16):
        # theta_user: shape (K-1,) e.g. [-1.0, 0.0, 1.0]
        # we internally pad with [-inf, ..., +inf]
        self.theta = np.concatenate(([-np.inf], np.array(theta), [np.inf]))
        self.eps = eps

    def _grad_hess(self, f, y):
        # f: shape (n,), latent predictions
        # y: shape (n,), integer classes in {1..K}

        u_k   = self.theta[y]   - f    # θ_{y_i}   - f_i
        u_km1 = self.theta[y-1] - f    # θ_{y_i-1} - f_i

        Phi_k    = norm.cdf(u_k)
        Phi_km1  = norm.cdf(u_km1)
        phi_k    = norm.pdf(u_k)
        phi_km1  = norm.pdf(u_km1)

        N  = phi_km1 - phi_k          # = φ(u_{k-1}) - φ(u_k)
        D  = Phi_k   - Phi_km1        # = Φ(u_k) - Φ(u_{k-1})
        
        # For probit: φ'(u) = -u * φ(u) and ϕ′(u)=0 where u = infinite
        # Suppress the “invalid value” warning just for this computation
        with np.errstate(invalid='ignore'):
            phi_prime_k    = -u_k * phi_k
            phi_prime_k[~np.isfinite(u_k)] = 0.0

            phi_prime_km1  = -u_km1 * phi_km1
            phi_prime_km1[~np.isfinite(u_km1)] = 0.0

        A = phi_prime_km1 - phi_prime_k

        # Stabilize & compute gradient of -logL and Hessian
        D_safe = np.clip(D, self.eps, None)
        grad = - N  / D_safe
        hess = -(A * D_safe - N**2) / (D_safe**2)
        return grad, hess

    def lgb_obj(self, preds, dataset):
        y = dataset.get_label().astype(int)
        return self._grad_hess(preds, y)

        
def predict_proba(f, theta):
    """
    Compute class probabilities for each sample.
    
    Parameters
    ----------
    f : array-like of shape (n_samples,)
        The latent predictions.
        
    Returns
    -------
    P : array-like of shape (n_samples, n_classes)
        The probability of each class for each sample.
    """
    # Compute U = θ - f for all thresholds
    U = theta[None, :] - f[:, None]       # shape (n, K+1)
    
    # Compute CDF values
    C = norm.cdf(U)                       # shape (n, K+1)
    
    # Compute probabilities as differences of CDFs
    P = C[:, 1:] - C[:, :-1]              # shape (n, K)
    
    return P


dtrain = lgb.Dataset(X, label=y)
loss = OrdinalProbitLoss(theta)

params = {
    'learning_rate': 0.1,
    'num_leaves':    7,
    'verbose':      -1,
    'objective': loss.lgb_obj,
}

bst = lgb.train(
    params,
    dtrain,
)

f = bst.predict(X)

y_proba_pred = predict_proba(f, theta)
y_pred = np.argmax(y_proba_pred, axis=1) + 1

kappa = cohen_kappa_score(y, y_pred, weights='quadratic')
print(f"Quadratic Cohen's Kappa: {kappa:.3f}")
```

## Conclusion

In this post, we've explored how to implement ordinal regression using gradient boosting with a probit link function. 
