---
title: "Ordinal Regression in LightGBM — 1"
date: "2025-04-14"
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

In this post, we'll derive the gradient and hessian of the log likelihood function for an Ordinal Regression Model. 
These derivatives are essential components for implementing ordinal regression within gradient boosting frameworks like LightGBM, as they guide the optimization process during model training.

### Ordinal Regression Model

We observe data $(x_i, y_i)_{i=1}^n$ with $x_i \in \mathbb{R}^p$ and ordinal responses $y_i \in\{1,2, \ldots, K\}$.
In ordinal regression, unlike classical multiclass classification, the target variable ($y_i$) represents ordered categories, where the relationship between categories matters.
To enforce the order, we introduce  cut‐points (thresholds) and a latent score $\eta \in \mathbb{R}$.
$$
-\infty=\theta_0<\theta_1<\theta_2<\cdots<\theta_{K-1}<\theta_K=+\infty
$$
The position of $\eta$ relative to the cut-points determines the probability distribution over the ordinal response categories.
$$
y = \begin{cases}1 & \text { if } \eta \leq \theta_1 \\ 2 & \text { if } \theta_1 < \eta \leq \theta_2 \\ \vdots & \\ k & \text { if } \theta_{k-1} < \eta \leq \theta_k \\ \vdots & \\ K & \text { if } \eta > \theta_{K-1} \end{cases}
$$
We will assume a mapping from feature (x_i) to the latent score $\eta$ given by a tree ensemble:
$$
\eta_i = f\left(x_i\right) + \epsilon_i \quad \text{where} \quad \epsilon_i \overset{\text{iid}}{\sim} \mathcal{N}(0, 1)
$$
The probability of an ordinal class is simply the probability that the latent score $\eta$ is between the right cut-points:
$$
\Pr(y_i=k \mid x_i) = P(\theta_{k-1} < \eta \leq \theta_k)
$$

Since $\eta_i$ follows a normal distribution with mean $f(x_i)$ and variance 1, we can rewrite the probability using the standard normal CDF $\Phi$:

$$
\begin{aligned}
\Pr(y_i=k \mid x_i) &= P(\theta_{k-1} < \eta \leq \theta_k) \\
&= P(\theta_{k-1} < f(x_i) + \epsilon_i \leq \theta_k) \\
&= P(\theta_{k-1} - f(x_i) < \epsilon_i \leq \theta_k - f(x_i)) \\
&= \Phi(\theta_k - f(x_i)) - \Phi(\theta_{k-1} - f(x_i))
\end{aligned}
$$

This formulation expresses the probability of observing class $k$ as the difference between two cumulative normal probabilities.

The log likelihood for our ordinal regression model is given by:

$$
\begin{aligned}
\ell(f, \theta) &= \sum_{i=1}^n \log \Pr(y_i \mid x_i) \\
&= \sum_{i=1}^n \log \left[ \Phi(\theta_{y_i} - f(x_i)) - \Phi(\theta_{y_i-1} - f(x_i)) \right]
\end{aligned}
$$

To learn the mapping from features to latent scores using gradient boosting, we need to compute the gradient and hessian of the log likelihood with respect to the function values $f(x_i)$. These derivatives will guide the gradient boosting algorithm in updating the model.

### Preliminary and Notations

To ease the derivation, finding the right intermediate variable is important. 
It makes the derivation digests. 
I will defer some parts on the appendix to make the derivation as clean as possible.

So, we will the following variables: 

$$
D=\Phi_{y_i}-\Phi_{y_i-1}, \quad N = \phi_{y_i-1}-\phi_{y_i}, \quad \Phi_k=\Phi\left(u_k\right), \quad u_k=\theta_k-f_i.
$$

As a result, the log likelihood for a single observation $i$ is:
$$
\ell_i=\log \left[\Phi\left(\theta_{y_i}-f_i\right)-\Phi\left(\theta_{y_i-1}-f_i\right)\right]=\log D
$$

### Derivation of $g_i = \frac{\partial \ell_i}{\partial f_i}$

The first object needed to use gradient boosting with regard to the log likehihood of an ordinal regression problem is the derivative of the loss (the log likelihood) with regard to the mapping between the feature and the latent score $\eta$. 

$$
g_i = \frac{\partial \ell_i}{\partial f_i} = \frac{\partial \log D}{\partial f_i} = \frac{1}{D} \frac{\partial D}{\partial f_i}
$$

We will defer the derivation of $\frac{\partial D}{\partial f_i}$ in the appendix. 
We can show that 

$$
g_i =\frac{N}{D}=\frac{\phi_{y_i-1}-\phi_{y_i}}{\Phi_{y_i}-\Phi_{y_i-1}},
$$


### Derivation of $h_i = \frac{\partial^2 \ell}{\partial f_i^2}=g^{\prime}\left(f_i\right)$

Based on previous computation of the $g_i$, $h_i$ can be compute via the derivative of the quotient:

$$
\frac{\partial^2 \ell}{\partial f_i^2}=g^{\prime}\left(f_i\right)=\frac{N^{\prime} D-N D^{\prime}}{D^2}
$$

I will provide the derivation of $N'$ in appendix, we already know that $D' = N$.

$$
\frac{\partial^2 \ell}{\partial f_i^2}=g^{\prime}\left(f_i\right)=\frac{N^{\prime} D-N^2}{D^2}
$$

Now we $N, N', D$ we have all the necessary tools to code $g_i$ and $h_i$ and use ordinal regression with a gradient boosting framework like lightgbm.


# Appendix

### The Normal Cumulative Distribution Function (CDF)

The Normal Cumulative Distribution Function (CDF) is a cornerstone in probability theory and statistics. For a normally distributed random variable $X$ with mean $\mu$ and standard deviation $\sigma$, denoted $X \sim \mathcal{N}(\mu, \sigma^2)$, its CDF, $F_X(x)$ or $\Phi\left(\frac{x-\mu}{\sigma}\right)$, gives the probability that $X$ will take a value less than or equal to $x$.

The probability density function (PDF) of a standard normal distribution (where $\mu=0$ and $\sigma=1$) is given by:
$$
\phi(z) = \frac{1}{\sqrt{2\pi}} e^{-\frac{z^2}{2}}
$$

The derivative of $\Phi(x)$ is $\phi(x)$:

$$
\frac{d}{dx}\Phi(x) = \phi(x)
$$

For the hessian, we need the derivative of $\phi(x)$, which is:

$$
\phi'(x) = -x \cdot \phi(x)
$$

This relationship between $\Phi$ and $\phi$ allows us to compute the necessary derivatives for gradient boosting in ordinal regression.


### Derivation of $D' = \frac{\partial D}{\partial f_i}$

Recall

$$
D=\Phi_{y_i}-\Phi_{y_i-1}
$$

Since each $\Phi\left(u_k\right)$ depends on $f_i$ only via $u_k=\theta_k-f_i$ and $\frac{d u_k}{d f_i}=-1$ we have:

$$
\begin{aligned} 
\frac{\partial \Phi_{y_i}}{\partial f_i} & =\phi\left(u_{y_i}\right) \frac{d u_{y_i}}{d f_i}=\phi_{y_i}(-1)=-\phi_{y_i}, \\[1.5em]
\frac{\partial \Phi_{y_i-1}}{\partial f_i} & =\phi\left(u_{y_i-1}\right) \frac{d u_{y_i-1}}{d f_i}=\phi_{y_i-1}(-1)=-\phi_{y_i-1} .
\end{aligned}
$$

Therefore

$$
D^{\prime}=\frac{\partial D}{\partial f_i}=\left(-\phi_{y_i}\right)-\left(-\phi_{y_i-1}\right)=\phi_{y_i-1}-\phi_{y_i} = N.
$$

### Derivation of $N' = \frac{\partial N}{\partial f_i}$
Recall both facts: 
$$ N=\phi_{y_i-1}-\phi_{y_i} \quad \text{and} \quad \phi^{\prime}(x)=-x \cdot \phi(x)$$

Applying the chain rule, we have 

$$
\frac{\partial}{\partial f_i} \phi(u)=\phi^{\prime}(u) \frac{\partial u}{\partial f_i}=\phi^{\prime}(u)(-1)=-\phi^{\prime}(u)
$$

As a result, we have:

$$
N_i^{\prime} =\frac{\partial}{\partial f_i}\left[\phi\left(u_{k-1}\right)-\phi\left(u_k\right)\right]=-\phi^{\prime}\left(u_{k-1}\right)-\left(-\phi^{\prime}\left(u_k\right)\right)  =\phi^{\prime}\left(u_k\right)-\phi^{\prime}\left(u_{k-1}\right) 
$$

Substituting back $u_{k-1}=\theta_{k-1}-f_i, u_k=\theta_k-f_i$, we get the fully explicit form

$$
N_i^{\prime}=\left(\theta_{k-1}-f_i\right) \phi\left(\theta_{k-1}-f_i\right)-\left(\theta_k-f_i\right) \phi\left(\theta_k-f_i\right)
$$

