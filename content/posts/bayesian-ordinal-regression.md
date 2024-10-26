---
title: "Bayesian Ordinal Regression - WIP"
date: "2024-10-01"
summary: "This is a test summary"
description: "This is a test description"
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

. . .[Conclusion]

## Bayesian Binary Classification with a twist
---

**Data Generative Process —** There is a latent continuous variable which censored yield to the ordinal probabilities. Features (or Covariate) influenced the latent variable and as a result influences the final ordinal probabilities.

See Betaandalpha blog (TODO CITE)
> Let's start by given a probabilistic structure to the latent continuous variable. First, the ambiant space is $\mathbb{R}$ equipped with a probability density function $\pi(x)$. Then, we introduce cut points that partition the ambiant space of the latent variable $\left\{c_0, c_1, c_2\right\}$ where the first cut points extend to $c_0=-\infty$ and the last cut points $c_2=\infty$. The non-extreme points (here only $c_1$ ) controls the partition of the space $X$.

> Given this partition we can define the the ordinal probabilities as the complementary probability allocated to each interval or, more equivalently, the integral of the complementary density across each interval.

. . .  [Continue]



