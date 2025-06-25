---
title: "Gradient Boosting — 1"
date: "2025-04-17"
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


Gradient boosting is an ensemble technique that iteratively builds an additive model of simpler trees (weak learners). At each iteration, a tree is trained to predict the gradient of the loss function with respect to the predictions of the existing model.

## Introduction

Given data points $\left(x_i, y_i\right)$, our objective is to minimize a loss function $L$ over our dataset using a predictive model, an ensemble of decision trees $F$. 
The optimization problem can be expressed as finding the function $F$ that minimizes the following loss function:

$$
\text { Loss }=\sum_{i=1}^n L\left(y_i, F\left(x_i\right)\right)
$$

The function $F$ represents our ensemble model, which is constructed as the sum of $T$ individual decision trees. This can be expressed mathematically as:

$$
F(x)=f_0(x)+f_1(x)+f_2(x)+\cdots+f_T(x)
$$

where $f_0(x)$ is initialized as the constant minimizing the loss function.

$$
f_0(x)=\underset{\gamma}{\arg \min } \sum_{i=1}^n L\left(y_i, \gamma\right)
$$

$F$ will be constructed iteratively starting from $f_0$.
At round $t$, the model $F_t$ is the sum of all previous trees plus the current tree $f_t$:

$$
F_t(x)=f_0(x)+f_1(x)+\cdots+f_{t-1}(x) + f_{t}(x) = F_{t-1}(x)+f_t(x)
$$

Understanding gradient boosting results in understanding how to search for  $f_t$.

## Iterative Improvement

At iteration $t$, the objective is the following:
$$\mathrm{Obj}^{(t)}=\sum_{i=1}^n L\left(y_i, F_{t-1}\left(x_i\right)+f_t\left(x_i\right)\right)$$

This function is too complex to optimize directly.
Our blueprint to find $f_t$ is simple.
Find a simple surrogate function of $\mathrm{Obj}^{(t)}$ and optimize it. 
We use the Taylor Expansion up to the second order to find our surrogate.
Around the point $F_{t-1}\left(x_i\right)$, an approximation of $L(y, \hat{y})$ is:
$$
L\left(y_i, F_{t-1}\left(x_i\right)+f_t\left(x_i\right)\right) \approx L\left(y_i, F_{t-1}\left(x_i\right)\right)+f_t\left(x_i\right) \cdot g_i+\frac{1}{2} f_t\left(x_i\right)^2 \cdot h_i
$$

where:

$$g_i = \left.\frac{\partial L\left(y_i, \hat{y}\right)}{\partial \hat{y}}\right|_{\hat{y}=F_{t-1}\left(x_i\right)}, \quad h_i = \left.\frac{\partial^2 L\left(y_i, \hat{y}\right)}{\partial \hat{y}^2}\right|_{\hat{y}=F_{t-1}\left(x_i\right)}$$

Substituting the Taylor expansion back into our objective, we obtain a simplified quadratic objective:

$$
\mathrm{Obj}^{(t)} \approx \sum_{i=1}^n\left(L\left(y_i, F_{t-1}\left(x_i\right)\right)+g_i f_t\left(x_i\right)+\frac{1}{2} h_i f_t\left(x_i\right)^2\right)
$$

Since the first term $L\left(y_i, F_{t-1}\left(x_i\right)\right)$ is constant w.r.t. $f_t\left(x_i\right)$, we ignore it. 
We are interested in this objective:

$$
\sum_{i=1}^n\left(g_i f_t\left(x_i\right)+\frac{1}{2} h_i f_t\left(x_i\right)^2\right)
$$

Recall that a decision tree model learns partitions of the input data into distinct regions (leaf nodes).
For each of the partitions $j$, the decision tree assignes a value $f_j$.
This observation helps us updating our objective into:

$$
\sum_{j=1}^J \sum_{i \in R_j}\left(g_i f_j+\frac{1}{2} h_i f_j^2\right)
$$

with $R_j$ the set of sample $i$ that fall into leaf $j$.

At this point, it may be useful to state what are objects we want to optimize.
We are looking for the prediction value per leaf $f_j$ and the best partition of the input space $R_j$.
Both types of variable will requires a different approaches.
$$
f_j \text{ for } j \in \big\{ 1, ... J \big\} \quad\text{and}\quad R_j \text{ for } j \in \big\{ 1, ... J \big\} 
$$

**Optimal Leaf Predictions.** How to find the optimal prediction $f_j^*$ in each leaf $j$.
We differentiate the expression with regard to each prediction in leaf $j$ (ie, $f_j$) :

$$
\frac{\partial}{\partial f_j} \sum_{i \in R_j}\left(g_i f_j+\frac{1}{2} h_i f_j^2\right)=0
$$

Solving this equation logically yields:

$$
f_j^*=-\frac{\sum_{i \in R_j} g_i}{\sum_{i \in R_j} h_i}
$$

which states that the optimal prediction per leaf is simply a function of gradient and hessian of the samples 
that falls in the leaf $j$.

**Optimal Splits.** How to best partition the input space into regions.

Here, we will use a brute force approach. 
We will compare the objective before a split compared to after.
The quality of each split is evaluated by its gain in reducing the objective function:

$$
\text{Gain} = \mathrm{Obj}_{\text{before split}} - \mathrm{Obj}_{\text{after split}}
$$

The current region before split is called the parent and after the split it is transformed in child nodes (left and right).
Recall that our objective (without the constant term) is: 

$$\mathrm{Obj}_{\text{parent}} = \sum_{i \in \text{parent}} \left(g_i f_{\text{parent}} + \frac{1}{2} h_i f_{\text{parent}}^2\right)$$

with optimal prediction in parent node as previously derived:

$$
f_{\text {parent }}^*=-\frac{\sum_{i \in \text { parent }} g_i}{\sum_{i \in \text { parent }} h_i+}
$$

After the split, the objective is:

$$
\mathrm{Obj}_{\text {after split }} = \mathrm{Obj}_{\text {left }}+\mathrm{Obj}_{\text {right }} =-\frac{1}{2} \frac{\left(\sum_{i \in \text { left }} g_i\right)^2}{\sum_{i \in \text { left }} h_i}-\frac{1}{2} \frac{\left(\sum_{i \in \text { right }} g_i\right)^2}{\sum_{i \in \text { right }} h_i}
$$

Finally, we have an expression of the gain:

$$
\boxed{
\text{Gain} = \frac{1}{2}\left[\frac{\left(\sum_{i \in \text{left}} g_i\right)^2}{\sum_{i \in \text{left}} h_i+\lambda}+\frac{\left(\sum_{i \in \text{right}} g_i\right)^2}{\sum_{i \in \text{right}} h_i+\lambda}-\frac{\left(\sum_{i \in \text{parent}} g_i\right)^2}{\sum_{i \in \text{parent}} h_i+\lambda}\right]
}
$$

To find the best split, we want to maximize this gain:

$$
\text { Optimal Split }=\arg \underset{\text { all candidate splits }}{\max } \text { Gain }
$$

A candidate split is defined by a feature and a threshold (for numerical features) or a partition (for categorical features).
A simple solution could be to simply do a brute force approach. 
In practice, other approaches exists to improve the search.

