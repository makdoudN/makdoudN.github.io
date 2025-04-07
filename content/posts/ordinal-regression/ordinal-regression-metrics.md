---
title: "How to evaluate Ordinal Regression"
date: "2025-04-01"
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

****
When working with ordinal regression models, a critical question often emerges: how do we properly evaluate their performance? Unlike classification or regression tasks, ordinal regression presents unique evaluation challenges that are frequently overlooked in practice.

Ordinal regression occupies a middle ground between classification and regression. 
The ordered nature of the categories (e.g., "poor" < "fair" < "good" < "excellent") means that not all misclassifications are equal. Consider a movie rating prediction:

- Predicting a 4-star movie as 5-star (off by 1) is less severe than
- Predicting a 1-star movie as 5-star (off by 4)

Standard classification metrics like accuracy treat all errors equally, while regression metrics might impose inappropriate assumptions about the distances between ordinal levels. This fundamental mismatch creates a significant evaluation challenge.

Despite these challenges, proper evaluation of ordinal regression models is crucial for several reasons:

1. **Decision-making impact**: In many domains (healthcare severity assessment, credit risk scoring, customer satisfaction), the ordered nature of outcomes directly impacts decision-making.

2. **Model selection**: Without appropriate metrics, we risk selecting models that perform well on conventional metrics but poorly on the ordinal structure of the data.

3. **Interpretability**: Stakeholders need to understand model performance in terms relevant to the ordinal nature of the problem.

So how should we approach this evaluation challenge? What metrics properly account for the ordinal structure while providing meaningful performance insights? The standard toolkit of classification and regression metrics falls short, requiring us to consider specialized approaches that respect the unique characteristics of ordinal data.

I will talk from first principle about **Quadratic Weighted Kappa Score** (or QWK) which appears to be a simple yet effective metric to rate ordinal classifications. 


## A Detour by Cohen's Kappa Score measures

The **Cohen's Kappa Score measures** the level of agreement between two raters (or classifiers) who are assigning categorical labels to a set of items, while accounting for the agreement that would be expected by chance.

Let's say we have:
- A set of $n$ items.
- Two raters (or classifiers) assigning each item to one of $k$ possible categories.

and let's define:
- $O=$ observed agreement $=$ proportion of cases where the two raters agree.
- $E=$ expected agreement by chance $=$ proportion of agreement that would be expected by random chance.

We would like to quantify by how much better the agreement is compared to random chance. 

First, we will need to quantify the **observed agreement**. 
Simply, this is the proportion of cases where the two raters agree:

$$
O=\frac{\text { Number of agreements }}{n}
$$

Secondly, we will need **Expected Agreement by Chance**.
I was a bit bugged when I saw the formula so it is useful to re-derive it from first principle. 

Let's formalize a bit this problem with introduction of the following objects:
- $\mathcal{C}=\{1,2, \ldots, k\}$ be the set of categories.
- Rater A assigns category $i$ with probability $p_i^A$.
- Rater B assigns category $i$ with probability $p_i^B$.

We model the decisions of Rater A and Rater B as two independent random variables:

$$X \sim \operatorname{Categorical}\left(p_1^A, \ldots, p_k^A\right) \quad \text{and} \quad Y \sim \operatorname{Categorical}\left(p_1^B, \ldots, p_k^B\right)$$

We define the event of agreement as $\{X=Y\}$, i.e., both raters assign the same category.
With this setup the probability of agreement by chance is $\{X=Y\}$, i.e., both raters assign the same category. Let's compute the probability of this event: 

$$
\mathbb{P}(X=Y)=\sum_{i=1}^k \mathbb{P}(X=i, Y=i)
$$
Since $X$ and $Y$ are independent:

$$
\mathbb{P}(X=i, Y=i)=\mathbb{P}(X=i) \cdot \mathbb{P}(Y=i)=p_i^A \cdot p_i^B
$$

Finally:

$$
E=\mathbb{P}(X=Y)=\sum_{i=1}^k p_i^A \cdot p_i^B
$$

We are ready to display the **Cohen's Kappa Formula** which measures how much better the agreement is compared to random chance, scaled between -1 and 1 :

$$
\boxed{\kappa=\frac{O-E}{1-E} = \frac{\text{Observed improvement over chance }}{\text { Maximum possible improvement over chance }}}
$$

where:
- $O=$ observed agreement.
- $E=$ expected agreement by chance.

The interpretation is simple:
- $\kappa=1 \rightarrow$ Perfect agreement.
- $\kappa=0 \rightarrow$ Agreement is no better than chance.
- $\kappa<0 \rightarrow$ Worse than chance (systematic disagreement).

However, the standard Cohen's Kappa does not account for the cost sensitivity of misclassification that naturally arises in ordinal regression problems. 

In ordinal regression, misclassifying a sample into a category that is far from the true category (e.g., predicting category 1 when the true category is 5) should be penalized more heavily than misclassifying into an adjacent category (e.g., predicting category 4 when the true category is 5). Therefore, we need to weight this metric to increase the penalty for misclassifications between labels that are distant from each other on the ordinal scale.

## Weighted Cohen’s Kappa

Basic idea is quite similar to Cohen's Kappa Score. 
We want to measure the agreement between two raters but we want to increase the error penalty in case of large disagreement. 

Let's first use an example to see what we want. 

Assume we have 10 items rated by two raters (or a model vs. ground truth). Categories are:
- $1 =$ Low
- $2=$ Medium
- $3=$ High

The confusion matrix of observed agreement is the following:

|  |  B: $\mathbf{1}$  | B: 2 | B: 3 |
| :--- | :--- | :--- | :--- |
| **A:** $\mathbf{1}$ |   1 | 0 | 2 |
| **A:** $\mathbf{2}$ |   3 | 1 | 1 |
| **A:** $\mathbf{3}$ |   1 | 1 | 0 |

We can define a **observed proportions** matrix $O=\left[o_{i j}\right]$

$$
O=\frac{1}{10}\left[\begin{array}{lll}
2 & 1 & 0 \\
1 & 3 & 1 \\
0 & 1 & 1
\end{array}\right]
$$

In this matrix we have the proportion of aggreement in the diagonal but also now the disagreement.

Now the **Expected (Dis)Agreement by Chance**  can also be computing per rater's choices

$$
E_{i j}=p_i^A \cdot p_j^B
$$


$$
E=\left[\begin{array}{lll}
0.09 & 0.15 & 0.06 \\
0.15 & 0.25 & 0.10 \\
0.06 & 0.10 & 0.04
\end{array}\right]
$$

Perfect now with both matrix we can adjust the cost based on the disagreement.
As the name indicate, we will use a quadratic weight

$$
w_{i j}=\left(\frac{i-j}{k-1}\right)^2, \quad k=3
$$

These weights penalize larger disagreements (e.g., 1 vs. 3) more heavily than small ones (e.g., 1 vs. 2). In our example:

$$
W=\begin{bmatrix}
0 & \left(\frac{1-2}{2}\right)^2 & \left(\frac{1-3}{2}\right)^2 \\
\left(\frac{2-1}{2}\right)^2 & 0 & \left(\frac{2-3}{2}\right)^2 \\
\left(\frac{3-1}{2}\right)^2 & \left(\frac{3-2}{2}\right)^2 & 0
\end{bmatrix}=\begin{bmatrix}
0 & 0.25 & 1 \\
0.25 & 0 & 0.25 \\
1 & 0.25 & 0
\end{bmatrix}
$$

We are ready to integrate the weight to each observed agreement and random agreement

$$
O = \text{Weighted observed Agreement} =1 - \sum_{i, j} w_{i j} \cdot o_{i j}
$$

and 

$$
E = \text{Weighted Random Agreement}  =1 - \sum_{i, j} w_{i j} \cdot e_{i j}
$$


We arrive to the same formula with more penalization on large disagreement.
$$
\boxed{\kappa=\frac{O-E}{1-E}}
$$

---

## Appendix

**How to compute the observed agreements $O$ in practice ?**

The observed agreements $O$ is computed as the *empirical* joint distribution of the ratings (ie, how often Rater A and Rater B actually agree on each label pair in practice)

$$
\begin{aligned}
&O_{i j}=\frac{1}{n} \sum_{t=1}^n \mathbb{I}\left[x_t=i \text{ and } y_t=j\right]\\
\end{aligned}
$$

So $O$ captures the actual frequency of label pairings.

**How to compute the Expected agreements by chance $E$ in practice ?**

This is what the agreement would look like by chance, assuming independence between the raters:

$$
E_{i j}=p_i^A \cdot p_j^B \quad \text { with } \quad p_i^A=\mathbb{P}[x=i], \quad p_j^B=\mathbb{P}[y=j]
$$


It uses the marginal distributions only, ignoring any correlation between Rater A and B with:

$$p_i^A=\frac{1}{n} \sum_{t=1}^n \mathbb{I}\left[x_t=i\right], \quad p_j^B=\frac{1}{n} \sum_{t=1}^n \mathbb{I}\left[y_t=j\right]$$
