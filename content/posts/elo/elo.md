---
title: "A Bayesian Rating Model — WIP"
date: "2024-11-06"
summary: "WIP"
description: "WIP"
toc: false
readTime: false
autonumber: false
math: true
tags: ["Machine Learning", "Bayesian Inference", "Skill Rating Models"]
showTags: false
hideBackToTop: false
---

How to rate players in a competitive game by leveraging data ?

Skill ratings in competitive games and sports serve several key purposes.
First, they enable players to be matched with others of similar skill, creating engaging, balanced matches.
Second, these ratings can be shared with players and the public, sparking interest and fostering competition (think ELO in chess).
Third, with the rise of online gaming, interest in rating systems has surged, as they impact the daily online experience of millions of players worldwide.

With the vast amount of sports and events, it is difficult if not impossible to accurately rate the skill of players withtout automatic approaches.
What better approach to rate the skill of players that leveraging the record of wins, draws and losses to automatically asses their skill.

Rating models in their simple form generaly assume that a player $A$ has a  **strengh** $s_A$, and that the higher the difference in strengh with a player $B$ the larger the probability of winning a match $P\left(\mathbf{x}_A \triangleright \mathbf{x}_B | s_A, s_B\right) = f(s_A - s_B)$. 
A subtile yet important common assumption (implied by our previous statement) is that only the skill gap impacts the probability of winning and not directly the absolute magnitude of the skill.

In summary, skill ratings generally assumes: 
- $ \textcolor{blue}{\raisebox{.5pt}{\textcircled{\scriptsize 1}}} $ Any player has an underlying strengh (or skill level)
- $ \textcolor{blue}{\raisebox{.5pt}{\textcircled{\scriptsize 2}}} $ Probability $A$ wins over $B$ is a fonction of the difference in strengh (skill gap):
$$
P\left(\mathbf{x}_A \triangleright \mathbf{x}_B | s_A, s_B\right) = f(s_A - s_B)
$$

Those two core assumptions underpins the majority of the skill rating approaches. 
As we will see, each approach will require a bit more hypothesis about the influence of the skill gap and the probability of wins.

To ease the derivation, let's note the skill gap $\Delta_{AB} = s_A - s_B$

## A Bayesian Rating Model
---

The ELO rating model is a statistical approach famous to be adopted by World Chess Federation (FIDE).
ELO adds another hypothesis upon  $ \textcolor{blue}{\raisebox{.5pt}{\textcircled{\scriptsize 1}}} $ and $ \textcolor{blue}{\raisebox{.5pt}{\textcircled{\scriptsize 2}}} $: 

- $ \textcolor{red}{\raisebox{.5pt}{\textcircled{\scriptsize 3}}}$ Probability of A wins over B is given by 
$$ f(\Delta_{AB}) = \Phi\left(\frac{\Delta_{AB}}{\sqrt{2} \beta}\right) $$
with $\Phi$ is the cumulative density of a zero-mean unit-variance Gaussian. 

In this case $ \textcolor{red}{\raisebox{.5pt}{\textcircled{\scriptsize 3}}}$ is more than a hypothesis but it describes functionally the relationship between the **skill gap** and the probability of winning.

Here is a plot of the probability of winning as a function of the skill gap.
![ELO Rating Graph](/elo1.png)

As you can see, the probability of winning is symmetric around zero (skill gap = 0) and the probability of winning increase when the skill gap increase. 
This is aligned with our previous hypothesis. 
The more the skill gap increase, the more the probability of winning increase. 
This relationship is decribed by the $\beta$ parameter.

To understand how to update this model, it is useful to make a detour by the specification of  full (bayesian) data generative process.
This specification is the hypothesis space upon which we believe the observations are generated. 
In our case, the foundation are our hypothesis about the process and be revised to derive different method. 
In this case,  

- $ \textcolor{blue}{\raisebox{.5pt}{\textcircled{\scriptsize 1}}}$ Each player $i$ has an underlying skill level $s_i$. In the Bayesian Framework, those unobserved skill level are assume to be normal random variable.
$$
s_i \sim \mathcal{N}\left(\mu_i, \sigma_i^2\right)
$$
Those are called prior that reflect our knowledge before any observation. 
Here, we simply assume that the skill of a player is distributed around a skill level $\mu_i$ but can fluctuate with a standard variation of $\sigma_i$.

- $ \textcolor{blue}{\raisebox{.5pt}{\textcircled{\scriptsize 2}}} $ Probability $A$ wins over $B$ is
$$
P\left(\mathbf{x}_A \triangleright \mathbf{x}_B | \Delta_{AB}\right) = \Phi\left(\frac{\Delta_{AB}}{\sqrt{2} \beta}\right)
$$

A simple graphical model can be derived from this data generative process: (TODO Explain)
$$
s_A \rightarrow \Delta_{A B} \leftarrow s_B \quad \text { and } \quad \Delta_{A B} \rightarrow X_{A \triangleright B}
$$

**Full Data Generation Process**. Based on the model's Bayesian specification and the dependencies in the graphical model, the joint density for the observed and latent variables can be derived as follows:

*Prior Densities for Skills*

$$
P\left(s_A\right)=\mathcal{N}\left(s_A \mid \mu_A, \sigma_A^2\right), \quad P\left(s_B\right)=\mathcal{N}\left(s_B \mid \mu_B, \sigma_B^2\right)
$$

*Outcome Probability Conditional on Skill Gap* 

$$
P\left(X_{A \triangleright B}=1 \mid \Delta_{A B}\right)=\Phi\left(\frac{\Delta_{A B}}{\sqrt{2} \beta}\right)
$$

*Full Joint Density over observed and latent variables*

$$
P\left(X_{A \triangleright B}, s_A, s_B\right)=P\left(X_{A \triangleright B} \mid \Delta_{A B}\right) P\left(s_A\right) P\left(s_B\right)
$$

$$
P\left(X_{A \triangleright B}, s_A, s_B\right)=\Phi\left(\frac{s_A-s_B}{\sqrt{2} \beta}\right) \cdot \mathcal{N}\left(s_A \mid \mu_A, \sigma_A^2\right) \cdot \mathcal{N}\left(s_B \mid \mu_B, \sigma_B^2\right)
$$

### Derivation of $P\left(s_A \mid X_{A \triangleright B}=1\right)$
---

Based on the full joint density, the posterior distribution of the skill of player $A$ given that $A$ wins over $B$ can be derived using the Bayes rule:

$$
P\left(s_A, s_B \mid X_{A \triangleright B}=1\right) \propto \Phi\left(\frac{s_A-s_B}{\sqrt{2} \beta}\right) \cdot \mathcal{N}\left(s_A \mid \mu_A, \sigma_A^2\right) \cdot \mathcal{N}\left(s_B \mid \mu_B, \sigma_B^2\right)
$$

Our goal is to find $P\left(s_A \mid X_{A \triangleright B}=1\right)$, which requires integrating out $s_B$ :

$$
P\left(s_A \mid X_{A \triangleright B}=1\right)=\int_{-\infty}^{\infty} P\left(s_A, s_B \mid X_{A \triangleright B}=1\right) d s_B
$$

This is a nasty integral, but we can solve it analytically.


---

{{< details  title="Tiny Test" >}} 
This is a test
$$
\boxed{\mathbb{E}\left[s_A \mid A \text { beats } B\right]=\mu_A+\frac{\sigma_A^2}{\sqrt{\sigma_A^2+\sigma_B^2+2 \beta^2}} \cdot \frac{\phi\left(\frac{\mu_A-\mu_B}{\sqrt{\sigma_A^2+\sigma_B^2+2 \beta^2}}\right)}{\Phi\left(\frac{\mu_A-\mu_B}{\sqrt{\sigma_A^2+\sigma_B^2+2 \beta^2}}\right)}}
$$
{{< /details >}}





