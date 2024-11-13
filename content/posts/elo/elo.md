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



{{< details  title="Full Derivation of the Full Joint Density" >}} 

$$
P\left(X_{A \triangleright B}, s_A, s_B\right)=P\left(X_{A \triangleright B} \mid \Delta_{A B}\right) P\left(s_A\right) P\left(s_B\right)
$$

{{< /details >}}

$$
P\left(X_{A \triangleright B}, s_A, s_B\right)=\Phi\left(\frac{s_A-s_B}{\sqrt{2} \beta}\right) \cdot \mathcal{N}\left(s_A \mid \mu_A, \sigma_A^2\right) \cdot \mathcal{N}\left(s_B \mid \mu_B, \sigma_B^2\right)
$$

TODO - Explain what is needed now to update the distribution of the skill of player $A$ and $B$ based on the observation of a win or loss.


---
### Derivation of $P\left(s_A \mid X_{A \triangleright B}=1\right)$

TODO — *Explain why we are interested in $P\left(s_A \mid X_{A \triangleright B}=1\right)$*

Our goal is to find $P\left(s_A \mid X_{A \triangleright B}=1\right)$, which requires integrating out $s_B$ :

$$
P\left(s_A \mid X_{A \triangleright B}=1\right)=\int_{-\infty}^{\infty} P\left(s_A, s_B \mid X_{A \triangleright B}=1\right) d s_B
$$

This is a nasty integral, but we can solve it analytically.
Let's be clear, the derivation is not trivial and requires a bit of knowledge of the Gaussian and a dose of coffee and patience.

{{< details  title="Full Derivation of the Posterior Distribution of the Skill of Player A" >}} 

Substituting the expressions into the integral

$$
P\left(s_A \mid X_{A \triangleright B}=1\right) \propto \mathcal{N}\left(s_A \mid \mu_A, \sigma_A^2\right) \cdot \int_{-\infty}^{\infty} \Phi\left(\frac{s_A-s_B}{\sqrt{2} \beta}\right) \cdot \mathcal{N}\left(s_B \mid \mu_B, \sigma_B^2\right) d s_B
$$

Let's simplify the integral. Let's denote:
- $\Delta=s_A-s_B$
- $\gamma=\sqrt{2} \beta$

Then the integral becomes:

$$
I\left(s_A\right)=\int_{-\infty}^{\infty} \Phi\left(\frac{\Delta}{\gamma}\right) \cdot \mathcal{N}\left(s_B \mid \mu_B, \sigma_B^2\right) d s_B
$$

Changing Variables. We can change the variable of integration from $s_B$ to $\Delta$ :

$$
d s_B=-d \Delta
$$


When $s_B \rightarrow-\infty, \Delta \rightarrow \infty$, and when $s_B \rightarrow \infty, \Delta \rightarrow-\infty$. Adjusting the limits accordingly:

$$
I\left(s_A\right)=\int_{-\infty}^{\infty} \Phi\left(\frac{\Delta}{\gamma}\right) \cdot \mathcal{N}\left(s_A-\Delta \mid \mu_B, \sigma_B^2\right)(-d \Delta)
$$


Rewriting:

$$
I\left(s_A\right)=\int_{-\infty}^{\infty} \Phi\left(\frac{\Delta}{\gamma}\right) \cdot \mathcal{N}\left(s_A-\Delta \mid \mu_B, \sigma_B^2\right) d \Delta
$$

Recognizing the Integral as a Convolution.

The integral $I\left(s_A\right)$ represents the convolution of $\Phi\left(\frac{\Delta}{\gamma}\right)$ with the normal distribution of $\Delta$. However, there exists an integral identity that can simplify this:

Integral Identity:

For constants $a$ and $b$ :

$$
\int_{-\infty}^{\infty} \Phi(a x+b) \cdot \phi(x) d x=\Phi\left(\frac{b}{\sqrt{1+a^2}}\right)
$$

where $\phi(x)$ is the standard normal density function.

In our case:
- Let $x=t$
- $\phi(t)=\frac{1}{\sqrt{2 \pi}} \exp \left(-\frac{t^2}{2}\right)$
- Set $a=-\frac{\gamma}{\sigma_B}=-\frac{\sqrt{2} \beta}{\sigma_B}$
- Set $b=\frac{s_A-\mu_B}{\sigma_B}$

Then the integral becomes:

$$
I\left(s_A\right)=\int_{-\infty}^{\infty} \Phi(a t+b) \cdot \phi(t) d t=\Phi\left(\frac{b}{\sqrt{1+a^2}}\right)
$$


Compute $1+a^2$ :

$$
1+a^2=1+\left(-\frac{\gamma}{\sigma_B}\right)^2=1+\frac{2 \beta^2}{\sigma_B^2}=\frac{\sigma_B^2+2 \beta^2}{\sigma_B^2}
$$

Compute the denominator:

$$
\sqrt{1+a^2}=\frac{\sqrt{\sigma_B^2+2 \beta^2}}{\sigma_B}
$$


Compute $\frac{b}{\sqrt{1+a^2}}$ :

$$
\frac{b}{\sqrt{1+a^2}}=\frac{\frac{s_A-\mu_B}{\sigma_B}}{\frac{\sqrt{\sigma_B^2+2 \beta^2}}{\sigma_B}}=\frac{s_A-\mu_B}{\sqrt{\sigma_B^2+2 \beta^2}}
$$


Therefore, the integral simplifies to:

$$
I\left(s_A\right)=\Phi\left(\frac{s_A-\mu_B}{\sqrt{\sigma_B^2+2 \beta^2}}\right)
$$

{{< /details >}}

$$
\boxed{P\left(s_A \mid X_{A \triangleright B}=1\right) \propto \mathcal{N}\left(s_A \mid \mu_A, \sigma_A^2\right) \cdot \Phi\left(\frac{s_A-\mu_B}{\sqrt{\sigma_B^2+2 \beta^2}}\right)}
$$




---







