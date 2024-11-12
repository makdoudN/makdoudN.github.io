---
title: "A Bayesian Rating Model — WIP"
date: "2024-11-06"
summary: "WIP"
description: "WIP"
toc: false
readTime: false
autonumber: false
math: true
tags: ["Machine Learning", "Bayesian Inference"]
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


--- 

## Derivation of $\mathbb{E}\left[s_A \mid A \text { beats } B\right]$

From Bayes Theorem, we have:

$$
P\left(s_A \mid A \text { beats } B\right) \propto P\left(A \text { beats } B \mid s_A\right) \cdot P\left(s_A\right)
$$

We do not have directly access to $P\left(A \text { beats } B \mid s_A\right)$ but we can use the sum rule of probability to rewrite it as:

$$
P\left(A \text { beats } B \mid s_A\right)=\int P\left(A \text { beats } B \mid s_A, s_B\right) \cdot P\left(s_B\right) d s_B
$$
Substituting the given probability function:

$$
P\left(A \text { beats } B \mid s_A\right)=\int \Phi\left(\frac{s_A-s_B}{\sqrt{2} \beta}\right) \cdot \mathcal{N}\left(s_B \mid \mu_B, \sigma_B^2\right) d s_B
$$

Thus, the posterior distribution of $s_A$ is proportional to:

$$
P\left(s_A \mid A \text { beats } B\right) \propto \mathcal{N}\left(s_A \mid \mu_A, \sigma_A^2\right) \cdot \mathbb{E}_{s_B}\left[\Phi\left(\frac{s_A-s_B}{\sqrt{2} \beta}\right)\right]
$$

Then: 

$$\small{\mathbb{E}\left[s_A \mid A \text { beats } B\right]=\int s_A \cdot P\left(s_A \mid A \text { beats } B\right) d s_A \propto
\mathbb{E}_{s_A}\left[s_A \cdot P\left(A \text { beats } B \mid s_A\right)\right]
}$$

**Stein's Lemma.** To continue the derivation we will need Stein's Lemma whichis a powerful tool in probability theory, particularly useful when dealing with expectations involving Gaussian random variables. It states that for a normally distributed random variable $X \sim \mathcal{N}\left(\mu, \sigma^2\right)$ and a differentiable function $f$ :

$$
\mathbb{E}[X f(X)]=\mu \mathbb{E}[f(X)]+\sigma^2 \mathbb{E}\left[f^{\prime}(X)\right]
$$

where $f^{\prime}(X)$ is the derivative of $f$ with respect to $X$.

Perfect so, we will set $f$ to: 

$$
f\left(s_A\right)=P\left(A \text { beats } B \mid s_A\right)=\mathbb{E}_{s_B}\left[\Phi\left(\frac{\Delta_{A B}}{\sqrt{2} \beta}\right)\right]=\Phi\left(\frac{s_A-\mu_B}{\sqrt{\sigma_B^2+2 \beta^2}}\right)
$$

$$
f\left(s_A\right)=\mathbb{E}_{s_B}\left[\Phi\left(\frac{s_A-s_B}{\sqrt{2} \beta}\right)\right]=\int_{-\infty}^{\infty} \Phi\left(\frac{s_A-s_B}{\sqrt{2} \beta}\right) P\left(s_B\right) d s_B
$$

why ? Recall:
$$
\Delta_{A B}=s_A-s_B \sim \mathcal{N}\left(\mu_A-\mu_B, \sigma_A^2+\sigma_B^2\right) .
$$
Then:

$$
Z = \frac{\Delta_{A B}}{\sqrt{2} \beta}=\frac{s_A-s_B}{\sqrt{2} \beta} \sim \mathcal{N}\left(\frac{\mu_A-\mu_B}{\sqrt{2} \beta}, \frac{\sigma_A^2+\sigma_B^2}{2 \beta^2}\right)
$$
This simplify the computation of the expectation as:
$$
f\left(s_A\right)=\mathbb{E}_Z[\Phi(Z)]
$$
Since $Z$ is normally distributed, this expectation can be further simplified.

To evaluate $\mathbb{E}[\Phi(Z)]$, where $Z \sim \mathcal{N}\left(\mu_Z, \sigma_Z^2\right)$

> We will use this property of gaussian : $$\mathbb{E}[\Phi(a X+b)]=\Phi\left(\frac{a \mu_X+b}{\sqrt{1+a^2 \sigma_X^2}}\right)$$

Which in our case give:

$$
\mathbb{E}[\Phi(Z)]=\Phi\left(\frac{\mu_Z}{\sqrt{1+\sigma_Z^2}}\right) = \Phi\left(\frac{\frac{s_A-\mu_B}{\sqrt{2} \beta}}{\sqrt{1+\frac{\sigma_B^2}{2 \beta^2}}}\right) = \cdots =\Phi\left(\frac{s_A-\mu_B}{\sqrt{\sigma_B^2+2 \beta^2}}\right)
$$

Let's summarise a bit: 

$$
\boxed{f\left(s_A\right)=P\left(A \text { beats } B \mid s_A\right) = \Phi\left(\frac{s_A-\mu_B}{\sqrt{\sigma_B^2+2 \beta^2}}\right)}
$$

Ok perfect, now we can go back to our original problem.

$$
P\left(s_A \mid A \text { beats } B\right)=\frac{P\left(A \text { beats } B \mid s_A\right) \cdot P\left(s_A\right)}{P(A \text { beats } B)}
$$

and thus:
$$
\mathbb{E}\left[s_A \mid A \text { beats } B\right]=\frac{\int s_A \cdot P\left(A \text { beats } B \mid s_A\right) \cdot P\left(s_A\right) d s_A}{P(A \text { beats } B)}
$$

TODO.. expand a bit

$$
\mathbb{E}\left[s_A \mid A \text { beats } B\right]=\frac{\mathbb{E}_{s_A}\left[s_A \cdot f\left(s_A\right)\right]}{\mathbb{E}_{s_A}\left[f\left(s_A\right)\right]}
$$

To compute $\mathbb{E}\left[s_A \cdot f\left(s_A\right)\right]$, we utilize Stein's Lemma.

$$
\mathbb{E}\left[s_A \cdot f\left(s_A\right)\right]=\mu_A \cdot \mathbb{E}\left[f\left(s_A\right)\right]+\sigma_A^2 \cdot \mathbb{E}\left[f^{\prime}\left(s_A\right)\right]
$$

We have:

$$
f^{\prime}\left(s_A\right)=\frac{d}{d s_A} \Phi\left(\frac{s_A-\mu_B}{\sqrt{\sigma_B^2+2 \beta^2}}\right)=\frac{1}{\sqrt{\sigma_B^2+2 \beta^2}} \cdot \phi\left(\frac{s_A-\mu_B}{\sqrt{\sigma_B^2+2 \beta^2}}\right)
$$

Then:

$$
\mathbb{E}\left[s_A \cdot f\left(s_A\right)\right]=\mu_A \cdot \mathbb{E}\left[f\left(s_A\right)\right]+\frac{\sigma_A^2}{\sqrt{\sigma_B^2+2 \beta^2}} \cdot \mathbb{E}\left[\phi\left(\frac{s_A-\mu_B}{\sqrt{\sigma_B^2+2 \beta^2}}\right)\right]
$$


---

Final Result

$$
\boxed{\mathbb{E}\left[s_A \mid A \text { beats } B\right]=\mu_A+\frac{\sigma_A^2}{\sqrt{\sigma_A^2+\sigma_B^2+2 \beta^2}} \cdot \frac{\phi\left(\frac{\mu_A-\mu_B}{\sqrt{\sigma_A^2+\sigma_B^2+2 \beta^2}}\right)}{\Phi\left(\frac{\mu_A-\mu_B}{\sqrt{\sigma_A^2+\sigma_B^2+2 \beta^2}}\right)}}
$$

{{< details  title="Learn More" >}} 
fdsfdsafadas
$$
\boxed{\mathbb{E}\left[s_A \mid A \text { beats } B\right]=\mu_A+\frac{\sigma_A^2}{\sqrt{\sigma_A^2+\sigma_B^2+2 \beta^2}} \cdot \frac{\phi\left(\frac{\mu_A-\mu_B}{\sqrt{\sigma_A^2+\sigma_B^2+2 \beta^2}}\right)}{\Phi\left(\frac{\mu_A-\mu_B}{\sqrt{\sigma_A^2+\sigma_B^2+2 \beta^2}}\right)}}
$$
{{< /details >}}



See [Appendix A](#appendix-a) for more details on Section A.
See [Appendix B](#appendix-b) for more details on Section B.

{{< appendix title="Appendix" sections="Section A, Section B" />}}


