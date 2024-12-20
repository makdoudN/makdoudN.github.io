---
title: "LoRa, An introduction — WIP"
date: "2024-11-15"
summary: "work in progress"
description: "work in progress"
toc: true
readTime: true
autonumber: false
math: true
tags: ["Generative AI", "LoRa"]
showTags: false
hideBackToTop: false
---


LoRA or Low Rank Adaptation is a technique to adapt pre-trained models to specific tasks without fine-tuning the entire model. 
It does so by adding low-rank matrices to the original model.


---
## What about Low Rank ?

So, let's refresh my old knowledge of linear algebra. 

**1. Rank of a Matrix** is the dimension of the vector space generated by its columns. 
Given a matrix $A \in \mathbb{R}^{m \times n}$, its rank, rank($A$), is the dimension of the vector space generated by its columns.
We know that $\operatorname{rank}(A) \leq \min (m, n)$

We have different types of matrices:
- **Full Rank**: $\operatorname{rank}(A) = \min (m, n)$
- **Low Rank**: $\operatorname{rank}(A) < \min (m, n)$

But why do we care about the rank of a matrix ?

---

Some great resources to understand LoRA:

- https://arxiv.org/abs/2106.09685
- https://arxiv.org/abs/2402.12354
- https://arxiv.org/abs/2405.09673
- https://arxiv.org/abs/2407.18242
- https://arxiv.org/abs/2402.09353
- https://blogs.rstudio.com/ai/posts/2023-06-22-understanding-lora/
- https://huggingface.co/blog/lora
- https://cameronrwolfe.substack.com/p/easily-train-a-specialized-llm-peft
- https://blog.stackademic.com/understanding-lora-ebebe26dae8e
- https://www.ml6.eu/blogpost/low-rank-adaptation-a-technical-deep-dive

---
- SVD (Singular Value Decomposition)
- matrix approximation lemma or Eckart–Young–Mirsky theorem