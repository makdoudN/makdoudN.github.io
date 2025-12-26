---
title: "The Geometry of the Differential"
date: "2025-12-26"
summary: "Building intuition for the differential, gradient, Jacobian, and Hessian from first principles using geometry and linear algebra"
description: "Understanding differentiation through the lens of linear approximation and geometric intuition"
toc: false
readTime: false
autonumber: false
math: true
tags: ["Mathematics", "Linear Algebra", "Calculus"]
showTags: false
hideBackToTop: false
---

What does it mean to differentiate a function?

Calculus courses teach differentiation as a collection of rules applied mechanically to compute derivatives. But these rules obscure a deeper idea: **differentiation is linear approximation**.

## The Core Idea: Replacing Curves with Lines

Consider $f: \mathbb{R} \to \mathbb{R}$. Curves are hard, lines are easy.
The fundamental question: *can we replace a curve with a line that approximates it well near a point?*

The line through $(a, f(a))$ with slope $m$ is:
$$L(x) = f(a) + m(x - a)$$

For this approximation to be "good", the error must shrink faster than the step size $h$:
$$\lim_{h \to 0} \frac{f(a + h) - f(a) - mh}{h} = 0$$

Rearranging:
$$\lim_{h \to 0} \frac{f(a + h) - f(a)}{h} = m$$

This is the definition of the derivative. The derivative $f'(a)$ is the unique slope making the linear approximation good.

## From Numbers to Vectors

Now consider $f: \mathbb{R}^n \to \mathbb{R}^m$. In one dimension:
- Input perturbation: a number $h$
- Output change: approximately $f'(a) \cdot h$

In higher dimensions:
- Input perturbation: a vector $\mathbf{h} \in \mathbb{R}^n$
- Output change: a vector in $\mathbb{R}^m$
- What multiplies input to give output? **A linear map.**

A function $f: \mathbb{R}^n \to \mathbb{R}^m$ is **differentiable** at $\mathbf{a}$ if there exists a linear map $L: \mathbb{R}^n \to \mathbb{R}^m$ such that:
$$f(\mathbf{a} + \mathbf{h}) = f(\mathbf{a}) + L(\mathbf{h}) + o(\|\mathbf{h}\|)$$

The linear map $L$ is the **differential** of $f$ at $\mathbf{a}$, written $Df(\mathbf{a})$.

The differential is not a number—it is a linear transformation.

## The Jacobian: Matrix of the Differential

Every linear map has a matrix representation. For $Df(\mathbf{a}): \mathbb{R}^n \to \mathbb{R}^m$, this matrix is the **Jacobian** $Jf(\mathbf{a})$.

The $j$-th column of $Jf(\mathbf{a})$ is $Df(\mathbf{a})(\mathbf{e}_j)$—what happens when perturbing only in direction $j$.

Moving in direction $\mathbf{e}_j$ by small $h$:
$$f(\mathbf{a} + h\mathbf{e}_j) - f(\mathbf{a}) \approx h \cdot Df(\mathbf{a})(\mathbf{e}_j)$$

So:
$$Df(\mathbf{a})(\mathbf{e}_j) = \lim_{h \to 0} \frac{f(\mathbf{a} + h\mathbf{e}_j) - f(\mathbf{a})}{h} = \frac{\partial f}{\partial x_j}(\mathbf{a})$$

Since $f = (f_1, \ldots, f_m)$:
$$\boxed{Jf(\mathbf{a}) = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}}$$

Entry $(i, j)$ answers: how does output $i$ change when perturbing input $j$?

### Example: Polar to Cartesian

Consider $f(r, \theta) = (r\cos\theta, r\sin\theta)^T$.

$$\frac{\partial f}{\partial r} = \begin{bmatrix} \cos\theta \\ \sin\theta \end{bmatrix}, \quad \frac{\partial f}{\partial \theta} = \begin{bmatrix} -r\sin\theta \\ r\cos\theta \end{bmatrix}$$

$$Jf(r, \theta) = \begin{bmatrix} \cos\theta & -r\sin\theta \\ \sin\theta & r\cos\theta \end{bmatrix}$$

First column: unit vector pointing radially. Second column: vector of length $r$ pointing tangentially.

## The Gradient

For scalar-valued $f: \mathbb{R}^n \to \mathbb{R}$, the Jacobian is a $1 \times n$ row vector:
$$Jf(\mathbf{a}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} & \cdots & \frac{\partial f}{\partial x_n} \end{bmatrix}$$

The **gradient** is the transpose:
$$\nabla f(\mathbf{a}) = (Jf(\mathbf{a}))^T = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

The linear approximation becomes:
$$f(\mathbf{a} + \mathbf{h}) \approx f(\mathbf{a}) + \langle \nabla f(\mathbf{a}), \mathbf{h} \rangle$$

### Direction of Steepest Ascent

The directional derivative in direction $\mathbf{v}$ (with $\|\mathbf{v}\| = 1$) is:
$$D_\mathbf{v}f(\mathbf{a}) = \langle \nabla f(\mathbf{a}), \mathbf{v} \rangle$$

Which direction maximizes this? By Cauchy-Schwarz:
$$\langle \nabla f(\mathbf{a}), \mathbf{v} \rangle \leq \|\nabla f(\mathbf{a})\|$$

Equality when $\mathbf{v}$ points in the direction of $\nabla f(\mathbf{a})$.

**The gradient points in the direction of steepest ascent.**

### Orthogonality to Level Sets

Let $\gamma(t)$ be a curve on a level set with $\gamma(0) = \mathbf{a}$. Since $f(\gamma(t)) = c$:
$$\frac{d}{dt}f(\gamma(t)) = \langle \nabla f(\gamma(t)), \gamma'(t) \rangle = 0$$

At $t = 0$: $\langle \nabla f(\mathbf{a}), \gamma'(0) \rangle = 0$.

**The gradient is orthogonal to level sets.**

## The Chain Rule

If $f: \mathbb{R}^n \to \mathbb{R}^m$ and $g: \mathbb{R}^m \to \mathbb{R}^p$, then near $\mathbf{a}$:
$$g(f(\mathbf{a} + \mathbf{h})) \approx g(f(\mathbf{a})) + Dg(f(\mathbf{a}))(Df(\mathbf{a})(\mathbf{h}))$$

The differential of $g \circ f$ is the composition of differentials:
$$D(g \circ f)(\mathbf{a}) = Dg(f(\mathbf{a})) \circ Df(\mathbf{a})$$

In matrix form:
$$\boxed{J(g \circ f)(\mathbf{a}) = Jg(f(\mathbf{a})) \cdot Jf(\mathbf{a})}$$

Dimensions: $(p \times m) \cdot (m \times n) = (p \times n)$.

The chain rule is matrix multiplication. Composing linear maps means multiplying matrices.

## The Hessian

For $f: \mathbb{R}^n \to \mathbb{R}$, the gradient $\nabla f: \mathbb{R}^n \to \mathbb{R}^n$ is itself a function.

Its Jacobian is an $n \times n$ matrix:
$$\boxed{Hf(\mathbf{a}) = J(\nabla f)(\mathbf{a}) = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{bmatrix}}$$

This is the **Hessian**. When second partials are continuous, $Hf$ is symmetric.

### Second-Order Approximation

$$f(\mathbf{a} + \mathbf{h}) \approx f(\mathbf{a}) + \nabla f(\mathbf{a})^T \mathbf{h} + \frac{1}{2}\mathbf{h}^T Hf(\mathbf{a}) \mathbf{h}$$

The quadratic form $\frac{1}{2}\mathbf{h}^T Hf(\mathbf{a}) \mathbf{h}$ captures curvature.

### Eigenstructure and Curvature

Since $Hf$ is symmetric, it has real eigenvalues $\lambda_1, \ldots, \lambda_n$ with orthonormal eigenvectors $\mathbf{v}_1, \ldots, \mathbf{v}_n$.

In the eigenbasis:
$$\mathbf{h}^T Hf(\mathbf{a}) \mathbf{h} = \sum_{i=1}^n \lambda_i (\mathbf{h} \cdot \mathbf{v}_i)^2$$

Each $\lambda_i$ is the curvature in direction $\mathbf{v}_i$:
- $\lambda_i > 0$: curves upward (bowl)
- $\lambda_i < 0$: curves downward (dome)
- $\lambda_i = 0$: flat

### Critical Points

At a critical point $\nabla f(\mathbf{a}) = \mathbf{0}$:
$$f(\mathbf{a} + \mathbf{h}) \approx f(\mathbf{a}) + \frac{1}{2}\mathbf{h}^T Hf(\mathbf{a}) \mathbf{h}$$

The Hessian determines the nature:
- All $\lambda_i > 0$ (positive definite): local minimum
- All $\lambda_i < 0$ (negative definite): local maximum
- Mixed signs: saddle point

### The Condition Number

For positive definite $Hf$, the condition number $\kappa = \lambda_{\max} / \lambda_{\min}$.

When $\kappa \approx 1$: curvatures similar, level sets nearly spherical, gradient descent converges fast.

When $\kappa \gg 1$: curvatures differ wildly, level sets elongated, gradient descent zigzags.

Newton's method adapts to curvature:
$$\mathbf{h} = -Hf(\mathbf{a})^{-1} \nabla f(\mathbf{a})$$

For quadratic $f$, Newton finds the minimum in one step.

## Summary

| Object | Definition | Meaning |
|--------|------------|---------|
| Differential $Df(\mathbf{a})$ | Linear map $\mathbb{R}^n \to \mathbb{R}^m$ | Best linear approximation |
| Jacobian $Jf(\mathbf{a})$ | $m \times n$ matrix | Matrix of $Df(\mathbf{a})$ |
| Gradient $\nabla f(\mathbf{a})$ | $Jf^T$ for $f: \mathbb{R}^n \to \mathbb{R}$ | Direction of steepest ascent |
| Hessian $Hf(\mathbf{a})$ | $J(\nabla f)$ | Curvature (Jacobian of gradient) |
| Chain rule | $J(g \circ f) = Jg \cdot Jf$ | Composing linear maps = multiplying matrices |

The derivative is the slope of the best linear approximation. The chain rule is matrix multiplication. The Hessian's eigenvalues are curvatures. These are not arbitrary definitions—they follow inevitably from asking: what is the best linear approximation?
