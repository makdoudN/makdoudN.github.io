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

What does it mean to differentiate a function? Most calculus courses teach differentiation as a collection of rules—power rule, chain rule, product rule—applied mechanically to compute derivatives. But these rules obscure a deeper, more beautiful idea.

This post builds differentiation from scratch, using only geometry and linear algebra. By the end, you'll own these concepts—not because you memorized formulas, but because you derived them yourself from simple principles.

## The Core Idea: Replacing Curves with Lines

Let's start with the simplest possible case: a function $f: \mathbb{R} \to \mathbb{R}$. You have a curve, and you want to understand its behavior near a point.

Here's the key insight: **curves are hard, lines are easy.**

A general curve can bend, twist, and do complicated things. But a line? A line is completely determined by two numbers: where it starts and how steeply it rises. We know everything about lines from basic algebra.

So here's the fundamental question of differential calculus: *Can we replace a curve with a line that approximates it well near a point?*

### What "Approximate Well" Means

Pick a point $a$ on the $x$-axis. We want to find a line $L(x)$ that passes through $(a, f(a))$ and matches the curve as closely as possible nearby.

The line through $(a, f(a))$ with slope $m$ is:

$$
L(x) = f(a) + m(x - a)
$$

What makes one slope better than another? Consider what happens when you zoom in. At the point $a + h$ for small $h$, the curve gives you $f(a + h)$ and the line gives you $f(a) + mh$. The error is:

$$
\text{error}(h) = f(a + h) - f(a) - mh
$$

For an approximation to be "good," we want this error to shrink faster than the size of the step $h$ itself. That is:

$$
\lim_{h \to 0} \frac{f(a + h) - f(a) - mh}{h} = 0
$$

Rearranging:

$$
\lim_{h \to 0} \frac{f(a + h) - f(a)}{h} = m
$$

This is the definition of the derivative! The derivative $f'(a)$ is the unique slope $m$ that makes the linear approximation "good" in this precise sense.

**First insight:** The derivative isn't fundamentally about "instantaneous rate of change" or limits. It's the slope of the best linear approximation.

### Why This Perspective Matters

When you think of the derivative as "the slope of the tangent line," you're thinking geometrically. But "slope of the best linear approximation" is more powerful—it generalizes naturally to higher dimensions where "slope" doesn't make sense, but "linear approximation" does.

## From Numbers to Vectors: The Leap to Higher Dimensions

Now comes the key step. We want to differentiate functions like $f: \mathbb{R}^n \to \mathbb{R}^m$—functions that take vectors in and produce vectors out. How should we think about this?

Let's reason from first principles. In one dimension:
- Input perturbation: a number $h$
- Output change: approximately $f'(a) \cdot h$
- The derivative $f'(a)$ is a number that multiplies the input perturbation

In higher dimensions:
- Input perturbation: a vector $\mathbf{h} \in \mathbb{R}^n$
- Output change: a vector in $\mathbb{R}^m$
- What should multiply the input to give the output?

**The answer comes from linear algebra.** A linear map from $\mathbb{R}^n$ to $\mathbb{R}^m$ takes vectors in $\mathbb{R}^n$ and produces vectors in $\mathbb{R}^m$. And every linear map can be represented by an $m \times n$ matrix.

So the natural generalization is: **the derivative of $f$ at $\mathbf{a}$ should be a linear map** $Df(\mathbf{a}): \mathbb{R}^n \to \mathbb{R}^m$.

### The Definition

A function $f: \mathbb{R}^n \to \mathbb{R}^m$ is **differentiable** at $\mathbf{a}$ if there exists a linear map $L: \mathbb{R}^n \to \mathbb{R}^m$ such that:

$$
f(\mathbf{a} + \mathbf{h}) = f(\mathbf{a}) + L(\mathbf{h}) + o(\|\mathbf{h}\|)
$$

where $o(\|\mathbf{h}\|)$ means "terms that shrink faster than $\|\mathbf{h}\|$."

This says: near $\mathbf{a}$, the function $f$ looks like a linear map $L$, up to negligible error.

The linear map $L$ is called the **differential** of $f$ at $\mathbf{a}$, written $Df(\mathbf{a})$.

**Second insight:** The differential isn't a number—it's a linear transformation. It tells you: "if you perturb the input by $\mathbf{h}$, the output changes by approximately $Df(\mathbf{a})(\mathbf{h})$."

## The Jacobian: Giving the Differential a Matrix

Every linear map has a matrix representation. For $Df(\mathbf{a}): \mathbb{R}^n \to \mathbb{R}^m$, this matrix is called the **Jacobian**, written $Jf(\mathbf{a})$.

But what are the entries of this matrix? Let's figure it out from first principles.

### Building the Matrix Column by Column

The matrix of a linear transformation is determined by what it does to the standard basis vectors. Consider the standard basis vectors $\mathbf{e}_1, \ldots, \mathbf{e}_n$ in $\mathbb{R}^n$.

The $j$-th column of $Jf(\mathbf{a})$ is $Df(\mathbf{a})(\mathbf{e}_j)$—what happens when you perturb only in the $j$-th direction?

Moving in direction $\mathbf{e}_j$ by a small amount $h$ means:

$$
\mathbf{a} + h\mathbf{e}_j = (a_1, \ldots, a_j + h, \ldots, a_n)
$$

The change in $f$ is:

$$
f(\mathbf{a} + h\mathbf{e}_j) - f(\mathbf{a}) \approx Df(\mathbf{a})(h\mathbf{e}_j) = h \cdot Df(\mathbf{a})(\mathbf{e}_j)
$$

So:

$$
Df(\mathbf{a})(\mathbf{e}_j) = \lim_{h \to 0} \frac{f(\mathbf{a} + h\mathbf{e}_j) - f(\mathbf{a})}{h}
$$

This limit is exactly the **partial derivative** of $f$ with respect to $x_j$, evaluated at $\mathbf{a}$. We write it as $\frac{\partial f}{\partial x_j}(\mathbf{a})$.

Since $f$ has $m$ components $f = (f_1, \ldots, f_m)$, this partial derivative is an $m$-vector:

$$
\frac{\partial f}{\partial x_j}(\mathbf{a}) = \begin{bmatrix} \frac{\partial f_1}{\partial x_j}(\mathbf{a}) \\ \vdots \\ \frac{\partial f_m}{\partial x_j}(\mathbf{a}) \end{bmatrix}
$$

### The Full Jacobian Matrix

Assembling all the columns, the Jacobian is:

$$
Jf(\mathbf{a}) = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}
$$

Each entry $(i, j)$ answers: "How does output component $i$ change when I perturb input component $j$?"

**Third insight:** The Jacobian isn't just a table of partial derivatives. It's the matrix of the linear approximation. Every entry has meaning: it tells you the sensitivity of one output to one input.

### A Concrete Example

Consider the polar-to-Cartesian transformation:

$$
f(r, \theta) = \begin{bmatrix} r\cos\theta \\ r\sin\theta \end{bmatrix}
$$

Let's compute the Jacobian. We need four partial derivatives:

$$
\frac{\partial}{\partial r}\begin{bmatrix} r\cos\theta \\ r\sin\theta \end{bmatrix} = \begin{bmatrix} \cos\theta \\ \sin\theta \end{bmatrix}, \quad
\frac{\partial}{\partial \theta}\begin{bmatrix} r\cos\theta \\ r\sin\theta \end{bmatrix} = \begin{bmatrix} -r\sin\theta \\ r\cos\theta \end{bmatrix}
$$

So:

$$
Jf(r, \theta) = \begin{bmatrix} \cos\theta & -r\sin\theta \\ \sin\theta & r\cos\theta \end{bmatrix}
$$

What does this mean geometrically?

The first column $[\cos\theta, \sin\theta]^T$ is a unit vector pointing radially outward. It tells you: if you increase $r$ by 1, you move 1 unit in the radial direction.

The second column $[-r\sin\theta, r\cos\theta]^T$ is a vector of length $r$ pointing tangentially. It tells you: if you increase $\theta$ by 1 radian, you move $r$ units in the tangential direction.

The Jacobian encodes how the coordinate system stretches and rotates locally.

## The Gradient: When Output is One-Dimensional

An important special case: scalar-valued functions $f: \mathbb{R}^n \to \mathbb{R}$.

The Jacobian of such a function is a $1 \times n$ matrix—a row vector:

$$
Jf(\mathbf{a}) = \begin{bmatrix} \frac{\partial f}{\partial x_1}(\mathbf{a}) & \cdots & \frac{\partial f}{\partial x_n}(\mathbf{a}) \end{bmatrix}
$$

The linear approximation is:

$$
f(\mathbf{a} + \mathbf{h}) \approx f(\mathbf{a}) + Jf(\mathbf{a}) \cdot \mathbf{h}
$$

This is a row vector times a column vector—an inner product.

We define the **gradient** $\nabla f(\mathbf{a})$ as the transpose of this row vector:

$$
\nabla f(\mathbf{a}) = \begin{bmatrix} \frac{\partial f}{\partial x_1}(\mathbf{a}) \\ \vdots \\ \frac{\partial f}{\partial x_n}(\mathbf{a}) \end{bmatrix}
$$

Now the linear approximation becomes:

$$
f(\mathbf{a} + \mathbf{h}) \approx f(\mathbf{a}) + \langle \nabla f(\mathbf{a}), \mathbf{h} \rangle
$$

where $\langle \cdot, \cdot \rangle$ is the inner product.

### Why Define the Gradient as a Column Vector?

This isn't just notational convenience. The gradient is a column vector because it lives in the same space as the input—it represents a direction you could actually move in.

The **directional derivative** of $f$ at $\mathbf{a}$ in direction $\mathbf{v}$ (with $\|\mathbf{v}\| = 1$) is:

$$
D_\mathbf{v}f(\mathbf{a}) = \langle \nabla f(\mathbf{a}), \mathbf{v} \rangle
$$

This measures how fast $f$ changes when you move in direction $\mathbf{v}$.

**Question:** Which direction maximizes this?

By Cauchy-Schwarz:

$$
\langle \nabla f(\mathbf{a}), \mathbf{v} \rangle \leq \|\nabla f(\mathbf{a})\| \cdot \|\mathbf{v}\| = \|\nabla f(\mathbf{a})\|
$$

Equality holds when $\mathbf{v}$ points in the same direction as $\nabla f(\mathbf{a})$.

**Fourth insight:** The gradient points in the direction of steepest ascent. Its magnitude is the rate of increase in that direction.

This is why gradient descent works: to minimize $f$, move opposite to $\nabla f$.

### Orthogonality to Level Sets

Here's another beautiful geometric fact. A **level set** of $f$ is the set of points where $f$ takes a constant value: $\{\mathbf{x} : f(\mathbf{x}) = c\}$.

**Claim:** The gradient is orthogonal to level sets.

*Proof:* Let $\gamma(t)$ be any curve lying on a level set, with $\gamma(0) = \mathbf{a}$. Since $f(\gamma(t)) = c$ for all $t$:

$$
\frac{d}{dt}f(\gamma(t)) = 0
$$

By the chain rule:

$$
\langle \nabla f(\gamma(t)), \gamma'(t) \rangle = 0
$$

At $t = 0$: $\langle \nabla f(\mathbf{a}), \gamma'(0) \rangle = 0$.

Since $\gamma'(0)$ is tangent to the level set, and this holds for any curve on the level set, the gradient is orthogonal to the level set.

**Geometric picture:** Think of a topographic map. Level sets are contour lines. The gradient at any point is perpendicular to the contour line, pointing uphill.

## The Chain Rule: Composition of Linear Maps

If differentiation gives you linear approximations, what happens when you compose functions?

Suppose $f: \mathbb{R}^n \to \mathbb{R}^m$ and $g: \mathbb{R}^m \to \mathbb{R}^p$. The composition $g \circ f: \mathbb{R}^n \to \mathbb{R}^p$ is defined by $(g \circ f)(\mathbf{x}) = g(f(\mathbf{x}))$.

Near $\mathbf{a}$:
- $f$ is approximately linear: $f(\mathbf{a} + \mathbf{h}) \approx f(\mathbf{a}) + Df(\mathbf{a})(\mathbf{h})$
- $g$ is approximately linear near $f(\mathbf{a})$: $g(\mathbf{y} + \mathbf{k}) \approx g(\mathbf{y}) + Dg(\mathbf{y})(\mathbf{k})$

Composing these approximations:

$$
g(f(\mathbf{a} + \mathbf{h})) \approx g(f(\mathbf{a}) + Df(\mathbf{a})(\mathbf{h})) \approx g(f(\mathbf{a})) + Dg(f(\mathbf{a}))(Df(\mathbf{a})(\mathbf{h}))
$$

The linear approximation of $g \circ f$ is the composition of the linear approximations:

$$
D(g \circ f)(\mathbf{a}) = Dg(f(\mathbf{a})) \circ Df(\mathbf{a})
$$

In terms of matrices:

$$
J(g \circ f)(\mathbf{a}) = Jg(f(\mathbf{a})) \cdot Jf(\mathbf{a})
$$

This is the **chain rule**: the Jacobian of a composition is the product of the Jacobians.

**Fifth insight:** The chain rule is just the fact that composing linear maps corresponds to multiplying their matrices. There's nothing to memorize—it follows inevitably from the definition of the differential as a linear approximation.

### Dimensional Analysis

The dimensions work out perfectly:
- $Jf(\mathbf{a})$: $m \times n$ (maps $\mathbb{R}^n \to \mathbb{R}^m$)
- $Jg(f(\mathbf{a}))$: $p \times m$ (maps $\mathbb{R}^m \to \mathbb{R}^p$)
- Product: $p \times n$ (maps $\mathbb{R}^n \to \mathbb{R}^p$)

The intermediate dimension $m$ gets "contracted away" in the matrix multiplication, just as the intermediate space $\mathbb{R}^m$ is traversed internally in the composition.

## The Hessian: When First Order Isn't Enough

The gradient tells us the slope—the first-order behavior. But what about curvature? Is the function bowl-shaped? Saddle-shaped? How does the gradient itself change?

For $f: \mathbb{R}^n \to \mathbb{R}$, we can ask: what is the differential of the gradient?

The gradient is a function $\nabla f: \mathbb{R}^n \to \mathbb{R}^n$. Its Jacobian is an $n \times n$ matrix:

$$
J(\nabla f)(\mathbf{a}) = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{bmatrix}
$$

This is the **Hessian** $Hf(\mathbf{a})$.

When the second partial derivatives are continuous, mixed partials are equal: $\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i}$. So the Hessian is symmetric.

**Sixth insight:** The Hessian is the Jacobian of the gradient. It tells you how the gradient changes as you move through space.

### Second-Order Approximation

With the Hessian, we get a second-order (quadratic) approximation:

$$
f(\mathbf{a} + \mathbf{h}) \approx f(\mathbf{a}) + \nabla f(\mathbf{a})^T \mathbf{h} + \frac{1}{2}\mathbf{h}^T Hf(\mathbf{a}) \mathbf{h}
$$

The new term $\frac{1}{2}\mathbf{h}^T Hf(\mathbf{a}) \mathbf{h}$ is a **quadratic form**. It captures how the function curves.

### Geometry of the Hessian: Eigenvalues and Curvature

Since the Hessian is symmetric, it has a full set of real eigenvalues and orthogonal eigenvectors. This spectral structure has beautiful geometric meaning.

Let $\lambda_1, \ldots, \lambda_n$ be the eigenvalues with corresponding orthonormal eigenvectors $\mathbf{v}_1, \ldots, \mathbf{v}_n$.

In the eigenbasis, the quadratic form becomes:

$$
\mathbf{h}^T Hf(\mathbf{a}) \mathbf{h} = \sum_{i=1}^n \lambda_i (\mathbf{h} \cdot \mathbf{v}_i)^2
$$

Each eigenvalue $\lambda_i$ tells you the **curvature** in the direction $\mathbf{v}_i$:
- $\lambda_i > 0$: function curves upward in direction $\mathbf{v}_i$ (like a bowl)
- $\lambda_i < 0$: function curves downward in direction $\mathbf{v}_i$ (like a dome)
- $\lambda_i = 0$: function is flat in direction $\mathbf{v}_i$ (no curvature)

### Critical Points and the Second Derivative Test

A **critical point** is where $\nabla f(\mathbf{a}) = \mathbf{0}$. At such a point, the first-order term vanishes, and the quadratic term dominates.

At a critical point:

$$
f(\mathbf{a} + \mathbf{h}) \approx f(\mathbf{a}) + \frac{1}{2}\mathbf{h}^T Hf(\mathbf{a}) \mathbf{h}
$$

The nature of the critical point depends entirely on the Hessian's eigenvalues:

- **All eigenvalues positive** (positive definite Hessian): local minimum—the function curves upward in every direction
- **All eigenvalues negative** (negative definite Hessian): local maximum—the function curves downward in every direction
- **Mixed signs**: saddle point—the function curves up in some directions and down in others

**Seventh insight:** The Hessian's eigenvalues are the principal curvatures. Its eigenvectors are the principal directions of curvature.

### The Condition Number and Optimization

The **condition number** of the Hessian is $\kappa = \lambda_{\max} / \lambda_{\min}$ (for positive definite Hessians).

When $\kappa \approx 1$: the curvatures are similar in all directions. Level sets near the minimum are nearly spherical. Gradient descent converges quickly.

When $\kappa \gg 1$: the curvatures differ wildly. Level sets are elongated ellipsoids. Gradient descent zigzags, converging slowly.

This is why **Newton's method** uses the Hessian. The Newton step is:

$$
\mathbf{h} = -Hf(\mathbf{a})^{-1} \nabla f(\mathbf{a})
$$

This adapts to the local curvature, taking larger steps in flat directions and smaller steps in curved directions. For a quadratic function, Newton's method finds the minimum in one step.

## Points and Directions: A Conceptual Distinction

As you internalize these concepts, a subtle but important distinction becomes clear: **points and directions are different kinds of objects.**

A point $\mathbf{a} \in \mathbb{R}^n$ is a location. A direction $\mathbf{h} \in \mathbb{R}^n$ is a displacement or velocity.

Notationally, they look the same—both are $n$-tuples of numbers. But conceptually:

- You can add two directions: "north + east = northeast"
- You can scale a direction: "twice as fast north"
- You cannot meaningfully add two points: what is "your location + my location"?

The set of all directions at a point $\mathbf{a}$ is called the **tangent space** at $\mathbf{a}$, written $T_\mathbf{a}\mathbb{R}^n$. For $\mathbb{R}^n$, every tangent space looks like $\mathbb{R}^n$ itself—but they're conceptually different copies attached to each point.

**The differential maps between tangent spaces:**

$$
Df(\mathbf{a}): T_\mathbf{a}\mathbb{R}^n \to T_{f(\mathbf{a})}\mathbb{R}^m
$$

The function $f$ tells you where points go. The differential $Df(\mathbf{a})$ tells you where directions go.

This perspective becomes essential when you move beyond flat Euclidean space to curved manifolds—but even in $\mathbb{R}^n$, keeping the conceptual distinction clear prevents confusion.

## Synthesis: The Hierarchy of Approximation

Let's step back and see the whole picture.

Differentiation builds a hierarchy of increasingly accurate approximations:

**Constant (zeroth-order):** $f(\mathbf{x}) \approx f(\mathbf{a})$
- No derivatives needed
- Only accurate extremely close to $\mathbf{a}$

**Linear (first-order):** $f(\mathbf{x}) \approx f(\mathbf{a}) + \nabla f(\mathbf{a})^T(\mathbf{x} - \mathbf{a})$
- Uses the gradient
- Captures slope/direction of change
- Good for small $\|\mathbf{x} - \mathbf{a}\|$

**Quadratic (second-order):** $f(\mathbf{x}) \approx f(\mathbf{a}) + \nabla f(\mathbf{a})^T(\mathbf{x} - \mathbf{a}) + \frac{1}{2}(\mathbf{x} - \mathbf{a})^T Hf(\mathbf{a})(\mathbf{x} - \mathbf{a})$
- Uses the Hessian
- Captures curvature
- Much more accurate

Each level adds more derivative information, giving a better local approximation.

## Summary: Owning the Concepts

Here's what we built, from the ground up:

1. **Differentiation is linear approximation.** We replace a complicated function with a simple linear one that matches it locally.

2. **The differential is a linear map.** For $f: \mathbb{R}^n \to \mathbb{R}^m$, the differential $Df(\mathbf{a}): \mathbb{R}^n \to \mathbb{R}^m$ is the linear map that best approximates $f$ near $\mathbf{a}$.

3. **The Jacobian is the matrix of the differential.** Its columns are the partial derivatives with respect to each input; its rows are the gradients of each output component.

4. **The gradient is the Jacobian transpose for scalar functions.** It points in the direction of steepest ascent and is orthogonal to level sets.

5. **The chain rule is matrix multiplication.** Composing functions means composing their linear approximations, which means multiplying Jacobians.

6. **The Hessian is the Jacobian of the gradient.** Its eigenvalues are curvatures; its eigenvectors are principal directions.

7. **Points and directions are different.** Functions map points; differentials map directions between tangent spaces.

These aren't arbitrary definitions—they're inevitable consequences of asking: "What's the best linear approximation?"

Once you see differentiation this way, the formulas become transparent. The gradient points uphill because that's what maximizes the inner product. The chain rule multiplies matrices because that's how linear maps compose. The Hessian determines critical points because curvature dominates when slope vanishes.

You don't need to memorize these facts. You can derive them yourself, starting from the simple idea: curves are hard, lines are easy, so let's approximate curves with lines.
