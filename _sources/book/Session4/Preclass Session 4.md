# **Preclass Session 4: NonLinear Predictor**

## **Transformations**
### **Linear Transformation: $\bf z' = Az$**
- Accompanied with the embedding vecor $\bf z = (z1,...,z_n)$ is an axis system, direction representing features $\bf \phi = (\phi_1,...,\phi_n)$. Similarly, accompanied with the $\bf z'$ is the axis system $\bf \phi'$
- If $\bf A=U_{n \times n}$ is an orthogonal matrix (as in PCA with centered input $\bf \dot{x} = x - \bar{x}$), the effect of $\bf z' = Uz$ is simply rotating $\bf z$ around the axis $O$ to get $\bf z'$. This is similar to rotating direction of axes $\phi$ of $\bf z$ around the axis $O$ but in the reverse direction, with the $\phi'$ is the row of the orthogonal matrix $\bf U$.
- if $\bf A=S_{n \times n}$ is an diagonal matrix: $\bf S_{n \times n} = diag(\lambda_1,...\lambda_n)$, the effect of $\bf z'=Sz$ is simply stretching/contracting each axis $\phi_i$ according to the inverse ratio $\bf \frac{1}{\lambda_i}$ to get a new axis $\lambda_i$ in a new system $\phi'$. This is similar to stretching/contracting each coordinate $\bf z_i' = \lambda_i z_i$ 
- If $A_{m \times n}$ is an random matrix, Singular Value Decompostion (SVD) gives us $\bf A = USV^\top$ including two transformations: rotation, flipping and a stretching/contraction. $\bf z$ will be rotated by $\bf V^\top$ and then stretched/contracted by $\bf S$,and finally rotated by $\bf U$ $\rightarrow$ we get $\bf z'$
<br>
<center><img src=https://d1q4qwyh0q55bh.cloudfront.net/images/q7jy9gUw2k9oRNTsM0ZoZOQqJfEAovAszBUB1ftlc7ipk2IgxMySysg2a7pjcYuH.png?d=desktop-thumbnail></center>
<br>
<center><img src=https://d1q4qwyh0q55bh.cloudfront.net/images/IKE6BEt4nwyL96iPXsrKOUuVFqMydOgt091ui8Xi3vDVQbkDVf1glIaal7zRKfyU.gif?d=desktop-thumbnail></center>

### **Nonlinear transformation $\bf z' = \gamma(Az + b)$**
- The rotation $\rightarrow$ stretching/contraction $\rightarrow$ rotation of coordinates $\bf z$ by $\bf z = Az$ in new axis system $\phi'$ .
- Move the coordinates by a segment of $\bf b$.
- Then each coordinate (element-wise) is "moulded" by thê activation function $\gamma$ to get the final $z'$ coordinates with the value in the desired range:
    - Sigmoid: [0,1]
    - Tanh: [-1,1] (see the image below)
    - Relu: [0, $\infty$]

<center><img src=https://d1q4qwyh0q55bh.cloudfront.net/images/6DBE2eEu33CcGZiSYCBfvJlcdERldVBvC5LRIy8ItS3xx5RXYQJOKvIBc0kguqCA.gif?d=desktop-thumbnail></center>

## **Kernels**
Supposing that we have two inputs $\bf x^1, x^2$ and the feature extractor $\phi$ $\rightarrow$ $\bf z^t = \phi(x^t)$
- In session 1, we know that this representation allows us to measure th similarity of $\bf x^1, x^2$ indirectly through dot product $\bf z^1 \cdot z^2 = \phi(x^1) \cdot \phi(x^2)$. Also, dot product in the space/axis system of coordinate z is used for linear predictors.
- However, instead of extracting features and then calculating the dot product in the coordinate embedding space, we will use the **nonlinear kernels** to measure the similarity right in the input space $\bf \mathcal{K}(x^1, x^2) = \phi(x^1) \cdot \phi(x^2)$. Especially, using the method usually simplify the computation (avoid complex computation in high-dimensional space)
- [**Polynomial Kernel**](https://en.wikipedia.org/wiki/Polynomial_kernel): 
$$\bf \mathcal{K}(x, x') = (x \cdot x' + c)^d \text{ with } x = (x_1,...,x_n)^\top \in \mathbb{R}^n$$

    With $n = 2, d = 2, c = 1$, we have quadratic kernel:
$$\bf \mathcal{K}(z, z') = (z_1z_1' + z_2z_2' + 1)^2 = 1 + (z_1z_1')^2 + (z_2z_2')^2 + 2z_1z_1' + 2z_2z_2' + 2z_1z_1'z_2z_2'.$$

    With $n$ and $d$ increase, the kernel would get more complicated.


