# **Session 8: Revision**

## **Theory**
### **Question 1**
General of CNN: ${\bf \hat{y}} = s(W^s \gamma(W^l * Z^l + {\bf b}^l) + {\bf b}^s)$ with convolutional layers $\gamma(W^l * Z^l + {\bf b}^l)$. Each row of $W^l$ is used as kernel $\mathcal{K_i} = {\bf w}^l_i$ to perform convolution (dot product + sliding) with low level feature $Z^l$ (or raw input $X$ at the first neuron layer $l = 1$), producing high level feature map $Z^{l+1}$. Explain briefly how convolution kernels works?
### **Answer**
- At layer $l = 1$:
    - Kernel ${\bf w}^l_i \in \mathbb{R}^{k \times k}$ sliding on an input image $X \in \mathbb{R}^{n \times n}$ and perform dot product (padding = same, stride = 1) $\rightarrow$ we get the feature map with size $\frac{n +2p - k}{1}+1$ (the same size input image). This high level feature map is add a bias $b^l_i$, producing $Z^{l+1} \in \mathbb{R}^{n \times n}$
    - If we have $W^l$ with $d$ rows $\rightarrow$ we have $d$ kernels $\rightarrow$ we have $d$ bias $b_i^l$ (i=1..d) stacked into vector $b^l$
    - Therefore, we have the $d$ output feature maps or output shape is $(n, n, d)$. These feature maps $Z^{l+1}$ is then perform next convolution or pooling
    - For example: $X^l \in R^{100 \times 100} \xrightarrow[\text{padding = 2, } \text{stride = 1}]{\text{32 kernels} \space 5\times 5} Z^{l+1} \in \mathbb{R}^{100 \times 100 \times 32} + {\bf b}^l \in \mathbb{R}^{32}$

### **Question 2**
Explain why the operation of convolutional kernels is called weight-sharing and its benefits? (compare with fully connected in MLP)

### **Answer**
- Convolution = Sliding + Dot product
    - Sliding: we use similar kernels and make it slide on the input image, but we only use a certain weights, $k^2$ numbers need optimizing if $W_i^l \in \mathbb{R}^k$ (e.g $9$ weights if the kernel is $3 \times 3$). Theses weight vector is then copied at every input's position to compute the output feature. $\rightarrow$ **weight-sharing in space**.
- However, with fully-connected layer, For each position in the input we need to learn 1 weight vector respectively, and the total number of weights/parameters to learn is equal to the $\text{output} \times \text{input size}$ $\rightarrow$ very large.
- Benefits: While $W$ in MLP need oprimizing $d \times n$ parameters, $W$ in convolution need optimizing $d \times k$ parameters with $k$ is kernel size ($k < n$). Fewer parameters means less data need for training (do not need augmentation technique or add more data) $\rightarrow$ reduce complexity $\rightarrow$ reduce overfitting.

### **Question 3**
Explain why the operation of convolutional kernels can create location/translation invariance and what is its effect? 
- Kernel sliding on the entire image $\rightarrow$ perform matching $\rightarrow$ extract local features. So it can recognize the same detail/feature/object regardless of where it appears in the input image. 
- One convolutional kernel $\mathcal{k}_i$ allows extracting/detecting one feature (is determined by the values of its parameters) at various positions in input thanks to the effect of sliding. The embedding coordinate $z^{l+1}$ in the high-level feature map $Z^{l+1}$ is of high value at the input position which pattern $\mathcal{K}$ locates (this value is maintained after max-pooling). This is called **invariance** with translation $\rightarrow$ can detect local feature in the input image despite its location.
- Its effect: Do not need use data augmentation or add more data with more variance to train, convolution can solve it $\rightarrow$ model is more simple but more efficient 

### **Question 4**
Effective receptive field. Calculate the ERF at layer $1$ for one pixel at layer $3$
<center><img src=https://d1q4qwyh0q55bh.cloudfront.net/images/aQY8ULRy1tggCe7LJVMAmB3pQvraMb2APACIaACjGt7zKf18CiW2fmORWlTODA1g.png?d=desktop-thumbnail></center><br>

$$
7 \times 7 \space \text{pixels}
$$

## **Summary - Convolutional Neural Network**
### **Tensors (image data)**
- 1D tensors = vectors $\in \mathbb{R}^d$
- 2D tensors = matrics $\in \mathbb{R}^{d \times n}$
- 3D tensors (e.g. RGB images) $\in \mathbb{R}^{m \times n \times k}$
- multi-dimensional tensors $T = [a_{ijkl...}] \in \mathbb{R}^{i \times j \times k \times l ...}$

### **Template matching** 
**1. Dot product (scaler product)** (not generalized dot product)
- $\mathbf{z} \cdot \mathbf{z'} = \mathbf{z'} \cdot \mathbf{z} = z_1z_1' + ... + z_nz_n' = \sum_{i=1}^nz_iz_i'$
    - The more positive the result of the dot product, the more similar of embedding coordinate vectors $\bf z$ and $\bf z$, which means the more similar of input $\bf x$ and $\bf x'$ when considering features $\phi_1,...,\phi_n$ being applied.
    - Conversely, the more negative the result of dot product, the more dissimilar of embedding coordinates $\bf z$ and $\bf z$. 
    - Specially, when the result of dot product is pretty small $\rightarrow$ two embedding coordinate vectors are unrelated to each other (orthogonal, uncorrelated).

**2. Template matching + nD Sliding = nD Convolution**
- Template matching + 1D Sliding = 1D Convolution
    - Sliding in 1 Dimension = time $\rightarrow$ Word Embedding
- Template matching + 2D Sliding = 2D Convolution
- Template matching + 3D Sliding = 3D Convolution

### **Feature Extraction: Difference between PCA and CNN**
- **PCA** 
    - Principal: $X = X_0 + \sum_{i=1}^n w_i U_i$ with $w_i = (X-X_0)\cdot U_i ~$ and $~ X,X_0,U_i\in\mathbb{R}^{m\times p}$ (already proved in Session 1 assignment)
    - Use the orthogonal matrix $\bf U$ with the shape similar to the image $\bf A$ $\rightarrow$ only extract **global** features, **not local** ones.

- **Convolution**:
    - Principal: Kernel $\mathcal{K}$ slide on $\bf A$ and then perform pattern matching with dot product at each position the kernel is placed $\rightarrow$ producing a new image called feature map $\bf Z$
    - Can extract **local** features (similar to human eyes when we need to observe closely to details to classify things)

### **CNN operations**
- Padding: Maintain or increase input size and avoid data leakage on the edge of input image
- Stride: (bước sải) $\rightarrow$ movement of filters & the rate of how input image gets smaller
- Pooling: 
    - Decrease the size of input
    - Feature detector (still stable with small  deviations)
    - Max pooling & average pooling
    - Implemented as $2 \times 2$ kernel with stride 2
- Formula to calculate the output size:
$$
n_{out} = \frac{n_{in} + 2*\text{paddding} - \text{kernel size}}{\text{stride size}} + 1
$$
- Sliding: we use similar kernels and make it slide on the input image, but we only use a certain weights (e.g 9 weights if the kernel is $3 \times 3$) $\rightarrow$ **weight-sharing** & location/translation invariance.

### **Flattening**
- Generalized dot product $\xrightarrow{\text{flattening}}$ dot product
- Flattening or vectorization is when we convert an array/tensor $\bf A$ of shape $m \times n \times p$... into a column of vector $\bf a$ containing all elements $a_{ijk}$ of $\bf A$ (hence $l$-dimensional vector $\bf a$ has $l = mnp...$) By vectorizing or flattening 2 arrays of same shape $\bf A$ and $\bf B$ into $\bf a$ and $\bf b$, we can compute their dot product as $\bf A \cdot B = a \cdot b = a^T b$