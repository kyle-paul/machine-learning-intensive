# **Preclass Session 8: Convolutional Neural Network**

## **Convolutional Neural Network**
### **Review part 1**
1. Pattern matchng for feature extraction: From the oberservation of input $\bf x \in \mathbb{R}^m$, we use $n$ features $\phi_1,...\phi_n$ to perform pattern matching with raw input $\bf x$ $\rightarrow$ producing embedding vector $\mathbf{z} = (z_1,...,z_n)$.
    - This process of sampling and matching is also called feature extraction, producing $\mathbf{z} \in \mathbb{R}^n$ or $z_i = \phi(\mathbf{x})$ with $i = 1,...,n$
    - For example, in PCA, each feature $\phi_i = \bf u_i$ and $z_i = \phi(\mathbf{x}) = \bf u_i \cdot \dot{x}$ with matching operation is the dot product between the principal component $\bf u_i$ with centered input $\bf \dot{x} = x - \bar{x}$ ($\bf \bar{x}$ is the mean of $\bf x$)

2. We can continue extract $d$ samples and perform matching right in the semantic space of extracted features $\phi_1,...,\phi_2$ to extract higher-level features.
    - In particular, we use features $\phi_1,..., \phi_d$ to perform pattern matching with $\bf z$, producing $\mathbf{z'} = (z_1',...z_d')$ with $z'_j = \phi_j(\mathbf{z})$
    - Another example, in Multilayer Perceptron with neural layers / fully-connected layers $\bf z' = \gamma(W^lz + b^l)$. Each row $\bf w_j$ of matrix $\bf W \in \mathbb{R}^{d \times n}$ is a high-dimensional feature $\phi_j$ (we call prototypes, landmarks) pattern matching with input (low-level feature cooridinates) $\mathbf{z} \in \mathbb{R}^n$ by dot product $z_j' = \phi(\mathbb{z}) = \gamma(\mathbf{z} \cdot \mathbf{z_j} + b_j)$. This process can be repeated (stacked neuron layers) to extract higher-dimensional features. And MLP is an example of model "Artificial Neural Network (ANNs)"

### **Review Part 2**
1. Dot product (scaler product) $\mathbf{z} \cdot \mathbf{z'} = \mathbf{z'} \cdot \mathbf{z} = z_1z_1' + ... + z_nz_n' = \sum_{i=1}^nz_iz_i'$
    - The more positive the result of the dot product, the more similar of embedding coordinate vectors $\bf z$ and $\bf z$, which means the more similar of input $\bf x$ and $\bf x'$ when considering features $\phi_1,...,\phi_n$ being applied.
    - Conversely, the more negative the result of dot product, the more dissimilar of embedding coordinates $\bf z$ and $\bf z$. 
    - Specially, when the result of dot product is pretty small $\rightarrow$ two embedding coordinate vectors are unrelated to each other (orthogonal, uncorrelated).

2. We can expand the notion of dot product for 2 tensors (e.g. 2 colored images) with the same shape/size: $\mathbf{A} \cdot \mathbf{B} = \sum_{ijjk} a_{ijk} b_{ijk}$. This is sum of element-wise products. 
    -  Dot product in math means sum of element-wise product of 2 same shape arrays (in code we call tensors) $\bf A,B$. Since $c = \mathbf{A \cdot B}$ is a scaler, it's also named scaler product, or inner product (collapsing dimensions, to differentiate with expanding dimensions by [outer product](https://en.wikipedia.org/wiki/Outer_product)). It's used as a similarity measure of 2 arrays (tensors) $\bf A$ and $\bf B$. It is also used to explain **convolutional operator** (dot product + sliding)
    -  In coding e.g `numpy.dot` means different things depending on the shapes of $\bf A$ and $\bf B$. For 2D array $\mathbf{A}^{m \times n}$ and $\mathbf{B}^{n \times p}$ it means matrix multiplication (which results in a matrix $\mathbf{C}^{m \times p}$), which is not a scaler.
    
    
3. Flattening or vectorization is when we convert an array/tensor $\bf A$ of shape $m \times n \times p$... into a column of vector $\bf a$ containing all elements $a_{ijk}$ of $\bf A$ (hence $l$-dimensional vector $\bf a$ has $l = mnp...$) By vectorizing or flattening 2 arrays of same shape $\bf A$ and $\bf B$ into $\bf a$ and $\bf b$, we can compute their dot product as $\bf A \cdot B = a \cdot b = a^T b$

<center><img src=https://d1q4qwyh0q55bh.cloudfront.net/images/JtY3EATzGim2Ojl3AmNsjfHzWxb0I3Y4ASLGtMIQteQpHFcrH5ogweioMf4Mj4tA.png?d=desktop-thumbnail></center><br>

Note: these above methods are very sensitive with the selection of origin (zero, centering) of coordinates.

### **Convolution**
- When two arrays/tensors are not in the same shape/size, e.g. $\bf A$ is a large colored image, $\mathcal{K}$ is a small colored image with size $3 \times 3$ called a kernel, template, feature, or filter. 
    - We can make kernel $\mathcal{K}$ slide on $\bf A$ and then perform pattern matching with dot product at each position the kernel is placed $\rightarrow$ producing a new image called **feature map** $Z$. This calculation is called convolution.
    - We can consider the feature map $Z$ is an image and continue using different kernels/filters $3 \times 3$ to slide and perform pattern matching to extract higher-level features $\rightarrow$ producing feature maps $Z'$.
    - We can take tons of kernels/filters $3 \times 3$ to extract tons of feature maps. Each kernel/filter can be designed/engineered with values of 9 numbers to create the desired effects. View more filters in [image processing](https://en.wikipedia.org/wiki/Kernel_(image_processing))
    - We can parameterize kernels/filters by optimizing $w_{11}, w_{12},..., w_{33}$ by learning from data input. The process in which we optimize parameters for kernels/filters and for the neural network is called end-to-end learning.

<center><img src=https://d1q4qwyh0q55bh.cloudfront.net/images/ZRmeTCCDT25fh2Qq1HSvFquHxQ4DrVKwVgIZMMZuekVM9Qn4HPjLZ1kYQsBtn28b.gif?d=desktop-thumbnail></center><br>

### **Arrays or Data Tensors**
The number of dimensions of array is the number of indices we have to use to access each element in that array/tensor.
- 1D tensors = vectors $\in \mathbb{R}^d$
- 2D tensors = matrics $\in \mathbb{R}^{d \times n}$
- 3D tensors (e.g. RGB images) $\in \mathbb{R}^{m \times n \times c}$
- multi-dimensional tensors $T = [a_{ijkl...}] \in \mathbb{R}^{i \times j \times k \times l ...}$

<center><img src=https://d1q4qwyh0q55bh.cloudfront.net/images/2ENBTdS41Z08prolFQGw9yhw0Co6BLblDX1RR5dSFDCYYjOgXI6Rs6XoxI9jRluj.png?d=desktop-thumbnail></center><br>

