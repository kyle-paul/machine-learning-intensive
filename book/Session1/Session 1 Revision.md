# **Session 1: Revision**

## **Theory**
### **Question 1:**
Prove that for the pca: the i-th coordinate is: $z_i = \bf{u}_i \cdot \dot{x}$ with centered input $\bf \dot{x} = (x - \bar{x})$

### **Answer**
**Basic math transformation:**
- PCA decomposes $\bf x=\bar{x} + z_1u_1 + ... + z_iu_i + ... + z_nu_n$ with orthonomal basis $\bf U = (u_1, u_2, ..., u_n)$
- Then $\bf \dot{x} = x - \bar{x} = z_1u_1 + ... + z_iu_i + ... + z_nu_n$
- Now dot product both size with $\bf u_i$
$$
\bf u_i \cdot \dot{x} = z_1 u_1 \cdot u_i  + ... + z_i u_i \cdot u_i + ... + z_n u_n \cdot u_i
$$

$$
\bf u_i \cdot \dot{x} = z_1 0  + ... + z_i + ... + z_n 0 (u_i \cdot u_j = 0, \forall i \neq j \space \& \space u_i \cdot u_i = ||u_i|| = 1)
$$

**Convert embeddings in different coordinate systems:**
<center><img src=https://hackmd.io/_uploads/SyYCLYwdn.png></center>

(See the image above) in the coordinate $O \bf e_1e_2$, point $\bf x$ has the coordinate $(x_1, x_2)$. But in the coordinate system $O \bf u_1u_2$, point $\bf x$ has the coordinate $(y_1, y_2)$

Now w have an orthogonal matrix: $\bf U = [u_1, u_2]$ serves as a new base (coordinate) system và vector $\bf y = [y_1, y_2]$ is the coordinate that represents $\bf x$ in this new $\bf U$ system.
Thereforee $\bf x$ is represented as the following formula:
$$\bf x = y_1u_1 + ... + y_n u_n$$ In the image above: $n = 2$
$$\bf x = Uy$$

So we will find $\bf y$ by multiplying the two sides of the above expression by $\bf U^{-1}$:

$$\bf U^{-1}x = U^{-1}Uy$$

$$\bf U^{-1}x = Iy$$

$$\bf y = U^{-1}x$$


Because $\bf U$ is an orthogonal matrix $\rightarrow$ $\bf U^{-1} = U^T$ $\rightarrow$ $\bf y = U^{T}x$


This is similar to the formula of **Principal Component Analysis**:
We have the formula in the Preclass session:

With $n$ is the number of dimension in new $\bf U$ system
$$\bf x = \bar{x} + z_1 u_1 + \dots + z_n u_n$$

$$\bf x - \bar{x} = z_1 u_1 + \dots + z_n u_n$$

$$\bf x - \bar{x} = z_1 u_1 + \dots + z_n u_n$$

$$\bf x - \bar{x} = Uz$$

$$\bf \dot{x} = Uz$$

$$\bf z = U^{-1}\dot{x}$$

$$\bf z = U^{T}\dot{x}$$


### **Question 2:**
Go to the sklearn `pca.transform` to reconfirm the math formula and then using code to verify using math formula instead of using `pca.transform` and see the result.

### **Answer**
```python
from sklearn.decomposition import PCA
pca = PCA().fit(X)

U = pca.components_
x_bar = pca.mean_
x_dot = X - x_bar
z = np.dot(x_dot, u.T)
x_pca = pca.transform(X)
```
You can compare two vectors `z` and `x_pca`. Then plot the reconstructed image
```python
reconstruct = x_bar + np.dot(z, u)
fig = plt.subplot(1,2,1)
fig.imshow(reconstruct[0].reshape(h, w), cmap="gray")
plt.show()
```

## **Summary - Representation**
### **Feature/basis $\rightarrow$ embedding coordinates**
#### 1. Fantastic function 
$$\text{real-word object} \xrightarrow[\text{brain & computer}]{\text{represented in}} \text{concepts, abstract "objects", thoughts}$$

#### 2. Feature Extraction
$$
\text{input } {\bf x} \xrightarrow[(\phi_1\dots\phi_n)]{\text{feature detectors}} \text{levels of activation/presence } {\bf z} =(z_1, \dots, z_n)
$$
#### 3. Similarity measure
- 17 types of similarity measures
#### 4. Manipulating "semantic" space $\bf z$
- "concept" formation in coordinate / semantic space: k-means clustering

#### 5. Predictors:
$$\text{fantastic function } f = \text{ feature extractors } \varphi + \text{ predictor } p$$

$$\text{input } {\bf x}\xrightarrow[\varphi_1,\dots,\varphi_n]{\text{feature extractors}} \text{embedding coordinates } {\bf z} \xrightarrow{\text{predictors}} \text{output } \bf \hat{y}$$ 


#### 6. Deep generative AI
$$\text{fantastic function } f = \text{ encoder } \varphi + \text{ decoder/generator } g$$

$$
{\bf x}\xrightarrow[\varphi]{\text{encoder}} {\bf z} \xrightarrow{\text{control}} {\bf z}' \xrightarrow[g]{\text{decoder}}
$$

### **Features/Kernels/Template/filter/basis** 
#### **1. Principal Component Analysis (PCA)**
- $\mathsf{Face} = \text{ Mean Face } + \sum_{i=1}^n w_i. i\text{-th } \text{Principal Face Component}$
- $X = X_0 + \sum_{i=1}^n w_i U_i$ with principal components $U_i$ (eigenfaces)
- Keep only first $k$ components <span><!-- .element: class="fragment highlight-yellow" -->$\to {\bf z}=(1, w_1,\dots, w_k)^{\top}$: dimensionality reduction.</span>

#### **2. Sparse encoding**
<span><!-- .element: class="fragment highlight-yellow" -->Embedding coordinates with very few nonzeros (i.e., sparse) coefficients $[a_1, \cdots, a_{64}] = [0, 0, \cdots, 0, 0.8, 0, \cdots, 0, 0.3, 0, \cdots, 0, 0.5, 0]$.</span>