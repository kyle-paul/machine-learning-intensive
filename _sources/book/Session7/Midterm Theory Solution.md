# Theory Midterm Solution

## **Theory 1 Representation**
### **Question 1**
Given a dataset $\bf D = \{x^1,...x^t,..., x^N\}$ with each $\bf x^t \in \mathbb{R}^d$. We need to use PCA to represent this dataset on the 2D graph. What is the shape of representation vector $\bf z^t$? What are the representation features? 
### **Answer**
On the 2D-graph, we need 2 axes (2 features), which are 2 first principal components $\bf u_1$ and $\bf u_2$ of PCA. Hence $\mathbf{z} = (z_1, z_2) \in \mathbb{R}^2$
s
### **Question 2**
Pinpoint main differences between 2 embedding vectors $z_{pca}$ by PCA and $z_{sparse}$ by sparse coding.
### **Answer**w
$z_{pca}$ only has $k$ principal components and most of them are different from $0$, while $z_{sparse}$ has the length of dictionary size but most of them are $0$, only some components are different from $0$ $\rightarrow$ that's the reason why it called sparse.

### **Question 3**
If we use k-means clustering to represent to dataset with k centroids $\bf (m^1, m^2,...,m^k)$ with each centroid $m^i \in \mathbb{R}^d$, how are the coordinates $z^t_i$ from the embedding coordinate vector $\bf z^t$ calculated?
### **Answer**
If we assign each raw input $\bf x^t$ into only one nearest cluster (hard assignment) $c = argmin_j d(x^t, m^j)$, we can represent $\bf z^t$ is one-hot vector $z^t_i = 1$ at the neerest centroid $i = c$ and zeros at other coordinates.
Otherwise, we can use similarity scores, e.g. cosine cos$(\bf x^t, m^j)$ or normalized distance $\frac{d(x^t, m^j)}{d_{max}}$ between input $\bf x^t$ and centroid $\bf m^j$ to represent coordinate values, called soft assignments: $z_j = s_j$ with $j = 1...k$


## **Theory 2 Recommender System**
### **Question 1**
Given the utility matrix as below
<center><img src=https://d1q4qwyh0q55bh.cloudfront.net/images/hL3EeAAWJ4bSgZLRTLlYy63KJOTnAejOY1QjTJLBsrqjz5kjaQrF0En6WxDQHBs0.png?d=desktop-thumbnail></center><br>

Measure the similarity between users $\bf A \& B$, $\bf A \& C$.
### **Answer**
We will center each feature vector (based on items) of users and then calculate the cosine similarity.

$\mathbf{u_A} = (5,5,3,1,1) \xrightarrow{\text{centering}} \mathbf{u_A} = (2,2,0,-2,-2)$
$\mathbf{u_A} = (5,2.5,4,1,0) \xrightarrow{\text{centering}} \mathbf{u_A} = (2.5,0,1.5,-1.5,-2.5)$
$\mathbf{u_A} = (0,2.5,1,4,5) \xrightarrow{\text{centering}} \mathbf{u_A} = (-2.5,0,-1.5,1.5,2.5)$

Then we measure the cosine similarity
$$
s(u_A,u_B) = \frac{u_A \cdot u_B}{||u_A||_2 ||u_B||_2} \approx 0.78824078
$$

$$
s(u_A,u_C) = \frac{u_A \cdot u_C}{||u_A||_2 ||u_C||_2} \approx -0.78824078
$$
$\rightarrow$ $A \& B$ are quite similar and $A \& C$ are quite different.

### **Question 2**
Measure the similarity between items $\bf m_1 \& m_3$, $\bf m_3 \& m_4$.

$\mathbf{m_1} = (5,5,0,0,1,2.2) \xrightarrow{\text{centering}} \mathbf{u_A} = (2.8,2.8,-2.2,-2.2,-1.2,0)$
$\mathbf{m_3} = (2,4,1,2,2,1) \xrightarrow{\text{centering}} \mathbf{u_A} = (0,2,-1,0,0,-1)$
$\mathbf{m_4} = (1,1,4,4,4,2.8) \xrightarrow{\text{centering}} \mathbf{u_A} = (-1.8,-1.8,1.2,1.2,0)$

Then we measure the cosine similarity
$$
s(m_1,m_3) = \frac{m_1 \cdot m_3}{||m_1||_2 ||m_3||_2} \approx 0.61510829
$$

$$
s(m_3,m_4) = \frac{m_3 \cdot m_4}{||m_3||_2 ||m_4||_2} \approx -0.59628479
$$

## **Theory 3 Linear Classifiers**
### **Question 1**
Logistic Regression and SVM are also linear binary classifiers with the same formula for prediction sore $\hat{y} = s(\mathbf{W} \cdot \mathbf{z} + b)$. What is the difference in the principal of each agorithm.
### **Answer**
- Transfer function $s$ is sigmois $\bf \sigma$ for logistic regression to get probability scores, whereas that for SVM is indentity/linear to get prediction "margin" (no probabilistic meaning). 
- Decision threshold of logistic regression is $0.5$, while that of SVM is $0$
- Loss function of losgitic regression is binary cross-entropy, while that of SVM is hinge loss.
- The objective of logistic regression algorithm is to find the  model/parameters/function space to classify (maximum likelihood) the dataset, while with SVM, we want to maximize the minimum margin

### **Question 2**
Why we need classification losses while we already have evaluation metrics? How do we usually design these loss functions (interpolating from entropy loss & SVM/large-margin hinge loss)?
### **Answer**
Evaluation metrics are often the noncontinuous parameterized function, having flat surface $\rightarrow$ difficult to for optimizers to perform operations (e.g. calcalating derivatives/gradients). In addition, it's hard to apply 'heuristics' into the model to get more generalization ability (inductive biases). For example, large-margin is more robust with noise, or regularizers reduce overfitting. Therefore, we usually approximate evaluation metrics with simpler and more smooth loss functions for more efficient optimization (or complementing sub-loss functions for better generalization ability).


### **Theory 4 Multilayer Perceptron (MLP)**
### **Question 1**
Write a general mathematic formula of 1 MLP with input feature coordinate $\bf z=\phi(x)$. 2 hidden (fully-connected) layers $\bf \gamma(z^l) = \gamma(W^{l-1} z^{l-1} + b^{l-1}$ and one output softmax layer.

### **Answer**
$$
\bf \hat{y} = s(W^3 \gamma(W^2 \gamma(W^1\phi(z) + b^1) + b^2) + b^3)
$$

### **Question 2**
Explain why we need to use MLP after extractin d features $\phi_i$ with embedding coordinate vector $\bf z \in \mathbb{R}^d$
### **Answer**
1. To reduce dimensions (compresing) of $\bf z$ by using matrix $\bf W \in \mathbb{R}^{m \times d}$ with $m < d$ if embedding coordinates $\bf z$ need refiltering.
2. To increase dimension (expanding) of $\bf z$ by using matrix $\bf W \in \mathbb{R}^{m \times d}$ with $m > d$ to transfer/transform $\bf z$ space into $\bf z'$ space to easily apply linear predictor in higher-dimensional space but nonlinear in origin space.

## **Theory 5 Optimizer - Gradient Descent**
### **Question 1**
In the Recommender System, supposing that we can extract $42$ features $\phi$ to represent each song: $\bf z = \phi(x) \in \mathbb{R}^{42}$. Then we use softmax regression $\bf \hat{y} = s(Wz + b)$ to classify ratings from $1$ to $5$ stars for each users. What is the shape of $\bf W,b$
### **Answer** 
- Since we have to classify $5$ classes $\rightarrow \bf \hat{y} \in \mathbb{R}^5 \rightarrow b \in \mathbb{R}^5$ 
- $\bf W \in \mathbb{R}^{5 \times 42}$ is a model/parameter of one user with each row is a vector $\bf w_j \in \mathbb{R}^{42}$, which is the model of that user for $j$-th song.

### **Question 2**
If we use Gradient Descent to opimize/train the model to find the optimal model/parameters for $i$-th user, how many dimension does the gradient vector have?

### **Answer** 
We have the math formula of gradient vector:
$$
\nabla_\theta J(\theta;X,{\bf y}) = \Big(\frac{\partial}{\partial\theta_1}J_\theta, \dots, \frac{\partial}{\partial\theta_i}J_\theta, \dots, \frac{\partial}{\partial\theta_n}J_\theta\Big)
$$

So the number of dimension of gradient vector is equal to the number of parameters need optimizing: $5 \times 42 + 5 = 251$

### **Question 3**
List main differences between Batch Gradient Descent and Stochastic Gradient Descent. Are their gradient directions different and why?
### **Answer** 
- Batch-GD computes the gradient vector $\nabla_\theta L$ with the loss function $L(\theta; D)$ on all samples of dataset $D = \{(x_t,y_t)\}_{t=1}^N$, while SGD computes the gradient vector $\nabla_\theta l_t$ with the loss function $l_t(\theta; (x_t,y_t))$ on each training data sample $(x_t,y_t)$. Hence:
    - Batch-GD is more accurate with less iterations/epochs but suffers large computation $\rightarrow$ needs more time for each iteration (sometimes encounters computation error since dataset is too large to be stored in RAM)
    - SGD is only approximate (stochastic) of batch-GD $\rightarrow$ not produce exact updating direction, need more iterations, but fast computation time (and fixed for each iteration). Additionally, SGD is also a good choice for better generalization ability.

## **Theory 6 Kernel Trick**
$\bf \mathcal{k}(x^1,x^2) = \phi(x^1) \cdot \phi(x^2)$
Why need to use kernel trick and why it is efficient?
### **Answer**
Kernel Trick is used to:
- Transform a linear predictor $\hat{y} = s(\mathbb{w} \cdot \mathbb{z} + b)$ in a feature space $\bf z$ into a nonlinear predictor $\hat{y} = s(\mathbb{w} \cdot \phi(\mathbb{x}) + b)$. However, with kernel tricks, we do not need to compute (bypass) complex operations in the high-level dimensional space by transforming the raw input $\bf x$ to embedding $\bf z =\phi(x)$
- By exploiting the kernel fnuction $\bf \mathcal{k}(x^1, x^2)$ to calculate the nonlinear similarity of each pair of raw input, which is similar to the linear similarity of each embedding cooridnate vector $\bf z^1 \cdot z^2 = \phi(x^1) \cdot \phi(x^2)$. 