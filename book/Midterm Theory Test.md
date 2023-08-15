# **Theory Midterm Test**

## **Theory 1 Representation**
### **Question 1**
Given a dataset $\bf D = \{x^1,...x^t,..., x^N\}$ with each $\bf x^t \in \mathbb{R}^d$. We need to use PCA to represent this dataset on the 2D graph. What is the shape of representation vector $\bf z^t$? What are the representation features? 

### **Question 2**
Pinpoint main differences between 2 embedding vectors $z_{pca}$ by PCA and $z_{sparse}$ by sparse coding.


### **Question 3**
If we use k-means clustering to represent to dataset with k centroids $\bf (m^1, m^2,...,m^k)$ with each centroid $m^i \in \mathbb{R}^d$, how are the coordinates $z^t_i$ from the embedding coordinate vector $\bf z^t$ calculated?


## **Theory 2 Recommender System**
### **Question 1**
Given the utility matrix as below
<center><img src=https://d1q4qwyh0q55bh.cloudfront.net/images/hL3EeAAWJ4bSgZLRTLlYy63KJOTnAejOY1QjTJLBsrqjz5kjaQrF0En6WxDQHBs0.png?d=desktop-thumbnail></center><br>

Measure the similarity between users $\bf A \& B$, $\bf A & C$.

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

### **Question 2**
Why we need classification losses while we already have evaluation metrics? How do we usually design these loss functions (interpolating from entropy loss & SVM/large-margin hinge loss)?


### **Theory 4 Multilayer Perceptron (MLP)**
### **Question 1**
Write a general mathematic formula of 1 MLP with input feature coordinate $\bf z=\phi(x)$. 2 hidden (fully-connected) layers $\bf \gamma(z^l) = \gamma(W^{l-1} z^{l-1} + b^{l-1}$ and one output softmax layer.

### **Question 2**
Explain why we need to use MLP after extractin d features $\phi_i$ with embedding coordinate vector $\bf z \in \mathbb{R}^d$


## **Theory 5 Optimizer - Gradient Descent**
### **Question 1**
In the Recommender System, supposing that we can extract $42$ features $\phi$ to represent each song: $\bf z = \phi(x) \in \mathbb{R}^{42}$. Then we use softmax regression $\bf \hat{y} = s(Wz + b)$ to classify ratings from $1$ to $5$ stars for each users. What is the shape of $\bf W,b$

### **Question 2**
If we use Gradient Descent to opimize/train the model to find the optimal model/parameters for $i$-th user, how many dimension does the gradient vector have?


### **Question 3**
List main differences between Batch Gradient Descent and Stochastic Gradient Descent. Are their gradient directions different and why?

## **Theory 6 Kernel Trick**
$\bf \mathcal{k}(x^1,x^2) = \phi(x^1) \cdot \phi(x^2)$
Why need to use kernel trick and why it is efficient?