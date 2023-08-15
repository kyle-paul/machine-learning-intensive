# **Session 2: Revision**

## **Theory**
### **Question 1:**
Classification Decision Boundary is a set of points that we can arrange into several classes/groups at the same time, that is, those with predicted scores of more than 1 class/group are the highest & equal. Demonstrate that with activation/transfer function, $s()$ is non-linear & monotonous (only increases or decreases as input data increases) such as tanh, sigmoid, softmax, decision boundary of linear classification models $s(Wz + b)$ are linear lines.


### **Answer:**
Decision Boundary is in the form: $\bf Wx + b = 0$ or if we combine bias into $\bf W$ $\rightarrow$ $\bf Wx = 0$ (this is the formula of linear equation/function). Because when appplying $\bf sigmoid(Wx) = sigmoid(0) = \frac{1}{1 - e^{-0}}= 0.5$ $\rightarrow \bf x$ can not be classified (located on the decision boundary)

Because the activation function $s$ is monotonous (only increase or decrease if the input increases). But with softmax, fishy, sigmoid increases if input increases **(covariate)** so that it keeps the order of the input value. For example: $\bf x > y$ $\rightarrow \bf s(x) > s(y)$. With softmax, if there is an input $\bf x$ plugged into $\bf s(W^Tx)$ $\rightarrow \bf x$ is predicted into class $\bf i$ if $\bf s(W^T_ix) > s(W^T_cx)$ with $\bf c$ are the remaining classes:

$$\bf s(W^T_ix) \geq  s(W^T_cx)$$

$$\bf \leftrightarrow W^T_ix \geq  W^T_cx$$

$$\bf \leftrightarrow W^T_ix - W^T_cx \geq 0$$

$$\bf \leftrightarrow (W^T_i - W^T_c)x \geq 0$$

Looking at the formula, we see $\bf W^T_i = W^T_c$, which means $\bf (W^T_i - W^T_c)x = 0$. This is the linear equation/function (decision boundary) when one datapoint $\bf x$ can not be determined to be belong to any class.


## **Summary Session 2 - Linear Predictors**

### **From feature extraction to Prediction**
Input ${\bf x}\xrightarrow[\varphi_1,\dots,\varphi_n]{\text{feature extractors}}$ embedding coordinates ${\bf z} \xrightarrow{\text{predictors}}$ output $\hat{y}$

Ta trích xuất đặc trưng từ input $\bf x$ bằng các hàm $\bf \phi$ ra được các tọa độ nhúng $\bf z$ rồi sau đó áp $\text{predictors}$ để cho ra $\bf \hat{y}$

### **Use dot product to predict $\bf \hat{y}$**

$\text{input } \bf x$ $\xrightarrow[\phi_1\dots\phi_d]{\text{fixed features}}$ embedding coordinates $\bf{z} \xrightarrow[\theta=(W,{\bf b})]{\text{linear-regressor}}$ regression value $\hat{y} = W{\bf z}+\bf b$

Simillarly, we use $\text{predictors}$ which is $\text{linear-regressor}$ with parameters need optimizing $\bf \theta$ including $\bf W$ and bias $\bf b$ then yield the regression value $\bf \hat{y}$ with the above formula.

### **Linear Regression**
#### **Singular Linear Regression**
- 2D: $\bf y = wz + b$ (straight line)
- 3D: $\bf y = w_1z_1 + w_2z_2 + b$ (plane)
- more than 3D: hyperplane
#### **Muliple Linear Regression**
$$
\begin{align}
\hat{y}_1 &= {\bf w}_1^\top{\bf z}+b_1 \\
\hat{y}_2 &= {\bf w}_2^\top{\bf z}+b_2 \\
 &\dots\\
\hat{y}_k &= {\bf w}_k^\top{\bf z}+b_k
\end{align}
$$

$$\rightarrow \hat{{\bf y}} = W{\bf z}+{\bf b}$$


### **Classification**
#### **1. Regression on prediction score of each class**
| index | class | score |
| -------- | -------- | -------- |
|   0   |   cat   | $\hat{y}_0 = w^\top_0 \cdot z + b_0$ |
|   1   |   dog   | $\hat{y}_1 = w^\top_1 \cdot z + b_1$ |
|   2   |   mouse   | $\hat{y}_2 = w^\top_2 \cdot z + b_2$ |
|   3   |   tiger   | $\hat{y}_3 = w^\top_3 \cdot z + b_3$ |

$$\text{vector} \space \bf\hat{y} = [\hat{y}_0, \hat{y}_1, \hat{y}_2, \hat{y}_3]$$

$$\text{matrix} \space \bf w = [w^\top_0; w^\top_1; w^\top_2; w^\top_3]$$

$$\text{vector} \space \bf b = [b_0, b_1, b_2, b_3]$$


#### **2. Convert prediction score to probability with softmax**
$$S({y_i}) = \frac{e^{y_i}}{\sum_{j=1}^k e^{y_j}}$$

$$\hat{y} = [\hat{y}_0, \hat{y}_1, \hat{y}_2, \hat{y}_3] \xrightarrow{\text{Softmax}} \hat{p} = [\hat{p}_0, \hat{p}_1, \hat{p}_2, \hat{p}_3] \xrightarrow{\text{argmax}} \text{label}$$

#### **3. Binary Cross Entropy**
$$E(y, \hat{y}) = -\sum_j y_j ln(\hat{y_i})$$


### **Optimization**
$\text{input } {\bf x} \xrightarrow[\phi_1\dots\phi_d]{\text{fixed features}}\text{ embedding coordinates }{\bf z} \xrightarrow[\theta=(W,{\bf b})]{\text{linear-predictor}}\text{ prediction }\hat{y} = s(W{\bf z}+\bf b)$

1. $N$ examples $\mathcal{D}=\{ (x^t,y^t) \}_{t=1}^N$
2. Pick a random $\hat{y} = f^0(z)$ with random $\theta^0 = (W^0,b^0)$ and then change step by step
3. Performance measure $P$ 
    - Regression $\rightarrow$ MSE (Mean Square Error)
    - Classification $\rightarrow$ (Cross Entropy)
4. Optimization algorithms (with gradient descent)    
    - Update $\theta$ based on metric, loss $\rightarrow$ get good $f$


### **TEFPA**
```mermaid
graph LR

Task --> Experience --> Function_Space --> Performance_measure --> Algorithm
```