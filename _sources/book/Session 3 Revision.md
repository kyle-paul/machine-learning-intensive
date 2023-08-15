# **Session 3: Revision**

## **Theory**
### **Question 1**
Collaborative filtering (user-based and item-based). Viết cụ các công thức Linear Regression dự đoán score $\hat{y}_{ij}$ của useser $u_i$ và item $m_j$. 
### **Answer**
Các công thức Regression dự đoán matching scores:
- Cosine Similarity giữa 2 vector:
$$\bf \text{cos_sim}(\mathbf{u}_1,\mathbf{u}_2)=\frac{\vec{u}_1\vec{u}_2}{ ||\mathbf{u}_1||_2.||\mathbf{u}_2||_2}$$
- Tính rating (user-based):
Generalized formula:
    $$\bf \frac{\sum( \text{rating} \space * \space \text{similarity})}{\sum|\text{similarity}|}$$
    
    Detailed formula:
    $$\bf \hat{y}_{u_i, m_j} = \frac{\sum_{k \in \mathcal{N(i)}} \bar{y}_{u_k, m_j} \text{cos_sim}(u_i, u_k)}{\sum_{} |\text{cos_sim}(u_i, u_k)|}$$
    
    $\bf \hat{y}_{u_i m_j}$ là rating cần tìm của user $\bf u_i$ cho item $\bf m_j$
    $\bf \bar{y}_{u_k m_j}$ là ratings của user $\bf u_k$ (nằm trong tập k neignbors $\mathcal{N}(i)$ có similarity gần với user $\bf u_i$) cho item $\bf m_j$
- Tính rating (item-based):
Tương tự:
$$\bf \hat{y}_{u_i, m_j} = \frac{\sum_{k \in \mathcal{N(j)}} \bar{y}_{u_i, m_k} \text{cos_sim}(m_j, m_k)}{\sum_{} |\text{cos_sim}(m_j, m_k)|}$$
    
    $\bf \hat{y}_{u_i m_j}$ là rating cần tìm của user $\bf u_i$ cho item $\bf m_j$
    $\bf \bar{y}_{u_i m_k}$ là rating cho item $\bf m_k$ (nằm trong tập k neignbors $\mathcal{N}(j)$ có similarity gần với item $\bf m_j$) đã được rated bởi uses $\bf u_i$

    
### **Question 2**
Model-based Content-based Recsys: dùng mô hình linear classifier để dự đoán ratings (1 $\rightarrow$ 5 stars) $\hat{y_{ij}}$ của useser $u_i$ và item $m_j$. Hãy thiết kế mô hình (task, experience, function, performance measure). Classifier là gì, bao nhiều class, training set, loss function nào.
### Answer
- **Task $\mathcal{T}$**: Cần đi dự đoán rating $y_{u_im_j}$ của user $\bf u_i$ khi biết vector feature của item $\bf m_j$
- **Experience $\mathcal{E}$**: là bộ utility matrix được cho ban đầu có các hàng là item và các cột là user.
- **Function Space ($\mathcal{F}$)** : Nếu xem vector feature của item $m_j$ là $z^j = (z_1^j,...,z_d^j)$ thì ta dùng classifier sau: $w_i \cdot z_j + b_k = \theta_i \cdot z_j = y_{u_i m_j}$ là weighted sum score hay rating. Với $w_i$ là model của user $i$. Ở đây ta cần tối ưu $d$ tham số trong $w_i$ và một bias $\rightarrow$ $d+1$ parameters need optimizing 
- **Performance Measure ($P$)**
Hàm mất mát $L$ (Ridge Regression)
$$
\mathcal{L}_i = \frac{1}{s_i} (\frac{1}{2}  ||\hat{\mathbf{Z}}_i\mathbf{\theta}_i  - \hat{\mathbf{y}}_i||_2^2 + \frac{\lambda}{2} ||\mathbf{\theta}_i||_2^2)
$$

    - Norm (Chuẩn) 2 được sử dụng để lấy tổng của các căn của cái sai số khi hệ trọng số $\theta_i$ được sử dụng. Sau đó bình phương mất căn.
    - $\bf Z_i$ là ma trận với các hàng (vector feature) của các item (hoặc item's profile) được rated bởi user thứ $i$
    - $\bf \hat{y}_i$ là rating đã biết từ utility matrix của user thứ $i$
    - Regularization term được áp dụng tránh overfitting giống $l_2$ với lambda $\lambda$ dương để điều chỉnh mức độ ảnh hưởng của regularization term. Hoặc nếu là Linear Rgression thì Loss function $L$:
$$
\mathcal{L}_i = \frac{1}{s_i}\frac{1}{2}  ||\hat{\mathbf{Z}}_i\mathbf{\theta}_i  - \hat{\mathbf{y}}_i||_2^2
$$
    - Sau đó lấy trung bình (**mean**) bằng cách chia cho $s_i$ là tổng số lượng các item mà user thứ $i$ rated.

## **Summary - Linear Predictors for RecSys**
### **1. Long-tail phenomenon**
- [The Pareto principle (or the 20/80 principle)](https://en.wikipedia.org/wiki/Pareto_principle)
- Recommend **popular** vs. highly **personalized** item or **Head-tail** vs. **Long-tail** products: mặc dù Long-tail products là niche items (less popular, less sales volumn) nhưng sẽ tăng customer experience và satisfaction $\rightarrow$ tăng loyalty 
### **2. Recsys**
- Key idea: similarity: **Linear Regression** để dự đoán rating score của users cho cách unrated item
#### **2.1 Neighborhood-based Collaborative Filtering**
- **Utility Matrix** and Embedding vector:
    - Ex: item $i_m$ có vector embedding ${\bf z}_{i_m} = (u^1_{i_m},...,u^k_{i_m},...,u^n_{i_m})$ với $n$ là số users đã rated cái iem $i_m$. Trong đó $u^k_{i_m}$ là rating của user thứ $k$ cho item  $i_m$
    - Ex: user $u_k$ có vector embedding ${\bf z}_{u_k} = (i^1_{u_k},...i^t_{u_k},...,i^m_{u_k})^\top$ với $m$ là số items mà user $u_k$ rated. Trong đó $i^t_{u_k}$ là item thứ $t$ mà user $u_k$ rated.
- **User-based:** 
    - B1: Normalize uitility matrix bằng cách trừ giá trị trong mỗi cột cho mean của vector cột (user). 
    - B2: Cosine Similarity: tính cos_sim của từng vector cột của một user so với tất cả user còn lại bằng công thức: $\text{cos_sim}(\mathbf{u}_1, \mathbf{u}_2)
=  \frac{\mathbf{u}_1^T\mathbf{u}_2}{ ||\mathbf{u}_1||_2.||\mathbf{u}_2||_2}$
    - B3: Fill vào ô trống bằng Person corelation: $\frac{\sum( \text{rating} \space * \space \text{similarity})}{\sum\text{similarity}}$
        $$\bf \hat{y}_{u_i, m_j} = \frac{\sum_{u_k \in \mathcal{N(i)}} \bar{y}_{u_k, m_j} \text{cos_sim}(u_i, u_)}{\sum_{} |\text{cos_sim}(u_i, u_k)|}$$
    
        $\bf \hat{y}_{u_i m_j}$ là rating cần tìm của user $\bf u_i$ cho item $\bf m_j$
        $\bf \bar{y}_{u_k m_j}$ là ratings (already normalized) của user $\bf u_k$ (nằm trong tập k neignbors $\mathcal{N}(i)$ có similarity gần với user $u_i$) cho item $\bf m_j$
$\rightarrow$ ta có được **ma trận đường chéo**
- **Item-based:**
     - Tương tự như user-based nhưng vector bedding sẽ nằm ngang và tính mean theo vector ngang của item
     - Ta có thể tranpose Utility Matrix và tính như bình thường
     - (more efficient since **#users ≫ #items**).

#### **2.2 Matrix Factorization Colaborative Filtering**
- Utility matrix $\bf Y$ được phân tích thành tích của hai ma trận low-rank $\bf X$ và $\bf W$
$$
\mathbf{Y} \approx \left[ \begin{matrix}
\mathbf{x}_1\mathbf{w}_1 & \mathbf{x}_1\mathbf{w}_2 & \dots & \mathbf{x}_1 \mathbf{w}_n \newline
\mathbf{x}_2\mathbf{w}_1 & \mathbf{x}_2\mathbf{w}_2 & \dots & \mathbf{x}_2 \mathbf{w}_n \newline
\dots & \dots & \ddots & \dots \newline
\mathbf{x}_m\mathbf{w}_1 & \mathbf{x}_m\mathbf{w}_2 & \dots & \mathbf{x}_m \mathbf{w}_n \newline
\end{matrix} \right]
 = \left[ \begin{matrix}
\mathbf{x}_1 \newline
\mathbf{x}_2 \newline
\dots \newline
\mathbf{x}_m \newline
\end{matrix} \right]
\left[ \begin{matrix}
\mathbf{w}_1 & \mathbf{w}_2 & \dots & \mathbf{w}_n
\end{matrix} \right] = \mathbf{XW}
$$

$$\bf Y = XW$$

trong đó $\bf Y \in \mathbb{R}^{m \times n}, X \in \mathbb{R}^{m \times k}, W \in \mathbb{R}^{k \times n}$ với m là số item, n là số user.


- Learning $\equiv$ optimizing embeddings $z_{\text{user}_i}, z_{\text{item}_j}$ of "*latent* features" for all $i,j$.

#### **2.3 Content-based (CB, personalized) RecSys**

- ${\bf w}_k \cdot {\bf z}_i + b_k =$ user$_k$--item$_i$ weighted sum score $\approx$ rating (could be formulated as a regression or a classification problem). 
- Learning $\equiv$ optimizing $\theta_k$.