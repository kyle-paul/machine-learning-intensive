# Session 5: Revision

## **Theory**
### **Question 1**
Giải thích ý nghĩa của công thức update parameters (learning) dùng gradient descent: $\theta \leftarrow \theta -\eta \nabla_\theta J$ (explain each symbol and its meaning)
### **Answer**
- Đây là công thứ để tìm ra bộ trọn số parameters $\theta$ mới mà tại đó value of cost function $J(\theta; X,y)$ nhỏ hơn bộ parameters $\theta$ cũ. 
    - $\eta$: learning_rate là tốc độ học - độ di chuyển nhanh hay chậm / lớn hay nhỏ của $\theta$ trong không gian tham số. 
    - $\nabla_\theta J$ là gradient vector: thể hiện cho hướng đi mà tại đó cost function $J$ tăng nhanh nhất - leo lên trên đỉnh (maximum rate of change):
     $$
    \nabla_\theta J(\theta;X,{\bf y}) = \Big(\frac{\partial}{\partial\theta_1}J_\theta, \dots, \frac{\partial}{\partial\theta_i}J_\theta, \dots, \frac{\partial}{\partial\theta_n}J_\theta\Big)
    $$
    Vì thế nếu ta muốn di chuyển (trượt) parameter $\theta$ đi xuống dưới để tìm global minimum thì cần trừ $\theta$ đi cái hướng leo lên (gradient vector) để đi xuống: $\theta -\eta \nabla_\theta J$

### **Question 2**
Khi ta áp dụng công thức này vào MLP ta cần 2 bước là feed forward and backpropagation. Hãy giải thích 2 bước này làm công việc gì dữa trên công thức trên.
### **Answer**
- Ở bước feed forward, model sử dụng hệ tham số $\theta$ hiện tại (nếu ở lần đầu là random) để tính ra giá trị đầu ra $\bf \hat{y}$ dựa vào input $\bf X$ (theo batch_size) và tính loss bằng function $J$ (MSE for regression, crossentropy for classification,..)
- Ở bước Backpropagation, ta tính đạo hàm một phần (partial derivative) của $J$ so với từng giá trị $\theta_i$ để tìm được gradient vector. 
- Áp dụng công thức trên: $J(\theta; X,y)$ nhỏ hơn bộ parameters $\theta$ để tìm được bộ $\theta$ mới tốt hơn.

## **Summary - Search: Optimization and Trainin**g
### **Refresh**
- TEFPA: The function space $f_\theta \rightleftharpoons  {\boldsymbol \theta} = (\theta_1, \dots,\theta_n)$
- How to find a good set of $\bf \theta = (W,b)$
    - Give the performance measure (metric, loss) $\mathcal{P}$ 
    - Optimization Algorithms $\mathcal{A}$ (e.g., GD, SGD, Adam, AdamW) can give a good set params $\theta^i = (W^i, b^i)$ at the $i$-th iteration.

### **Loss Function**
- Max performance = min cost/ loss: $\max_\theta\mathcal{P}(f_\theta,{\bf D}) \to \min_\theta J(\theta; \bf X, y)$
$$\theta^*  = \arg\min_\theta J(\theta; X,\bf y)$$

- Search / Optimization : di chuyển (trượt) trong không gian tham số $\bf \theta = (\theta_1, ... \theta_n) \rightarrow$ good function $f_\theta$ 
- Find the direction in không gian $\theta$ để có được $\theta_{i+1}$ tốt hơn $\theta{i}$: Follow the steepest direction $\rightarrow$ **Gradient**
    #### **Gradient vector**
    - The direction of maximum rate of change in function value:
    $$
    \nabla_\theta J(\theta;X,{\bf y}) = \Big(\frac{\partial}{\partial\theta_1}J_\theta, \dots, \frac{\partial}{\partial\theta_i}J_\theta, \dots, \frac{\partial}{\partial\theta_n}J_\theta\Big)
    $$
    
    $$
\text{with } \frac{\partial}{\partial\theta_i}J_\theta = \lim_{\Delta \theta_i\to 0}\frac{J(\theta_1,\dots,\theta_i + \Delta \theta_i,\dots,\theta_n) - J(\theta_1,\dots,\theta_i,\dots,\theta_n)}{\Delta \theta_i}
    $$
    
    - Gradient vector $\nabla_\theta J(\theta;X,{\bf y})$ = hướng của $\theta$ mà làm cho cost function $J$ tăng nhanh nhất (trong một thay đổi $\theta$ rất nhỏ). Vì thế ta cần trừ đi để đến điểm thấp nhất (ngược hướng đạo hàm) $\theta_{i+1} \leftarrow \theta_i -\eta \nabla_\theta J$
    - Optimal point $\nabla_\theta J(\theta^*;X,{\bf y}) \approx 0$ (hoặc cũng đúng với saddle, flat, local)
    
### **Realistic Loss function**
- Linear Predictors $\rightarrow$ often convex $\rightarrow$ có một global minimum
- Nonlinear Predictors $\rightarrow$ highly complex $\rightarrow$ highly noncovex $\rightarrow$ hard to search  
    #### Challenging issues:
    - Local optimum 
    - Saddle Point
    - Flat area
    - Narrow valleys
    
### **Generalization**
- Issues: underfitting (model is too simple) and overfitting (too complicated model trying to memorize training dataset)
- Variance - bias tradeoff $\rightarrow$ model "complexity" control: **regularization**
- Heuristics:
    - Early stopping (stop when validation loss increases)
    - Dropout: networks would learn more robust features
    - Cross validation: (hyperparameter tuning): **grid $\rightarrow$ average performance $\rightarrow$ best hyperparameters**



0.     
    - Fight overfitting: 
        - Feature selection
        - Weight regularization / weight sharing
        - Transfer & few-shot learning, data-augmentation
        - Ensemblee methods
        - Add more training data
        - Reduce model size
    - Underfitting:
        - Increase model capacity
        - Reduce Regularization
        - Error Analysis
        - Choose a more advanced architecture
        - Tune hyperparameters
        - Add features
    - Improving Search:
        - Weight initialization
        - Learning-rate decay (schedule)
        - Batch-normalization
        - Gradient clipping against exploding gradient
        - Skip connection against vanishing gradient
    
    - Addressing distribution shift:
        - Error Analysis
        - Synthesize data (by augmentation)
        - Domain application techniques
        
        
        
    