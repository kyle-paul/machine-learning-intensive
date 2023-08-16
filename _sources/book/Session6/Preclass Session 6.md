# **Preclass Session 6: Metrics and Losses**

## **Precision, Recall and Accuracy**
### **Theory**
- **Binary Classification**: we assume that Binary Classifier as a detector
    - Predicted & ground-truth labels for a particular class is of the same type (Positive/1) or not of the same type (Negative/0/-1).
    - When predicted label = ground-truth label $\rightarrow$ True (correctly predicted), otherwise False (mispredicted).
    - Therefore we have 4 case occuring for each prediction: True Positive (TP), True Negative (TN), False Positive (FP), False Negative (FN).
    - False Positive (FP) is also called type-I error. False Negative (FN) is type-II error.
- **Precision and Recall**: 
    - Precision only considers the positive labels of predictions (all detections): 
    $$
    \text{Precision} = \frac{\sum{TP}}{\sum (TP + FP)}
    $$
    - Recall (sensitivity) only consider ground-truth labels (all ground-truth): 
    $$
    \text{Recall} = \frac{\sum{TP}}{\sum (TP + FN)}
    $$
    
- **It depends on the application to determine how low and high each value is**
<center><img src="https://d1q4qwyh0q55bh.cloudfront.net/images/mcGvf602jq0619QpmuYTnTQe3yYMHtPYlCEOH0DGAvQsF1gfkCHueBWqNlJ9U0qr.png?d=desktop-thumbnail"></center><br>

- **Other evaluation metrics for the Classification problem**.
$$
f_B \text{ score} = (1 + B^2)\frac{Precision \cdot Recall}{Precision + Recall} \text{ with } B \in [0,1]
$$

$$
f_1 \text{ score} = 2 \frac{Precision \cdot Recall}{Precision + Recall}
$$
    - True Positive rate (TPR) = recall (probability of detection)
    - False Positive rate (FPR) = posibility of false alarm
    $$\frac{TN}{TN + FP} = 1 - TNR$$
    - Mis-classification rate = 1 - Accuracy


#### **Question 1**

<center><img src="https://d1q4qwyh0q55bh.cloudfront.net/images/WLoCHbUNJEiHEZsDT3zBx9OwdGyZ7oCUBeLvO9qtCDBW3QfmmlqFOOpNN9e1O5Mb.png?d=desktop-thumbnail
"></center><br>


| | | 
| -------- | -------- | 
| True Positive     | 5     | 
| False Positive     | 45     | 
| True Negative     | 940     | 
| False Negative     | 10     | 



$$\text{Precision} = \frac{\sum{TP}}{\sum (TP + FP)} = 0.1$$

$$\text{Recall} = \frac{\sum{TP}}{\sum (TP + FN)} = 0.33$$

$$\text{Accuracy} = \frac{\sum{TP + TN}}{\sum (TP + FP + FN + TN)} = 0.945$$

$$FPR = \frac{TN}{TN + FP} = 0.95$$


#### **Question 2**
<center><img src="https://d1q4qwyh0q55bh.cloudfront.net/images/EQ3h60MgsXXuKur2DNLchwblYyEbVboTp22WPV4eJ2gfKB4mMUlHWQ2ZZsamgYBv.png?d=desktop-thumbnail
"></center><br>

##### **Cat detector**
| | | 
| -------- | -------- | 
| True Positive     | 3     | 
| False Positive     | 1     | 
| True Negative     | 3     | 
| False Negative     | 1     | 

$$\text{Precision} = \frac{\sum{TP}}{\sum (TP + FP)} = 0.75$$

$$\text{Recall} = \frac{\sum{TP}}{\sum (TP + FN)} = 0.75$$

$$\text{Accuracy} = \frac{\sum{TP + TN}}{\sum (TP + FP + FN + TN)} = 0.75$$

$$\text{F1} = 2\frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = 0.75$$

##### **Dog Detector**
| | | 
| -------- | -------- | 
| True Positive     | 3     | 
| False Positive     | 0     | 
| True Negative     | 4    | 
| False Negative     | 1     | 

$$\text{Precision} = \frac{\sum{TP}}{\sum (TP + FP)} = 1$$

$$\text{Recall} = \frac{\sum{TP}}{\sum (TP + FN)} = 0.75$$

$$\text{Accuracy} = \frac{\sum{TP + TN}}{\sum (TP + FP + FN + TN)} = 0.875$$

$$\text{F1} = 2\frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = 0.86$$

### **Object Detection & Information Retrieval**
- In the database (let take an example is a image with multiple objects) containing some items we consider (for example: cat) and the Object Detection Algorithm can:
    - Detect/select the defined object (Postives) or not detect
    - Indentify correctly (True Positive) or misidentify (False Positive) after detecting/selecting.
    - If there is no detection or selection, the predicted labels are Negatives. Depending on the ground-truth labels $\rightarrow$ False Negative or True Negative.

### **Support Vector Machine and Hinge Loss**

SVM is a linear classifier with prediction score $\bf \hat{y} = w \cdot z - b$ and predicted label is $\bf sgn(\hat{y})$. We use $\bf -b$ for the sake of convenience in math computation and the classification threshold is $\bf \hat{y} >< 0$

The idea is that ww will maximize the minimum margin to get large margin classifiers. This task requires 2 things:
- Predicted labels = Ground-truth labels $\bf sgn(\hat{y}) = y \in \{\pm 1\}$
- Absolute value of prediction score $\bf |\hat{y}| = |w \cdot z - b|$ must also be $> 0$.

Large Classification margin will be robust to noise: when input features have small noises $\bf z_i \leftarrow z_i + \epsilon$, the predicted label $\bf sgn(\hat{y})$ will not be changed sign.

Combining these two conditions, we get the Sign Prediction Margin $\bf m = y(w \cdot - b)$ must be $> 0$.

Next, we only consider $sign$, so we can scale prediction score $\bf \hat{y} = (w \cdot z - b)$ with a random postive integer e.g. $\bf \hat{y} = 100(w \cdot z - b)$ $\rightarrow$ get the same $sign$. Therefore, we should choose the scale for magin $> 1$ for convenience.