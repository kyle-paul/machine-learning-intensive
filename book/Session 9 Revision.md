# **Session: 9 Revision**

## **Theory**
<center><img src=https://d1q4qwyh0q55bh.cloudfront.net/images/BAWEdRauwTwPOZylWaWaNbm9fqWNSQnxhOR6Q0OQk3a4pyrrKo2oOfuhfNjM5KBG.png?d=desktop-thumbnail></center>

### **Question 1**
Design a formula to preict output $o_t$ for the problem **NER** (name-entity recognition) simply based on the above diagram/flow chart. 

### **Answer**
output label $o_t = \arg\max_{i \in \text{NER}} {\bf \hat{y}}_t$ is $i$-th having the greatest prediction score $\hat{y}_t^i$ with ${\bf \hat{y}}_t = s(W {\bf h}_t + b_y$ is a softmax classifier with input is the hidden state ${\bf h}_t = \gamma(V {\bf h}_{t-1} + U {\bf z}_t) + b_h$ with ${\bf z}_t$ is the embedding vector of input ${\bf x}_t$ 

### **Question 2**
Why weight-sharing principal (in time for RNNs and on space for CNNs) is the latent feature to counter overfitting?

### **Answer**
- Weight-sharing makes the model still powerful in the task of feature extraction (representation power), but still simple and "light-weight" (need optimizing few parameters $\rightarrow$ low complexity). Due to this low complexity $\rightarrow$ reduce overcomplications of model $\rightarrow$ latently countering overfitting./ 

## **Summary - Recurrent Neural Network**