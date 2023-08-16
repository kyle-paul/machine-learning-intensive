# Session 6: Revision
## **Theory**
### **Question 1**
Consider the following applications, let's explain why accuracy is not the right metric, and precision or recall or both are very important. Note these systems the human staff reviewed:
- Car insurance application app using AI to detect and localize losses 
- Credit card fraudulent transaction detection [Kaggle Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Face-recognition attendance checking system

### **Answer**
Use this hint to memorize better: Precision $\rightarrow$ Predicted, Recall $\rightarrow$ Missed. Also, we need to consider two settings, which are classification and retrieval (detect-and-classify)

- Car insurance application app using AI to detect and localize damages:
    - AI need to detect damages and then classify them (scratches, dents, rust,...)
    - Since one object is scarely damaged (most of cars, objects are normal when owners apply for an insurance policy) $\rightarrow$ accuracy is not a suitable metric for this case with unbalanced dataset
    - Low precision: normal parts are detected/misclassified as damaged ones $\rightarrow$ users review and edit $\rightarrow$ inconveninent
    - Low recall: A lot of damaged parts are dismissed $\rightarrow$ without the staff's awareness $\rightarrow$ mistaken application issurance, an critical issue.
    - Therefore, Precision should only moderate/relaxed, but Recall should be high (critical in this usecase).

- Credit card fraudulent transaction detection
    - Case of fraudulence is so scare in daily transactions $\rightarrow$ accuracy metric is unsuitable
    - Precision must be extremely high to avoid normal transactions being mispredicted as fradulent ones (disturb customers)
    - Recall must also be high to avoid missing any case of fraudulent transactions (financial losses for customers and complicated consequences)
    
- Face-recognition attendance checking system
    - Most of time, cameras record non-face images. Only in attendance checking time are face-recoginition and face-classification functions used.
    - High precision since detecting non-face into face image will cost more to continue classifying faces (costly when running on cloud and making system "busy"), or misclassifying person will be disturbing when it has to update the databasee.
    - Recall is low or high... based on product's direction. Recall can be low $\rightarrow$ no detection $\rightarrow$ user's adjustment of light, distance,.. However if we aim for the user experience $\rightarrow$ Recall must be high.


## **Summary**