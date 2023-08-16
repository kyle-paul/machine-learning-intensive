# **Session 12: final theory test**
## **Theory 1**
1. Build a face recogtnition system for attendance checking
    - Task: Recognize and classify faces
    - Experience: Image data with each sample $X \in \mathbb{R}^{m \times n \times k}$ and labels $\bf y$ with number unique values are number of attendees
    - Function space: MLP $\bf \hat{y} = s(W^{l+2} \gamma(W^{l+1} \gamma(W^l\phi(z) + b^{l+1}) + b^2) + b^{l+2})$
    - Performance Measure: we use loss metric instead accuracy $\rightarrow$ loss function "Categorical Cross Entropy"
    2. Algorithm: SGD or Adam, and use backpropagation to optimize parameters

2. Build a recsys
    - Task: e.g Classify rating $1 \rightarrow 5$ stars for an unknown item for a particular user
    - Experience: Utility matrix with columns are feature vector of users and rows are feature vector of items
    - Function space:  softmax regression $\bf \hat{y} = s(Wz + b)$
    - Performance Measure: we use loss metric instead accuracy $\rightarrow$ loss function "Categorical Cross Entropy"
    2. Algorithm: SGD or Adam, and use backpropagation to optimize parameters 

3. Customer Segmentation Application
    - Task:  Grouping customers based on their shared features
    - Experience:  matrix $X \in \mathbb{R}^{n \times d}$ with $n$ is number of sample (customers). Each row of $X$ has $d$ features. use KNN (need label), K-means (no need label)
    - Function space: KNN / K-means clustering
    - Performance Measure: MSE, MAE
    2. Algorithm: SGD or Adam, and use backpropagation to optimize parameters 

4. Sentiment Analysis model
    - Task: Classify input into 3 classes (good, bad, neutral)
    - Experience: Train data is tokenized sentence with each word in the sentence having an ID of $n$ features (embedding vector of that token)
    - Function space: RNNs (or LSTM, GRU) + Dense
    - Performance Measure: we use loss metric instead accuracy $\rightarrow$ loss function "Categorical Cross Entropy"
    - Algorithm: SGD or Adam, and use backpropagation to optimize parameters 

## **Theory 2**
1. Convolution = Dot product + Sliding: a feature map is obtained by applying a filter or a kernel to the input image or feature map and performing a dot product operation at each position. So we get the **feature** output image by **mapping = dot + sliding**

2. ${\bf \hat{y}} = s(W^s \gamma(W^l * Z^l + {\bf b}^l) + {\bf b}^s)$ with convolutional layers $\gamma(W^l * Z^l + {\bf b}^l)$. Each row of $W^l$ is used as kernel $\mathcal{K_i} = {\bf w}^l_i$ to perform convolution (dot product + sliding) with low level feature $Z^l$ (or raw input $X$ at the first neuron layer $l = 1$), producing highlevel feature map $Z^{l+1}$. This is the highly abstract features.
    - Kernel ${\bf w}^l_i \in \mathbb{R}^{k \times k}$ sliding on an input image $X \in \mathbb{R}^{n \times n}$ and perform dot product (padding = same, stride = 1) $\rightarrow$ we get the feature map with size $\frac{n +2p - k}{1}+1$ (the same size input image). This high level feature map is add a bias $b^l_i$, producing $Z^{l+1} \in \mathbb{R}^{n \times n}$
    - If we have $W^l$ with $d$ rows $\rightarrow$ we have $d$ kernels $\rightarrow$ we have $d$ bias $b_i^l$ (i=1..d) stacked into vector $b^l$
    - Therefore, we have the $d$ output feature maps or output shape is $(n, n, d)$. These highly abstract feature maps $Z^{l+1}$ is then perform next convolution or pooling


## **Theory 3**
**1. BPTT vs normal BPT**
- Output is recurrently used as input: Back-propagation through time (BPTT) is a technique of updating tuned parameters within recurrent neural networks (RNNs), which can process sequential data (in time series) 
- Difference:
    - Input and output is **separate/independent** in Norm BP $\rightarrow$ not work for sequential data, while BPTT allows the RNN to learn from the errors made at each time step and adjust its parameters accordingly.
    - BPTT has more problems: gradient vanishing, complex computations.
**
2. Weight-sharing in time in RNNs.**
    - The same weight matrices are applied to the input and the hidden state at every time step, dont care about the length and position of sequence.
    - Advantages:
        - Reduce complexity and computation time
        - Do not need to learn more parameter/weights when input changes (cause only use one and share for all)
        - Capture the pattern of sequential data. If the weights not shared $\rightarrow$ network treat each time step independently $\rightarrow$ ignore the sequential nature of the data.

**2. Classify sentiments**
- Output label $y = \arg\max {\bf \hat{y}} \in \mathbb{R}^3$ (3 classes: good, bad, neutral) is $i$-th having the greatest prediction score $\hat{y}^i$
- ${\bf \hat{y}} = s(W {\bf h} + b)$ is a softmax classifier with input is the hidden state ${\bf h}_t = \gamma(W {\bf h}_{t-1} + H {\bf z}_t + \bf b_h$) with $W \in \mathbb{R}^{n \times d}$,  ${\bf h}_{t-1}\in \mathbb{R}^n$, $\bf b \in \mathbb{R}^n$,  ${\bf z}_t \in \mathbb{R}^d$ is the embedding vector of $t-th$ word in the sentence (embedding vector has $d$ features)

## **Theory 4**
1. Equation for $G = \sum_{t=0}^9 \gamma^t r_t$, and $a_t = \pi(s_t)$ or $a_t = \arg \max Q(s,a)$
2. $Q^\pi(\text{home, wait-bus}) = \frac1{10} \sum_{k=1}^{10} \Big(10 \space \text{mins} + \gamma G^\pi_k(s') \Big)$
3. Prediction $\bf \hat{y}$ can be image (tensor) = state / environment that agent can observe $o_t$ and make optimal action under optimal policy $\pi*$


## **Theory 5**
1. This is the MDP planning problem because the agent need to estimate its chance of winning by simulating the future over many courses of actions based on the rewards provided. State: position of the "quân cờ" on the chess table. And action here is how to move each "quân cờ" (make its position change)
2. Range within $-1 \rightarrow +1$
3. 
