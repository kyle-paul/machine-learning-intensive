# **Preclass Session 9: Natural Language Processing**

# **Recurrent Neural Network (RNN)**
## **Vanilla RNNs: Basic Recurrent neural network**
- Large language models (LLMs) (e.g. ChatGPT) has the main principle of operation based on predicting the next word.

<center><img src=https://d1q4qwyh0q55bh.cloudfront.net/images/RWKrPMxCp3hrKi6psqp2MMBzGUb30uo6wVRHYuw1gX8uBMUF4eqkquQsKBHn58hK.gif?d=desktop-thumbnail></center>

- We can describe this problem as follows: given a lot of (billion) series of words $(x_t)^T_{t=1}$ of sample sentences on the internet. We need to train a predictor to correctly predict the probability of the following word $x_t$ in the sentence if we input a series of previous words (called **context**).
$$
\bf x_0 \text{(start)}, x_1, ... x_{t-1} \xrightarrow{\text{predictors}} x_t
$$

    1. First, we need to have embedding vector ${\bf z_t} \in \mathbb{R}^d$ to represent word $x_t$ (simplest example is one-hot-encoding)
    2. Next, we need embedding vector $\bf h_{t-1} \in \mathbb{R}^n$ to represent the series of previous words (context): $\bf x_0, x_1, ..., x_{t-1}$ to predict the following word $\bf \hat{x}_t$. Take an example with linear predictor ${\bf \hat{x}_t} = s(\bf U \cdot h_{t-1} + b)$.
    3. We have the ground truth (the real next word) from the sample sentence $\bf x_t$, we continue to create the embedding vector (hidden, history) $\bf h_t$ to represent the series of context $\bf x_0, x_1,.., x_t$

- Recurrent Neural Network ([RNN](https://en.wikipedia.org/wiki/Recurrent_neural_network)) is a classic model and it is easy to compute the embedding vector $\bf h_t$ for each series of context $\bf x_0, x_1,..., x_t$ by linear combination of embedding vector of the previous series context $\bf h_{t-1}$ with embedding vector $\bf x_t$ of the new word included.

$$
\bf h_t \leftarrow W \cdot z_t + H \cdot h_{t-1} + b \space | \space W \in \mathbb{R}^{n \times d}, z_t \in \mathbb{R}^d, H \in \mathbb{R}^{n \times n}, h_t \in \mathbb{R}^n
$$
- Note: 
    - Review: **feed-forward** neural networks (e.g. MLPs): $\bf h_t = \gamma(W z_t + b)$ with activation function $\gamma = \text{sigmoid, tanh, relu,...}$ (element-wise operation)
    - We expand the above formula to plug in embedding vector of the next word (by linear transform) in RNN: $\bf h_t = \gamma(W \cdot z_t + H \cdot h_{t-1} + b)$
    - This formula is of 1 vanilla RNN unit. We can rearrange it into $\bf h_t = \gamma (\bar{W} \left[\begin{matrix}   \mathbf{z}_t\newline\mathbf{h}_{t-1}\newline\end{matrix} \right] + b)$ with $\bf \bar{W} = [W, H] \in \mathbb{R}^{n \times (d + n)}$ and the input vector $\left[\begin{matrix}   \mathbf{z}_t\newline\mathbf{h}_{t-1}\newline\end{matrix} \right]$ is concatenated from $\bf z_t$ and $\bf h_{t-1}$.
    - Important notion: parameters $\bf W$ is reused through all time-step $\rightarrow$ called **weight-sharing in time**. 
    - The output $\bf h_{t-1}$ of the previous time-step is plugged in as the input for RNN to comput $\bf h_t$ of the next time-step $rightarrow$ **Recurrent connection**

<center><img src=https://d1q4qwyh0q55bh.cloudfront.net/images/NEdRKsROW2uxgUOUjCmCytlbgUfnJKHTgXPNeRTeHN3ay1Ql5FHy6VAxqHHNcmMx.png?d=desktop-thumbnail></center><br>

Note: the below image uses the math notations different from the our lecture, but you can figure it out easily based on the you knowledge of each component's function.
<center><img src=https://d1q4qwyh0q55bh.cloudfront.net/images/BAWEdRauwTwPOZylWaWaNbm9fqWNSQnxhOR6Q0OQk3a4pyrrKo2oOfuhfNjM5KBG.png?d=desktop-thumbnail></center>

# **Natural Language Processing (NLP)**
## **Terminology**
- Corpus: dataset of language (consist of multiple sentences or texts)
- Document: each sentence (or each text) in corpus
## **Method**
- There are many ways to digitalize document to make it an input for AI model. You have learn about `CountVectorizer` and `TF-IDF`. 
- There is another popular method is to use `Word Embedding` to represent document.
- There are few basic steps below to digitalize document input:
    1. Prepocessing
    2. Tokenizer
    3. Embedding / CountVectorizer, TF-IDF... (represented as embedding vector)

### **Preprocessing**
- The most popular way is to remove `stop words`, lemmatization
- Depends on the each problem we are solving, there are other ways: removing special characters (e.g. emoji), normalize text (`"123"` $\rightarrow$ `"one two three"`), or convert `"teencode"` into normal language. Specially, in some other language (not English), there is a phenomenon which many words onyl represent one word; in this case, we have to use Word Segmentator. For example, in Vietnamese, `"đẹp trai"` $\rightarrow$ `"đẹp_trai"` (handsome).

### **Tokenizer**
- The purpose of this step is to seperate each word in document and assign a different `id` to each word. We can seperate multiple words or single word, yet simplest way is `White Space Tokenizer` (e.g. `"I am a student"` $\rightarrow$ `["I", "am", "a", "student"]`)
- Then, create a dictionary to perform mapping unique words with a distinct `id`. The number of elements in this dictionary is called `vocab length`.

```python
word2index = {
 "good": 0,
 "bad": 1,
  ...,
 "movie": 3511,
  ...,
 "learning": 6000,
  ...
}

index2word = {
 0:, "good",
 1:, "bad",
 ...,
 3511: "movie",
 ...,
 6000: "learning",
 ...
}
```

- Finally, each word in a document is represented by each of its distinct id. For example, `"I am a a student"` $\rightarrow$ `[0, 10, 15]`. These id numbers are called token.
- AI language models will only receive input of the same length, but there will be documents with different length in the dataset. If we declare the maximum length is 100 token, we need to remove redundant token in long sentence/document and add padding to short sentence/document. For example:

```python
word2index = {
 "<pad>": 0,
 "good": 1,
 ...
}
```

### **Word Embedding**
- How to combine pretrained Word Embedding model into Tensorflow. In `tensorflow.keras`, thers is a layer called `Embedding`
    - Layer embedding can map (convert) one token in the dictionary into a **dense vector**. There are two primary components in `Layer Embebedding`:
    - Vocab size: the size (the number of words) of a dictionary. 
    - Embedding dimension: the size of representation space

```python
Embedding(
	input_dim = ..., # vocab size  
	output_dim = ..., # embedding dimension 
)
```
- For example, we create a layer embedding with `input_dim=6000` and `output_dim=50` (which means your dictionary has 6000 unique token, each token is represented as a vector of `50` dimensions)
- There will be a matrix created with `shape=(6000, 50)`; each value of this matrix is randomly initialized and each row of matrix is the representation of each token in the dictionary (corresponding to positions).
- Recall, the input of the model will be the result of the tokenizer `"I am a student"` → `[0, 10, 15]`. When putting vectors `[0, 10, 15]` into the Embedding layer, the Embedding layer will retrieve the vectors at lines `0, 10, 15` to represent the input sentence. At this point, the input of the AI model will be a matrix with `shape=(3,50)`. The numbers in the Embedding layer will be learned (updated) by Gradient Descent during training. Of course, in the Embedding layer, there will exist lines to represent special tokens (padding, ...)
- Instead of randomly initializing numbers in the Embedding layer, we can instantiate them from Pretrained Word Embedding. The problem is that the `vocab_size` of the Embedding layer will be different `vocab_size` of Pretrained Word Embedding, so we have to process this. Look at the pseudo code below:
```python
# Initialize an empty list my_embeddings
my_embeddings = []

#  Loop through each word in the dictionary
for word in dictionary:
    # If the current Word exists in Pretrained Word Embedding
    if word in pretrained_model_embedding:
        # Take out the representation vector and put it in the my_embeddings
        my_embeddings.append(word.representation_vector)
        
    else:
    # Randomly generated and put in my_embeddings
    my_embeddings.append(random(word))
    
# Casting my_embeddings into numpy array
my_embeddings = np.array(my_embeddings)
# Assign this numpy array to the Embedding layer of the tensorflow
```
- Then we have 2 options (for example, the problem of classifying Good - Bad customer comments):

    - Do not update the numbers in the Embedding layer, but only other weights in the model.
    - Update the numbers in Embedding layer. After training, we will receive a new Embedding, this new Embedding may no longer have the original properties of Pretrained Word Embedding, but only the best representation of words to produce the best classification results.