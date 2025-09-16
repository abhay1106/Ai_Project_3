# Sentiment Analysis Using Recurrent Neural Network

## **Problem Statement**
The goal of this assignment is to **build a sentiment analysis model** using a **Recurrent Neural Network (RNN)** to classify movie reviews from the **IMDB dataset** into **positive** or **negative** sentiments.

## **Dataset**
- Dataset: **IMDB reviews** (provided by Keras)
- Total samples: **50,000** (25,000 for training and 25,000 for testing)
- Labels:  
  - `0` → Negative review  
  - `1` → Positive review
- Preprocessing:
  - Only the **top 10,000 most frequent words** are considered.
  - The top **20 most common words** are skipped.
  - Reviews are encoded as sequences of integers (word indices).

## **Data Preprocessing**
```python
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# **Load dataset**
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000, skip_top=20)

# **Pad sequences so all reviews are of equal length**
x_train_padded = pad_sequences(x_train, maxlen=100, padding='post', truncating='post')
x_test_padded  = pad_sequences(x_test,  maxlen=100, padding='post', truncating='post')
```
# **Final shapes:**

x_train_padded.shape → (25000, 100)

x_test_padded.shape → (25000, 100)

y_train.shape → (25000,)

y_test.shape → (25000,)

# Model Architecture

## **A Sequential RNN model built with TensorFlow Keras:**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.layers import Dropout as dropout

vocab_size = 10000
embedding_dim = 32
maxlen = 100

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen),
    SimpleRNN(32, return_sequences=False),
    dropout(0.5),
    Dense(1, activation='sigmoid')
])
```
## **Model Summary**

- Embedding Layer: Converts word indices into 32-dimensional dense vectors.
- SimpleRNN Layer: Captures sequential dependencies (32 units).
- Dropout (0.5): Prevents overfitting.
- Dense Layer: Single neuron with sigmoid activation → outputs probability of positive sentiment.

## **Compilation**
```python
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
```
- Loss function: Binary Crossentropy
- Optimizer: Adam
- Metric: Accuracy

## **Training**
```python
history = model.fit(
    x_train_padded,
    y_train,
    batch_size=128,
    epochs=10,
    validation_split=0.2,
    verbose=2
)
```
- Batch size: 128
- Epochs: 10
- Validation split: 20% of training set used for validation

## **Evaluation**
```python
test_loss, test_accuracy = model.evaluate(x_test_padded, y_test, verbose=2)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
```
Reports the final accuracy on the test set.
Expected test accuracy: around 80–85% (may vary by run).

## **Expected Outcome**

- A trained RNN model capable of classifying movie reviews into positive or negative.
- Accuracy metric printed after evaluation (e.g., Test Accuracy: 83.45%).

## **Possible Improvements**

- Replace SimpleRNN with LSTM or GRU for better handling of long-term dependencies.

- Ensure consistent maxlen usage (same in padding and embedding input).

- Use EarlyStopping to avoid overfitting.

- Try pre-trained embeddings (e.g., GloVe, Word2Vec).

## Key Learnings
1. **Recurrent Neural Networks (RNNs):** Learned how RNNs capture sequential information in text data, making them suitable for sentiment analysis.  
2. **IMDB Dataset:** Gained experience in handling a real-world dataset of movie reviews with 25,000 training and 25,000 testing samples.  
3. **Data Preprocessing:** Understood the importance of restricting vocabulary size, removing overly common words, and padding sequences to ensure equal input length.  
4. **Embedding Layer:** Learned how embeddings convert sparse word indices into dense vector representations, improving model efficiency.  
5. **Regularization with Dropout:** Implemented dropout to reduce overfitting and improve model generalization.  
6. **Model Evaluation:** Practiced evaluating models on test data and monitoring validation accuracy to check overfitting/underfitting.  
7. **Limitations of SimpleRNN:** Noticed that while RNNs capture short-term dependencies, they struggle with long-term dependencies — highlighting the need for LSTM/GRU in future work.  

## Conclusion
This assignment successfully demonstrated the process of building a **sentiment analysis model** using **Recurrent Neural Networks** on the IMDB dataset.  
The model was able to classify reviews into positive or negative sentiments with a decent accuracy (~80–85%).  

Through this exercise, we understood the full workflow — from **dataset preparation** and **sequence padding** to **model building, training, and evaluation**. While the RNN approach provides a solid baseline, replacing it with **LSTM or GRU** and integrating **pre-trained embeddings** can significantly improve performance on more complex datasets.  

Overall, this assignment highlighted the importance of **sequence modeling** in NLP and served as a practical foundation for working with advanced architectures in deep learning.  
