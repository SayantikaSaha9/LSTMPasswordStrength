# **LSTM Password Strength Model**  

A Deep Learning model using **LSTMs & BiLSTMs** to classify passwords as **Weak, Medium, or Strong** based on their security level.  

## **Overview**  
This project applies **Natural Language Processing (NLP) and Deep Learning** techniques to evaluate password strength. It uses:  
- **Character-level tokenization** for feature extraction  
- **Optimal sequence padding** to handle variable password lengths  
- **Bidirectional LSTM model** for sequence learning  

## **Project Structure**  
```
📁 LSTM-Password-Strength  
│── README.md  
│── requirements.txt  
│── LSTM_Password_Classifier.ipynb  # Main Jupyter Notebook  
│── data/  
│   └── password_data.sqlite  # SQLite database with passwords & strength labels  
│── models/  
│   └── trained_model.h5  # Saved LSTM model  
│── results/  
│   ├── accuracy_loss_plot.png  # Training evaluation  
│   ├── password_length_distribution.png  # EDA visualization  
│   └── sample_predictions.txt  # Model predictions  
```

## **Key Features**  
- **Deep Learning-based classification** with LSTM & BiLSTM models  
- **Character-level tokenization** for effective NLP processing  
- **Data visualization** to analyze password length distribution  
- **Fine-tuned hyperparameters** (LSTM units, dropout, batch size)  
- **Real-time password strength prediction**  

## **Data Preprocessing & Feature Engineering**  
- **Tokenization**: Converts passwords into numerical sequences  
- **Padding & Truncation**: Uses optimal sequence length (16) based on data analysis  
- **Train-test split (80-20)** with stratification to maintain class balance  

## **Model Architecture**  
- **Embedding Layer**: Converts character sequences into dense vectors  
- **BiLSTM Layer**: Enhances sequence learning  
- **Dense Output Layer**: Uses softmax activation for classification  

```python
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=16))
model.add(Bidirectional(LSTM(units=128, dropout=0.3)))
model.add(Dense(units=3, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
```

## **Training & Evaluation**  
The model was trained using **10 epochs**, batch size of **16**, and learning rate **0.001**.  

```python
history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))
```

### **Results:**  
- **Test Accuracy:** 100%  
- **F1 Score:** 1.00  

## **Example Predictions**  
```python
def predict_password_strength(password):
    sequence = tokenizer.texts_to_sequences([password])
    padded = pad_sequences(sequence, maxlen=16, padding='pre')
    prediction = model.predict(padded)
    strength = np.argmax(prediction)
    return {0: "Weak", 1: "Medium", 2: "Strong"}[strength]

print(predict_password_strength("123456"))  # Weak  
print(predict_password_strength("SecurePass@123"))  # Strong  
print(predict_password_strength("P@ssword12"))  # Medium  
```

## **Installation & Usage**  

### **1️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **2️⃣ Run the Model**  
```bash
python LSTM_Password_Classifier.py
```

## **Author**  
**Sayantika Saha**  
[GitHub Profile](https://github.com/SayantikaSaha9)  
