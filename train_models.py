import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, Flatten,
    Dropout, GlobalAveragePooling1D, Activation,
    RepeatVector, Permute, Multiply, Layer
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K

# Custom attention layer to replace Lambda
class AttentionSum(Layer):
    def call(self, inputs):
        return K.sum(inputs, axis=1)

# Attention mechanism function
def attention_layer(inputs):
    attention = Dense(1, activation='tanh')(inputs)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(inputs.shape[-1])(attention)
    attention = Permute([2, 1])(attention)
    output = Multiply()([inputs, attention])
    return AttentionSum()(output)

# Load dataset
data = pd.read_csv("Datasets.csv")
data.dropna(inplace=True)

# Preprocess
X = data['url'].astype(str)
y = data['label']

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=200)

# Save tokenizer
with open("model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)

vocab_size = len(tokenizer.word_index) + 1

# LSTM Model
input_lstm = Input(shape=(200,))
embed = Embedding(vocab_size, 64)(input_lstm)
lstm_out = LSTM(64, return_sequences=True)(embed)
attn_out = attention_layer(lstm_out)
dense1 = Dense(64, activation='relu')(attn_out)
output_lstm = Dense(1, activation='sigmoid')(dense1)
lstm_model = Model(inputs=input_lstm, outputs=output_lstm)
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=64)
lstm_model.save("model/lstm_model.h5")

# ANN Model
ann_model = Sequential([
    Embedding(vocab_size, 64, input_length=200),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=64)
ann_model.save("model/ann_model.h5")

# Evaluation
y_pred_lstm = (lstm_model.predict(X_test) > 0.5).astype("int32")
y_pred_ann = (ann_model.predict(X_test) > 0.5).astype("int32")

acc_lstm = accuracy_score(y_test, y_pred_lstm)
acc_ann = accuracy_score(y_test, y_pred_ann)

# Plot accuracies
plt.figure(figsize=(6, 4))
sns.barplot(x=["LSTM", "ANN"], y=[acc_lstm, acc_ann], palette="viridis")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1)
plt.savefig("static/accuracy.png")
plt.close()

# Confusion matrix for LSTM and ANN
conf_mat = confusion_matrix(y_test, y_pred_lstm)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues")
plt.title("LSTM & ANN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("static/confusion.png")
plt.close()

print("✅ Models trained and saved. Plots generated.")
