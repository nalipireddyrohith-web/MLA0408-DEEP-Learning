import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, RepeatVector

np.random.seed(42)
tf.random.set_seed(42)

vocab_size = 20
sequence_length = 5

X = np.array([[1,2,3,0,0],[1,4,3,0,0],[1,5,6,3,0],[1,2,7,3,0]])
y = np.array([[1,8,9,3,0],[1,10,9,3,0],[1,11,12,9,3],[1,8,13,9,0]])

encoder_input = Input(shape=(sequence_length,))
embedding = Embedding(vocab_size, 32)(encoder_input)
encoder_output = Bidirectional(LSTM(32))(embedding)
context = RepeatVector(sequence_length)(encoder_output)
decoder = LSTM(64, return_sequences=True)(context)
output = Dense(vocab_size, activation="softmax")(decoder)

model = Model(encoder_input, output)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

target = y.reshape(y.shape[0], y.shape[1], 1)
model.fit(X, target, epochs=20, batch_size=2, verbose=0)

test_sentence = np.array([[1,2,3,0,0]])
prediction = model.predict(test_sentence, verbose=0)
translation = np.argmax(prediction, axis=-1)

print("Bidirectional RNN Encoder-Decoder")
print("Translation Prediction:")
print(translation)
print("\nPrediction Shape:")
print(prediction.shape)
