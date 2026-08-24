import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model

np.random.seed(42)
tf.random.set_seed(42)

cnn = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(224,224,3))

for layer in cnn.layers:
    layer.trainable = False

frames = np.random.rand(5,224,224,3).astype("float32")
features = cnn.predict(frames, verbose=0)
features = features.reshape(1,5,1280)

encoder_input = Input(shape=(5,1280))
encoder = LSTM(128, return_state=True)
encoder_output, state_h, state_c = encoder(encoder_input)

decoder_input = Input(shape=(5,9))
decoder = LSTM(128, return_sequences=True)
decoder_output = decoder(decoder_input, initial_state=[state_h,state_c])
output = Dense(9, activation="softmax")(decoder_output)

model = Model([encoder_input,decoder_input], output)
model.compile(optimizer="adam", loss="categorical_crossentropy")

decoder_data = np.random.rand(1,5,9).astype("float32")
prediction = model.predict([features,decoder_data], verbose=0)

words = ["a","boy","is","playing","football","girl","running","man","walking"]
predicted_words = np.argmax(prediction[0], axis=1)
caption = [words[i] for i in predicted_words]

print("Video Caption Generation")
print("Feature Shape:", features.shape)
print("Prediction Shape:", prediction.shape)
print("\nGenerated Caption:")
print(" ".join(caption))
