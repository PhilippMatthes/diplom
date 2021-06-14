import tempfile
import os

import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, LeakyReLU, concatenate, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K


matusek_model = Sequential()
matusek_model.add(Dense(256, input_shape=(47,), activation="relu"))
matusek_model.add(Dense(256, activation="relu"))
matusek_model.add(Dense(7, activation="softmax"))
matusek_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# plot_model(matusek_model, to_file="matusek-model.pdf", show_shapes=True, show_layer_names=False)

with tempfile.TemporaryDirectory() as tmp_dir:
    filename = f"{tmp_dir}/model.h5"
    matusek_model.save(filename)
    print(f"Matusek model size in Bytes: {os.path.getsize(filename)}")
    trainable_count = int(np.sum([K.count_params(p) for p in set(matusek_model.trainable_weights)]))
    print(f"Matusek model params: {trainable_count}")

werner_model_left = Sequential()
werner_model_left.add(LSTM(64, input_shape=(64,1), return_sequences=True))
werner_model_left.add(LSTM(64, return_sequences=True))
werner_model_left.add(Dense(64, activation="relu"))
werner_model_left.add(LeakyReLU(alpha=0.05))
werner_model_left.add(Dense(32, activation="relu"))
werner_model_left.add(LeakyReLU(alpha=0.05))

werner_model_center = Sequential()
werner_model_center.add(Dense(64, input_shape=(64,1)))
werner_model_center.add(LeakyReLU(alpha=0.05))
werner_model_center.add(Dense(48, activation="relu"))
werner_model_center.add(LeakyReLU(alpha=0.05))
werner_model_center.add(Dense(32, activation="relu"))
werner_model_center.add(LeakyReLU(alpha=0.05))

werner_model_right = Sequential()
werner_model_right.add(LSTM(64, input_shape=(64,1), return_sequences=True))
werner_model_right.add(LSTM(64, return_sequences=True))
werner_model_right.add(Dense(64, activation="relu"))
werner_model_right.add(LeakyReLU(alpha=0.05))
werner_model_right.add(Dense(32, activation="relu"))
werner_model_right.add(LeakyReLU(alpha=0.05))

werner_model_conc = concatenate([
    werner_model_left.output,
    werner_model_center.output,
    werner_model_right.output
], axis=-1)
werner_model_conc = Dense(128, activation="relu")(werner_model_conc)
werner_model_conc = LeakyReLU(alpha=0.05)(werner_model_conc)
werner_model_conc = Dense(64, activation="relu")(werner_model_conc)
werner_model_conc = LeakyReLU(alpha=0.05)(werner_model_conc)
werner_model_conc = Dense(7, activation="softmax")(werner_model_conc)

werner_model = Model(inputs=[
    werner_model_left.input,
    werner_model_center.input,
    werner_model_right.input
], outputs=werner_model_conc)

werner_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

with tempfile.TemporaryDirectory() as tmp_dir:
    filename = f"{tmp_dir}/model.h5"
    werner_model.save(filename)
    print(f"Werner model size in Bytes: {os.path.getsize(filename)}")
    trainable_count = int(np.sum([K.count_params(p) for p in set(werner_model.trainable_weights)]))
    print(f"Werner model params: {trainable_count}")

# plot_model(werner_model, to_file="werner-model.pdf",show_shapes=True, show_layer_names=False)


# Unklarheiten: Ort des Dropout, Ort der L2-Regularisierung, Input-Shape
# Und genaue Reihenfolge der conv dimensionen

stojanov_model = Sequential()
stojanov_model.add(Conv2D(32, (3,3), activation="relu", input_shape=(90,43,1)))
stojanov_model.add(MaxPooling2D((2, 2)))
stojanov_model.add(Conv2D(48, (3,3), activation="relu"))
stojanov_model.add(MaxPooling2D((2, 2)))
stojanov_model.add(Conv2D(646, (3,3), activation="relu"))
stojanov_model.add(MaxPooling2D(2, 5))
stojanov_model.add(Dense(32))
stojanov_model.add(Dropout(0.5))
stojanov_model.add(Dense(64, activation="softmax"))

stojanov_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

with tempfile.TemporaryDirectory() as tmp_dir:
    filename = f"{tmp_dir}/model.h5"
    stojanov_model.save(filename)
    print(f"Stojanov model size in Bytes: {os.path.getsize(filename)}")
    trainable_count = int(np.sum([K.count_params(p) for p in set(stojanov_model.trainable_weights)]))
    print(f"Stojanov model summary: {trainable_count}")

# plot_model(stojanov_model, to_file="stojanov-model.pdf",show_shapes=True, show_layer_names=False)
