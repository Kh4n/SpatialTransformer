import tensorflow as tf
import cv2 as cv
import numpy as np
import SpatialTransformer as st


image_in = tf.keras.layers.Input(shape=(400, 400, 3))

loc_f = tf.keras.layers.Flatten()(image_in)
local = tf.keras.layers.Dense(6)(loc_f)

st = st.SpatialTransformer()([local, image_in])
flat = tf.keras.layers.Flatten()(st)
dense1 = tf.keras.layers.Dense(256)(flat)
act2 = tf.keras.layers.Activation("relu")(dense1)
out = tf.keras.layers.Dense(1)(act2)

model = tf.keras.models.Model(inputs=[image_in], outputs=[out])
print(model.summary())

model.compile(
    optimizer="rmsprop",
    loss="mse",
)

x = np.array([cv.imread("square.jpg").astype(np.float32), cv.imread("square.jpg").astype(np.float32)])
y = np.array([[1.0], [0.0]])

model.fit(x, y, batch_size=1, epochs=10, callbacks=[tf.keras.callbacks.TensorBoard(log_dir='./logs')])
