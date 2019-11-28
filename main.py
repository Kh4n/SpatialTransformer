import tensorflow as tf
import cv2 as cv
import numpy as np
import SpatialTransformer as st
import ImageOutputAtLayerTensorboard as it
import shutil
import os

if os.path.isdir("./logs"):
    shutil.rmtree("./logs")

def identity_flat(shape, dtype=None):
    return tf.convert_to_tensor([1,0,1, 0,1,0], np.float32)

image_in = tf.keras.layers.Input(shape=(400, 400, 3))

loc_f = tf.keras.layers.Flatten()(image_in)
loc_a = tf.keras.layers.Activation("sigmoid")(loc_f)
loc_d = tf.keras.layers.Dense(6, kernel_initializer="zeros", bias_initializer=identity_flat)(loc_a)
local = loc_d
# local = tf.keras.layers.Activation("sigmoid")(loc_d)

st = st.SpatialTransformer(name="st")([local, image_in])
flat = tf.keras.layers.Flatten()(st)
dense1 = tf.keras.layers.Dense(64)(flat)
act2 = tf.keras.layers.Activation("relu")(dense1)
out = tf.keras.layers.Dense(1, activation="sigmoid")(act2)

model = tf.keras.models.Model(inputs=[image_in], outputs=[out])
print(model.summary())

sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(
    optimizer=sgd,
    loss="mse",
)

x = np.array([cv.imread("square.jpg").astype(np.float32), cv.imread("square.jpg").astype(np.float32)])
y = np.array([[1.0], [0.0]])

intermediate_layer_model = tf.keras.models.Model(inputs=model.input,
                                 outputs=model.get_layer("st").output)
intermediate_output = intermediate_layer_model.predict(x)
cv.imshow("test", intermediate_output[0])
cv.waitKey()

model.fit(x, y, batch_size=1, epochs=10, callbacks=[it.ImageOutputAtLayerTensorboard("st", x, log_dir='./logs')])

intermediate_output = intermediate_layer_model.predict(x)
cv.imshow("test", intermediate_output[0])
cv.waitKey()