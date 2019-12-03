import tensorflow as tf
import cv2 as cv
import numpy as np
import LocScaleTransform as mt
import ImageOutputAtLayerTensorboard as it
import shutil
import os
import datagen as dg
import tensorflow as tf
from tensorflow.keras import layers

# toy problem demonstrating use of Spatial Transformer. generates randomly placed
# circles and zooms in on them, but only zoom in far enough to fit the screen
# consumes a fair bit of ram, reduce step/batch sizes accordingly
# can also turn off multiprocessing as well to reduce ram usage

if os.path.isdir("./logs"):
    shutil.rmtree("./logs")

def identity_flat(shape, dtype=None):
    return tf.convert_to_tensor(
        [0,0,1],
        np.float32
    )


img_dims = (240,240,3)

imgs_in = layers.Input(shape=img_dims)

conv = layers.Conv2D(32, (5,5), activation="relu")(imgs_in)
conv = layers.MaxPooling2D(pool_size=[2,2])(conv)
conv = layers.Conv2D(64, (5,5), activation="relu")(conv)
conv = layers.MaxPooling2D(pool_size=[4,4])(conv)
conv = layers.Conv2D(128, (5,5), activation="relu")(conv)
conv = layers.MaxPooling2D(pool_size=[6,6])(conv)

flat = layers.Flatten()(conv)
flat = layers.Activation("tanh")(flat)
dense = layers.Dense(32, activation="tanh")(flat)
local = layers.Dense(3, kernel_initializer="zeros")(dense)
# local = layers.Activation("sigmoid")(local)
mt = mt.LocScaleTransform(name="mt")([local, imgs_in])
# out = layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(st)

model = tf.keras.Model(inputs=[imgs_in], outputs=[mt])
print(model.summary())

sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(
    optimizer=sgd,
    loss="mse",
)

batch_size = 32
step_size = 100
# consumes a fair bit of ram/vram, reduce step/batch sizes accordingly
gen, vgen = dg.STNGeneratorBasic.create_data_generators(
    img_dims[:-1], step_size, batch_size=batch_size, val_split=0.2
)

intermediate_layer_model = tf.keras.models.Model(inputs=model.input,
                                 outputs=model.get_layer("mt").output)
intermediate_output = intermediate_layer_model.predict(gen[0][0])
cv.imshow("initial untrained. should be indentity transform", intermediate_output[0])
cv.waitKey()
cv.destroyAllWindows()

history = model.fit_generator(
    gen, epochs=100, validation_data=vgen, shuffle=False,
    callbacks=[it.ImageOutputAtLayerTensorboard("mt", gen[0][0], log_dir='./logs')],
    # there is a bug in vscode when using multiprocessing. either comment them out or run
    # from command line. if you choose the latter, use Ctrl+\ (SIGQUIT) instead of Ctrl+c
    # for a cleaner exit (it will take much longer to quit, however)
    workers=8, use_multiprocessing=True
)

intermediate_layer_model = tf.keras.models.Model(inputs=model.input,
                                 outputs=model.get_layer("mt").output)
intermediate_output = intermediate_layer_model.predict(gen[0][0])
cv.imshow("trained output. should be more zoomed in over the circle", intermediate_output[0])
cv.waitKey()
cv.destroyAllWindows()