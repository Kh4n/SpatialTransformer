import tensorflow as tf
import cv2 as cv
import numpy as np
import time

class SpatialTransformer(tf.keras.layers.Wrapper):
    def __init__(self, layer, **kwargs):
        super(SpatialTransformer, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        self.h = input_shape[1]
        self.w = input_shape[2]
        self.c = input_shape[3]
        x_t, y_t = tf.meshgrid(tf.linspace(-1.0, 1.0, self.w), tf.linspace(-1.0, 1.0, self.h))
        self.sampling_grid = tf.stack([tf.reshape(x_t, [self.h*self.w]), tf.reshape(y_t, [self.h*self.w]), tf.ones(self.h*self.w, tf.float32)])
        if not self.layer.built:
            self.layer.build(input_shape[1:])
            self.layer.built = True
        super(SpatialTransformer, self).build(input_shape)
  
    def call(self, inputs):
        batches = tf.shape(inputs)[0]
        sampling_grid_curr = tf.matmul(tf.reshape(self.layer(inputs), [batches, 2, 3]), self.sampling_grid)

        x = ((sampling_grid_curr[:, 0] + 1) * self.w) * 0.5
        y = ((sampling_grid_curr[:, 1] + 1) * self.h) * 0.5

        x0 = tf.floor(x)
        x1 = x0 + 1
        y0 = tf.floor(y)
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, 0, self.w-1)
        x1 = tf.clip_by_value(x1, 0, self.w-1)
        y0 = tf.clip_by_value(y0, 0, self.h-1)
        y1 = tf.clip_by_value(y1, 0, self.h-1)

        wa = (x1-x) * (y1-y)
        wb = (x1-x) * (y-y0)
        wc = (x-x0) * (y1-y)
        wd = (x-x0) * (y-y0)

        wa = tf.expand_dims(wa, axis=2)
        wb = tf.expand_dims(wb, axis=2)
        wc = tf.expand_dims(wc, axis=2)
        wd = tf.expand_dims(wd, axis=2)

        x0 = tf.cast(x0, tf.int32)
        x1 = tf.cast(x1, tf.int32)
        y0 = tf.cast(y0, tf.int32)
        y1 = tf.cast(y1, tf.int32)

        batch_indices = tf.tile(tf.range(batches), [self.h*self.w])
        batch_indices = tf.transpose(tf.reshape(batch_indices, [self.h*self.w, batches]))
        batch_indices = tf.reshape(batch_indices, [self.h*self.w*batches])

        indices = tf.stack([batch_indices, tf.reshape(y0, [batches*self.w*self.h]), tf.reshape(x0, [batches*self.w*self.h])], axis=1)
        Ia = tf.reshape(tf.gather_nd(inputs, indices), [batches, self.h*self.w, self.c])

        indices = tf.stack([batch_indices, tf.reshape(y1, [batches*self.w*self.h]), tf.reshape(x0, [batches*self.w*self.h])], axis=1)
        Ib = tf.reshape(tf.gather_nd(inputs, indices), [batches, self.h*self.w, self.c])

        indices = tf.stack([batch_indices, tf.reshape(y0, [batches*self.w*self.h]), tf.reshape(x1, [batches*self.w*self.h])], axis=1)
        Ic = tf.reshape(tf.gather_nd(inputs, indices), [batches, self.h*self.w, self.c])

        indices = tf.stack([batch_indices, tf.reshape(y1, [batches*self.w*self.h]), tf.reshape(x1, [batches*self.w*self.h])], axis=1)
        Id = tf.reshape(tf.gather_nd(inputs, indices), [batches, self.h*self.w, self.c])

        out = wa*Ia + wb*Ib + wc*Ic + wd*Id
        out = tf.reshape(out, [batches, self.h, self.w, self.c])
        return out
  
    def compute_output_shape(self, input_shape):
        return input_shape
  
    def get_config(self):
        base_config = super(SpatialTransformer, self).get_config()
        return base_config
  
    @classmethod
    def from_config(cls, config):
        return cls(**config)

# x = tf.constant([
#         [
#             [[0],[1]],
#             [[1],[2]]
#         ],
#         [
#             [[2],[3]],
#             [[3],[4]]
#         ]
#     ])
# batches = 2
# x0 = [[0, 1, 1, 0], [1, 1, 0, 1]]
# y0 = [[0, 0, 1, 1], [0, 0, 1, 0]]
# indices = tf.tile(tf.range(batches), [2**2])
# indices = tf.transpose(tf.reshape(indices, [2**2, batches]))
# indices = tf.reshape(indices, [2**2*batches])
# indices = tf.stack([indices, tf.reshape(x0, [batches*2*2]), tf.reshape(y0, [batches*2*2])], axis=1)
# print(tf.stack([x0,y0], axis=2))
# print(tf.gather_nd(x, tf.stack([x0,y0], axis=2)))
# exit()


model = tf.keras.models.Sequential()

local = tf.keras.models.Sequential()
local.add(tf.keras.layers.Input(shape=(400, 400, 3)))
local.add(tf.keras.layers.Flatten())
local.add(tf.keras.layers.Dense(6))

model.add(SpatialTransformer(local, input_shape=(400, 400, 3)))
model.add(tf.keras.layers.AveragePooling2D((10, 10)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.Dense(1))

model.compile(
    optimizer="rmsprop",
    loss="mse",
)

x = np.array([[1.0,0.0], [1.0,1.0]])
x = np.array([cv.imread("square.jpg").astype(np.float32), cv.imread("square.jpg").astype(np.float32)])
y = np.array([[1.0], [0.0]])

print(model.summary())
model.fit(x, y, batch_size=2, epochs=10, callbacks=[tf.keras.callbacks.TensorBoard(log_dir='.\\logs')])

'''
img = cv.imread("random.jpg")
h, w, c = img.shape
print(h, w, c)

x = np.linspace(-1, 1, w)
y = np.linspace(-1, 1, h)
# x, y = np.arange(w), np.arange(h)
x_t, y_t = np.meshgrid(x, y)

M = np.array([[0.707, -0.707, 0.], [0.707, 0.707, 0.]])

sampling_grid = np.vstack([x_t.flatten(), y_t.flatten(), np.ones(h*w)])
num_batches = 4
sampling_grid = np.resize(sampling_grid, (num_batches, 3, h*w))
sampling_grid = np.matmul(M, sampling_grid)

x = ((sampling_grid[:, 0] + 1) * w) * 0.5
y = ((sampling_grid[:, 1] + 1) * h) * 0.5

x0 = np.floor(x).astype(np.int64)
x1 = x0 + 1
y0 = np.floor(y).astype(np.int64)
y1 = y0 + 1

x0 = np.clip(x0, 0, w-1)
x1 = np.clip(x1, 0, w-1)
y0 = np.clip(y0, 0, h-1)
y1 = np.clip(y1, 0, h-1)

Ia = img[y0, x0]
Ib = img[y1, x0]
Ic = img[y0, x1]
Id = img[y1, x1]

wa = (x1-x) * (y1-y)
wb = (x1-x) * (y-y0)
wc = (x-x0) * (y1-y)
wd = (x-x0) * (y-y0)

wa = np.expand_dims(wa, axis=1)
wb = np.expand_dims(wb, axis=1)
wc = np.expand_dims(wc, axis=1)
wd = np.expand_dims(wd, axis=1)

out = wa*Ia + wb*Ib + wc*Ic + wd*Id
out = np.reshape(out, (num_batches, h, w, c)).astype(np.uint8)
cv.imshow("dd", img)
cv.imshow("hi", out[3])
cv.waitKey(0)
cv.destroyAllWindows()
'''