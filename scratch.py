import tensorflow as tf
import numpy as np
import cv2 as cv
import math

@tf.function
def affine2d(imgs):
    batches = 2
    img_shape = np.shape(imgs[0])
    # img_shape = [3,4]
    h = img_shape[0]
    w = img_shape[1]
    # imgs = np.reshape(np.arange(np.prod([batches, *img_shape]), dtype=np.float32), [batches, *img_shape])

    x_t, y_t = tf.meshgrid(tf.linspace(-1.0, 1.0, w), tf.linspace(-1.0, 1.0, h))
    sampling_grid = tf.stack([tf.reshape(x_t, [h*w]), tf.reshape(y_t, [h*w]), tf.ones(h*w, tf.float32)])

    transforms = np.asarray([[math.cos(math.pi/6),-math.sin(math.pi/6),0.],[math.sin(math.pi/6),math.cos(math.pi/6),0.] ]*batches, dtype=np.float32)
    transforms = tf.reshape(transforms, [-1, 2, 3])
    samples = tf.matmul(transforms, sampling_grid)
    
    x = ((samples[:, 0] + 1) * w) * 0.5
    y = ((samples[:, 1] + 1) * h) * 0.5

    x0 = tf.floor(x)
    x1 = x0 + 1
    y0 = tf.floor(y)
    y1 = y0 + 1

    x0 = tf.clip_by_value(x0, 0, w-1)
    x1 = tf.clip_by_value(x1, 0, w-1)
    y0 = tf.clip_by_value(y0, 0, h-1)
    y1 = tf.clip_by_value(y1, 0, h-1)

    wa = tf.expand_dims((y1-y) * (x1-x), axis=-1)
    wb = tf.expand_dims((y1-y) * (x-x0), axis=-1)
    wc = tf.expand_dims((y-y0) * (x1-x), axis=-1)
    wd = tf.expand_dims((y-y0) * (x-x0), axis=-1)

    x0 = tf.cast(x0, tf.int32)
    x1 = tf.cast(x1, tf.int32)
    y0 = tf.cast(y0, tf.int32)
    y1 = tf.cast(y1, tf.int32)

    y0_x0 = tf.stack([y0, x0], axis=-1)
    Ia = tf.gather_nd(imgs, tf.stack([y0, x0], axis=-1), batch_dims=1)
    Ib = tf.gather_nd(imgs, tf.stack([y0, x1], axis=-1), batch_dims=1)
    Ic = tf.gather_nd(imgs, tf.stack([y1, x0], axis=-1), batch_dims=1)
    Id = tf.gather_nd(imgs, tf.stack([y1, x1], axis=-1), batch_dims=1)
    
    out = tf.reshape(wa*Ia + wb*Ib + wc*Ic + wd*Id, [-1, h, w, 3])

    # tf.print(imgs, summarize=-1)
    # # tf.print(x_t, summarize=-1)
    # # tf.print(y_t, summarize=-1)
    # # tf.print(sampling_grid, summarize=-1)
    # # tf.print(samples, summarize=-1)
    # # tf.print(x, summarize=-1)
    # # tf.print(y, summarize=-1)
    # tf.print(y0_x0, summarize=-1)
    # tf.print(Ia, summarize=-1)
    # # tf.print(out, summarize=-1)

    return out

t = np.float32(affine2d(np.asarray([cv.imread("random.jpg"), cv.imread("random.jpg")]).astype(np.float32))) / 255
print(np.shape(t))
cv.imshow("out", t[0])
cv.waitKey()