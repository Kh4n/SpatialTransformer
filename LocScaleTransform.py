import tensorflow as tf
import cv2 as cv

class LocScaleTransform(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LocScaleTransform, self).__init__(**kwargs)

    def build(self, input_shape):
        # grab the dimensions of the image here so we can use them later. also will throw errors early for users
        self.h = input_shape[1][1]
        self.w = input_shape[1][2]
        self.c = input_shape[1][3]

        # get smallest axis to scale relatively to that
        self.max_hw = max(self.h, self.w)
        self.min_hw = min(self.h, self.w)
        self.ratio = self.max_hw/self.min_hw

        # we scale to the smaller axis and then apply transforms to that resulting square
        # originally was [0.0, 1.0], but this resulted in the model being unable to learn. not sure why
        x_t, y_t = tf.meshgrid(tf.linspace(-1.0, 1.0, self.min_hw), tf.linspace(-1.0, 1.0, self.min_hw))
        # x_t, y_t = tf.meshgrid(tf.linspace(0.0, 1.0, self.min_hw), tf.linspace(0.0, 1.0, self.min_hw))
        self.sampling_grid = tf.stack([
            tf.reshape(x_t, [self.min_hw*self.min_hw]), tf.reshape(y_t, [self.min_hw*self.min_hw])
        ])

        super(LocScaleTransform, self).build(input_shape)
  
    def call(self, inputs):
        transforms = inputs[0]
        imgs = inputs[1]

        # -1 as reshape automatically infers batch dimension
        scale = tf.reshape(transforms[:, -1] + 1, [-1, 1, 1])
        translate = tf.reshape(transforms[:, 0:2], [-1, 2, 1])
        samples = (tf.expand_dims(self.sampling_grid, axis=0) * scale) + translate

        # have to adjust to the relative scaling done earlier
        # x = samples[:, 0] * self.min_hw
        x = ((samples[:, 0] + 1) * self.min_hw) * 0.5
        # y = samples[:, 1] * self.min_hw
        y = ((samples[:, 1] + 1) * self.min_hw) * 0.5

        x0 = tf.floor(x)
        x1 = x0 + 1
        y0 = tf.floor(y)
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, 0, self.w-1)
        x1 = tf.clip_by_value(x1, 0, self.w-1)
        y0 = tf.clip_by_value(y0, 0, self.h-1)
        y1 = tf.clip_by_value(y1, 0, self.h-1)

        wa = tf.expand_dims((y1-y) * (x1-x), axis=-1)
        wb = tf.expand_dims((y1-y) * (x-x0), axis=-1)
        wc = tf.expand_dims((y-y0) * (x1-x), axis=-1)
        wd = tf.expand_dims((y-y0) * (x-x0), axis=-1)

        x0 = tf.cast(x0, tf.int32)
        x1 = tf.cast(x1, tf.int32)
        y0 = tf.cast(y0, tf.int32)
        y1 = tf.cast(y1, tf.int32)

        Ia = tf.gather_nd(imgs, tf.stack([y0, x0], axis=-1), batch_dims=1)
        Ib = tf.gather_nd(imgs, tf.stack([y0, x1], axis=-1), batch_dims=1)
        Ic = tf.gather_nd(imgs, tf.stack([y1, x0], axis=-1), batch_dims=1)
        Id = tf.gather_nd(imgs, tf.stack([y1, x1], axis=-1), batch_dims=1)

        out = tf.reshape(wa*Ia + wb*Ib + wc*Ic + wd*Id, [-1, self.min_hw, self.min_hw, self.c])
        return out
  
    def compute_output_shape(self, input_shape):
        return [None, self.min_hw, self.min_hw, self.c]
  
    def get_config(self):
        base_config = super(LocScaleTransform, self).get_config()
        return base_config
  
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ZoomOutMT(tf.keras.layers.Layer):
    def __init__(self, zoom_amount=2, **kwargs):
        self.zoom_amount = zoom_amount
        super(ZoomOutMT, self).__init__(**kwargs)

    def build(self, input_shape):
        self.zoom_tensor = tf.convert_to_tensor([[1, 1, self.zoom_amount]], tf.float32)
        super(ZoomOutMT, self).build(input_shape)
  
    def call(self, inputs):
        return inputs*self.zoom_tensor
  
    def compute_output_shape(self, input_shape):
        return input_shape
  
    def get_config(self):
        base_config = super(ZoomOutMT, self).get_config()
        return base_config
  
    @classmethod
    def from_config(cls, config):
        return cls(**config)



if __name__ == "__main__":
    import numpy as np
    import math

    batches = 4
    # identity transform, for testing purposes
    transforms = np.asarray(
        [[4.0, 4.0, 0.0]]*batches,
        dtype=np.float32
    )
    imgs = np.asarray([cv.imread("random.jpg")]*batches).astype(np.float32)

    # different image (width == height)
    # imgs = np.asarray([cv.imread("square.jpg"), cv.imread("square.jpg")]).astype(np.float32)
    out = LocScaleTransform()([transforms, imgs])
    print(np.shape(out))
    cv.imshow("the image should look almost the same, just cropped from the right", np.float32(out[0]) / 255)
    cv.waitKey()