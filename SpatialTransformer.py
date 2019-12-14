import tensorflow as tf
import cv2 as cv

class SpatialTransformer(tf.keras.layers.Layer):
    def __init__(self, seq_len=None, **kwargs):
        self.seq_len = seq_len
        super(SpatialTransformer, self).__init__(**kwargs)

    def build(self, input_shape):
        # grab the dimensions of the image here so we can use them later. also will throw errors early for users
        self.h = input_shape[1][1]
        self.w = input_shape[1][2]
        self.c = input_shape[1][3]

        if self.seq_len:
            self.out_shape = [-1, self.seq_len, self.h, self.w, self.c]
        else:
            self.out_shape = [-1, self.h, self.w, self.c]

        # need to get the axis that has the max length so we can scale relatively to that
        self.max_hw = max(self.h, self.w)

        # this where many other implementations do not work correctly. you must create the meshgrid with sizes relative to the max dimension.
        # this is what it used to look like for reference: x_t, y_t = tf.meshgrid(tf.linspace(-1.0, 1.0, self.w), tf.linspace(-1.0, 1.0, self.h))
        # essentially, one dimension gets skewed because it has less samples (last arg of tf.linspace) but is still scaled to [-1,1]
        x_t, y_t = tf.meshgrid(
            tf.linspace(-self.w/self.max_hw, self.w/self.max_hw, self.w),
            tf.linspace(-self.h/self.max_hw, self.h/self.max_hw, self.h)
        )
        self.sampling_grid = tf.stack([tf.reshape(x_t, [self.h*self.w]), tf.reshape(y_t, [self.h*self.w]), tf.ones(self.h*self.w, tf.float32)])

        super(SpatialTransformer, self).build(input_shape)
  
    def call(self, inputs):
        local = inputs[0]
        imgs = tf.reshape(inputs[1], [-1, self.h, self.w, self.c])

        # -1 as reshape automatically infers batch dimension
        transforms = tf.reshape(local, [-1, 2, 3])
        samples = tf.matmul(transforms, self.sampling_grid)

        # have to adjust to the relative scaling done earlier
        x = ((samples[:, 0] + self.w/self.max_hw) * self.max_hw) * 0.5
        y = ((samples[:, 1] + self.h/self.max_hw) * self.max_hw) * 0.5

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
        a = tf.stack([y0, x0], axis=-1)
        print(tf.shape(y0))
        print(tf.shape(a))
        print(tf.shape(wa))
        print(tf.shape(Ia))

        out = tf.reshape(wa*Ia + wb*Ib + wc*Ic + wd*Id, self.out_shape)
        return out
  
    def compute_output_shape(self, input_shape):
        return input_shape[1]
  
    def get_config(self):
        base_config = super(SpatialTransformer, self).get_config()
        return base_config
  
    @classmethod
    def from_config(cls, config):
        return cls(**config)

if __name__ == "__main__":
    import numpy as np
    import math

    batches = 4
    transforms = np.asarray(
        [[math.cos(math.pi/6), -math.sin(math.pi/6), 0.,
          math.sin(math.pi/6),  math.cos(math.pi/6), 0.]]*batches,
        dtype=np.float32
    )

    # identity transform, for testing purposes
    # transforms = np.asarray(
    #     [[1., 0., 0.,
    #       0., 1., 0.]]*batches,
    #     dtype=np.float32
    # )
    imgs = np.asarray([cv.imread("random.jpg")]*batches).astype(np.float32)

    # different image (width == height)
    # imgs = np.asarray([cv.imread("square.jpg"), cv.imread("square.jpg")]).astype(np.float32)
    out = SpatialTransformer()([transforms, imgs])
    print(np.shape(out))
    cv.imshow("the image should be rotated 30deg ccw", np.float32(out[0]) / 255)
    cv.waitKey()