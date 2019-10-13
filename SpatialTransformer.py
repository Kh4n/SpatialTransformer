import tensorflow as tf

class SpatialTransformer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SpatialTransformer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.h = input_shape[1][1]
        self.w = input_shape[1][2]
        self.c = input_shape[1][3]
        x_t, y_t = tf.meshgrid(tf.linspace(-1.0, 1.0, self.w), tf.linspace(-1.0, 1.0, self.h))
        self.sampling_grid = tf.stack([tf.reshape(x_t, [self.h*self.w]), tf.reshape(y_t, [self.h*self.w]), tf.ones(self.h*self.w, tf.float32)])

        super(SpatialTransformer, self).build(input_shape)
  
    def call(self, inputs):
        local = inputs[0]
        image = inputs[1]
        batches = tf.shape(image)[0]
        sampling_grid_curr = tf.matmul(tf.reshape(local, [batches, 2, 3]), self.sampling_grid)

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
        Ia = tf.reshape(tf.gather_nd(image, indices), [batches, self.h*self.w, self.c])

        indices = tf.stack([batch_indices, tf.reshape(y1, [batches*self.w*self.h]), tf.reshape(x0, [batches*self.w*self.h])], axis=1)
        Ib = tf.reshape(tf.gather_nd(image, indices), [batches, self.h*self.w, self.c])

        indices = tf.stack([batch_indices, tf.reshape(y0, [batches*self.w*self.h]), tf.reshape(x1, [batches*self.w*self.h])], axis=1)
        Ic = tf.reshape(tf.gather_nd(image, indices), [batches, self.h*self.w, self.c])

        indices = tf.stack([batch_indices, tf.reshape(y1, [batches*self.w*self.h]), tf.reshape(x1, [batches*self.w*self.h])], axis=1)
        Id = tf.reshape(tf.gather_nd(image, indices), [batches, self.h*self.w, self.c])

        out = wa*Ia + wb*Ib + wc*Ic + wd*Id
        out = tf.reshape(out, [batches, self.h, self.w, self.c])
        return out
  
    def compute_output_shape(self, input_shape):
        return input_shape[1]
  
    def get_config(self):
        base_config = super(SpatialTransformer, self).get_config()
        return base_config
  
    @classmethod
    def from_config(cls, config):
        return cls(**config)