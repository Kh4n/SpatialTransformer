import tensorflow as tf
import cv2 as cv

class MotionTracking(tf.keras.layers.Layer):
    def __init__(self, num_tracks=3, window_pixel_wh=15, **kwargs):
        assert(num_tracks > 1)
        assert(window_pixel_wh > 3 and window_pixel_wh % 2 != 0)
        self.num_tracks = num_tracks
        self.window_pixel_wh = window_pixel_wh
        super(MotionTracking, self).__init__(**kwargs)

    def build(self, input_shape):
        # grab the dimensions of the image here so we can use them later. also will throw errors early for users
        self.h = input_shape[1][1+1]
        self.w = input_shape[1][2+1]
        self.c = input_shape[1][3+1]

        self.scale = self.window_pixel_wh / min(self.w, self.h)

        # get smallest axis to scale relatively to that
        self.win_pixel_wh = int(min(self.h, self.w)*self.scale)

        # we scale to the smaller axis and then apply transforms to that resulting square
        # originally was [0.0, 1.0], but this resulted in the model being unable to learn. not sure why. possibly because tanh learns better than sigmoid
        x_t, y_t = tf.meshgrid(
            tf.linspace(-1.0, 1.0, self.win_pixel_wh),
            tf.linspace(-1.0, 1.0, self.win_pixel_wh),
        )
        self.sampling_grid = tf.stack([
            tf.reshape(x_t, [self.win_pixel_wh*self.win_pixel_wh]),
            tf.reshape(y_t, [self.win_pixel_wh*self.win_pixel_wh]),
        ])

        self.sobel_x = tf.reshape(
            tf.constant([
                [-1.,  0.,  1.],
                [-2.,  0.,  2.],
                [-1.,  0.,  1.],
            ]),
            [3, 3, 1, 1]
        )
        self.sobel_y = tf.reshape(
            tf.constant([
                [-1., -2., -1.],
                [ 0.,  0.,  0.],
                [ 1.,  2.,  1.],
            ]),
            [3, 3, 1, 1]
        )

        self.scharr_x = tf.reshape(
            tf.constant([
                [-3.,   0.,  3.],
                [-10.,  0.,  10.],
                [-3.,   0.,  3.],
            ]),
            [3, 3, 1, 1]
        )
        self.scharr_y = tf.reshape(
            tf.constant([
                [-3., -10., -3.],
                [ 0.,   0.,  0.],
                [ 3.,  10.,  3.],
            ]),
            [3, 3, 1, 1]
        )

        super(MotionTracking, self).build(input_shape)
  
    def call(self, inputs):
        init_track_locs = tf.reshape(inputs[0], [-1, self.num_tracks, 2, 1]) * 1.0/self.scale
        imgs = inputs[1]

        samples = tf.reshape(self.sampling_grid, [1, 1, 2, -1]) + init_track_locs

        # have to adjust to the relative scaling done earlier
        x = ((samples[:, :, 0]) * self.win_pixel_wh) * 0.5
        y = ((samples[:, :, 1]) * self.win_pixel_wh) * 0.5

        x = tf.tile(tf.expand_dims(x, axis=1), [1, 2, 1, 1])
        y = tf.tile(tf.expand_dims(y, axis=1), [1, 2, 1, 1])

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

        # we are sampling the image num_tracks times, so we need to tile here. unlike other
        # operations, there is no way for gather_nd to broadcast, so we need to tile explicitly
        tiled_imgs = tf.tile(tf.expand_dims(imgs[:, 0:2], axis=2), [1, 1, self.num_tracks, 1, 1, 1])

        # there are not actually 2 batch dimensions. however this works as intended: performing the
        # indexing for each batch and motion track
        Ia = tf.gather_nd(tiled_imgs, tf.stack([y0, x0], axis=-1), batch_dims=3)
        Ib = tf.gather_nd(tiled_imgs, tf.stack([y0, x1], axis=-1), batch_dims=3)
        Ic = tf.gather_nd(tiled_imgs, tf.stack([y1, x0], axis=-1), batch_dims=3)
        Id = tf.gather_nd(tiled_imgs, tf.stack([y1, x1], axis=-1), batch_dims=3)

        # a = tf.stack([y0, x0], axis=-1)
        # print(tf.shape(y0))
        # print(tf.shape(a))
        # print(tf.shape(wa))
        # print(tf.shape(Ia))
        # print(tf.shape(tiled_imgs))

        out = tf.reshape(wa*Ia + wb*Ib + wc*Ic + wd*Id, [-1, 2, self.num_tracks, self.win_pixel_wh, self.win_pixel_wh, self.c])

        first_frame  = out[:, 0]
        second_frame = out[:, 1]
        ff_comb = tf.reshape(first_frame, [-1, self.win_pixel_wh, self.win_pixel_wh, self.c])

        Ix = tf.reshape(
            tf.nn.convolution(ff_comb, self.sobel_x, padding="SAME"),
            [-1, self.num_tracks, self.win_pixel_wh, self.win_pixel_wh, self.c]
        )
        sum_Ix2 = tf.reduce_sum(Ix*Ix, axis=[2,3,4])

        Iy = tf.reshape(
            tf.nn.convolution(ff_comb, self.sobel_y, padding="SAME"),
            [-1, self.num_tracks, self.win_pixel_wh, self.win_pixel_wh, self.c]
        )
        sum_Iy2 = tf.reduce_sum(Iy*Iy, axis=[2,3,4])

        sum_IxIy = tf.reduce_sum(Ix*Iy, axis=[2,3,4])
        ATA = tf.reshape(tf.stack([sum_Iy2, -sum_IxIy, -sum_IxIy, sum_Ix2], axis=-1), [-1, self.num_tracks, 2,2])
        ATA_det = tf.reshape(1.0/(sum_Ix2*sum_Iy2 - sum_IxIy*sum_IxIy), [-1, self.num_tracks, 1,1])
        ATA_1 = ATA_det*ATA

        It = second_frame-first_frame
        sum_IxIt = tf.reduce_sum(Ix*It, axis=[2,3,4])
        sum_IyIt = tf.reduce_sum(Iy*It, axis=[2,3,4])
        b = tf.reshape(tf.stack([-sum_IxIt, -sum_IyIt], axis=-1), [-1, self.num_tracks, 2,1])

        VxVy = tf.matmul(ATA_1, b)

        # print(tf.shape(Ix))
        # print(tf.shape(sum_Ix2))
        # print(tf.shape(Iy))
        # print(tf.shape(sum_Iy2))
        # print(tf.shape(It))
        # print(tf.shape(ATA_1))
        # print(tf.shape(b))
        # print(tf.shape(VxVy))
        # tf.print(sum_Ix2, summarize=-1)
        # tf.print(sum_Iy2, summarize=-1)
        # tf.print(sum_IxIy, summarize=-1)
        # tf.print(ATA, summarize=-1)
        # tf.print(ATA_det, summarize=-1)
        # tf.print(ATA_1, summarize=-1)
        # tf.print(VxVy, summarize=-1)

        return out
  
    def compute_output_shape(self, input_shape):
        return [None, self.win_pixel_wh, self.win_pixel_wh, self.c]
  
    def get_config(self):
        base_config = super(MotionTracking, self).get_config()
        return base_config
  
    @classmethod
    def from_config(cls, config):
        return cls(**config)

if __name__ == "__main__":
    import numpy as np
    import math

    window_pixel_wh = 21
    num_tracks = 3
    # seq_len = 5
    seq_len = 2
    batches = 4
    # identity transform, for testing purposes
    transforms = np.asarray(
        [[0.95, 1.15]*num_tracks]*batches,
        dtype=np.float32
    )
    imgs = np.asarray([
        [
            np.expand_dims(cv.imread("car_dashcam0.png", cv.IMREAD_GRAYSCALE) / 255, axis=-1),
            np.expand_dims(cv.imread("car_dashcam1.png", cv.IMREAD_GRAYSCALE) / 255, axis=-1),
        ]
    ]*batches).astype(np.float32)

    # different image (width == height)
    # imgs = np.asarray([cv.imread("square.jpg"), cv.imread("square.jpg")]).astype(np.float32)
    out = np.float32(MotionTracking(num_tracks=num_tracks, window_pixel_wh=window_pixel_wh)([transforms, imgs]))
    print(out.shape)
    cv.imshow(
        f"should be {num_tracks} zoomed in images on the tail of the elephant, stacked vertically",
        np.reshape(out, [-1, window_pixel_wh*num_tracks, window_pixel_wh, 1])[0]
        # out[0]
        # out[0, 0]
    )
    cv.waitKey()
    for i in range(2):
        for j in range(num_tracks):
            cv.imshow(f"frame {i}, track {j}", out[0, i,j])
            cv.waitKey()

    with open("OUT", "w") as f:
        with np.printoptions(threshold=np.inf):
            f.write(str(out))