import tensorflow as tf
import cv2 as cv

def gaussian(x, sigma):
    return 1/(sigma*math.sqrt(2*math.pi))*math.e**(-1/2*(x/sigma)**2)

class MotionTracking(tf.keras.layers.Layer):
    def __init__(self, num_tracks=3, window_pixel_wh=21, sigma=0.3, **kwargs):
        self.sigma = sigma
        assert(num_tracks > 1)
        assert(window_pixel_wh >= 3)
        self.num_tracks = num_tracks
        self.win_pixel_wh = window_pixel_wh
        super(MotionTracking, self).__init__(**kwargs)

    def build(self, input_shape):
        # grab the dimensions of the image here so we can use them later. also will throw errors early for users
        self.h = input_shape[1][1+1]
        self.w = input_shape[1][2+1]
        self.c = input_shape[1][3+1]

        self.scale = self.win_pixel_wh / min(self.w, self.h)

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

        weights = np.ones([window_pixel_wh, window_pixel_wh])
        # weights = np.empty([window_pixel_wh, window_pixel_wh])
        # center = window_pixel_wh//2
        # for y in range(window_pixel_wh):
        #     for x in range(window_pixel_wh):
        #         weights[y, x] = (x-center)**2 + (y-center)**2

        # weights = gaussian(np.sqrt(weights), self.sigma)
        self.win_weights = tf.constant(weights, shape=[1, 1, window_pixel_wh, window_pixel_wh, 1], dtype=tf.float32)
        # print(weights)
        # tf.print(weights)
        # tf.print(tf.reduce_max(weights))

        super(MotionTracking, self).build(input_shape)
    
    def sample_ntracks_from_2frames(self, samples, frames):
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
        tiled_imgs = tf.tile(tf.expand_dims(imgs, axis=2), [1, 1, self.num_tracks, 1, 1, 1])

        # there are not actually 3 batch dimensions. however this works as intended: performing the
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

        return tf.reshape(wa*Ia + wb*Ib + wc*Ic + wd*Id, [-1, 2, self.num_tracks, self.win_pixel_wh, self.win_pixel_wh, self.c])
  
    def calc_velocity_2frames_ntracks_LK(self, first_frame, second_frame):
        ff_comb = tf.reshape(first_frame, [-1, self.win_pixel_wh, self.win_pixel_wh, self.c])

        Ix = tf.reshape(
            tf.nn.convolution(ff_comb, self.sobel_x, padding="SAME"),
            [-1, self.num_tracks, self.win_pixel_wh, self.win_pixel_wh, self.c]
        )
        sum_Ix2 = tf.reduce_sum(Ix*Ix*self.win_weights, axis=[2,3,4])

        Iy = tf.reshape(
            tf.nn.convolution(ff_comb, self.sobel_y, padding="SAME"),
            [-1, self.num_tracks, self.win_pixel_wh, self.win_pixel_wh, self.c]
        )
        sum_Iy2 = tf.reduce_sum(Iy*Iy*self.win_weights, axis=[2,3,4])

        sum_IxIy = tf.reduce_sum(Ix*Iy*self.win_weights, axis=[2,3,4])
        ATA = tf.reshape(tf.stack([sum_Iy2, -sum_IxIy, -sum_IxIy, sum_Ix2], axis=-1), [-1, self.num_tracks, 2,2])
        ATA_det = tf.reshape(1.0/(sum_Ix2*sum_Iy2 - sum_IxIy*sum_IxIy) , [-1, self.num_tracks, 1,1])
        ATA_1 = ATA_det*ATA

        It = second_frame-first_frame
        sum_IxIt = tf.reduce_sum(Ix*It*self.win_weights, axis=[2,3,4])
        sum_IyIt = tf.reduce_sum(Iy*It*self.win_weights, axis=[2,3,4])
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

        return VxVy

    def iterative_LK(self, samples, frames, iterations):
        out = self.sample_ntracks_from_2frames(samples, frames)
        second_frame = out[:, 1]

        VxVy = self.calc_velocity_2frames_ntracks_LK(out[:, 0], second_frame)
        samples -= VxVy
        sum_VxVy = VxVy

        i = tf.constant(1)
        check = lambda i, s, f, sf, svv: tf.less(i, iterations)

        def iterate(i, samples, frames, second_frame, sum_VxVy):
            out = self.sample_ntracks_from_2frames(samples, frames)
            VxVy = self.calc_velocity_2frames_ntracks_LK(out[:, 0], second_frame)
            samples -= VxVy
            i += 1
            sum_VxVy += VxVy
            return i, samples, frames, second_frame, sum_VxVy

        _, samples, _, _, sum_VxVy = tf.while_loop(check, iterate, [i, samples, frames, second_frame, sum_VxVy])
        out = self.sample_ntracks_from_2frames(samples, frames)
        tf.print(sum_VxVy)
        return out[:, 0], second_frame


    def call(self, inputs):
        init_track_locs = tf.reshape(inputs[0], [-1, self.num_tracks, 2, 1]) * 1.0/self.scale
        imgs = inputs[1]

        samples = tf.reshape(self.sampling_grid, [1, 1, 2, -1]) + init_track_locs
        tmp = self.sample_ntracks_from_2frames(samples, imgs[:, 0:2])

        out, sf = self.iterative_LK(samples, imgs[:, 0:2], 20)
        
        return tf.stack([out, sf], axis=1)
  
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
    sigma = 2
    # seq_len = 5
    seq_len = 2
    batches = 1
    # identity transform, for testing purposes
    transforms = np.asarray(
        [[0.95+0.57, 1.15-0.06]*num_tracks]*batches,
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
    out = np.float32(MotionTracking(
        num_tracks=num_tracks, window_pixel_wh=window_pixel_wh, sigma=sigma)([transforms, imgs]
    ))
    print(out.shape)
    cv.imshow(
        f"should be {num_tracks} zoomed in images",
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