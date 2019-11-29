import tensorflow as tf
import numpy as np
import cv2 as cv
import os
import random


class STNGeneratorSequence(tf.keras.utils.Sequence):
    def __init__(self, path, format, img_size, indices, y_train, seq_len, split, batch_size=32, shuffle=True):
        self.path = path
        self.format = format
        self.img_size = img_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.indices, self.y_train = indices, y_train
        self.count = 0
        self.mem = 10*(10**9)*split
        self.shuffle = shuffle

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(self.mem//(self.batch_size*self.seq_len*np.prod(self.img_size)*4))

    def __getitem__(self, index):
        'Generate one batch of data'
        x = np.empty([self.batch_size, self.seq_len, *self.img_size], dtype=np.float32)
        y = np.empty([self.batch_size], dtype=np.float32)
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        for i,fnum in enumerate(batch_indices):
            sum_speeds = 0
            for k in range(self.seq_len):
                file = os.path.join(self.path, self.format.format(fnum + k))
                self.count += 1
                # print(file)
                x[i][k] = cv.resize(cv.imread(file, cv.IMREAD_UNCHANGED), self.img_size[-2:None:-1]) / 255
                sum_speeds += self.y_train[(fnum-1)+k]
                # print(i,k,fnum+k,self.y_train[(fnum-1)+k])
            y[i] = sum_speeds / self.seq_len
        # print("count:", self.count)
        return x, y / 30
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    @classmethod
    def create_data_generators(cls, path, format, img_size, seq_len, batch_size=32, val_split=0.2, shuffle=True):
        with open("train.txt", 'r') as f:
            y_train = np.asfarray([float(k) for k in f.read().strip().splitlines()])
        
        data_size = len([file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]) - seq_len
        data_size_train = data_size - int(val_split*data_size)

        indices = np.arange(data_size)[::seq_len] + 1
        if shuffle:
            np.random.shuffle(indices)

        train_gen = cls(path, format, img_size, indices[0:data_size_train], y_train, seq_len, 1-val_split, batch_size, shuffle)
        val_gen = cls(path, format, img_size, indices[data_size_train:], y_train, seq_len, val_split, batch_size, shuffle)

        return (train_gen, val_gen)


class STNGeneratorBasic(tf.keras.utils.Sequence):
    def __init__(self, hw, steps, batch_size=32):
        self.hw = hw
        self.steps = steps
        self.batch_size = batch_size

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(self.steps)

    def __getitem__(self, index):
        'Generate one batch of data'
        x = np.zeros([self.batch_size, *self.hw, 3], dtype=np.float32)
        min_wh = min(self.hw)
        for i,pic in enumerate(x):
            radius = random.randint(min_wh//20, min_wh//3)
            cv.circle(
                x[i],
                (random.randint(radius, self.hw[1]-radius), random.randint(radius, self.hw[0]-radius)),
                radius, (1., 1., 0.), thickness=-1
            )
        
        y = np.zeros([1, *self.hw, 3], dtype=np.float32)
        cv.circle(y[0], (self.hw[1]//2, self.hw[0]//2), min_wh//2, (1., 1., 0.), thickness=-1)
        return x, np.repeat(y, self.batch_size, axis=0)
    
    @classmethod
    def create_data_generators(cls, hw, steps, batch_size=32, val_split=0.2):
        val_steps = int(val_split*steps)
        train_gen = cls(hw, steps - val_split, batch_size=32)
        val_gen = cls(hw, val_steps, batch_size=32)

        return (train_gen, val_gen)

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

if __name__ == "__main__":
    gen, vgen = STNGeneratorBasic.create_data_generators(
        (240, 240), 100,
    )
    print(np.shape(gen[0][0][0]))
    cv.imshow("out", gen[0][0][0])
    cv.waitKey()
    cv.imshow("out", gen[0][1][0])
    cv.waitKey()