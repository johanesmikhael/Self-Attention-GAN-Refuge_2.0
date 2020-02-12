import scipy.misc
import numpy as np
import os
from glob import glob
import imageio

import tensorflow as tf
import tensorflow.contrib.slim as slim
from keras.datasets import cifar10, mnist

class ImageData:

    def __init__(self, load_size, channels, crop_pos='center'):
        self.load_size = load_size
        self.channels = channels
        self.crop_pos = crop_pos

    def image_processing(self, filename):
        x = tf.io.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels)
        s = tf.shape(x_decode)
        w, h = s[0], s[1]
        # height, width, channel = x_decode.eval(session=self.sess).shape
        c = tf.minimum(w, h)
        zoom_factor = 0.15
        c_ = tf.cast(tf.cast(c, dtype=tf.float32) * (1 - tf.random.uniform(shape=[])*zoom_factor), dtype=tf.int32)
        if self.crop_pos == 'random':
            print('crop random')
            k = tf.random.uniform(shape=[])
            l = tf.random.uniform(shape=[])
            w_start = tf.cast(tf.cast((w - c_), dtype=tf.float32) * k, dtype=tf.int32)
            h_start = tf.cast(tf.cast((h - c_), dtype=tf.float32) * l, dtype=tf.int32)
        else:
            w_start = (w - c_) // 2
            h_start = (h - c_) // 2
        img = x_decode[w_start:w_start + c_, h_start:h_start + c_]
        img = tf.image.resize_images(img, [self.load_size, self.load_size])
        img = tf.cast(img, tf.float32) / 127.5 - 1
        return img


def load_mnist(size=64):
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
    train_data = normalize(train_data)
    test_data = normalize(test_data)

    x = np.concatenate((train_data, test_data), axis=0)
    # y = np.concatenate((train_labels, test_labels), axis=0).astype(np.int)

    seed = 777
    np.random.seed(seed)
    np.random.shuffle(x)
    # np.random.seed(seed)
    # np.random.shuffle(y)
    # x = np.expand_dims(x, axis=-1)

    x = np.asarray([scipy.misc.imresize(x_img, [size, size]) for x_img in x])
    x = np.expand_dims(x, axis=-1)
    return x

def load_cifar10(size=64) :
    (train_data, train_labels), (test_data, test_labels) = cifar10.load_data()
    train_data = normalize(train_data)
    test_data = normalize(test_data)

    x = np.concatenate((train_data, test_data), axis=0)
    # y = np.concatenate((train_labels, test_labels), axis=0).astype(np.int)

    seed = 777
    np.random.seed(seed)
    np.random.shuffle(x)
    # np.random.seed(seed)
    # np.random.shuffle(y)

    x = np.asarray([scipy.misc.imresize(x_img, [size, size]) for x_img in x])

    return x

def load_data(dataset_name, size=64) :

    x = glob(f'{dataset_name}/*/*.jpg')
    x.extend(glob(f'{dataset_name}/*.jpg'))
    x.extend(glob(f'{dataset_name}/*/*.png'))
    x.extend(glob(f'{dataset_name}/*.png'))

    return x


def preprocessing(x, size):
    x = scipy.misc.imread(x, mode='RGB')
    x = scipy.misc.imresize(x, [size, size])
    x = normalize(x)
    return x

def normalize(x) :
    return x/127.5 - 1

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def save_images_plt(images, size, image_path):
    pass

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
    # image = np.squeeze(merge(images, size)) # 채널이 1인거 제거 ?
    return imageio.imwrite(path, merge(images, size))


def inverse_transform(images):
    return (images+1.)/2.


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def show_all_variables():
    model_vars = tf.compat.v1.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def str2bool(x):
    return x.lower() in ('true')
