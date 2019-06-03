import os
from os import path
import pathlib
import random
import argparse
import glog as log

import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

from trainer.global_config import cfg
CFG = cfg

u_trans = 1 / 0.43601035
v_trans = 1 / 0.61497538

# Y U V -> [-1, 1]
def _normalize(image):
    return tf.stack([
        image[:,:,0]*2-1,
        image[:,:,1]*u_trans,
        image[:,:,2]*v_trans
    ], axis=-1)

def preprocess_image(image):
    #TODO: add size
    image = tf.image.decode_png(image, channels=3)
    image_float = tf.image.convert_image_dtype(image, tf.float32)
    image_yuv = tf.image.rgb_to_yuv(image_float)
    return _normalize(image_yuv)


def _process_rain_gt(record):
    return (preprocess_image(record['rain']), preprocess_image(record['gt']))


def augment_for_train(rain_image, clean_image):
    rain_image = tf.cast(rain_image, tf.float32)
    clean_image = tf.cast(clean_image, tf.float32)

    return random_crop_batch_images(
        rain_image=rain_image,
        clean_image=clean_image,
        cropped_size=[CFG.TRAIN.CROP_IMG_WIDTH, CFG.TRAIN.CROP_IMG_HEIGHT])


def augment_for_test(rain_image, clean_image):
    return rain_image, clean_image


def random_crop_batch_images(rain_image, clean_image, cropped_size):
    concat_images = tf.concat([rain_image, clean_image], axis=-1)

    concat_cropped_images = tf.image.random_crop(
        concat_images,
        [cropped_size[1], cropped_size[0],
         tf.shape(concat_images)[-1]],
        seed=tf.random.set_random_seed(1234))

    cropped_rain_image = tf.slice(concat_cropped_images,
                                  begin=[0, 0, 0],
                                  size=[cropped_size[1], cropped_size[0], 3])
    cropped_clean_image = tf.slice(concat_cropped_images,
                                   begin=[0, 0, 3],
                                   size=[cropped_size[1], cropped_size[0], 3])

    return cropped_rain_image, cropped_clean_image


image_feature_description = {
    'rain': tf.FixedLenFeature([], tf.string),
    'gt': tf.FixedLenFeature([], tf.string),
}


def _parse_image_function(example_proto):
    return tf.parse_single_example(example_proto, image_feature_description)


class DataLoader(object):
    def __init__(self, save_dir, flag='train'):
        self._save_dir = save_dir

        if not tf.gfile.Exists(self._save_dir):
            raise ValueError('{:s} not exist, please check again'.format(
                self._save_dir))

        self._dataset_flags = flag.lower()
        if self._dataset_flags not in ['train', 'test', 'val']:
            raise ValueError(
                'flags of the data feeder should be \'train\', \'test\', \'val\''
            )

    def inputs(self, batch_size, num_epochs=1):
        """
        dataset feed pipline input
        :param batch_size:
        :param num_epochs:
        :return: A tuple (images, labels), where:
                    * images is a float tensor with shape [batch_size, H, W, C]
                      in the range [-0.5, 0.5].
                    * labels is an int32 tensor with shape [batch_size] with the true label,
                      a number in the range [0, CLASS_NUMS).
        """
        # if not num_epochs:
        #     num_epochs = None

        tfrecords_file_path = path.join(
            self._save_dir, '{}.tfrecords'.format(self._dataset_flags))

        with tf.name_scope('input_tensor'):
            dataset = tf.data.TFRecordDataset(tfrecords_file_path)

            dataset = dataset.map(
                map_func=_parse_image_function,
                num_parallel_calls=CFG.TRAIN.CPU_MULTI_PROCESS_NUMS)
            dataset = dataset.map(
                map_func=_process_rain_gt,
                num_parallel_calls=CFG.TRAIN.CPU_MULTI_PROCESS_NUMS)
            if self._dataset_flags != 'test':
                dataset = dataset.map(
                    map_func=augment_for_train,
                    num_parallel_calls=CFG.TRAIN.CPU_MULTI_PROCESS_NUMS)
            else:
                dataset = dataset.map(
                    map_func=augment_for_test,
                    num_parallel_calls=CFG.TRAIN.CPU_MULTI_PROCESS_NUMS)

            # The shuffle transformation uses a finite-sized buffer to shuffle elements
            # in memory. The parameter is the number of elements in the buffer. For
            # completely uniform shuffling, set the parameter to be the same as the
            # number of elements in the dataset.
            if self._dataset_flags != 'test':
                dataset = dataset.shuffle(buffer_size=1024)
                # repeat 不加参数可以无限循环，否则需要catch OutOfRangeError
                dataset = dataset.repeat()

            dataset = dataset.batch(batch_size, drop_remainder=True)
            dataset = dataset.prefetch(buffer_size=AUTOTUNE)

            iterator = dataset.make_one_shot_iterator()

        return iterator.get_next(
            name='{:s}_IteratorGetNext'.format(self._dataset_flags))


if __name__ == "__main__":
    dl = DataLoader("/Users/zeroslope/ai/a2net/trainer/data/tfrecord", 'train')
    r = dl.inputs(4)

    print(r)