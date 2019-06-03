import os
from os import path
import pathlib
import random
import argparse
import glog as log

import tensorflow as tf


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _get_img_bytes(img_path):
    r = None
    with tf.gfile.Open(img_path, 'rb') as fd:
        r = fd.read()
    return r


def _encode2feature(rain_img_path, gt_img_path):
    rain_img_raw = _get_img_bytes(rain_img_path)
    gt_img_raw = _get_img_bytes(gt_img_path)

    feature = {
        'rain': _bytes_feature(rain_img_raw),
        'gt': _bytes_feature(gt_img_raw)
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir',
                        type=str,
                        help='The source nsfw data dir path')
    parser.add_argument('--tfrecords_dir',
                        type=str,
                        help='The dir path to save converted tfrecords')

    return parser.parse_args()


class DataProducer(object):
    def __init__(self, dataset_dir):
        self._dataset_dir = dataset_dir
        self._rain_image_dir = path.join(dataset_dir, 'data')
        self._gt_image_dir = path.join(dataset_dir, 'gt')

        self._train_index_file_path = path.join(self._dataset_dir, 'train.txt')
        self._test_index_file_path = path.join(self._dataset_dir, 'test.txt')
        self._val_index_file_path = path.join(self._dataset_dir, 'val.txt')

        if not self._is_exist_dataset():
            raise ValueError('Please confirm your dataset is completed.')

        if not self._is_exist_index_file():
            self._generate_training_index_file()

    def _is_exist_dataset(self):
        return tf.gfile.Exists(self._rain_image_dir) and tf.gfile.Exists(
            self._gt_image_dir)

    def _is_exist_index_file(self):
        return tf.gfile.Exists(self._train_index_file_path) and \
            tf.gfile.Exists(self._test_index_file_path) and \
            tf.gfile.Exists(self._val_index_file_path)

    def _generate_training_index_file(self):
        def get_all_img_path(dir_path):
            dir_path = pathlib.Path(dir_path)
            img_paths = list(dir_path.glob('*.png'))
            img_paths.sort(key=lambda x: x.name)
            img_paths = [str(path) for path in img_paths]
            print(img_paths[:10])
            return img_paths

        def split_train_val_test(all_img_paths):
            image_count = len(all_img_paths)
            random.shuffle(all_img_paths)

            m1 = int(image_count * 0.85)
            m2 = int(image_count * 0.9)

            train_set = all_img_paths[:m1]
            val_set = all_img_paths[m1:m2]
            test_set = all_img_paths[m2:]

            return train_set, val_set, test_set

        rain_paths = get_all_img_path(self._rain_image_dir)
        gt_paths = get_all_img_path(self._gt_image_dir)

        train_set_paths, val_set_paths, test_set_paths = split_train_val_test(
            list(zip(rain_paths, gt_paths)))

        with tf.gfile.Open(path.join(self._dataset_dir, 'train.txt'), 'w') as fd:
            for r, c in train_set_paths:
                s = "{} {}\n".format(r, c)
                fd.write(s)

        with tf.gfile.Open(path.join(self._dataset_dir, 'val.txt'), 'w') as fd:
            for r, c in val_set_paths:
                s = "{} {}\n".format(r, c)
                fd.write(s)

        with tf.gfile.Open(path.join(self._dataset_dir, 'test.txt'), 'w') as fd:
            for r, c in test_set_paths:
                s = "{} {}\n".format(r, c)
                fd.write(s)

        log.info('Generate training example index file complete')

    def generate_tfrecords(self, save_dir):
        def generate(index_file_path, set_name):
            img_paths = []
            with tf.gfile.Open(index_file_path, 'r') as fd:
                for line in fd:
                    r, c = line.rstrip('\r').rstrip('\n').split(' ')
                    img_paths.append((r, c))

            save_path = "{}.tfrecords".format(set_name)
            tfrecords_path = path.join(save_dir, save_path)
            with tf.python_io.TFRecordWriter(tfrecords_path) as _writer:
                for r, c in img_paths:
                    example = _encode2feature(r, c)
                    _writer.write(example.SerializeToString())

        tf.gfile.MakeDirs(save_dir)
        # os.makedirs(save_dir, exist_ok=True)

        log.info('Generating training example tfrecords')

        generate(self._train_index_file_path, 'train')

        log.info('Generate training example tfrecords complete')

        log.info('Generating validation example tfrecords')

        generate(self._val_index_file_path, 'val')

        log.info('Generate validation example tfrecords complete')

        log.info('Generating testing example tfrecords')

        generate(self._test_index_file_path, 'test')

        log.info('Generate testing example tfrecords complete')


if __name__ == '__main__':
    args = init_args()

    assert tf.gfile.Exists(args.dataset_dir), '{:s} not exist'.format(
        args.dataset_dir)

    producer = DataProducer(dataset_dir=args.dataset_dir)
    producer.generate_tfrecords(save_dir=args.tfrecords_dir)