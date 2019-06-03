import os
from os import path
import pathlib

import numpy as np

import tensorflow as tf
import tensorlayer as tl

import skimage.io as io
from skimage.transform import resize
from skimage.color import rgb2yuv, yuv2rgb, rgb2gray
from skimage.measure import compare_ssim, compare_psnr

from trainer.global_config import cfg
CFG = cfg

from trainer.model import a2net

flags = tf.app.flags
flags.DEFINE_string('dataset_dir', None, 'The dataset dir path. [None]')
flags.DEFINE_string('weights_path', None, 'The pretrained weights path. [None]')
flags.DEFINE_string('save_dir', None, 'save_dir. [None]')
flags.DEFINE_integer('height', 480, 'image height [240]')
flags.DEFINE_integer('width', 720, 'image height [360]')

flags.mark_flag_as_required('dataset_dir')
flags.mark_flag_as_required('weights_path')
flags.mark_flag_as_required('save_dir')
FLAGS = flags.FLAGS

def preprocess_image(image):
    # image = tf.image.decode_png(image, channels=3)
    # image = tf.image.resize(image, [FLAGS.height, FLAGS.width])
    # image_float = tf.image.convert_image_dtype(image, tf.float32)
    # image_yuv = tf.image.rgb_to_yuv(image_float)
    image = resize(image, (FLAGS.height, FLAGS.width))
    image_yuv = rgb2yuv(image)
    return image_yuv

def load_and_process_image(image_path):
    # img_raw = tf.read_file(image_path)
    img_raw = io.imread(image_path)
    return preprocess_image(img_raw)

def get_img_paths(dataset_dir):
    assert path.exists(dataset_dir), '{:s} not exist'.format(dataset_dir)

    def get_all_img_path(dir_path):
        dir_path = pathlib.Path(dir_path)
        img_paths = list(dir_path.glob('*.png'))
        img_paths.sort(key=lambda x: x.name)
        img_paths = [str(path) for path in img_paths]
        return img_paths

    rain_img_dir = path.join(dataset_dir, 'data')
    gt_img_dir = path.join(dataset_dir, 'gt')

    rain_paths = get_all_img_path(rain_img_dir)
    gt_paths = get_all_img_path(gt_img_dir)

    return list(zip(rain_paths, gt_paths))

def main(_):
    tl.files.exists_or_mkdir(FLAGS.save_dir)

    img_paths = get_img_paths(FLAGS.dataset_dir)


    with tf.device('/cpu:0'):
        x = tf.placeholder(dtype=tf.float32, 
            shape=[1, None, None, 3],
            name='input_tensor'
        )
        o_Y, o_UV, net_out = a2net(x, is_train=False, reuse=False)
        out_tensor = net_out.outputs

    config = tf.ConfigProto()
    config.allow_soft_placement = True

    sess = tf.InteractiveSession(config=config)

    saver = tf.train.Saver()
    saver.restore(sess, FLAGS.weights_path)

    ssim_list = []
    psnr_list = []

    for r, g in img_paths:

        rain_img = load_and_process_image(r)
        gt_img = load_and_process_image(g)

        out = sess.run(out_tensor, 
            feed_dict={
                x: np.expand_dims(rain_img, 0)
            })
    
        out_img = out[0]

        rain_img = yuv2rgb(rain_img)
        out_img = yuv2rgb(out_img)
        gt_img = yuv2rgb(gt_img)

        ssim = compare_ssim(rgb2gray(gt_img), rgb2gray(out_img))
        psnr = compare_psnr(rgb2gray(gt_img), rgb2gray(out_img))

        ssim_list.append(ssim)
        psnr_list.append(psnr)

        print('SSIM: {:.5f}'.format(ssim))
        print('PSNR: {:.5f}'.format(psnr))

        gen_img = np.array([rain_img, out_img, gt_img])

        num = r.split('/')[-1].split('_')[0]

        tl.visualize.save_images(gen_img, [1, 3], path.join(FLAGS.save_dir, 'out_{}.png'.format(num)))

    sess.close()

    print('SSIM: {:.5f}'.format(np.mean(ssim_list)))
    print('PSNR: {:.5f}'.format(np.mean(psnr_list)))


if __name__ == "__main__":
    tf.app.run()