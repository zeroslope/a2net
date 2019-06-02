import os
from os import path

import numpy as np

import tensorflow as tf
# tf.enable_eager_execution()
import tensorlayer as tl

import skimage.io as io
from skimage.transform import resize
from skimage.color import rgb2yuv, yuv2rgb, rgb2gray
from skimage.measure import compare_ssim, compare_psnr

from config import global_config
CFG = global_config.cfg

from model import a2net

flags = tf.app.flags
flags.DEFINE_string('rain_img_path', None, 'The input rain image path. [None]')
flags.DEFINE_string('gt_img_path', None, 'The ground truth image path. [None]')
flags.DEFINE_string('weights_path', None, 'The pretrained weights path. [None]')
flags.DEFINE_string('save_dir', None, 'save_dir. [None]')
flags.DEFINE_integer('height', 240, 'image height [240]')
flags.DEFINE_integer('width', 360, 'image height [360]')

flags.mark_flag_as_required('rain_img_path')
flags.mark_flag_as_required('gt_img_path')
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

def main(_):
    x = tf.placeholder(dtype=tf.float32, 
        shape=[1, FLAGS.height, FLAGS.width, 3],
        name='input_tensor'
    )

    o_Y, o_UV, net_out = a2net(x, is_train=False, reuse=False)
    out_tensor = net_out.outputs

    rain_img = load_and_process_image(FLAGS.rain_img_path)
    gt_img = load_and_process_image(FLAGS.gt_img_path)

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.99

    sess = tf.InteractiveSession(config=config)

    saver = tf.train.Saver()
    saver.restore(sess, FLAGS.weights_path)

    out = sess.run([out_tensor], 
        feed_dict={
            x: np.expand_dims(rain_img, 0)
        })
    
    out_img = out[0][0]

    print(out_img.shape, gt_img.shape)

    out_img = yuv2rgb(out_img)
    gt_img = yuv2rgb(gt_img)

    ssim = compare_ssim(rgb2gray(out_img), rgb2gray(gt_img))
    psnr = compare_ssim(rgb2gray(out_img), rgb2gray(gt_img))

    print('SSIM: {:.5f}'.format(ssim))
    print('PSNR: {:.5f}'.format(psnr))

    io.imshow(out_img)
    io.imsave(path.join(FLAGS.save_dir, 'gen.png'), out_img / 255)

    sess.close()


if __name__ == "__main__":
    tf.app.run()