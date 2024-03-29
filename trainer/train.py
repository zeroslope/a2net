import os
from os import path
import time
import glog as log
import numpy as np
import tensorflow as tf
import tensorlayer as tl 

import imageio
from tempfile import NamedTemporaryFile

from trainer.data_loader import DataLoader
from trainer.global_config import cfg
CFG = cfg

from trainer.model import a2net, a2net_loss

flags = tf.app.flags
flags.DEFINE_string('dataset_dir', None, 'The dataset dir. [None]')
flags.DEFINE_string('save_dir', None, 'The model saving dir. [None]')
flags.DEFINE_string('weights_path', None, 'The pretrained weights path. [None]')
flags.DEFINE_float('alpha', 0.6, 'loss = l_Y + alpha * l_UV [0.6]')
flags.DEFINE_float('beta1', 0.5, 'beta1 [0.5]')
flags.DEFINE_float('lr', 0.0002, 'learning_rate [0.0002]')
flags.DEFINE_float('lr_decay', 0.90, 'lr_decay [0.5]')
flags.DEFINE_integer('decay_every', 50, 'decay_every [200]')
flags.DEFINE_integer('train_epochs', 15000, 'train_epochs')
flags.DEFINE_integer('batch_size', 32, 'batch_size [32]')
flags.DEFINE_integer('test_batch_size', 16, 'batch_size [16]')


flags.mark_flag_as_required('dataset_dir')
flags.mark_flag_as_required('save_dir')
FLAGS = flags.FLAGS

def _save_images(images, size, image_path='_temp.png'):
    if len(images.shape) == 3:  # Greyscale [batch, h, w] --> [batch, h, w, 1]
        images = images[:, :, :, np.newaxis]

    def merge(images, size):
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1], 3), dtype=images.dtype)
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img

    def imsave(images, size, path):
        if np.max(images) <= 1 and (-1 <= np.min(images) < 0):
            images = ((images + 1) * 127.5).astype(np.uint8)
        elif np.max(images) <= 1 and np.min(images) >= 0:
            images = (images * 255).astype(np.uint8)

        return imageio.imwrite(path, merge(images, size))

    if len(images) > size[0] * size[1]:
        raise AssertionError("number of images should be equal or less than size[0] * size[1] {}".format(len(images)))

    return imsave(images, size, image_path)

def save_images(images, size, image_path):
    if image_path[:2] == 'gs':
        with NamedTemporaryFile(suffix='.png') as temp:
            _save_images(images, size, temp.name)
            with tf.gfile.GFile(temp.name, mode='rb') as fr:
                d = fr.read()
                with tf.gfile.GFile(image_path, 'wb') as fw:
                    fw.write(d) 
    else:
        _save_images(images, size, image_path)

def _yuv_to_rgb(images):
    val_out_tensor = tf.stack([
        (images[:,:,:,0] + 1) / 2,
        images[:,:,:,1] / 2,
        images[:,:,:,2] / 2,
    ], axis=-1)
    val_out_rgb_float = tf.image.yuv_to_rgb(val_out_tensor)
    return tf.image.convert_image_dtype(val_out_rgb_float, tf.uint8)

def main(_):
    tboard_save_dir = path.join(FLAGS.save_dir, 'tboard')
    sample_save_dir = path.join(FLAGS.save_dir, 'sample')
    model_save_dir = path.join(FLAGS.save_dir, 'model')

    tl.files.exists_or_mkdir(path.join(FLAGS.save_dir, 'model'))
    tl.files.exists_or_mkdir(path.join(FLAGS.save_dir, 'tboard'))
    tl.files.exists_or_mkdir(path.join(FLAGS.save_dir, 'sample'))

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    # config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.99

    with tf.device('/cpu:0'):
        sess = tf.Session(config=config)
        ###======================== DEFIINE MODEL =======================###
        train_dataset = DataLoader(save_dir=FLAGS.dataset_dir, flag='train')
        val_dataset = DataLoader(save_dir=FLAGS.dataset_dir, flag='val')

        # iterator_train = train_dataset.inputs(FLAGS.batch_size)
        # train_x, train_y = iterator_train.get_next()
        # iterator_val = val_dataset.inputs(FLAGS.test_batch_size)
        # val_x, val_y = iterator_val.get_next()
        train_x, train_y = train_dataset.inputs(FLAGS.batch_size)
        val_x, val_y = val_dataset.inputs(FLAGS.test_batch_size)

    with tf.device('/gpu:0'):
        
        train_O_Y, train_O_UV, train_out = a2net(train_x, is_train=True, reuse=False)
        val_O_Y, val_O_UV, val_out = a2net(val_x, is_train=False, reuse=True)

        train_out_tensor = train_out.outputs
        
        val_out_tensor = val_out.outputs
        
        val_out_rgb = _yuv_to_rgb(val_out_tensor)

        val_in_x_rgb = _yuv_to_rgb(val_x)

        val_in_y_rgb = _yuv_to_rgb(val_y)

        ###======================== DEFINE LOSS =========================###

        train_loss, train_l_ssim_Y, train_l_ssim_UV = a2net_loss(train_O_Y, train_O_UV, train_y, name='a2net_loss', reuse=False)
        val_loss, val_l_ssim_Y, val_l_ssim_UV = a2net_loss(val_O_Y, val_O_UV, val_y, name='a2net_loss', reuse=True)
        

        ####======================== DEFINE TRAIN OPTS ==============================###
        a2net_vars = tl.layers.get_variables_with_name('a2net', train_only=True, verbose=True)


        global_step = tf.Variable(0, trainable=False)
        # learning_rate = tf.train.exponential_decay(FLAGS.lr, global_step, FLAGS.decay_every, FLAGS.lr_decay, staircase=True)
        learning_rate = tf.train.noisy_linear_cosine_decay(learning_rate=FLAGS.lr, global_step=global_step, decay_steps=FLAGS.decay_every, initial_variance=0.01, variance_decay=0.1, num_periods=0.2, alpha=0.5, beta=0.2)
        train_op = tf.train.AdamOptimizer(learning_rate, beta1=FLAGS.beta1).minimize(train_loss, var_list=a2net_vars, global_step=global_step)

        train_loss_scalar = tf.summary.scalar('train_loss', train_loss)
        train_l_ssim_Y_scalar = tf.summary.scalar('train_l_ssim_Y', train_l_ssim_Y)
        train_l_ssim_UV_scalar = tf.summary.scalar('train_l_ssim_UV', train_l_ssim_UV)
        val_loss_scalar = tf.summary.scalar('val_loss', val_loss)
        val_l_ssim_Y_scalar = tf.summary.scalar('val_l_ssim_Y', val_l_ssim_Y)
        val_l_ssim_UV_scalar = tf.summary.scalar('val_l_ssim_UV', val_l_ssim_UV)

        lr_scalar = tf.summary.scalar(name='learning_rate', tensor=learning_rate)
        global_step_scalar = tf.summary.scalar(name='global_step', tensor=global_step)


        train_summary = tf.summary.merge(
            [train_loss_scalar, train_l_ssim_Y_scalar, train_l_ssim_UV_scalar, lr_scalar, global_step_scalar]
        )
        val_summary = tf.summary.merge(
            [val_loss_scalar, val_l_ssim_Y_scalar, val_l_ssim_UV_scalar]
        )

        ###======================== LOAD MODEL ==============================###
        tl.layers.initialize_global_variables(sess)
        # tl.files.load_and_assign_npz(sess=sess, name=FLAGS.save_dir+'/u_net_{}.npz'.format(task), network=net)

        train_out.print_params(details=False)
        train_out.print_layers()


        # set tf saver
        saver = tf.train.Saver()

        train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        model_name = 'a2net_{:s}.ckpt'.format(str(train_start_time))
        model_save_path = path.join(model_save_dir, model_name)

        summary_writer = tf.summary.FileWriter(tboard_save_dir)
        summary_writer.add_graph(sess.graph)

        log.info('Global configuration is as follows:')
        # log.info(FLAGS)

        tf.train.write_graph(graph_or_graph_def=sess.graph, logdir='', name='{:s}/a2net.pb'.format(model_save_dir))

        if FLAGS.weights_path is None:
            log.info('Training from scratch')
            # init = tf.global_variables_initializer()
            # sess.run(init)
        else:
            log.info('Restore model from last model checkpoint {:s}'.format(FLAGS.weights_path))
            saver.restore(sess=sess, save_path=FLAGS.weights_path)

        for epoch in range(FLAGS.train_epochs):
            t_start = time.time()
            t_out, t_l, t_l_Y, t_l_UV, t_s, v_s, _ = sess.run([train_out_tensor, train_loss, train_l_ssim_Y, train_l_ssim_UV, train_summary, val_summary, train_op])
            # t_out, t_l, t_l_Y, t_l_UV, t_s, _ = sess.run([train_out_tensor, train_loss, train_l_ssim_Y, train_l_ssim_UV, train_summary, train_op])
            cost_time = time.time() - t_start

            summary_writer.add_summary(t_s, global_step=epoch)
            summary_writer.add_summary(v_s, global_step=epoch)

            log.info('Epoch_Train: {:d} train_loss: {:.5f} train_l_ssim_Y: {:.5f} train_l_ssim_UV: {:.5f} Cost_time: {:.5f}s'.format(epoch, t_l, t_l_Y, t_l_UV, cost_time))

            # Evaluate model
            if (epoch+1) % 50 == 0:
                v_in_x_rgb, v_in_y_rgb, v_out_rgb, v_l, v_l_Y, v_l_UV = sess.run([val_in_x_rgb, val_in_y_rgb, val_out_rgb, val_loss, val_l_ssim_Y, val_l_ssim_UV])
                
                gen_img = np.concatenate((v_in_x_rgb, v_out_rgb, v_in_y_rgb), axis=0)

                save_images(gen_img, [3, FLAGS.test_batch_size], path.join(sample_save_dir, 'val_{}.png'.format(epoch)))

                log.info('Epoch_Val: {:d} val_loss: {:.5f} val_l_ssim_Y: {:.5f} val_l_ssim_UV: {:.5f}  Cost_time: {:.5f}s'.format(epoch, v_l, v_l_Y, v_l_UV, cost_time))

            # Save Model
            if (epoch+1) % 250 == 0:
                saver.save(sess=sess, save_path=model_save_path, global_step=epoch)

    summary_writer.close()
    sess.close()


if __name__ == "__main__":
    tf.app.run()