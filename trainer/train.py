import os
from os import path
import time
import glog as log
import numpy as np
import tensorflow as tf
import tensorlayer as tl

from data_loader import DataLoader
from config import global_config
CFG = global_config.cfg

from model import a2net, a2net_loss

flags = tf.app.flags
flags.DEFINE_string('dataset_dir', None, 'The dataset dir. [None]')
flags.DEFINE_string('save_dir', None, 'The model saving dir. [None]')
flags.DEFINE_string('weights_path', None, 'The pretrained weights path. [None]')
flags.DEFINE_float('alpha', 0.6, 'loss = l_Y + alpha * l_UV [0.6]')
flags.DEFINE_float('beta1', 0.5, 'beta1 [0.5]')
flags.DEFINE_float('lr', 0.0002, 'learning_rate [0.0002]')
flags.DEFINE_float('lr_decay', 0.5, 'lr_decay [0.5]')
flags.DEFINE_integer('decay_every', 100, 'decay_every [100]')
flags.DEFINE_integer('train_epochs', 10000, 'train_epochs')
flags.DEFINE_integer('batch_size', 4, 'batch_size [4]')
flags.DEFINE_integer('test_batch_size', 1, 'batch_size [1]')


flags.mark_flag_as_required('dataset_dir')
flags.mark_flag_as_required('save_dir')
FLAGS = flags.FLAGS


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

        with tf.device('/gpu:0'):
            ###======================== DEFIINE MODEL =======================###
            train_dataset = DataLoader(save_dir=FLAGS.dataset_dir, flag='train')
            val_dataset = DataLoader(save_dir=FLAGS.dataset_dir, flag='val')

            train_x, train_y = train_dataset.inputs(FLAGS.batch_size)
            val_x, val_y = val_dataset.inputs(FLAGS.batch_size)

            train_O_Y, train_O_UV, train_out = a2net(train_x, is_train=True, reuse=False)
            val_O_Y, val_O_UV, val_out = a2net(val_x, is_train=False, reuse=True)

            train_out_tensor = train_out.outputs
            
            val_out_tensor = val_out.outputs
            val_out_rgb_float = tf.image.yuv_to_rgb(val_out_tensor)
            val_out_rgb = tf.image.convert_image_dtype(val_out_rgb_float, tf.uint8)

            val_in_x_float = tf.image.yuv_to_rgb(val_x)
            val_in_x_rgb = tf.image.convert_image_dtype(val_in_x_float, tf.uint8)

            val_in_y_float = tf.image.yuv_to_rgb(val_y)
            val_in_y_rgb = tf.image.convert_image_dtype(val_in_y_float, tf.uint8)

            ###======================== DEFINE LOSS =========================###

            train_loss, train_l_ssim_Y, train_l_ssim_UV = a2net_loss(train_O_Y, train_O_UV, train_y, name='a2net_loss', reuse=False)
            val_loss, val_l_ssim_Y, val_l_ssim_UV = a2net_loss(val_O_Y, val_O_UV, val_y, name='a2net_loss', reuse=True)
        

        ####======================== DEFINE TRAIN OPTS ==============================###
        a2net_vars = tl.layers.get_variables_with_name('a2net', train_only=True, verbose=True)

        with tf.device('/gpu:0'):
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(FLAGS.lr , global_step, 100000, FLAGS.lr_decay, staircase=True)
            train_op = tf.train.AdamOptimizer(learning_rate, beta1=FLAGS.beta1).minimize(train_loss, var_list=a2net_vars)

        train_loss_scalar = tf.summary.scalar('train_loss', train_loss)
        train_l_ssim_Y_scalar = tf.summary.scalar('train_l_ssim_Y', train_l_ssim_Y)
        train_l_ssim_UV_scalar = tf.summary.scalar('train_l_ssim_UV', train_l_ssim_UV)
        val_loss_scalar = tf.summary.scalar('val_loss', val_loss)
        val_l_ssim_Y_scalar = tf.summary.scalar('val_l_ssim_Y', val_l_ssim_Y)
        val_l_ssim_UV_scalar = tf.summary.scalar('val_l_ssim_UV', val_l_ssim_UV)

        lr_scalar = tf.summary.scalar(name='learning_rate', tensor=learning_rate)

        train_summary = tf.summary.merge(
            [train_loss_scalar, train_l_ssim_Y_scalar, train_l_ssim_UV_scalar, lr_scalar]
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
        cost_time = time.time() - t_start

        summary_writer.add_summary(t_s, global_step=epoch)
        summary_writer.add_summary(v_s, global_step=epoch)

        log.info('Epoch_Train: {:d} train_loss: {:.5f} train_l_ssim_Y: {:.5f} train_l_ssim_UV: {:.5f} Cost_time: {:.5f}s'.format(epoch, t_l, t_l_Y, t_l_UV, cost_time))

        # Evaluate model
        if (epoch+1) % 50 == 0:
            v_in_x_rgb, v_in_y_rgb, v_out_rgb, v_l, v_l_Y, v_l_UV = sess.run([val_in_x_rgb, val_in_y_rgb, val_out_rgb, val_loss, val_l_ssim_Y, val_l_ssim_UV])

            gen_img = np.concatenate((v_in_x_rgb, v_out_rgb, v_in_y_rgb), axis=0)

            tl.visualize.save_images(gen_img, [3, FLAGS.batch_size], path.join(sample_save_dir, 'val_{}.png'.format(epoch)))
            log.info('Epoch_Val: {:d} val_loss: {:.5f} val_l_ssim_Y: {:.5f} val_l_ssim_UV: {:.5f}  Cost_time: {:.5f}s'.format(epoch, v_l, v_l_Y, v_l_UV, cost_time))

        # Save Model
        if (epoch+1) % 200 == 0:
            saver.save(sess=sess, save_path=model_save_path, global_step=epoch)

    summary_writer.close()
    sess.close()


if __name__ == "__main__":
    tf.app.run()