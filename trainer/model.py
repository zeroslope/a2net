#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
# tf.enable_eager_execution()
import tensorlayer as tl
from tensorlayer.layers import InputLayer, Conv2d, ConcatLayer, DeConv2d, ElementwiseLambdaLayer, LambdaLayer, BatchNormLayer

UV_SIZE = 24
BETA = 0.65


def a2net(x, is_train=True, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope('a2net', reuse=reuse):
        net_in = InputLayer(x, name='input')
        inputY = InputLayer(x[:, :, :, :1], name='inputY')
        inputUV = InputLayer(x[:, :, :, 1:], name='inputUV')

        # Encoder

        conv1 = Conv2d(net_in, 32, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None, name='encoder/conv1')
        conv1 = BatchNormLayer(conv1, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='encoder/bn1')
        conv2 = Conv2d(conv1, 32, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None, name='encoder/conv2')
        conv2 = BatchNormLayer(conv2, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='encoder/bn2')

        concat1 = ConcatLayer([conv1, conv2], concat_dim=-1, name='encoder/concat1')
        aggregation1 = Conv2d(concat1, 32, (4, 4), (2, 2), act=None, W_init=w_init, b_init=None, name='encoder/aggregation1')
        aggregation1 = BatchNormLayer(aggregation1, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='encoder/bn3')

        conv3 = Conv2d(aggregation1, 32, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None, name='encoder/conv3')
        conv3 = BatchNormLayer(conv3, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='encoder/bn4')

        concat2 = ConcatLayer([aggregation1, conv3], concat_dim=-1, name='encoder/concat2')
        aggregation2 = Conv2d(concat2, 32, (4, 4), (2, 2),act=None, W_init=w_init, b_init=None, name='encoder/aggregation2')
        aggregation2 = BatchNormLayer(aggregation2, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='encoder/bn5')

        conv4 = Conv2d(aggregation2, 32, (3, 3), (1, 1),act=None, W_init=w_init, b_init=None, name='encoder/conv4')
        conv4 = BatchNormLayer(conv4, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='encoder/bn6')

        concat3 = ConcatLayer([aggregation2, conv4], concat_dim=-1, name='encoder/concat3')
        aggregation3 = Conv2d(concat3, 32, (4, 4), (2, 2),act=None, W_init=w_init, b_init=None, name='encoder/aggregation3')
        aggregation3 = BatchNormLayer(aggregation3, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='encoder/bn7')

        # DecoderY

        convY_1 = Conv2d(aggregation3, 32, (3, 3), (1, 1),act=None, W_init=w_init, b_init=None, name='decoderY/conv1')
        convY_1 = BatchNormLayer(convY_1, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='decoderY/bn1')

        concatY_1 = ConcatLayer([aggregation3, convY_1], concat_dim=-1, name='decoderY/concat1')
        aggregationY_1 = DeConv2d(concatY_1, 32, (2, 2), (2, 2), act=None, W_init=w_init, b_init=None, name='decoderY/aggregation1')
        aggregationY_1 = BatchNormLayer(aggregationY_1, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='decoderY/bn2')

        copyY_1 = ConcatLayer([conv4, aggregationY_1], concat_dim=-1, name='decoderY/copy1')
        convY_2 = Conv2d(copyY_1, 32, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None, name='decoderY/conv2')
        convY_2 = BatchNormLayer(convY_2, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='decoderY/bn3')

        concatY_2 = ConcatLayer([copyY_1, convY_2], concat_dim=-1, name='decoderY/concat2')
        aggregationY_2 = DeConv2d(concatY_2, 32, (2, 2), (2, 2), act=None, W_init=w_init, b_init=None, name='decoderY/aggregation2')
        aggregationY_2 = BatchNormLayer(aggregationY_2, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='decoderY/bn4')

        copyY_2 = ConcatLayer([conv3, aggregationY_2], concat_dim=-1, name='decoderY/copy2')
        convY_3 = Conv2d(copyY_2, 32, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None, name='decoderY/conv3')
        convY_3 = BatchNormLayer(convY_3, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='decoderY/bn5')

        concatY_3 = ConcatLayer([copyY_2, convY_3], concat_dim=-1, name='decoderY/concat3')
        aggregationY_3 = DeConv2d(concatY_3, 32, (2, 2), (2, 2), act=None, W_init=w_init, b_init=None, name='decoderY/aggregation3')
        aggregationY_3 = BatchNormLayer(aggregationY_3, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='decoderY/bn6')

        copyY_3 = ConcatLayer([conv2, aggregationY_3], concat_dim=-1, name='decoderY/copy3')

        outputY = Conv2d(copyY_3, 1, (3, 3), (1, 1), act=tf.nn.tanh, name='decoderY/output')

        # DecoderUV

        convUV_1 = Conv2d(aggregation3, UV_SIZE, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None, name='decoderUV/conv1')
        convUV_1 = BatchNormLayer(convUV_1, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='decoderUV/bn1')

        concatUV_1 = ConcatLayer([aggregation3, convUV_1], concat_dim=-1, name='decoderUV/concat1')
        aggregationUV_1 = DeConv2d(concatUV_1, UV_SIZE, (2, 2), (2, 2),act=None, W_init=w_init, b_init=None, name='decoderUV/aggregation1')
        aggregationUV_1 = BatchNormLayer(aggregationUV_1, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='decoderUV/bn2')

        copyUV_1 = ConcatLayer([conv4, aggregationUV_1], concat_dim=-1, name='decoderUV/copy1')
        convUV_2 = Conv2d(copyUV_1, UV_SIZE, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None, name='decoderUV/conv2')
        convUV_2 = BatchNormLayer(convUV_2, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='decoderUV/bn3')

        concatUV_2 = ConcatLayer([copyUV_1, convUV_2], concat_dim=-1, name='decoderUV/concat2')
        aggregationUV_2 = DeConv2d(concatUV_2, UV_SIZE, (2, 2), (2, 2),act=None, W_init=w_init, b_init=None, name='decoderUV/aggregation2')
        aggregationUV_2 = BatchNormLayer(aggregationUV_2, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='decoderUV/bn4')

        copyUV_2 = ConcatLayer([conv3, aggregationUV_2], concat_dim=-1, name='decoderUV/copy2')
        convUV_3 = Conv2d(copyUV_2, UV_SIZE, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None, name='decoderUV/conv3')
        convUV_3 = BatchNormLayer(convUV_3, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='decoderUV/bn5')
        
        concatUV_3 = ConcatLayer([copyUV_2, convUV_3], concat_dim=-1, name='decoderUV/concat3')
        aggregationUV_3 = DeConv2d(concatUV_3, UV_SIZE, (2, 2), (2, 2), act=None, W_init=w_init, b_init=None, name='decoderUV/aggregation3')
        aggregationUV_3 = BatchNormLayer(aggregationUV_3, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='decoderUV/bn6')

        copyUV_3 = ConcatLayer([conv2, aggregationUV_3], concat_dim=-1, name='decoderUV/copy3')

        outputUV = Conv2d(copyUV_3, 2, (3, 3), (1, 1), act=tf.nn.tanh, name='decoderUV/output')

        outY_plus_Y = ElementwiseLambdaLayer(
            [outputY, inputY],
            fn=lambda x, y: BETA * x + (1 - BETA) * y,
            name='outY_plus_Y')

        outUV_plus_UV = ElementwiseLambdaLayer(
            [outputUV, inputUV],
            fn=lambda x, y: BETA * x + (1 - BETA) * y,
            name='outUV_plus_UV')

        net_out = ConcatLayer([outY_plus_Y, outUV_plus_UV], concat_dim=-1, name='net_out')

        return outY_plus_Y, outUV_plus_UV, net_out


def a2net_loss(o_Y, o_UV, gt, name, alpha=0.6, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        o_Y = (o_Y.outputs + 1) / 2 # [-1, 1] -> [0, 1]
        o_UV = (o_UV.outputs + 1) / 2 # [-1, 1] -> [0, 1]
        gt = (gt + 1) / 2 # [-1, 1] -> [0, 1]
        t_Y = gt[:,:,:,:1]
        t_UV = gt[:,:,:,1:]

        l_mse_Y = tl.cost.mean_squared_error(o_Y, t_Y, is_mean=True, name='loss/mse_Y')
        l_ssim_Y = tf.reduce_mean(tf.image.ssim(o_Y, t_Y, max_val=1.0), name='loss/ssim_Y')
        l_Y = l_mse_Y - l_ssim_Y

        l_mse_UV = tl.cost.mean_squared_error(o_UV, t_UV, is_mean=True, name='loss/mse_UV')
        l_ssim_UV = tf.reduce_mean(tf.image.ssim(o_UV, t_UV, max_val=1.0), name='loss/ssim_UV')
        l_UV = l_mse_UV - l_ssim_UV

        # TODO: I am too stupid.
        loss = l_Y + alpha * l_UV

        return loss, l_ssim_Y, l_ssim_UV