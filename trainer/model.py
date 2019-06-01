#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
# tf.enable_eager_execution()
import tensorlayer as tl
from tensorlayer.layers import InputLayer, Conv2d, ConcatLayer, DeConv2d, ElementwiseLambdaLayer, LambdaLayer


def a2Net(x, is_train=True, reuse=False):
    with tf.variable_scope('a2net', reuse=reuse):
        net_in = InputLayer(x, name='input')
        inputY = InputLayer(x[:, :, :, :1], name='inputY')
        inputUV = InputLayer(x[:, :, :, 1:], name='inputUV')

        # Encoder

        conv1 = Conv2d(net_in, 32, (3, 3), (1, 1), act=tf.nn.relu, name='encoder/conv1')
        conv2 = Conv2d(conv1, 32, (3, 3), (1, 1), act=tf.nn.relu, name='encoder/conv2')

        concat1 = ConcatLayer([conv1, conv2], concat_dim=-1, name='encoder/concat1')
        aggregation1 = Conv2d(concat1, 32, (4, 4), (2, 2), act=tf.nn.relu, name='encoder/aggregation1')
        conv3 = Conv2d(aggregation1, 32, (3, 3), (1, 1), act=tf.nn.relu, name='encoder/conv3')

        concat2 = ConcatLayer([aggregation1, conv3], concat_dim=-1, name='encoder/concat2')
        aggregation2 = Conv2d(concat2, 32, (4, 4), (2, 2), act=tf.nn.relu, name='encoder/aggregation2')
        conv4 = Conv2d(aggregation2, 32, (3, 3), (1, 1), act=tf.nn.relu, name='encoder/conv4')

        concat3 = ConcatLayer([aggregation2, conv4], concat_dim=-1, name='encoder/concat3')
        aggregation3 = Conv2d(concat3, 32, (4, 4), (2, 2), act=tf.nn.relu, name='encoder/aggregation3')

        # DecoderY

        convY_1 = Conv2d(aggregation3, 32, (3, 3), (1, 1), act=tf.nn.relu, name='decoderY/conv1')

        concatY_1 = ConcatLayer([aggregation3, convY_1], concat_dim=-1, name='decoderY/concat1')
        aggregationY_1 = DeConv2d(concatY_1, 32, (2, 2), (2, 2), act=tf.nn.relu, name='decoderY/aggregation1')
        copyY_1 = ConcatLayer([conv4, aggregationY_1], concat_dim=-1, name='decoderY/copy1')
        convY_2 = Conv2d(copyY_1, 32, (3, 3), (1, 1), act=tf.nn.relu, name='decoderY/conv2')

        concatY_2 = ConcatLayer([copyY_1, convY_2], concat_dim=-1, name='decoderY/concat2')
        aggregationY_2 = DeConv2d(concatY_2, 32, (2, 2), (2, 2), act=tf.nn.relu, name='decoderY/aggregation2')
        copyY_2 = ConcatLayer([conv3, aggregationY_2], concat_dim=-1, name='decoderY/copy2')
        convY_3 = Conv2d(copyY_2, 32, (3, 3), (1, 1), act=tf.nn.relu, name='decoderY/conv3')

        concatY_3 = ConcatLayer([copyY_2, convY_3], concat_dim=-1, name='decoderY/concat3')
        aggregationY_3 = DeConv2d(concatY_3, 32, (2, 2), (2, 2), act=tf.nn.relu, name='decoderY/aggregation3')
        copyY_3 = ConcatLayer([conv2, aggregationY_3], concat_dim=-1, name='decoderY/copy3')

        outputY = Conv2d(copyY_3, 1, (3, 3), (1, 1), act=tf.nn.tanh, name='decoderY/output')

        # DecoderUV

        convUV_1 = Conv2d(aggregation3, 24, (3, 3), (1, 1), act=tf.nn.relu, name='decoderUV/conv1')

        concatUV_1 = ConcatLayer([aggregation3, convUV_1], concat_dim=-1, name='decoderUV/concat1')
        aggregationUV_1 = DeConv2d(concatUV_1, 24, (2, 2), (2, 2), act=tf.nn.relu, name='decoderUV/aggregation1')
        copyUV_1 = ConcatLayer([conv4, aggregationUV_1], concat_dim=-1, name='decoderUV/copy1')
        convUV_2 = Conv2d(copyUV_1, 24, (3, 3), (1, 1), act=tf.nn.relu, name='decoderUV/conv2')

        concatUV_2 = ConcatLayer([copyUV_1, convUV_2], concat_dim=-1, name='decoderUV/concat2')
        aggregationUV_2 = DeConv2d(concatUV_2, 24, (2, 2), (2, 2), act=tf.nn.relu, name='decoderUV/aggregation2')
        copyUV_2 = ConcatLayer([conv3, aggregationUV_2], concat_dim=-1, name='decoderUV/copy2')
        convUV_3 = Conv2d(copyUV_2, 24, (3, 3), (1, 1), act=tf.nn.relu, name='decoderUV/conv3')
        
        concatUV_3 = ConcatLayer([copyUV_2, convUV_3], concat_dim=-1, name='decoderUV/concat3')
        aggregationUV_3 = DeConv2d(concatUV_3, 24, (2, 2), (2, 2), act=tf.nn.relu, name='decoderUV/aggregation3')
        copyUV_3 = ConcatLayer([conv2, aggregationUV_3], concat_dim=-1, name='decoderUV/copy3')

        outputUV = Conv2d(copyUV_3, 2, (3, 3), (1, 1), act=tf.nn.tanh, name='decoderUV/output')


        outY_plus_Y = ElementwiseLambdaLayer(
            [outputY, inputY],
            fn=lambda x, y: (2 * x + y + 1) / 4.,
            name='outY_plus_Y')

        outUV_plus_UV = ElementwiseLambdaLayer(
            [outputUV, inputUV],
            fn=lambda x, y: (2 * x + y) / 4.,
            name='outUV_plus_UV')

        net_out = ConcatLayer([outY_plus_Y, outUV_plus_UV], concat_dim=-1, name='net_out')

        return net_out
