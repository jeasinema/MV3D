from net.utility.file import *
from net.blocks import *
from net.rpn_nms_op import tf_rpn_nms
from net.roipooling_op import roi_pool as tf_roipooling
from config import cfg
from net.resnet import ResnetBuilder
from keras.models import Model
import keras.applications.xception as xcep
from keras.preprocessing import image
from keras.models import Model
from keras import layers
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    SeparableConv2D,
    Conv2D,
    BatchNormalization, 
    MaxPooling2D
    )
import numpy as np

top_view_rpn_name = 'top_view_rpn'
imfeature_net_name = 'image_feature'
frontfeature_net_name = 'front_feature'
fusion_net_name = 'fusion'
conv3d_net_name = 'conv3d_for_regress'


def top_feature_net(input, anchors, inds_inside, num_bases, nms_thresh):
    """temporary net for debugging only. may not follow the paper exactly .... 
    :param input: 
    :param anchors: 
    :param inds_inside: 
    :param num_bases: 
    :return: 
            top_features, top_scores, top_probs, top_deltas, proposals, proposal_scores
    """
    stride=1.
    #with tf.variable_scope('top-preprocess') as scope:
    #    input = input

    with tf.variable_scope('top-block-1') as scope:
        block = conv2d_bn_relu(input, num_kernels=32, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        block = conv2d_bn_relu(block, num_kernels=32, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        block = maxpool(block, kernel_size=(2,2), stride=[1,2,2,1], padding='SAME', name='4' )
        stride *=2

    with tf.variable_scope('top-block-2') as scope:
        block = conv2d_bn_relu(block, num_kernels=64, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        block = conv2d_bn_relu(block, num_kernels=64, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        block = maxpool(block, kernel_size=(2,2), stride=[1,2,2,1], padding='SAME', name='4' )
        stride *=2 

    with tf.variable_scope('top-block-3') as scope:
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='3')
        block = maxpool(block, kernel_size=(2,2), stride=[1,2,2,1], padding='SAME', name='4' )
        stride *=2

    with tf.variable_scope('top-block-4') as scope:
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='3')


    with tf.variable_scope('top-for-rpn') as scope:
        up   = upsample2d(block, factor = 2, has_bias=True, trainable=True, name='1')
        top_rpn_stride = stride/2
        up      = conv2d_bn_relu(up, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        scores  = conv2d(up, num_kernels=2*num_bases, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='score')
        probs   = tf.nn.softmax( tf.reshape(scores,[-1,2]), name='prob')
        deltas  = conv2d(up, num_kernels=4*num_bases, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='delta')

    with tf.variable_scope('top-for-rcnn') as scope:
        block = upsample2d(block, factor=4, has_bias=True, trainable=True, name='1')
        top_rcnn_stride = stride/4

    #<todo> flip to train and test mode nms (e.g. different nms_pre_topn values): use tf.cond
    with tf.variable_scope('top-nms') as scope:    #non-max
        batch_size, img_height, img_width, img_channel = input.get_shape().as_list()
        img_scale = 1
        rois, roi_scores = tf_rpn_nms( probs, deltas, anchors, inds_inside,
                                       top_rpn_stride, img_width, img_height, img_scale,
                                       nms_thresh=nms_thresh, min_size=top_rpn_stride, #nms_pre_topn=500, nms_post_topn=100, too less!
                                       name ='nms')
    feature = block

    print ('top: rpn_scale=%f, rpn_stride=%d, rcnn_scale:%f, rcnn_stride=%d'%(1./top_rpn_stride, top_rpn_stride, 1./top_rcnn_stride, top_rcnn_stride))
    return feature, scores, probs, deltas, rois, roi_scores, top_rpn_stride, top_rcnn_stride


def top_feature_net_r(input, anchors, inds_inside, num_bases, nms_thresh):
    """
    :param input: 
    :param anchors: 
    :param inds_inside: 
    :param num_bases: 
    :return: 
            top_features, top_scores, top_probs, top_deltas, proposals, proposal_scores
    """
    stride=1.
    #with tf.variable_scope('top-preprocess') as scope:
    #    input = input
    batch_size, img_height, img_width, img_channel = input.get_shape().as_list()

    with tf.variable_scope('feature-extract-resnet') as scope:
        print('build_resnet')
        block = ResnetBuilder.resnet_tiny(input)

        # resnet_input = resnet.get_layer('input_1').input
        # resnet_output = resnet.get_layer('add_7').output
        # resnet_f = Model(inputs=resnet_input, outputs=resnet_output)  # add_7
        # # print(resnet_f.summary())
        # block = resnet_f(input)
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(1, 1), stride=[1, 1, 1, 1], padding='SAME', name='2')
        stride = 8


    with tf.variable_scope('predict-for-rpn') as scope:
        up = upsample2d(block, factor = 2, has_bias=True, trainable=True, name='1')
        top_rpn_stride = stride/2
        up = conv2d_bn_relu(up, num_kernels=128, kernel_size=(3, 3), stride=[1, 1, 1, 1], padding='SAME', name='2')
        scores = conv2d(up, num_kernels=2 * num_bases, kernel_size=(1, 1), stride=[1, 1, 1, 1], padding='SAME',name='score')
        probs = tf.nn.softmax(tf.reshape(scores, [-1, 2]), name='prob')
        deltas = conv2d(up, num_kernels=4 * num_bases, kernel_size=(1, 1), stride=[1, 1, 1, 1], padding='SAME',name='delta')

    with tf.variable_scope('for-rcnn') as scope:
        block = upsample2d(block, factor=4, has_bias=True, trainable=True, name='1')
        top_rcnn_stride = stride/4

    #<todo> flip to train and test mode nms (e.g. different nms_pre_topn values): use tf.cond
    with tf.variable_scope('NMS') as scope:    #non-max
        img_scale = 1
        rois, roi_scores = tf_rpn_nms( probs, deltas, anchors, inds_inside,
                                       top_rpn_stride, img_width, img_height, img_scale,
                                       nms_thresh=nms_thresh, min_size=top_rpn_stride, #nms_pre_topn=500, nms_post_topn=100,
                                       name ='nms')

    feature = block

    print ('top: rpn_scale=%f, rpn_stride=%d, rcnn_scale:%f, rcnn_stride=%d'%(1./top_rpn_stride, top_rpn_stride, 1./top_rcnn_stride, top_rcnn_stride))
    return feature, scores, probs, deltas, rois, roi_scores, top_rpn_stride, top_rcnn_stride


# def top_feature_net_feature_only(input):
#     stride=1.
#     #with tf.variable_scope('top-preprocess') as scope:
#     #    input = input
#     with tf.variable_scope('feature_only'):
#         with tf.variable_scope('top-block-1') as scope:
#             block = conv2d_bn_relu(input, num_kernels=32, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
#             block = conv2d_bn_relu(block, num_kernels=32, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
#             block = maxpool(block, kernel_size=(2,2), stride=[1,2,2,1], padding='SAME', name='4' )
#             stride *=2

#         with tf.variable_scope('top-block-2') as scope:
#             block = conv2d_bn_relu(block, num_kernels=64, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
#             block = conv2d_bn_relu(block, num_kernels=64, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
#             block = maxpool(block, kernel_size=(2,2), stride=[1,2,2,1], padding='SAME', name='4' )
#             stride *=2

#         with tf.variable_scope('top-block-3') as scope:
#             block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
#             block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
#             block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='3')
#             block = maxpool(block, kernel_size=(2,2), stride=[1,2,2,1], padding='SAME', name='4' )
#             stride *=2

#         with tf.variable_scope('top-block-4') as scope:
#             block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
#             block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
#             block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='3')

#     #<todo> feature = upsample2d(block, factor = 4,  ...)
#     feature = block

#     print ('top: scale=%f, stride=%d'%(1./stride, stride))
#     return feature, stride


# def top_feature_net_r_feature_only(input):
#     stride=1.
#     #with tf.variable_scope('top-preprocess') as scope:
#     #    input = input
#     batch_size, img_height, img_width, img_channel = input.get_shape().as_list()

#     with tf.variable_scope('feature_only'):
#         with tf.variable_scope('feature-extract-resnet') as scope:
#             print('build_resnet')
#             block = ResnetBuilder.resnet_tiny(input)

#             # resnet_input = resnet.get_layer('input_1').input
#             # resnet_output = resnet.get_layer('add_7').output
#             # resnet_f = Model(inputs=resnet_input, outputs=resnet_output)  # add_7
#             # # print(resnet_f.summary())
#             # block = resnet_f(input)
#             block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(1, 1), stride=[1, 1, 1, 1], padding='SAME', name='2')
#             stride = 8

#     #<todo> feature = upsample2d(block, factor = 4,  ...)
#     feature = block

#     print ('top: scale=%f, stride=%d'%(1./stride, stride))
#     return feature, stride

#------------------------------------------------------------------------------
def rgb_feature_net(input):

    stride=1.
    #with tf.variable_scope('rgb-preprocess') as scope:
    #   input = input-128

    with tf.variable_scope('rgb-block-1') as scope:
        block = conv2d_bn_relu(input, num_kernels=32, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        block = conv2d_bn_relu(block, num_kernels=32, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        block = maxpool(block, kernel_size=(2,2), stride=[1,2,2,1], padding='SAME', name='4' )
        stride *=2

    with tf.variable_scope('rgb-block-2') as scope:
        block = conv2d_bn_relu(block, num_kernels=64, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        block = conv2d_bn_relu(block, num_kernels=64, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        block = maxpool(block, kernel_size=(2,2), stride=[1,2,2,1], padding='SAME', name='4' )
        stride *=2

    with tf.variable_scope('rgb-block-3') as scope:
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='3')
        block = maxpool(block, kernel_size=(2,2), stride=[1,2,2,1], padding='SAME', name='4' )
        stride *=2

    with tf.variable_scope('rgb-block-4') as scope:
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='3')

    with tf.variable_scope('rgb-block-5') as scope:
        block = upsample2d(block, factor=2, has_bias=True, trainable=True, name='1')
        stride /= 2

    feature = block


    print ('rgb : scale=%f, stride=%d'%(1./stride, stride))
    return feature, stride

def rgb_feature_net_r(input):

    #with tf.variable_scope('rgb-preprocess') as scope:
    #   input = input-128

    batch_size, img_height, img_width, img_channel = input.get_shape().as_list()

    with tf.variable_scope('resnet-block-1') as scope:
        print('build_resnet')
        block = ResnetBuilder.resnet_tiny(input)
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(1, 1), stride=[1, 1, 1, 1], padding='SAME', name='2')
        stride = 8

    with tf.variable_scope('resnet-block-2') as scope:
        block = upsample2d(block, factor=2, has_bias=True, trainable=True, name='1')
        stride /= 2
    
    feature = block

    print ('rgb : scale=%f, stride=%d'%(1./stride, stride))
    return feature, stride


def rgb_feature_net_x(input):

    # Xception feature extractor

    batch_size, img_height, img_width, img_channel = input.get_shape().as_list()

    # with tf.variable_scope('xception_model'):
    #     base_model= xcep.Xception(include_top=False, weights=None,
    #                               input_shape=(img_height, img_width, img_channel ))
    # # print(base_model.summary())
    #
    #     base_model_input = base_model.get_layer('input_2').input
    #     base_model_output = base_model.get_layer('block12_sepconv3_bn').output
    # # print(model.summary())

    with tf.variable_scope('preprocess'):
        block = maxpool(input, kernel_size=(2,2), stride=[1,2,2,1], padding='SAME', name='4' )
        block = xcep.preprocess_input(block)

    with tf.variable_scope('feature_extract'):
        # keras/applications/xception.py
        print('build Xception')
        x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, name='block1_conv1')(block)
        x = BatchNormalization(name='block1_conv1_bn')(x)
        x = Activation('relu', name='block1_conv1_act')(x)
        x = Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
        x = BatchNormalization(name='block1_conv2_bn')(x)
        x = Activation('relu', name='block1_conv2_act')(x)

        residual = Conv2D(128, (1, 1), strides=(2, 2),
                          padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
        x = BatchNormalization(name='block2_sepconv1_bn')(x)
        x = Activation('relu', name='block2_sepconv2_act')(x)
        x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
        x = BatchNormalization(name='block2_sepconv2_bn')(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
        x = layers.add([x, residual])

        residual = Conv2D(256, (1, 1), strides=(2, 2),
                          padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = Activation('relu', name='block3_sepconv1_act')(x)
        x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
        x = BatchNormalization(name='block3_sepconv1_bn')(x)
        x = Activation('relu', name='block3_sepconv2_act')(x)
        x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
        x = BatchNormalization(name='block3_sepconv2_bn')(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(x)
        x = layers.add([x, residual])

        residual = Conv2D(728, (1, 1), strides=(2, 2),
                          padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = Activation('relu', name='block4_sepconv1_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
        x = BatchNormalization(name='block4_sepconv1_bn')(x)
        x = Activation('relu', name='block4_sepconv2_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
        x = BatchNormalization(name='block4_sepconv2_bn')(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(x)
        x = layers.add([x, residual])

        i = None
        for i in range(7):
            residual = x
            prefix = 'block' + str(i + 5)

            x = Activation('relu', name=prefix + '_sepconv1_act')(x)
            x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(x)
            x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
            x = Activation('relu', name=prefix + '_sepconv2_act')(x)
            x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(x)
            x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
            x = Activation('relu', name=prefix + '_sepconv3_act')(x)
            x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(x)
            x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)
            x = layers.add([x, residual])

        i += 1
        prefix = 'block' + str(i + 5)
        x = Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(x)
        x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(x)
        x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(x)
        x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)

        block = x
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(1, 1), stride=[1, 1, 1, 1],
                               padding='SAME', name='conv')
        stride = 32

        with tf.variable_scope('resnet-block-2') as scope:
            block = upsample2d(block, factor=8, has_bias=True, trainable=True, name='1')
            stride /= 8

        feature = block

    print ('rgb : scale=%f, stride=%d'%(1./stride, stride))
    return feature, stride

#------------------------------------------------------------------------------
def front_feature_net(input):
    stride=1.
    #with tf.variable_scope('top-preprocess') as scope:
    #    input = input
    with tf.variable_scope('feature_extraction'):
        if not cfg.USE_FRONT:
            tmp = tf.Variable(tf.random_normal(shape=[1]), name='tmp') # since we have set saver for front_view subnet, it must contain at least 1 variable
            return None, stride
        with tf.variable_scope('front-block-1') as scope:
            block = conv2d_bn_relu(input, num_kernels=32, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
            block = conv2d_bn_relu(block, num_kernels=32, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
            block = maxpool(block, kernel_size=(2,2), stride=[1,2,2,1], padding='SAME', name='4' )
            stride *=2

        with tf.variable_scope('front-block-2') as scope:
            block = conv2d_bn_relu(block, num_kernels=64, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
            block = conv2d_bn_relu(block, num_kernels=64, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
            block = maxpool(block, kernel_size=(2,2), stride=[1,2,2,1], padding='SAME', name='4' )
            stride *=2

        with tf.variable_scope('front-block-3') as scope:
            block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
            block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
            block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='3')
            block = maxpool(block, kernel_size=(2,2), stride=[1,2,2,1], padding='SAME', name='4' )
            stride *=2

        with tf.variable_scope('front-block-4') as scope:
            block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
            block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
            block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='3')

        with tf.variable_scope('front-block-5') as scope:
            block = upsample2d(block, factor=4, has_bias=True, trainable=True, name='1')
            stride /= 4

    feature = block

    print ('front: scale=%f, stride=%d'%(1./stride, stride))
    return feature, stride


def front_feature_net_r(input):
    stride=1.
    #with tf.variable_scope('top-preprocess') as scope:
    #    input = input
    batch_size, img_height, img_width, img_channel = input.get_shape().as_list()

    with tf.variable_scope('feature_extraction'):
        if not cfg.USE_FRONT:
            tmp = tf.Variable(tf.random_normal(shape=[1]), name='tmp') # since we have set saver for front_view subnet, it must contain at least 1 variable
            return None, stride
        with tf.variable_scope('front-feature-extract-resnet') as scope:
            print('build_resnet')
            block = ResnetBuilder.resnet_tiny(input)

            # resnet_input = resnet.get_layer('input_1').input
            # resnet_output = resnet.get_layer('add_7').output
            # resnet_f = Model(inputs=resnet_input, outputs=resnet_output)  # add_7
            # # print(resnet_f.summary())
            # block = resnet_f(input)
            block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(1, 1), stride=[1, 1, 1, 1], padding='SAME', name='2')
            stride = 8

    with tf.variable_scope('front-block-upsample') as scope:
        block = upsample2d(block, factor=4, has_bias=True, trainable=True, name='1')
        stride /= 4

    feature = block

    print ('front: scale=%f, stride=%d'%(1./stride, stride))
    return feature, stride

# def front_feature_net(input):
#     batch_size, img_height, img_width, img_channel = input.get_shape().as_list()

#     with tf.variable_scope('feature_extraction'):



#     # unimplemented
#     feature = None
#     return feature

# feature_list:
# ( [top_features,     top_rois,     6,6,1./stride],
#   [front_features,   front_rois,   0,0,1./stride],  #disable by 0,0
#   [rgb_features,     rgb_rois,     6,6,1./stride],)
#
def fusion_net(feature_list, num_class, out_shape=(8,3)):

    with tf.variable_scope('fuse-net') as scope:
        num = len(feature_list)
        feature_names = ['top', 'front', 'rgb'] if cfg.USE_FRONT else ['top', 'rgb'] 
        roi_features_list = []
        ctx_roi_features_list = []
        for n in range(num):
            feature = feature_list[n][0]
            roi = feature_list[n][1]
            pool_height = feature_list[n][2]
            pool_width = feature_list[n][3]
            pool_scale = feature_list[n][4]
            if (pool_height == 0 or pool_width == 0): continue

            with tf.variable_scope(feature_names[n] + '-roi-pooling'):
                roi_features, roi_idxs = tf_roipooling(feature, roi, pool_height, pool_width,
                                                       pool_scale, name='%s-roi_pooling' % feature_names[n])
            with tf.variable_scope(feature_names[n]+ '-feature-conv'):

                with tf.variable_scope('block1') as scope:
                    block = conv2d_bn_relu(roi_features, num_kernels=128, kernel_size=(3, 3),
                                           stride=[1, 1, 1, 1], padding='SAME',name=feature_names[n]+'_conv_1')
                    residual=block

                    block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3, 3), stride=[1, 1, 1, 1],
                                           padding='SAME',name=feature_names[n]+'_conv_2')+residual
 
                    block = avgpool(block, kernel_size=(2, 2), stride=[1, 2, 2, 1],
                                    padding='SAME', name=feature_names[n]+'_max_pool')
                with tf.variable_scope('block2') as scope:

                    block = conv2d_bn_relu(block, num_kernels=256, kernel_size=(3, 3), stride=[1, 1, 1, 1], padding='SAME',
                                           name=feature_names[n]+'_conv_1')
                    residual = block
                    block = conv2d_bn_relu(block, num_kernels=256, kernel_size=(3, 3), stride=[1, 1, 1, 1], padding='SAME',
                                           name=feature_names[n]+'_conv_2')+residual

                    block = avgpool(block, kernel_size=(2, 2), stride=[1, 2, 2, 1],
                                    padding='SAME', name=feature_names[n]+'_max_pool')
                with tf.variable_scope('block3') as scope:

                    block = conv2d_bn_relu(block, num_kernels=512, kernel_size=(3, 3), stride=[1, 1, 1, 1], padding='SAME',
                                           name=feature_names[n]+'_conv_1')
                    residual = block
                    block = conv2d_bn_relu(block, num_kernels=512, kernel_size=(3, 3), stride=[1, 1, 1, 1], padding='SAME',
                                           name=feature_names[n]+'_conv_2')+residual

                    block = avgpool(block, kernel_size=(2, 2), stride=[1, 2, 2, 1],
                                    padding='SAME', name=feature_names[n]+'_max_pool')

                roi_features = flatten(block) # now roi_feture shape is (N, X)  N is the amount of rois
                #tf.summary.histogram(feature_names[n], roi_features)
                roi_features_list.append(roi_features)

            if cfg.USE_SIAMESE_FUSION:  # seperately introduce context info by enlarge the roi
                def enlarge_roi(roi, ratio):
                    #x1, y1, x2, y2 = roi[1:5]
                    roi_T = tf.transpose(roi)
                    x1 = tf.gather(roi_T, 1)
                    y1 = tf.gather(roi_T, 2)
                    x2 = tf.gather(roi_T, 3)
                    y2 = tf.gather(roi_T, 4)
                    center_x, center_y = (x1+x2)//2, (y1+y2)//2
                    width, height = x2 - x1, y2 - y1
                    new_width, new_height = tf.to_float(width)*ratio, tf.to_float(height)*ratio
                    return tf.stack([
                        tf.zeros_like(roi[:, 0]),
                        center_x - new_width/2.,
                        center_y - new_height/2.,
                        center_x + new_width/2.,
                        center_y + new_height/2.
                    ], axis=1)

                # ctx_pool_height, ctx_pool_width = cfg.ROI_ENLARGE_RATIO*(pool_height, pool_width)  ?? don't know If I shoud enlarge it.
                ctx_pool_height, ctx_pool_width = pool_height, pool_width
                ctx_roi = enlarge_roi(roi, cfg.ROI_ENLARGE_RATIO)
                with tf.variable_scope(feature_names[n] + '-roi-pooling-ctx'):
                    ctx_roi_features, ctx_roi_idxs = tf_roipooling(feature, ctx_roi, ctx_pool_height, ctx_pool_width,
                                                       pool_scale, name='%s-roi_pooling-ctx' % feature_names[n])
                with tf.variable_scope(feature_names[n]+ '-feature-conv-ctx'):

                    with tf.variable_scope('block1-ctx') as scope:
                        block = conv2d_bn_relu(ctx_roi_features, num_kernels=128, kernel_size=(3, 3),
                                               stride=[1, 1, 1, 1], padding='SAME',name=feature_names[n]+'_conv_1-ctx')
                        residual=block

                        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3, 3), stride=[1, 1, 1, 1],
                                               padding='SAME',name=feature_names[n]+'_conv_2-ctx')+residual
     
                        block = avgpool(block, kernel_size=(2, 2), stride=[1, 2, 2, 1],
                                        padding='SAME', name=feature_names[n]+'_max_pool-ctx')
                    with tf.variable_scope('block2-ctx') as scope:

                        block = conv2d_bn_relu(block, num_kernels=256, kernel_size=(3, 3), stride=[1, 1, 1, 1], padding='SAME',
                                               name=feature_names[n]+'_conv_1-ctx')
                        residual = block
                        block = conv2d_bn_relu(block, num_kernels=256, kernel_size=(3, 3), stride=[1, 1, 1, 1], padding='SAME',
                                               name=feature_names[n]+'_conv_2-ctx')+residual

                        block = avgpool(block, kernel_size=(2, 2), stride=[1, 2, 2, 1],
                                        padding='SAME', name=feature_names[n]+'_max_pool-ctx')
                    with tf.variable_scope('block3-ctx') as scope:

                        block = conv2d_bn_relu(block, num_kernels=512, kernel_size=(3, 3), stride=[1, 1, 1, 1], padding='SAME',
                                               name=feature_names[n]+'_conv_1-ctx')
                        residual = block
                        block = conv2d_bn_relu(block, num_kernels=512, kernel_size=(3, 3), stride=[1, 1, 1, 1], padding='SAME',
                                               name=feature_names[n]+'_conv_2-ctx')+residual

                        block = avgpool(block, kernel_size=(2, 2), stride=[1, 2, 2, 1],
                                        padding='SAME', name=feature_names[n]+'_max_pool-ctx')
                        ctx_roi_features = flatten(block)
                        #tf.summary.histogram(feature_names[n], ctx_roi_features)
                        ctx_roi_features_list.append(ctx_roi_features)
        
        if cfg.USE_SIAMESE_FUSION:
            # now the shape of (ctx_)roi_features_list is (3(2), N, X)
            roi_feature_list = concat([roi_features_list, ctx_roi_features_list], axis=2, name='concat_ctx_roi_feature')

        with tf.variable_scope('rois-without-rgb-feature-concat'):
            block_without_rgb = concat(roi_features_list[0:2] if cfg.USE_FRONT else roi_features_list[0], axis=1, name='concat_without_rgb')

        with tf.variable_scope('fusion-without-rgb-feature-fc'):
            block_without_rgb = linear_bn_relu(block_without_rgb, num_hiddens=512, name='1')
            block_without_rgb = linear_bn_relu(block_without_rgb, num_hiddens=512, name='2')
            if cfg.USE_SIAMESE_FUSION:
                block_without_rgb = linear_bn_relu(block_without_rgb, num_hiddens=512, name='3')  # add an extra dense layer for siamese feature mixed

        with tf.variable_scope('rois-all-feature-concat'):
            block = concat(roi_features_list, axis=1, name='concat') # since roi_feature_list is not Tensor, so the output shape is (N, 3(2)*X)

        with tf.variable_scope('fusion-feature-fc'):
            print('\nUse fusion-feature-2fc')
            block = linear_bn_relu(block, num_hiddens=512, name='1')
            block = linear_bn_relu(block, num_hiddens=512, name='2')
            if cfg.USE_SIAMESE_FUSION:
                block = linear_bn_relu(block, num_hiddens=512, name='3')

    return  block_without_rgb, block


def fuse_loss(scores, deltas, rcnn_labels, rcnn_targets):

    def modified_smooth_l1( deltas, targets, sigma=3.0):
        '''
            ResultLoss = outside_weights * SmoothL1(inside_weights * (box_pred - box_targets))
            SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                          |x| - 0.5 / sigma^2,    otherwise
        '''
        sigma2 = sigma * sigma
        diffs  =  tf.subtract(deltas, targets)
        smooth_l1_signs = tf.cast(tf.less(tf.abs(diffs), 1.0 / sigma2), tf.float32)

        smooth_l1_option1 = tf.multiply(diffs, diffs) * 0.5 * sigma2
        smooth_l1_option2 = tf.abs(diffs) - 0.5 / sigma2
        smooth_l1_add = tf.multiply(smooth_l1_option1, smooth_l1_signs) + tf.multiply(smooth_l1_option2, 1-smooth_l1_signs)
        smooth_l1 = smooth_l1_add
        
        return smooth_l1


    _, num_class = scores.get_shape().as_list()
    dim = np.prod(deltas.get_shape().as_list()[1:])//num_class

    with tf.variable_scope('get_scores'):
        rcnn_scores   = tf.reshape(scores,[-1, num_class], name='rcnn_scores')
        pos_inds = tf.where(tf.not_equal(rcnn_labels, 0), name='pos_inds')
        rcnn_cls_loss_pos = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=tf.gather(rcnn_scores, pos_inds), labels=tf.gather(rcnn_labels, pos_inds)))
        rcnn_cls_loss_all = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=rcnn_scores, labels=rcnn_labels))
        rcnn_cls_loss = tf.add(tf.multiply(rcnn_cls_loss_pos, 2.0-1.0), tf.multiply(rcnn_cls_loss_all, 1.0))

    with tf.variable_scope('get_detals'):
        num = tf.identity( tf.shape(deltas)[0], 'num')
        idx = tf.identity(tf.range(num)*num_class + rcnn_labels,name='idx')
        deltas1      = tf.reshape(deltas,[-1, dim],name='deltas1')
        rcnn_deltas_with_fp  = tf.gather(deltas1,  idx, name='rcnn_deltas_with_fp')  # remove ignore label
        rcnn_targets_with_fp =  tf.reshape(rcnn_targets,[-1, dim], name='rcnn_targets_with_fp')

        #remove false positive
        fp_idxs = tf.where(tf.not_equal(rcnn_labels, 0), name='fp_idxs')
        rcnn_deltas_no_fp  = tf.gather(rcnn_deltas_with_fp,  fp_idxs, name='rcnn_deltas_no_fp')
        rcnn_targets_no_fp =  tf.gather(rcnn_targets_with_fp,  fp_idxs, name='rcnn_targets_no_fp')

    with tf.variable_scope('modified_smooth_l1'):
        rcnn_smooth_l1 = modified_smooth_l1(rcnn_deltas_no_fp, rcnn_targets_no_fp, sigma=3.0)

    rcnn_reg_loss = tf.reduce_mean(tf.reduce_sum(rcnn_smooth_l1, axis=1))
    # remove nans(when there are no any bbox, the tf.reduce_mean will generate nan)
    rcnn_reg_loss = tf.where(tf.is_nan(rcnn_reg_loss), tf.zeros_like(rcnn_reg_loss), rcnn_reg_loss)
    rcnn_cls_loss = tf.where(tf.is_nan(rcnn_cls_loss), tf.zeros_like(rcnn_cls_loss), rcnn_cls_loss)

    return rcnn_cls_loss, rcnn_reg_loss

def rpn_loss(scores, deltas, inds, pos_inds, rpn_labels, rpn_targets):

    def modified_smooth_l1( box_preds, box_targets, sigma=2.0):
        '''
            ResultLoss = outside_weights * SmoothL1(inside_weights * (box_pred - box_targets))
            SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                          |x| - 0.5 / sigma^2,    otherwise
        '''
        sigma2 = sigma * sigma
        diffs  =  tf.subtract(box_preds, box_targets)
        smooth_l1_signs = tf.cast(tf.less(tf.abs(diffs), 1.0 / sigma2), tf.float32)

        smooth_l1_option1 = tf.multiply(diffs, diffs) * 0.5 * sigma2
        smooth_l1_option2 = tf.abs(diffs) - 0.  / sigma2
        smooth_l1_add = tf.multiply(smooth_l1_option1, smooth_l1_signs) + tf.multiply(smooth_l1_option2, 1-smooth_l1_signs)
        smooth_l1 = smooth_l1_add   #tf.multiply(box_weights, smooth_l1_add)  #

        return smooth_l1

    scores1      = tf.reshape(scores,[-1,2])
    rpn_scores   = tf.gather(scores1,inds)  # remove ignore label
    # for more sample in RPN training, we need to use loss weight to balance the pos/neg sample
    rpn_scores_pos = tf.gather(scores1, pos_inds)
    rpn_labels_pos = tf.ones_like(pos_inds)

    rpn_cls_loss_pos = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_scores_pos, labels=rpn_labels_pos))
    rpn_cls_loss_all = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_scores, labels=rpn_labels))
    rpn_cls_loss = tf.add(tf.multiply(rpn_cls_loss_pos, 2.0-1.0), tf.multiply(rpn_cls_loss_all, 1.0))

    deltas1       = tf.reshape(deltas,[-1,4])
    rpn_deltas    = tf.gather(deltas1, pos_inds)  # remove ignore label

    with tf.variable_scope('modified_smooth_l1'):
        rpn_smooth_l1 = modified_smooth_l1(rpn_deltas, rpn_targets, sigma=3.0)

    rpn_reg_loss = tf.reduce_mean(tf.reduce_sum(rpn_smooth_l1, axis=1))
    # remove nans
    rpn_reg_loss = tf.where(tf.is_nan(rpn_reg_loss), tf.zeros_like(rpn_reg_loss), rpn_reg_loss)
    rpn_cls_loss = tf.where(tf.is_nan(rpn_cls_loss), tf.zeros_like(rpn_cls_loss), rpn_cls_loss)

    return rpn_cls_loss, rpn_reg_loss

def remove_empty_anchor(top_view, top_anchors, top_inside_inds):
    # input:
    # top_view: (B, H, W, C)
    # top_anchors: (N, 4)
    # top_inside_inds: (n)
    def cond(no_empty_inds, index):
        nonlocal top_view, top_anchors
        return tf.less(index, cfg.ANCHOR_AMOUNT)

    def body(no_empty_inds, index):
        nonlocal top_view, top_anchors
        # doing integration
        x1 = top_anchors[index][0]
        y1 = top_anchors[index][1]
        x2 = top_anchors[index][2]
        y2 = top_anchors[index][3]
        res = tf.reduce_sum(top_view[:, x1:x2, y1:y2, :])
        no_empty_inds = tf.cond(
            tf.less_equal(res, 0.),
            lambda: no_empty_inds,
            lambda: tf.concat([no_empty_inds, index*tf.ones(1, dtype=tf.int32)], axis=0),
        )
        index = tf.add(index, 1)
        return no_empty_inds, index

    index = tf.constant(0)
    top_no_empty_inds = tf.zeros(1, dtype=tf.int32)
    top_no_empty_inds, _ = tf.while_loop(
        cond,
        body,
        [top_no_empty_inds, index],
        shape_invariants=[tf.TensorShape([None]), index.get_shape()],
        parallel_iterations=512,
        back_prop=False,
        name='remove_empty_anchor'
    )
    top_no_empty_inds = top_no_empty_inds[1:]  # remove the first redundant one
    return top_no_empty_inds


def load(top_shape, front_shape, rgb_shape, num_class, len_bases):

    out_shape = (8, 3)
    stride = 8

    # for mimic batch size
    top_cls_loss_sum = tf.placeholder(shape=[], dtype=tf.float32, name='top_cls_loss_sum')
    top_reg_loss_sum = tf.placeholder(shape=[], dtype=tf.float32, name='top_reg_loss_sum')
    fuse_cls_loss_sum = tf.placeholder(shape=[], dtype=tf.float32, name='fuse_cls_loss_sum')
    fuse_reg_loss_sum = tf.placeholder(shape=[], dtype=tf.float32, name='fuse_reg_loss_sum')

    top_anchors = tf.placeholder(shape=[None, 4], dtype=tf.int32, name='anchors')
    top_inside_inds = tf.placeholder(shape=[None], dtype=tf.int32, name='inside_inds')  # use this to trunc empty anchor

    top_view = tf.placeholder(shape=[None, *top_shape], dtype=tf.float32, name='top')
    front_view = tf.placeholder(shape=[None, *front_shape], dtype=tf.float32, name='front')
    rgb_images = tf.placeholder(shape=[None, *rgb_shape], dtype=tf.float32, name='rgb')
    top_rois = tf.placeholder(shape=[None, 5], dtype=tf.float32, name='top_rois')  # todo: change to int32???
    front_rois = tf.placeholder(shape=[None, 5], dtype=tf.float32, name='front_rois')
    rgb_rois = tf.placeholder(shape=[None, 5], dtype=tf.float32, name='rgb_rois')

    # naive implementation of using pointnet and 3dconv for bbox regress
    raw_lidar = tf.placeholder(shape=[None, cfg.POINT_AMOUNT_LIMIT, 4], dtype=tf.float32, name='raw_lidar')
    point_cloud_rois = tf.placeholder(shape=[None, None, cfg.POINT_AMOUNT_LIMIT, 4], dtype=tf.float32, name='point_cloud_rois')
    voxel_rois = tf.placeholder(shape=[None, None, cfg.VOXEL_ROI_L, cfg.VOXEL_ROI_W, cfg.VOXEL_ROI_H], dtype=tf.float32, name='voxel_rois')

    with tf.variable_scope('remove_empty_anchor'):
        top_no_empty_inds = remove_empty_anchor(top_view, top_anchors, top_inside_inds)

    with tf.variable_scope(top_view_rpn_name):
        # top feature
        if cfg.USE_RESNET_AS_TOP_BASENET==True:
            top_features, top_scores, top_probs, top_deltas, proposals, proposal_scores, top_feature_rpn_stride, top_feature_rcnn_stride = \
                top_feature_net_r(top_view, top_anchors, top_inside_inds, len_bases, cfg.RPN_NMS_THRESHOLD)
        else:
            top_features, top_scores, top_probs, top_deltas, proposals, proposal_scores, top_feature_rpn_stride, top_feature_rcnn_stride = \
                top_feature_net(top_view, top_anchors, top_inside_inds, len_bases, cfg.RPN_NMS_THRESHOLD)

        with tf.variable_scope('loss'):
            # RPN
            top_inds = tf.placeholder(shape=[None], dtype=tf.int32, name='top_ind')
            top_pos_inds = tf.placeholder(shape=[None], dtype=tf.int32, name='top_pos_ind')
            top_labels = tf.placeholder(shape=[None], dtype=tf.int32, name='top_label') # only contain the labels of selected sample after rpn_target
            top_targets = tf.placeholder(shape=[None, 4], dtype=tf.float32, name='top_target')
            top_cls_loss_cur, top_reg_loss_cur = rpn_loss(top_scores, top_deltas, top_inds, top_pos_inds,
                                                  top_labels, top_targets)
    # with tf.variable_scope('top_feature_only'):
    #     if cfg.USE_RESNET_AS_TOP_BASENET==True:
    #         top_features, top_feature_stride = top_feature_net_r_feature_only(top_view)
    #     else:
    #         top_features, top_feature_stride = top_feature_net_feature_only(top_view)


    with tf.variable_scope(imfeature_net_name) as scope:
        if cfg.RGB_BASENET =='resnet':
            rgb_features, rgb_stride= rgb_feature_net_r(rgb_images)
        elif cfg.RGB_BASENET =='xception':
            rgb_features, rgb_stride = rgb_feature_net_x(rgb_images)
        elif cfg.RGB_BASENET =='VGG':
            rgb_features, rgb_stride = rgb_feature_net(rgb_images)

    with tf.variable_scope(frontfeature_net_name) as scope:
        if cfg.USE_RESNET_AS_FRONT_BASENET:
            front_features, front_stride = front_feature_net_r(front_view)
        else:
            front_features, front_stride = front_feature_net(front_view)
    #debug roi pooling
    # with tf.variable_scope('after') as scope:
    #     roi_rgb, roi_idxs = tf_roipooling(rgb_images, rgb_rois, 100, 200, 1)
    #     tf.summary.image('roi_rgb',roi_rgb)

    with tf.variable_scope(fusion_net_name) as scope:
        if cfg.IMAGE_FUSION_DISABLE==True:
            fuse_output_without_rgb, fuse_output_with_rgb = fusion_net(
                    ([top_features, top_rois, cfg.ROI_POOLING_HEIGHT, cfg.ROI_POOLING_WIDTH, 1. / top_feature_rcnn_stride],
                     [front_features, front_rois, 0, 0, 1. / front_stride],  # disable by 0,0
                     [rgb_features, rgb_rois*0, cfg.ROI_POOLING_HEIGHT, cfg.ROI_POOLING_WIDTH, 1. / rgb_stride],),
                    num_class, out_shape)
            print('\n\n!!!! disable image fusion\n\n')

        elif 0:
            # for test
            fuse_output_without_rgb, fuse_output_with_rgb = fusion_net(
                    ([top_features, top_rois*0, cfg.ROI_POOLING_HEIGHT, cfg.ROI_POOLING_WIDTH, 1. / top_feature_rcnn_stride],
                     [front_features, front_rois, cfg.ROI_POOLING_HEIGHT, 0, 1. / front_stride],  # disable by 0,0
                     [rgb_features, rgb_rois, cfg.ROI_POOLING_HEIGHT, cfg.ROI_POOLING_WIDTH, 1. / rgb_stride],),
                    num_class, out_shape)
            print('\n\n!!!! disable top view fusion\n\n')

        else:
            if cfg.USE_FRONT:
                fuse_output_without_rgb, fuse_output_with_rgb = fusion_net(
                    ([top_features, top_rois, cfg.ROI_POOLING_HEIGHT, cfg.ROI_POOLING_WIDTH, 1. / top_feature_rcnn_stride],
                     [front_features, front_rois, cfg.ROI_POOLING_HEIGHT, cfg.ROI_POOLING_WIDTH, 1. / front_stride],  # disable by 0,0
                     [rgb_features, rgb_rois, cfg.ROI_POOLING_HEIGHT, cfg.ROI_POOLING_WIDTH, 1. / rgb_stride],),
                    num_class, out_shape)
            else:
                fuse_output_without_rgb, fuse_output_with_rgb = fusion_net(
                    ([top_features, top_rois, cfg.ROI_POOLING_HEIGHT, cfg.ROI_POOLING_WIDTH, 1. / top_feature_rcnn_stride],
                     [rgb_features, rgb_rois, cfg.ROI_POOLING_HEIGHT, cfg.ROI_POOLING_WIDTH, 1. / rgb_stride],),
                    num_class, out_shape) 


        # include background class
        if cfg.USE_HANDCRAFT_FUSION or cfg.USE_LEARNABLE_FUSION:
            with tf.variable_scope('predict-without-rgb'):
                dim = np.product([*out_shape])
                fuse_scores_without_rgb = linear(fuse_output_without_rgb, num_hiddens=num_class, name='score')  # input shape is [N, X](just see what fusion_net and flatten do)
                fuse_probs_without_rgb = tf.nn.softmax(fuse_scores_without_rgb, name='prob')  # fuse_prob is [N, 2]
                fuse_deltas_without_rgb = linear_bn_relu(fuse_output_without_rgb, num_hiddens=256, name='box_1')
                fuse_deltas_without_rgb = linear_bn_relu(fuse_output_without_rgb, num_hiddens=256, name='box_2')
                fuse_deltas_without_rgb = linear(fuse_output_without_rgb, num_hiddens=dim * num_class, name='box_3') # now is [N, dim*num_class]
                fuse_deltas_without_rgb = tf.reshape(fuse_deltas_without_rgb, (-1, num_class, *out_shape)) # now is [N, num_class, 8, 3]

        with tf.variable_scope('predict-with-rgb') as scope:
            dim = np.product([*out_shape])
            # here is the output of fusenet(final output), since here it used the 
            # num_class, so it can just classify the car and the background
            fuse_scores_with_rgb = linear(fuse_output_with_rgb, num_hiddens=num_class, name='score')
            fuse_probs_with_rgb = tf.nn.softmax(fuse_scores_with_rgb, name='prob')
            fuse_deltas_with_rgb = linear_bn_relu(fuse_output_with_rgb, num_hiddens=256, name='box_1')
            fuse_deltas_with_rgb = linear_bn_relu(fuse_output_with_rgb, num_hiddens=256, name='box_2')
            fuse_deltas_with_rgb = linear(fuse_output_with_rgb, num_hiddens=dim * num_class, name='box_3')
            fuse_deltas_with_rgb = tf.reshape(fuse_deltas_with_rgb, (-1, num_class, *out_shape))

        # output concat
        # 9->2 conditions:
        # H: cls>0.9 L: cls<0.3
        # No-RGB    RGB    REG    CLS
        #    H       x     max's  max's
        #    x       H     max's  max's
        #    -       -     mean   mean
        with tf.variable_scope('predict-fuse') as scope:
            dim = np.product([*out_shape])
            if cfg.USE_HANDCRAFT_FUSION:
                max_op_mask = tf.logical_or(
                    cfg.HIGH_SCORE_THRESHOLD < fuse_probs_with_rgb, # score is not allowded here
                    cfg.HIGH_SCORE_THRESHOLD < fuse_probs_without_rgb
                )
                max_mask = fuse_probs_with_rgb > fuse_probs_without_rgb

                fuse_probs = tf.map_fn(lambda x: tf.cond(
                    max_op_mask[x],
                    lambda: tf.cond(
                        max_mask[x],
                        lambda: fuse_probs_with_rgb[x],
                        lambda: fuse_probs_without_rgb[x]
                    ),
                    lambda: tf.reduce_mean([fuse_probs_with_rgb[x], fuse_probs_without_rgb[x]], axis=0),
                ), tf.range(fuse_scores_with_rgb.shape[0]), dtype=tf.float32)
                
                # fuse_score is not available here since we cannot cal mean for 2 scores without softmax
                # TODO: for convinience, just use fuse_score here
                fuse_scores = tf.map_fn(lambda x: tf.cond(
                    max_op_mask[x],
                    lambda: tf.cond(
                        max_mask[x],
                        lambda: fuse_scores_with_rgb[x],
                        lambda: fuse_scores_without_rgb[x]
                    ),
                    lambda: tf.reduce_mean([fuse_probs_with_rgb[x], fuse_probs_without_rgb[x]], axis=0)* \
                        tf.sqrt(
                            tf.multiply(
                                tf.reduce_sum(
                                    tf.div(fuse_scores_with_rgb[x], fuse_probs_with_rgb[x])
                                ),
                                tf.reduce_sum(
                                    tf.div(fuse_scores_without_rgb[x], fuse_probs_without_rgb[x])
                                )
                            )
                        )
                ), tf.range(fuse_scores_with_rgb.shape[0]), dtype=tf.float32)

                fuse_deltas = tf.map_fn(lambda x: tf.cond(  # TODO for correct axis
                    max_op_mask[x],
                    lambda: tf.cond(
                        max_mask[x],
                        lambda: fuse_deltas_with_rgb[x],
                        lambda: fuse_deltas_without_rgb[x]
                    ),
                    lambda: tf.reduce_mean([fuse_deltas_with_rgb[x], fuse_deltas_without_rgb[x]], axis=0),
                ), tf.range(fuse_scores_with_rgb.shape[0]), dtype=tf.float32)
                fuse_deltas = tf.reshape(fuse_deltas, (-1, num_class, *out_shape))
            elif cfg.USE_LEARNABLE_FUSION:
                fuse_scores = linear(concat([fuse_scores_with_rgb, fuse_scores_without_rgb], axis=1), num_hiddens=num_class, name='fuse_scores')
                fuse_probs = linear(concat([fuse_probs_with_rgb, fuse_probs_without_rgb], axis=1), num_hiddens=num_class, name='fuse_probs')
                fuse_deltas = linear_bn_relu(concat([
                    tf.reshape(fuse_deltas_with_rgb, (-1, dim * num_class)), 
                    tf.reshape(fuse_deltas_without_rgb, (-1, dim * num_class))
                ], axis=1), num_hiddens=dim*num_class, name='fuse_deltas')
                fuse_deltas = tf.reshape(fuse_deltas, (-1, num_class, *out_shape))
            else:
                fuse_scores = fuse_scores_without_rgb = fuse_scores_with_rgb
                fuse_probs = fuse_probs_without_rgb = fuse_probs_with_rgb
                fuse_deltas = fuse_deltas_without_rgb = fuse_deltas_with_rgb

        with tf.variable_scope('loss') as scope:
            fuse_labels = tf.placeholder(shape=[None], dtype=tf.int32, name='fuse_label')
            fuse_targets = tf.placeholder(shape=[None, *out_shape], dtype=tf.float32, name='fuse_target')
            # in this implementation, it does not do NMS at final step, so the only NMS is done before fusenet(using score from PRN)
            # no, the line above is not correct, just see function 'predict' in mv3d.py, it does nms at final step.
            with tf.variable_scope('loss-fuse'):
                fuse_cls_loss_cur, fuse_reg_loss_cur = fuse_loss(fuse_scores, fuse_deltas, fuse_labels, fuse_targets)

            if cfg.USE_HANDCRAFT_FUSION or cfg.USE_LEARNABLE_FUSION:
                with tf.variable_scope('loss-with-rgb'):
                    fuse_cls_loss_with_rgb, fuse_reg_loss_with_rgb = fuse_loss(fuse_scores_with_rgb, fuse_deltas_with_rgb, fuse_labels, fuse_targets)

                with tf.variable_scope('loss-without-rgb'):
                    fuse_cls_loss_without_rgb, fuse_reg_loss_without_rgb = fuse_loss(fuse_scores_without_rgb, fuse_deltas_without_rgb, fuse_labels, fuse_targets)
            else:
                fuse_cls_loss_with_rgb = fuse_cls_loss_without_rgb = fuse_cls_loss_cur 
                fuse_reg_loss_with_rgb = fuse_reg_loss_without_rgb = fuse_reg_loss_cur

        # These for generate loss for optimizer
        with tf.variable_scope('loss') as scope:
            # optimize is controlled in fit_iterations, so there is no need for returning 0 when not doing optimize
            top_cls_loss = tf.add(top_cls_loss_cur, top_cls_loss_sum)  # top_cls_loss_cur will hold the path for backwards
            top_reg_loss = tf.add(top_reg_loss_cur, top_reg_loss_sum)
            fuse_cls_loss = tf.add(fuse_cls_loss_cur, fuse_cls_loss_sum)
            fuse_reg_loss = tf.add(fuse_reg_loss_cur, fuse_reg_loss_sum)


    # with tf.variable_scope(conv3d_net_name) as scope:
    #     if cfg.USE_CONV3D:
    #         conv3d_output = conv3d_for_bbox_regress(

    #         )
    #     elif cfg.USE_POINTNET:
    #         conv3d_output = pointnet_for_bbox_regress(

    #         )

    #     with tf.variable_scope('predict') as scope:
            

    #     with tf.variable_scope('loss') as scope:



    return {
        # for mimic batch size
        # These for feed in cumulative loss
        'top_cls_loss_sum': top_cls_loss_sum,
        'top_reg_loss_sum': top_reg_loss_sum,
        'fuse_cls_loss_sum': fuse_cls_loss_sum,
        'fuse_reg_loss_sum': fuse_reg_loss_sum,
        # These for return loss for cumulation
        'top_cls_loss_cur': top_cls_loss_cur,
        'top_reg_loss_cur': top_reg_loss_cur,
        'fuse_cls_loss_cur': fuse_cls_loss_cur,
        'fuse_reg_loss_cur': fuse_reg_loss_cur,

        'top_anchors':top_anchors,
        'top_inside_inds':top_inside_inds,
        'top_no_empty_inds':top_no_empty_inds,
        'top_view':top_view,
        'front_view':front_view,
        'rgb_images':rgb_images,
        'top_rois':top_rois,
        'front_rois':front_rois,
        'rgb_rois': rgb_rois,

        # These are loss for optimizer
        'top_cls_loss': top_cls_loss,
        'top_reg_loss': top_reg_loss,
        'fuse_cls_loss': fuse_cls_loss,
        'fuse_reg_loss': fuse_reg_loss,
        'fuse_cls_loss_with_rgb' : fuse_cls_loss_with_rgb,
        'fuse_reg_loss_with_rgb' : fuse_reg_loss_with_rgb,
        'fuse_cls_loss_without_rgb' : fuse_cls_loss_without_rgb,
        'fuse_reg_loss_without_rgb' : fuse_reg_loss_without_rgb,

        'top_features': top_features,
        'top_scores': top_scores,
        'top_probs': top_probs,
        'top_deltas': top_deltas,
        'proposals': proposals,
        'proposal_scores': proposal_scores,

        'top_inds': top_inds,
        'top_pos_inds':top_pos_inds,

        'top_labels':top_labels,
        'top_targets' :top_targets,

        'fuse_labels':fuse_labels,
        'fuse_targets':fuse_targets,

        'fuse_probs':fuse_probs,
        'fuse_scores':fuse_scores,  # useless, unnormalized
        'fuse_deltas':fuse_deltas,

        'fuse_probs_with_rgb': fuse_probs_with_rgb,
        'fuse_scores_with_rgb': fuse_scores_with_rgb,
        'fuse_deltas_with_rgb': fuse_deltas_with_rgb,
        
        'fuse_probs_without_rgb': fuse_probs_without_rgb,
        'fuse_scores_without_rgb': fuse_scores_without_rgb,
        'fuse_deltas_without_rgb': fuse_deltas_without_rgb,

        'top_feature_rpn_stride':top_feature_rpn_stride

    }

def test_roi_pooling():
    import numpy as np 
    rgb_images = tf.placeholder(shape=[None, *(375, 1242, 3)], dtype=tf.float32, name='rgb_images')
    rgb_rois = tf.placeholder(shape=[None, 5], dtype=tf.float32, name='rgb_rois')
    res = tf_roipooling(rgb_images, rgb_rois, 100, 200, 1)
    sess = tf.Session()
    with sess.as_default():
        ret = sess.run(res, feed_dict={
            rgb_images:np.ones((1, 375, 1242, 3)),
            rgb_rois:np.ones((1, 5)),
        })
        print(ret)

def test_nms():
    import numpy as np

if __name__ == '__main__':
    test_roi_pooling()
    # import  numpy as np
    # x =tf.placeholder(tf.float32,(None),name='x')
    # y = tf.placeholder(tf.float32,(None),name='y')
    # idxs = tf.where(tf.not_equal(x,0))
    # # weights = tf.cast(tf.not_equal(x,0),tf.float32)
    # y_w = tf.gather(y,idxs)
    # sess = tf.Session()
    # with sess.as_default():
    #     ret= sess.run(y_w, feed_dict={
    #         x: np.array([1.0,1.0,0.,2.]),
    #         y: np.array([1., 2., 2., 3.]),
    #     })
    #     print(ret)
