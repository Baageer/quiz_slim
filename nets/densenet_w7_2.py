from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

def densenet_arg_scope(weight_decay=0.0005):
  """Defines the VGG arg scope.

  Args:
    weight_decay: The l2 regularization coefficient.

  Returns:
    An arg_scope.
  """
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
      return arg_sc

def dense_block(inputs, block_size, k=12, scope=None, reuse=None):
    """Builds dense_block for dense net. """
    with slim.arg_scope([slim.conv2d],
                        stride=1, padding='SAME',
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm):
        with tf.variable_scope(scope, 'dense_block', [inputs], reuse=reuse):
            for idx in range(block_size):
                last_inputs = inputs
                scope_name = scope + chr(ord('a') + idx)
                inputs = slim.conv2d(inputs, k, [1, 1], scope=scope_name)
                scope_name = scope + chr(ord('A') + idx)
                inputs = slim.conv2d(inputs, k, [3, 3], scope=scope_name)
                inputs = tf.concat([last_inputs, inputs], axis=3)

    return inputs

def transition_layer(inputs, num_ker, scope=None, reuse=None):
    with slim.arg_scope([slim.conv2d, slim.avg_pool2d],
                        stride=2, padding='SAME'):
        with tf.variable_scope(scope, 'transition_layer', [inputs], reuse=reuse):
            scope_name = scope +'conv2d_1x1'
            conv2d_1x1 = slim.conv2d(inputs, num_ker, [1,1], scope=scope_name)
            scope_name = scope +'avg_pool2d_2x2'
            avg_pool2d_2x2 = slim.avg_pool2d(conv2d_1x1, [2,2], scope=scope_name)

    return avg_pool2d_2x2

def densenet_w7_2(inputs, num_classes=1000, is_training=True,
                  k=12,
                  reuse=None,
                  scope='DenseNet'):
    """Create the densenet model"""
    with tf.variable_scope(scope, 'DenseNet', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'

        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=2,padding='SAME',
                            outputs_collections=end_points_collection):
            inputs = slim.conv2d(inputs, 12, [7,7], scope='Conv2d_7x7')
            inputs = slim.max_pool2d(inputs, [3,3], scope='MaxPool_3x3')

            dense_block_1 = dense_block(inputs, 6, k, 'dense_block_1')
            transition_layer_1 = transition_layer(dense_block_1, 12, 'transition_layer_1')

            dense_block_2 = dense_block(transition_layer_1, 12, k, 'dense_block_2')
            transition_layer_2 = transition_layer(dense_block_2, 12, 'transition_layer_2')

            dense_block_3 = dense_block(transition_layer_2, 24, k, 'dense_block_3')
            transition_layer_3 = transition_layer(dense_block_3, 12, 'transition_layer3')

            dense_block_4 = dense_block(transition_layer_3, 16, k, 'dense_block_4')
            
            avg_pool2d_7x7 = slim.avg_pool2d(dense_block_4, [7,7], scope='avg_pool2d_7x7')

            end_points = slim.utils.convert_collection_to_dict(end_points_collection)

            end_points['global_pool'] = avg_pool2d_7x7
            
            flatten = slim.flatten(avg_pool2d_7x7, scope='PreLogitsFlatten')
            end_points['PreLogitsFlatten'] = flatten

            logits = slim.fully_connected(flatten, num_classes, activation_fn=None,
                                          scope='Logits')
            end_points['Logits'] = logits
            end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')

        return logits, end_points
densenet_w7_2.default_image_size = 32