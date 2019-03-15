# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 09:59:52 2018

@author: bx_chen
"""
from __future__ import print_function
import config as cfg
import tensorflow as tf
import tensorflow.contrib.slim as slim

class CNNnet(object):
    def __init__(self):        
        
        self.raw_input_image = tf.placeholder(tf.float32, [None, 384, 384, 7], name='image')            
#       self.input_images = tf.reshape(self.raw_input_image, [-1, 32, 32, 3])
        self.raw_input_label = tf.placeholder(tf.float32, [None, 1], name='label')
#       self.input_labels = tf.cast(self.raw_input_label, tf.int32)
        self.dropout = cfg.KEEP_PROB

        with tf.variable_scope("cnn_net") as scope:
            self.train_digits = self.construct_net(is_trained=True)
            '''在cnn_net下共享变量'''
            scope.reuse_variables()# or 
            #tf.get_variable_scope().reuse_variables() 
            self.pred_digits = self.construct_net(is_trained=False)
            
        with tf.name_scope('loss'):
            self.loss = slim.losses.mean_squared_error(self.raw_input_label, self.train_digits)
            tf.summary.scalar('loss', self.loss)
            
        self.lr = cfg.LEARNING_RATE
        with tf.name_scope('train_op'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        
        with tf.name_scope('sserror'):
            self.sserror = tf.reduce_sum(tf.square(tf.subtract(self.raw_input_label, self.train_digits)), name='sserror')
            tf.summary.scalar('sserror', self.sserror)
            
        #R square
        with tf.name_scope('R_squared'):
            self.prediction = self.pred_digits
            self.total_error = tf.reduce_sum(tf.square(tf.subtract(self.raw_input_label, tf.reduce_mean(self.raw_input_label))))
            self.unexplained_error = tf.reduce_sum(tf.square(tf.subtract(self.raw_input_label, self.prediction)))
            self.R_squared = tf.subtract(1.0, tf.div(self.unexplained_error, self.total_error))
            tf.summary.scalar('R_squared', self.R_squared)
            
        self.merged = tf.summary.merge_all()
        
    def construct_net(self, is_trained=True):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu6,
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            ):
            
            net = slim.conv2d(self.raw_input_image, 16, [3,3], padding='VALID', scope='conv1')
            net = slim.conv2d(net, 32, [3,3], padding='VALID', scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')           
            net = slim.conv2d(net, 64, [3,3], padding='VALID', scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.batch_norm(net, is_training=is_trained, scope='bn1')
            
            net = slim.conv2d(net, 64, [3, 3], padding='VALID', scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.batch_norm(net, is_training=is_trained, scope='bn2')
            
            net = slim.conv2d(net, 128, [3, 3], padding='VALID', scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.batch_norm(net, is_training=is_trained, scope='bn3')   
            
            net = slim.conv2d(net, 128, [3, 3], padding='VALID', scope='conv6')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            net = slim.batch_norm(net, is_training=is_trained, scope='bn4')
            
            net = slim.conv2d(net, 256, [3, 3], padding='VALID', scope='conv7')
            net = slim.conv2d(net, 256, [3, 3], padding='VALID', scope='conv8')
            net = slim.max_pool2d(net, [2, 2],  scope='pool6')
            net = slim.batch_norm(net, is_training=is_trained, scope='bn5')
            
            net = slim.conv2d(net, 512, [3, 3], padding='SAME', scope='conv9')
            net = slim.conv2d(net, 512, [3, 3], padding='SAME', scope='conv10')
            net = slim.max_pool2d(net, [2, 2], scope='pool7')
            net = slim.batch_norm(net, is_training=is_trained, scope='bn6')
            
            net = slim.flatten(net, scope='flat1')
            
            net = slim.fully_connected(net, 4096, scope='fc1')
            net = slim.dropout(net, self.dropout, is_training=is_trained, scope='dropout1')
            net = slim.batch_norm(net, scope='bn7')
 
            net = slim.fully_connected(net, 2048, scope='fc2')
            net = slim.dropout(net, self.dropout, is_training=is_trained, scope='dropout2')
            net = slim.batch_norm(net, is_training=is_trained, scope='bn8')
  
            net = slim.fully_connected(net, 1024, scope='fc3')
            net = slim.dropout(net, self.dropout, is_training=is_trained, scope='dropout3')
            net = slim.batch_norm(net, is_training=is_trained, scope='bn9')
  
            net = slim.fully_connected(net, 256, scope='fc4')
            net = slim.dropout(net, self.dropout, is_training=is_trained, scope='dropout4')
            net = slim.batch_norm(net, is_training=is_trained, scope='bn10')
  
            net = slim.fully_connected(net, 1, activation_fn=None, scope='fc5')
        return net