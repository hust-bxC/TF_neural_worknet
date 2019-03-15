#-*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import rnn
import tensorflow.contrib.slim as slim
import config as cfg


class Birnn_Atten(object):
    """
    BiRNN with Attention mechanism 
    """
    def __init__(self):
        self.images = tf.placeholder(tf.float32, [None, 28, 28], name='image')            
        self.labels = tf.placeholder(tf.float32, [None, 10], name='label')
        self.keep_prob = cfg.KEEP_PROB       
        self.dropout = cfg.DROPOUT
        self.hidden_num = cfg.HIDDEN_NUM
        self.layers_num = cfg.LAYERS_NUM
        self.attention = cfg.ATTENTIONS
        
        
        with tf.variable_scope("rnn_net") as scope:
            self.train_digits = self.construct_net(is_trained=True)
            '''在rnn_net下共享变量'''
            scope.reuse_variables()# or 
            #tf.get_variable_scope().reuse_variables() 
            self.pred_digits = self.construct_net(is_trained=False)
        
        with tf.name_scope('loss'):
#        self.loss = slim.losses.mean_squared_error(self.raw_input_label, self.train_digits)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.train_digits, labels=self.labels))
            tf.summary.scalar('loss', self.loss)
        
        self.lr = cfg.LEARNING_RATE
        with tf.name_scope('train_op'):
            self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)

        
        with tf.name_scope('acc'):
            self.prediction = tf.nn.softmax(self.train_digits)
            self.correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)
        
        self.merged = tf.summary.merge_all()
        
    def construct_net(self, is_trained=True):        

        # Define Basic RNN Cell
        def basic_rnn_cell():
#            return tf.contrib.rnn.GRUCell(self.hidden_num)
            return tf.contrib.rnn.LSTMCell(self.hidden_num)

        # Define Forward RNN Cell
        with tf.name_scope('fw_rnn'):
            fw_cell = tf.contrib.rnn.MultiRNNCell([basic_rnn_cell() for _ in range(self.layers_num)])
            fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=self.keep_prob)

        # Define Backward RNN Cell
        with tf.name_scope('bw_rnn'):
            bw_cell = tf.contrib.rnn.MultiRNNCell([basic_rnn_cell() for _ in range(self.layers_num)])
            bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=self.keep_prob)

        with tf.name_scope('bi_rnn'):
            '''outputs: A tuple (output_fw, output_bw)'''
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs=self.images, dtype=tf.float32)
            
            # Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_input)
#            x = tf.unstack(self.images, 28, 1)      
#            try:
#                outputs, _, _ = rnn.static_bidirectional_rnn(fw_cell, bw_cell, x, dtype=tf.float32)
#            except Exception: # Old TensorFlow version only returns outputs not states
#                outputs = rnn.static_bidirectional_rnn(fw_cell, bw_cell, x, dtype=tf.float32)
       
        if isinstance(outputs, tuple):
            outputs = tf.concat(outputs, 2)
                
        with tf.name_scope('attention'):
            input_shape = outputs.shape # (batch_size, sequence_length, fw_hidden_size+bw_hidden_size)
            sequence_size = input_shape[1].value  # sequence_length
            hidden_size = input_shape[2].value  # fw_hidden_size+bw_hidden_size
            
            attention_size = self.attention
            attention_w = tf.Variable(tf.truncated_normal([hidden_size, attention_size], stddev=0.1), name='attention_w')
            attention_b = tf.Variable(tf.constant(0.1, shape=[attention_size]), name='attention_b')
            attention_u = tf.Variable(tf.truncated_normal([attention_size], stddev=0.1), name='attention_u')
            z_list = []
            for t in range(sequence_size):
                u_t = tf.tanh(tf.matmul(outputs[:, t, :], attention_w) + tf.reshape(attention_b, [1, -1]))
                z_t = tf.matmul(u_t, tf.reshape(attention_u, [-1, 1]))
                z_list.append(z_t)
            # Transform to batch_size * sequence_size
            attention_z = tf.concat(z_list, axis=1)
            self.alpha = tf.nn.softmax(attention_z)
            # Transform to batch_size * sequence_size * 1 , same rank as rnn_output
            attention_output = tf.reduce_sum(outputs * tf.reshape(self.alpha, [-1, sequence_size, 1]), 1)
            
            final_output = tf.nn.dropout(attention_output, self.keep_prob) #(?, 256)

        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.relu6,
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005)):               
            net = slim.flatten(final_output, scope='flat1')
            
            net = slim.fully_connected(net, 4096, scope='fc1')
            net = slim.dropout(net, self.dropout, is_training=is_trained, scope='dropout1')
            net = slim.batch_norm(net, scope='bn1')   
            
            net = slim.fully_connected(net, 2048, scope='fc2')
            net = slim.dropout(net, self.dropout, is_training=is_trained, scope='dropout2')
            net = slim.batch_norm(net, scope='bn2')   
            
            net = slim.fully_connected(net, 1024, scope='fc3')
            net = slim.dropout(net, self.dropout, is_training=is_trained, scope='dropout3')
            net = slim.batch_norm(net, scope='bn3')  
            
            net = slim.fully_connected(net, 256, scope='fc4')
            net = slim.dropout(net, self.dropout, is_training=is_trained, scope='dropout4')
            net = slim.batch_norm(net, scope='bn4') 
            
            net = slim.fully_connected(net, 10, activation_fn=None, scope='fc5')
                
        return net
    
            