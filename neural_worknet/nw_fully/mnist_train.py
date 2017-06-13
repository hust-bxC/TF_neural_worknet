# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 10:13:30 2017

@author: rogue
"""

import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#加载mnist_inference.py中定义的常量和浅香传播函数
import mnist_inference

#配置神经网络参数
batch_size = 100  #批量随机梯度下降算法中的批量里训练数据个数
learning_rate_base = 0.8  #基础学习率
learning_rate_decay = 0.99 #学习率的衰减率
regularization_rate = 0.0001 #描述model复杂度的正则化项在损失函数的系数
training_steps = 30000 #训练次数
moving_average_decay = 0.99 #滑动平均衰减率
#模型保存的路径和名称
model_save_path = 'F:\python\tensorflow_training\mnist_fully'
model_name = 'model.ckpt'

def train(mnist):
	#d定义输入输出placeholder
	x = tf.placeholder(tf.float32, [None, mnist_inference.input_node], name='x-input')
	y_ = tf.placeholder(tf.float32, [None, mnist_inference.output_node], name='y-input')
	regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
	
	#直接使用mnist_inference中定义的前向传播过程
	y = mnist_inference.inference(x, regularizer)
	
	#定义存储训练轮数的变量，一般是不可训练的参数
	global_step = tf.Variable(0, trainable=False)
	
	#给定滑动平均衰减率和训练轮数的变量，初始化平均滑动类。（给定训练轮数的变量可以加快训练早期变量的更新速度）
	variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
	
	#在给定的所有神经网络的参数变量使用平均滑动类，tf.trainable_variables返回的就是图上集合元素，该元素就是所有没有指定trainable=Flase的参数
	variables_averages_op = variable_averages.apply(tf.trainable_variables())
	
	#计算交叉熵作为预测值和真实值差距的损失函数，其中第一个参数是神经网络不包括softmax层的前向传播结果，第二个是真实值
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=tf.argmax(y_, 1))
	
	#计算当前batch中所有样例的交叉熵平均值
	cross_entropy_mean = tf.reduce_mean(cross_entropy)
	
	#总损失等于交叉熵损失和正则化损失函数
	loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
	
	#设置指数衰减的学习率
	learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, mnist.train.num_examples/batch_size, learning_rate_decay)
	
	#用小批量梯度随机下降算法优化损失函数	
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)
	
	#更新反向传播后的参数，更新每一个参数的滑动平均值	
	with tf.control_dependencies([train_step, variables_average_op]):
		train_op = tf.no_op(name = 'train')
	
	#初始化Tensorflow持久化类
	saver = tf.train.Saver()
	with tf.Session() as sess:
		tf.initialize_all_variables().run()
		
		for i in range(training_steps):
		
			xs, ys = mnist.train.next_batch(batch_size)
			_, loss_value, step = sess.run([train_op, loss, global_step],feed_dict = {x: xs, y_: ys})
			
			#1000轮输出验证结果
			if i % 1000 == 0:
				#训练当前的训练情况，这里输出当前的训练batch上的损失函数大小
			
				print('After %d training steps, loss on training batch is %g' % (step, loss_value))
				
				#保存当前模型，这里有global_step参数，保存名称‘model.ckpy-100’表示村联1000轮之后的model
				saver.save(sess, os.path.join(model_save_path, model_name), global_step=global_step)

#主程序入口	
def main(argv=None):
	mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
	train(mnist)

#tf.app.run会调用main函数	
if __name__ == '__main__':
	tf.app.run()
		
				
	
	
	
	
	
	
	
	
	
	
	
	