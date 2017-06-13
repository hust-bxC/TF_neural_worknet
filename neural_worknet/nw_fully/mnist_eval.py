# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:20:25 2017

@author: rogue
"""

import time 
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#加载mnist_inference和mnist_train中定义的常量和模型
import mnist_inference 
import mnist_train

#每10秒加载一次最新的模型，并在测试数据上测试最新模型的正确率
eval_interval_secs = 10

def evaluate(mnist):
	with tf.Graph().as_default() as g:
		#定义输入输出格式
		x = tf.placeholder(tf.float32,[None, mnist_inference.input_node], name='x-input')
		y_ = tf.placeholder(tf.float32,[None, mnist_inference.output_node], name='x-input')
		validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
		
		#直接利用封装好的函数来计算前向传播的结果
		y = mnist_inference.inference(x, None)
		
		#使用前向传播的结果计算准确率
		#判断两个张量的索引是否相等，即统计正确率
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

		#首先将布尔型数值转换成实数型，然后计算均值，即时model在这组数据上的正确率	
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		
		#通过变量名重命名的方式来加载模型
		variable_averages = tf.train.ExponentialMovingAverage(mnist_train.moving_average_decay)
		variable_to_restore = variable_averages.variables_to_restore()
		saver = tf.train.Saver(variable_to_restore)
		
		#每隔eval_interval_secs秒调用一次计算正确率的过程检测训练过程中正确率的变化
		while True:
			with tf.Session() as sess:
				#tf.train.get_checkpoint_state函数会通过checkpoint文件自动找到目录中最新模型的文件名
				ckpt = tf.train.get_checkpoint_state(mnist_train.model_save_path)
				if ckpt and ckpt.model_checkpoint_path:
					#加载模型
					saver.restore(sess, ckpt.model_checkpoint_path)
					global_step = ckpt.ckpt.model_checkpoint_path.split('/')[-1].spilt('-')[-1]
					accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
					print('After %d training steps, validate accuracy = %g' % (global_step, accuracy_score))
				else:
					print('No checkpoint file found')
					return
					time.sleep(eval_interval_secs)
					
#主程序入口	
def main(argv=None):
	mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
	evaluate(mnist)

#tf.app.run会调用main函数	
if __name__ == '__main__':
	tf.app.run()
		

		