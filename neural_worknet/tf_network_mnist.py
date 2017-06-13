# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 15:15:03 2017

@author: rogue
"""


from tensorflow.examples.tutorials.mnist import input_data


import tensorflow as tf

#手写体的相关常数
input_node = 784 #输入层节点，等于图片的像素28*28
output_node = 10 #输出层节点，0-9的数字

#配置神经网络参数
layer1_node = 500 #隐藏层节点
batch_size = 100  #批量随机梯度下降算法中的批量里训练数据个数

learning_rate_base = 0.8  #基础学习率
learning_rate_decay = 0.99 #学习率的衰减率
regularization_rate = 0.0001 #描述model复杂度的正则化项在损失函数的系数
training_steps = 30000 #训练次数
moving_average_decay = 0.99 #滑动平均衰减率

#定义一个函数，给定所有的输入和参数，计算前向传播结果
#使用Relu函数实现全连接，通过其实现去线性化
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
	
	if avg_class ==None:  #没有提供提供滑动平均类，直接使用当前的参数
	
		layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1) #计算隐藏层前向传播结果
		'''
		在后面的处理中计算损失函数时会一并计算softmax函数，因而这里可以不加入激活函数
		预测时使用的是不同类别对应节点输出值的相对大小，有没用softmax计算对后面的分类结果没有影响
		因而可以不加入softmax层。
		'''
		return tf.matmul(layer1, weights2) + biases2

	else:
		#同上，不过之前先用函数计算出变量的滑动平均模型
		layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
		return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)

#训练模型			
def train(mnist):
	x = tf.placeholder(tf.float32, [None ,input_node], name='x-input')
	y_ = tf.placeholder(tf.float32, [None ,input_node], name='y-input')
	
	#生成隐藏层的参数
	weights1 = tf.Variable(tf.truncated_normal([input_node, layer1_node], stddev=0.1))
	biases1 = tf.Variable(tf.constant(0.1, shape=[layer1_node]))
	
	#生成输出层的参数
	weights2 = tf.Variable(tf.truncated_normal([layer1_node, output_node], stddev=0.1))
	biases2 = tf.Variable(tf.constant(0.1, shape=[output_node]))
	
     #调用函数inference，求出下一层的变量关系
	y = inference(x, None, weights1, biases1, weights2, biases2)
	
	#定义存储训练轮数的变量，一般是不可训练的参数
	global_step = tf.Variable(0, trainable=False)
	
	#给定滑动平均衰减率和训练轮数的变量，初始化平均滑动类。（给定训练轮数的变量可以加快训练早期变量的更新速度）
	variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
	
	#在给定的所有神经网络的参数变量使用平均滑动类，tf.trainable_variables返回的就是图上集合元素，该元素就是所有没有指定trainable=Flase的参数
	variables_averages_op = variable_averages.apply(tf.trainable_variables())
	
	#调用函数inference，计算使用了平均滑动类之后的前向传播结果
	average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)
	
	#计算交叉熵作为预测值和真实值差距的损失函数，其中第一个参数是神经网络不包括softmax层的前向传播结果，第二个是真实值
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=tf.argmax(y_, 1))
	
	#计算当前batch中所有样例的交叉熵平均值
	cross_entropy_mean = tf.reduce_mean(cross_entropy)
	
	#计算L2正则化损失函数
	regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
	
	#计算模型的正则化损失，仅计算权重的正则化损失
	regularization = regularizer(weights1) + regularizer(weights2)
	
	#总损失等于交叉熵损失和正则化损失函数
	loss = cross_entropy_mean + regularization
	
	#设置指数衰减的学习率
	learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, mnist.train.num_examples/batch_size, learning_rate_decay)

	#用小批量梯度随机下降算法优化损失函数	
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)

	#更新反向传播后的参数，更新每一个参数的滑动平均值	
	with tf.control_dependencies([train_step, variables_average_op]):
		train_op = tf.no_op(name = 'train')
	
	#判断两个张量的索引是否相等，即统计正确率
	correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))

	#首先将布尔型数值转换成实数型，然后计算均值，即时model在这组数据上的正确率	
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	#建立会话进行训练
	with tf.Session() as sess:
		tf.initialize_all_variables().run()
	
		#准备验证条件
		validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
	
		#测试数据
		test_feed = {x: mnist.test.images, y_: mnist.test.labels}

		#迭代训练神经网络
		for i in range(training_steps):
		
			#1000轮输出验证结果
			if i % 1000 == 0:
			
				validate_acc = sess.run(accuracy, feed_dict=validate_feed)
			
				print('After %d training steps, validation accuracy on average model is %g' % (i, validate_acc))
		
				#产生这一轮使用的bacth的训练数据，运行训练结果
			xs, ys = mnist.train.next_batch(batch_size)
			sess.run(train_op, feed_dict = {x: xs, y_: ys})
	
			#在测试数据上检测model的最终正确率
			test_acc = sess.run(accuracy, feed_dict = test_feed)

			print('After %d training steps, test accuracy using averagsmodel is %g' % (traing_steps, test.acc))

#主程序入口	
def main(argv=None):
	mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
	train(mnist)

#tf.app.run会调用main函数	
if __name__ == '__main__':
	tf.app.run()
		
	
	
	
	
	
	
	
	
	
	
	
	
	
	