# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 09:28:31 2017

@author: rogue
"""

import tensorflow as tf

#定义神经网络相关参数
input_node = 784 #输入层节点，等于图片的像素28*28
output_node = 10 #输出层节点，0-9的数字
layer1_node = 500 #隐藏层节点

#通过tf.get_variable函数来获取变量，在训练时创建变量，在测试时会通过保存的为模型加载这些变量的取值
def get_weight_variable(shape, regularizer):
	weights = tf.get_variable("weioghts", shape, initializer = tf.truncated_normal_initializer(stddev=0.1))
	
	#给出正则化生成函数，将当前变量的正则化损失加入名字losses的集合
	if regularizer != None:
		tf.add_to_collection('losses', regularizer(weights))
	return weights
	
def inference(input_tensor, regularizer):
	
	#声明第一层神经网络变量和前向传播过程
	with tf.variable_scope('layer1'):
		
		weights = get_weight_variable([input_node, layer1_node], regularizer)
		biases = tf.get_variable("biases",[layer1_node],initializer = tf.constant_initializer(0.0))
		layer1 = tf.nn.relu(tf.matmul(input_tensor,weights) + biases)
		
	#声明第二层神经网络变量和前向传播过程
	with tf.variable_scope('layer2'):
		
		weights = get_weight_variable([layer1_node, output_node], regularizer)
		biases = tf.get_variable("biases",[output_node],initializer = tf.constant_initializer(0.0))
		layer2 = tf.matmul(layer1, weights) + biases
	
	return layer2
		 