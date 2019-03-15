from __future__ import print_function
import tensorflow as tf
from BiRnnAtten import Birnn_Atten


# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

with tf.Graph().as_default():
    with tf.Session() as sess:
        net = Birnn_Atten()
        writer = tf.summary.FileWriter("logs/", sess.graph)  
        
        sess.run(tf.global_variables_initializer())
        
        for step in range(1, 1000+1):
            batch_x, batch_y = mnist.train.next_batch(128)
            # Reshape data to get 28 seq of 28 elements
            batch_x = batch_x.reshape((128, 28, 28))
            # Run optimization op (backprop)
            sess.run(net.train_op, feed_dict={net.images: batch_x, net.labels: batch_y})
            if step % 200 == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([net.loss, net.accuracy], feed_dict={net.images: batch_x,
                                                                     net.labels: batch_y})
                merged = sess.run(net.merged, feed_dict={net.images: batch_x,
                                                                     net.labels: batch_y})
                writer.add_summary(merged, step)
                
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))
        writer.close()
    
        print("Optimization Finished!")  
        # Calculate accuracy for 128 mnist test images
        test_len = 128
        test_data = mnist.test.images[:test_len].reshape((-1, 28, 28))
        test_label = mnist.test.labels[:test_len]
        print("Testing Accuracy:", \
              sess.run(net.accuracy, feed_dict={net.images: test_data, net.labels: test_label}))
