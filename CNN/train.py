# -*- coding: utf-8 -*-

from cnnnet import CNNnet
import config as cfg
import dataprocessing as dp
import tensorflow as tf
import os


def main():
    with tf.Graph().as_default():
        with tf.Session() as sess:
            batch_size = cfg.BATCH_SIZE
            epochs = cfg.EPOCHS
            parameter_path = cfg.PARAMETER_FILE
        
            net = CNNnet()
            writer = tf.summary.FileWriter("logs/", sess.graph)
            saver = tf.train.Saver()
            if os.path.exists(parameter_path):
                saver.restore(parameter_path)
            else:
                sess.run(tf.global_variables_initializer())
            number = 0
            # Loop over all batches
            for epoch in range(epochs):
                for i in [1, 4, 6]:
                    features = dp.get_figures('bigdata', i)
                    labels = dp.get_labels('data', i)
                    for batch_features, batch_labels in dp.get_batches_from(features, labels, batch_size):
                        sess.run(net.train_op, feed_dict={net.raw_input_image: batch_features,
                                                              net.raw_input_label: batch_labels})

                        loss, merged = sess.run([net.loss, net.merged], feed_dict={net.raw_input_image: batch_features,
                                                                 net.raw_input_label: batch_labels})

                        sserror, R_squared = sess.run([net.sserror, net.R_squared], feed_dict={net.raw_input_image: batch_features,
                                                                 net.raw_input_label: batch_labels})


                        writer.add_summary(merged, number)
                        number = number + batch_size
                        print('Epoch {:>2}, Loss: {:>10.4f}, sserror: {:>10.4f}, R_squared: {:>10.4f}'
                              .format(epoch, loss, sserror, R_squared))
            writer.close()
            saver.save(sess, parameter_path)
        
if __name__ == '__main__':
    main()