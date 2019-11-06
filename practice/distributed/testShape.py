import  tensorflow as tf

a = tf.constant([2, 1, 4, 8], shape=[2, 6])

with tf.Session() as sess:
    print(sess.run(a))