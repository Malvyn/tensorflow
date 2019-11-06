import tensorflow as tf

w1 = tf.Variable(1.0, name='w1')
w2 = tf.placeholder(tf.float32, [1])
w3 = tf.add(w1, w2)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(w3, feed_dict={w2: [2.0]}))


