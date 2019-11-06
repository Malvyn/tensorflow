import tensorflow as tf

with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1.], dtype=tf.int32)

with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v", [1])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(v))
    print(sess.run(v1))
