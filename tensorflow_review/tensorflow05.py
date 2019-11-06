import tensorflow as tf

g = tf.Graph()
with g.as_default():
    a1 = tf.random_normal(shape=[], dtype=tf.float32)
    b1 = tf.random_normal(shape=[], dtype=tf.float32)
    y = a1 * b1
    config = tf.ConfigProto(log_device_placement = True, allow_soft_placement = True)
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.allow_growth = True
    # sess = tf.Session()
    # print(sess.run(y))
    # sess.close()

    with tf.Session() as sess:
        print(sess.run([a1, b1]))
        print(sess.run(y))
        print(sess.run([a1, b1, y]))
        print(y.eval())

H = tf.Graph()
with H.as_default():
    a2 = tf.constant(6, tf.float32)
    b2 = tf.constant(7, tf.float32)

with tf.Session(graph=H) as sess:
    print(sess.run(a2))