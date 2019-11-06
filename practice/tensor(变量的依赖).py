import tensorflow as tf

#创建一个变量
w1 = tf.Variable(tf.random_normal([10], stddev=0.5, dtype=tf.float32), name = 'w1')

#基于第一个变量创建第二个变量
a = tf.constant(2, dtype=tf.float32)
# w2 = tf.Variable(w1.initialized_value()*a, name='w2')
w2 = tf.Variable(w1*a, name='w2')


#进行全局初始化
init_op = tf.global_variables_initializer()

#启动图
# allow_soft_placement = True : 如果你指定的设备不存在，允许TF自动分配设备
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    #运行init op
    sess.run(init_op)
    #获取值(对于多个值的获取建立获取方式)
    result = sess.run([w1, w2])
    print("w1={}\nw2={}".format(result[0], result[1]))