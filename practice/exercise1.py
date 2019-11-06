import tensorflow as tf

#1. 定义一个累加器，并且每一步均输出累加器的结果值
#(1)定义变量
x = tf.Variable(0)
#(2)变量的更新操作op
op_add_x = tf.assign(ref=x, value=tf.add(x, 1))
#(3) 循环run这个op操作，实现变量的数值变化
sess = tf.Session()
#初始化变量
sess.run(tf.global_variables_initializer())
for i in range(10):
    print(sess.run(x))
    print(sess.run((op_add_x)))
    print('*'*100)