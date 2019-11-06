import tensorflow as tf

#实现一个求解阶乘的代码
#定义一个变量(自增变量)或者定义一个占位符，使用fetch的方式传入每次要阶乘乘以的那个值
# i = tf.placeholder(dtype=tf.int32, shape=None)
i = tf.Variable(1, dtype=tf.int32)
#定义一个变量(存储每次阶乘结果)
x=tf.Variable(1,dtype=tf.int32)
#定义一个操作(更新变量(阶乘))
op_i = tf.assign(ref=i, value=tf.add(i, 1))
#控制依赖
with tf.control_dependencies([op_i]):
    op_x = tf.assign(ref=x, value=tf.multiply(x, i))
#会话执行更新变量的操作
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i_ in range(2,5):
    # sess.run(op_x ,feed_dict={i:i_})
    sess.run(op_x)
    print(sess.run(x))