import tensorflow as tf

#2. 编写一段代码，实现动态的更行变量的维度数目
#[0,0]==>[0,0,0,0]==>[0,0,0,0,0,0,0,0]
#trainable默认为True,表示参与梯度下降
x = tf.Variable([], validate_shape=False, trainable=False)
#validate_shape 默认为True，表示形状不可变
assign_op = tf.assign(ref=x, value=tf.concat([x, [0., 0.]], axis=0),
                      validate_shape=False)
sess=tf.Session()
sess.run(tf.global_variables_initializer())
with tf.control_dependencies([assign_op]):
    print_x = tf.Print(x, data=[x, x.read_value()], message='x2, x2 read=')
for i in range(5):
    # sess.run(assign_op)
    print(sess.run(print_x))