import tensorflow as tf
# #1.实现一个累加器，并且每一步均输出累加器的结果值。
# # （1）定义变量
# x=tf.Variable(0)
# # （2）变量的更新操作op
# op_add_x=tf.assign(ref=x,value=tf.add(x,1))
# # （3）循环run这个op操作，实现变量的数值变化
# sess=tf.Session()
# # 初始化变量
# sess.run(tf.global_variables_initializer())
# for i in range(10):
#     print(type(x))
#     print(sess.run(x))
#     #x=tf.add(x,1)
#     print(sess.run(op_add_x))
#     print('*'*100)
# print(sess.run(x))
#2.编写一段代码，实现动态的更新变量的维度数目
# [0,0]=>[0,0,0,0]=>[0,0,0,0,0,0]
# trainable 默认为True，表示参与梯度下降计算
x2=tf.Variable([],validate_shape=False,trainable=False)
# validate_shape 默认为True，表示形状不可变
assign_op=tf.assign(ref=x2,value=tf.concat([x2,[0.,0.]],axis=0),validate_shape=False)
with tf.control_dependencies([assign_op]):
    print_x2 = tf.Print(x2, data=[x2, x2.read_value()], message='x2,x2 read=')
sess=tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(5):
    print(sess.run(print_x2))
#3.实现一个求解阶乘的代码
# 占位符+assign