import tensorflow as tf

#查看添加的操作是否使添加到默认图中
# a = tf.constant(4.0)
# print("变量a定义在默认图上:{}".format(a.graph is tf.get_default_graph()))

#明确指定一个新图
# g = tf.Graph()
# with g.as_default():
#     #定义一个新的操作在图g上
#     b = tf.constant(3.0)
#     c = tf.constant(5.0)
#     print("变量b定义在图g上:{}".format(b.graph is g))
#     print("变量c定义在图g上:{}".format(c.graph is g))

# 设置操作
# 张量
mat1 = tf.constant([3,3],dtype=tf.float32,shape=[1,2])
mat2 = tf.constant([[2.],[2.]])

# 计算操作
product = tf.matmul(mat1, mat2)
w1 = tf.Variable(tf.random_normal([1]))

#启动默认图
# sess = tf.Session()
sess = tf.InteractiveSession()
# #变量初始化
sess.run(tf.global_variables_initializer())
print(w1.eval())
# reuslt = sess.run(product)
# print(reuslt)
# sess.close()

#使用with代码块,自动完成关闭操作
# with tf.Session() as sess2:
#     # result = sess2.run(product)
#     result = product.eval()
#     print(result)



