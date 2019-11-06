# 引入包
import tensorflow as tf

# # 设置操作
# # 张量
# mat1=tf.constant([3,3],dtype=tf.float32,shape=[1,2])
# mat2=tf.constant([[2.],[2.]])
#
# # 计算操作
# product=tf.matmul(mat1,mat2)
# w1=tf.Variable(tf.random_normal([1]))
# # 执行操作
# print(product)
# # Tensor("MatMul:0", shape=(1, 1), dtype=float32)
#
# g2=tf.Graph()
# with g2.as_default():
#     e=tf.constant(2,shape=[1,1])
#     # 中间空间
#     tf.add_to_collection('var',e)
# cpu 取一个结果和gpu中一个结果做计算，中间变量（设备间通信，以及图的通信）
#add_n=tf.add(product,tf.get_collection('var'))

#sess=tf.Session()
# sess1=tf.InteractiveSession()
# # 变量初始化
# sess1.run(tf.global_variables_initializer())
# print(w1.eval())
#print(sess.run([w1,product]))
w1=tf.Variable(1.0,name='w1')
sess=tf.Session()
sess.run(tf.global_variables_initializer())
w2=tf.placeholder(tf.float32,[1])
w3=tf.add(w1,w2)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(w3,feed_dict={w2:[2.0]}))