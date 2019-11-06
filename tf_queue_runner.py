# 队列管理器
import tensorflow as tf
#创建一个先进先出的队列，长度为1000
q=tf.FIFOQueue(1000,tf.float32)

# 定义一个计数器
counter=tf.Variable(0.0)
# 定义计数器自加的操作
counter_add=tf.assign_add(counter,tf.constant(1.0))

# 入队操作
with tf.control_dependencies([counter_add]):
    # 换成本地读数据的操作
    enque_op=q.enqueue([counter])

# 创建队列管理器
qr=tf.train.QueueRunner(q,enqueue_ops=[enque_op,counter_add])

# 创建会话
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 创建一个线程协调器
    coord=tf.train.Coordinator()
    # 启动入队线程
    enqueue_threads=qr.create_threads(sess,coord=coord,start=True)
    # 1 =>
    # 2=>
    # 3=>
    # 3<=
    # 4=>
    # 5=>
    # 6<=
    # 主线程
    for i in range(10):
        print(sess.run(q.dequeue()))
    # 通知其他线程关闭
    coord.request_stop()
    # 等待其他线程结束，当其他线程都关闭之后，函数才返回结果
    coord.join(enqueue_threads)