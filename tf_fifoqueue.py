# 先进先出
# 队列就是日常生活中排队，先来先的
import tensorflow as tf

# 定义一个先入先出的队列，初始化队列，设置队列大小为5
q=tf.FIFOQueue(5,tf.float32)

# 入队操作
init=q.enqueue_many(([1,2,3,4,5],))

# 定义出队操作
x=q.dequeue()

# 将出队的数据+1，然后再加入队列中
y=x+1
q_in=q.enqueue([y])
# 创建会话执行队列操作
with tf.Session() as sess:
    sess.run(init)

    # 执行多次出队操作,然后进行+1再入队
    for i in range(3):
        sess.run(q_in)

    # 获取队列的程度
    que_len=sess.run(q.size())

    # 将所有元素出队
    for i in range(que_len):
        print(sess.run(x))