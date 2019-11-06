#先进先出
#队列就是日常生活中排队，先来先到
import tensorflow as tf
#定义一个随机的队列，初始化队列，设置队列大小为5，最小长度为2
q = tf.RandomShuffleQueue(capacity=5,min_after_dequeue=0, dtypes=tf.float32)

#入队操作da
init = q.enqueue_many(([1,2,3,4,5],))

#定义出对操作
x = q.dequeue()

#将出对的数据+1，然后再加入队列中
y = x + 1
q_in = q.enqueue([y])
#创建会话执行队列操作
with tf.Session() as sess:
    sess.run(init)

    #执行多次出对操作,然后进行+1再入队
    for i in range(2):
        sess.run(q_in)

    #获取队列的长度
    que_len = sess.run(q.size())
    #将所有元素出队
    for i in range(que_len):
        print(sess.run(x))