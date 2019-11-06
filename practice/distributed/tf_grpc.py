#分布式
#多台机器，分配不同的任务
#一般1个主节点--》n个子节点

#集群
#多台机器，协作完成同一个任务

import tensorflow as tf

with tf.device(tf.train.replica_device_setter(worker_device='/job:cpu/task:0')):
    mat1 = tf.constant([3,3], dtype=tf.float32, shape=[1, 2])
    mat2 = tf.constant([[3.],[3.]])
    product = tf.matmul(mat1, mat2)

sess = tf.Session(target='grpc://localhost:10004', config=tf.ConfigProto(
    log_device_placement=True))
print(sess.run(product))
