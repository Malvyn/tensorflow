import tensorflow as tf
import sys
#生成work server
#3个cpu节点和一个gpu节点
#jobs
arg = sys.argv
if len(arg) == 3:
    cluster = tf.train.ClusterSpec(
        {'cpu': ['localhost:10001', 'localhost:10002'],
         'gpu': ['localhost:10004']
         }
    )

#注册server
server = tf.train.Server(cluster, job_name=arg[1], task_index=int(arg[2]))