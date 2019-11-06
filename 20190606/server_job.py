import tensorflow as tf
import sys
# 生成workner server
# 3个cpu节点和一个gpu节点
# jobs
arg=sys.argv
if len(arg)==3:
    cluser=tf.train.ClusterSpec(
    {'worker':['localhost:20001','localhost:20002'],
     'master': ['localhost:20003'],
     'ps':['localhost:20004']
     }
    )
    # 注册server
    server=tf.train.Server(cluser,job_name=arg[1],task_index=int(arg[2]))
    #python server_job.py worker 0
    #python server_job.py worker 1
    #python server_job ps 0