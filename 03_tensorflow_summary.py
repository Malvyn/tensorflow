#编写一个累加器；定义一个变量x(初值随机给定)，定义一个占位符y，迭代4次，每个迭代返回x*y的值，并且在计算x*y乘积前，先对x进行累加操作。并且将这个程序信息输出到文件以TensorBoard展示。
from tensorflow.python import debug as tf_debug
import tensorflow as tf
x=tf.Variable(tf.random_normal([]),name='x')
y=tf.placeholder(tf.float32,[])
assign_add_x=tf.assign(x,tf.add(x,1))
with tf.control_dependencies([assign_add_x]):
    tf.summary.scalar('x_value', x)
    op=tf.multiply(x,y)
    tf.summary.histogram('result',op)


sess = tf.Session()
#sess = tf_debug.TensorBoardDebugWrapperSession(sess, "DESKTOP-VBLAA6J:6007")
#sess.run(tf.global_variables_initializer())
# 收集summary scalar、histogram、image。。。。
merge=tf.summary.merge_all()
# 记录网络图
summary_write=tf.summary.FileWriter('summary',sess.graph)
for i in range(1000000):
    _,summary_result=sess.run([op,merge],feed_dict={y:0.1})
    summary_write.add_summary(summary_result,i)
summary_write.close()