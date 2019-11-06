import tensorflow as tf
a=tf.Variable(1.0)

op_assign_a=tf.assign(a,tf.add(a,1))
sess=tf.Session()
sess.run(tf.global_variables_initializer())
saver=tf.train.Saver()# 持久化模型类
def train():
    # 训练
    for i in range(100):
        sess.run(op_assign_a)
        saver.save(sess,'model/assign_a.ckpt',global_step=i)# 模型持久化
    sess.run(a)

def predict():
    print(sess.run(a))
    ckpt_file=tf.train.latest_checkpoint('model')
    print(ckpt_file)
    # 加载
    saver.restore(sess,ckpt_file)
    print(sess.run(a))

#train()
predict()

