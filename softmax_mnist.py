# 使用softmax逻辑回归实现mnist数据集的预测
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 数据准备
mnist=input_data.read_data_sets(train_dir='data/',one_hot=True)
# train 和 test
# trian数据量是55000条，图片数据一个样本是一个784的一维向量，label数据一个样本是一个10的一维向量
# test数据量有10000条
# (数据预处理)
#  构建模型
# 1.超参数和参数设置
learning_rate=1e-2#学习率
epoch_num=50#迭代次数
input_size=784#输入数据大小
out_size=10#类别数
batch_size=100# 批数据量
display_step=5#每多少次显示一下评估指数
# 2.占位符设置
x=tf.placeholder(tf.float32,[None,input_size])
y=tf.placeholder(tf.float32,[None,out_size])
# 3.模型设置
def softmax_model():
    # y=w*x+b
    W=tf.Variable(tf.random_uniform([input_size,out_size]))
    b=tf.Variable(tf.zeros([out_size]))
    pre=tf.nn.bias_add(tf.matmul(x,W),b)
    actv=tf.nn.softmax(pre)
    # softamx(y)
    return actv
def train():
    out=softmax_model()
    # 构建损失、准确率、优化器
    cost=tf.reduce_mean(-tf.reduce_sum(y*tf.log(out),axis=1))# 损失
    # argmax 取得最大值所在的下标
    # equal计算相等，返回一个boolean矩阵
    # cast转换数据类型=》True变成1.0，False变成0.0
    # reduce_mean计算平均值
    pred=tf.equal(tf.argmax(out,1),tf.argmax(y,1))
    accr=tf.reduce_mean(tf.cast(pred,tf.float32))#准确率
    optm=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)# 优化器
    # 迭代训练模型、并持久化
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())#全局变量
    saver=tf.train.Saver()
    for epoch in range(epoch_num):
        avg_cost=0
        num_batch=mnist.train.num_examples//batch_size
        for i in range(num_batch):
            # 从数据集抽样batch_size条数据出来
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            sess.run(optm,feed_dict={x:batch_xs,y:batch_ys})
        if epoch%display_step==0:
            # 使用测试集评估
            feeds_test={x:mnist.test.images,y:mnist.test.labels}
            test_acc=sess.run(accr,feed_dict=feeds_test)
            print('Epoch {},test acc:{}'.format(epoch,test_acc))
            if test_acc>0.9:
                saver.save(sess,'model/softmax.ckpt')
                break
    saver.save(sess,'model/softmax.ckpt')
#train()
# 载入模型，进行预测
def predict():
    import numpy as np
    out=softmax_model()
    sess=tf.Session()
    saver=tf.train.Saver()
    saver.restore(sess,'model/softmax.ckpt')
    feeds_test = {x: mnist.test.images[0:1]}
    y_true=mnist.test.labels[0:1]
    pred=sess.run(out,feed_dict=feeds_test)
    print(np.argmax(pred,axis=1),np.argmax(y_true,1))
predict()