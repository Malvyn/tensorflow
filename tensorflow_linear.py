# 实现线性回归
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# 1.生成一个y=w*x+b的数据集
# y=0.1*x+0.3
# 生成1000个点，围绕在y=0.1*x+0.3直线周围
num_points=1000
vectors_set=[]
for i in range(num_points):
    x1=np.random.normal(0.0,0.55)#均值为0,方差为0.55的高斯分布
    y1=0.1*x1+0.3+np.random.normal(0.0,0.03)# 加入一些抖动
    vectors_set.append([x1,y1])

# 构建X，Y
x_data=[v[0] for v in vectors_set]
y_data=[v[1] for v in vectors_set]
# plt.scatter(x_data,y_data,c='r')
# plt.show()
# 2.tensorflow构建一个线性回归的模型
# W*x+b
x_input=tf.placeholder(tf.float32,[None])
y_out=tf.placeholder(tf.float32,[None])
def linear_net():
    W=tf.Variable(tf.random_uniform([1],-1.0,1.0),name='W')
    b=tf.Variable(tf.zeros([1]),name='b')
    # 预测
    out=tf.add(tf.multiply(W,x_input),b)
    return out,W,b

y_,W,b=linear_net()
# loss
# 均方差 mean
# 平方和误差 sum
loss=tf.reduce_mean(tf.square(y_-y_out),name='loss')
# 优化器
# minimize中间一共有两步，第一步计算梯度（compute_gradients），第二步应用梯度（apply_gradients）
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)

# 3.调用模型进行训练
with tf.Session() as sess:
    # 全局变量初始化
    tf.global_variables_initializer().run()
    # 初始的W，b
    print("W={},b={}".format(sess.run(W),sess.run(b)))
    for step in range(20):
        weight,bias,cost,_=sess.run([W,b,loss,optimizer],feed_dict={x_input:x_data,y_out:y_data})
        print("W={},b={},loss={}".format(weight,bias,cost))
        # y=0.1*X+0.3

    # 4.实现预测
    plt.scatter(x_data,y_data,c='r')
    plt.plot(x_data,weight*x_data+bias)
    plt.show()