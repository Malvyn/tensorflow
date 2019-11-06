import tensorflow as tf

a1 = tf.Variable([2, 3], dtype=tf.float32, name='a1')
"""
               initial_value=None, 初始化值，可以是python类型的，也可以是tensors对线
               trainable=True,  给定的该变量是否参与模型训练，也就是说在模型训练的时候是否进行更行
               collections=None,
               validate_shape=True,  更新该变量前后，是否要求形状一致
               caching_device=None,
               name=None,
               variable_def=None,
               dtype=None,
               expected_shape=None,
               import_scope=None,
               constraint=None 
"""
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(a1))