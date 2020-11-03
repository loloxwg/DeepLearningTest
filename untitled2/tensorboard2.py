import tensorflow as tf
import numpy as np


def add_layer(inputs,
              in_size,
              out_size,
              n_layer,
              activation_function=None):
    # add one more layer and return the output of this layer
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            ##接下来,我们层中的Weights设置变化图, tensorflow中提供了tf.histogram_summary()方法,
            # 用来绘制图片, 第一个参数是图表的名称, 第二个参数是图表要记录的变量
            tf.summary.histogram(layer_name + 'weights', Weights)
        ##同样的方法我们对biases进行绘制图标:
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.summary.histogram(layer_name + 'biases', biases)

        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
            # 至于activation_function
            # 可以不绘制.我们对output
            # 使用同样的方法:
        if activation_function is None:
            outputs = Wx_plus_b
            tf.summary.histogram(layer_name + 'outputs', outputs)
        else:
            outputs = activation_function(Wx_plus_b, )
            tf.summary.histogram(layer_name + 'outputs', outputs)
        return outputs


# Make up some real data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise
# define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# add hidden layer
l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None)

# the error between prediciton and real data
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                        reduction_indices=[1]))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# 接下来， 开始合并打包。 tf.merge_all_summaries() 方法会对我们所有的
# summaries 合并到一起. 因此在原有代码片段中添加:
sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)
init = tf.global_variables_initializer()
sess.run(init)
#sess.run(tf.initialize_all_variables())
for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
        writer.add_summary(result, i)
# direct to the local dir and run this in terminal:
# $ tensorboard --logdir=logs

#运行以后在Pycharm的Terminal输入tensorboard --logdir logs即可打开浏览器观看
