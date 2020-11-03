# 搭建图纸 #
# 首先从 Input 开始：
# define placeholder for inputs to network
import tensorflow as tf


# xs = tf.placeholder(tf.float32, [None, 1])
# ys = tf.placeholder(tf.float32, [None, 1])
# 对于input我们进行如下修改： 首先，可以为xs指定名称为x_in:
# xs = tf.placeholder(tf.float32, [None, 1], name='x_in')
# 然后再次对ys指定名称y_in:
# ys = tf.placeholder(tf.float32, [None, 1], name='y_in')
# 这里指定的名称将来会在可视化的图层inputs中显示出来
# 使用with tf.name_scope('inputs')可以将xs和ys包含进来，
# 形成一个大的图层，图层的名字就是with tf.name_scope()方法里的参数。


def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    with tf.name_scope('layer'):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        return outputs


# define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_in')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_in')
# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

# the error between prediction and real data
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
writer = tf.summary.FileWriter(" logs/", sess.graph)
# important step
init = tf.global_variables_initializer()
sess.run(init)

##最后这两行很重要，没有的话会打不开tensorboard
##运行以后在Pycharm的Terminal输入tensorboard --logdir logs即可打开浏览器观看