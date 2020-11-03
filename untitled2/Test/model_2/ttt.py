# encoding=utf-8
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
mnist = input_data.read_data_sets(".\mni",one_hot=True)

class Net:
    def  __init__(self):
        self.x = tf.placeholder(dtype=tf.float32,shape=[None, 784])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None,10])
        self.w1 = tf.Variable(tf.truncated_normal(shape=[784,256],stddev=0.1,dtype=tf.float32))
        self.b1 = tf.Variable(tf.zeros(shape=[256],dtype=tf.float32))
        self.w2 = tf.Variable(tf.truncated_normal(shape=[256, 128], stddev=0.1, dtype=tf.float32))
        self.b2 = tf.Variable(tf.zeros(shape=[128], dtype=tf.float32))
        self.w3 = tf.Variable(tf.truncated_normal(shape=[128,10], stddev=0.1, dtype=tf.float32))
        self.b3 = tf.Variable(tf.zeros(shape=[10], dtype=tf.float32))

    def forward(self):
        y1=tf.nn.relu(tf.matmul(self.x,self.w1)+self.b1)
        y2 = tf.matmul(y1, self.w2) + self.b2
        self.y3= tf.matmul(y2, self.w3) + self.b3
        self.output = tf.nn.softmax(self.y3)

    def loss(self):
        self.error=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y,logits=self.y3))

    def backward(self):
        self.optimizer=tf.train.GradientDescentOptimizer(0.001).minimize(self.error)

    def accuracy(self):
        y = tf.equal(tf.argmax(self.output,axis=1),tf.argmax(self.y,axis=1))
        self.acc = tf.reduce_mean(tf.cast(y,dtype=tf.float32))


if __name__ == '__main__':
    net=Net()
    net.forward()
    net.loss()
    net.backward()
    net.accuracy()
    init_op=tf.global_variables_initializer()

    plt.ion()
    a=[]
    b=[]
    c=[]
    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(20000):
            xs,ys=mnist.train.next_batch(100)
            error,_ = sess.run([net.error,net.optimizer],feed_dict={net.x:xs,net.y:ys})
            if i%100 == 0:
                xss,yss = mnist.validation.next_batch(100)
                _error,_output,acc = sess.run([net.error,net.output,net.acc],feed_dict={net.x:xss,net.y:yss})
                label = np.argmax(yss[0])
                out = np.argmax(_output[0])
                print('error:',error)
                print('label:',label,"output:",out)
                print('accuracy:',acc)
                a.append(i)
                b.append(error)
                c.append(_error)
                plt.clf()
                train, =plt.plot(a,b,linewidth=1,color='red')
                validate, = plt.plot(a,c,linewidth=1,color='blue')
                plt.legend([train,validate],['train','validate'],loc='right top',fontsize=10)
                plt.pause(0.01)

    plt.ioff()