import tensorflow as tf
import numpy as np
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Training Code of tensorflow

W = tf.Variable(0.3)
b = tf.Variable(-0.3)
x = tf.placeholder(dtype=tf.float32)

linear_model = W*x + b

y = tf.placeholder(dtype=tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init)
for _ in range(1000):
    sess.run(train, {x:[1, 2, 3, 4], y:[0, -1, -2, -3]})
    print(sess.run(loss, {x:[1, 2, 3, 4], y:[0, -1, -2, -3]}))

curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:[1, 2, 3, 4], y:[0, -1, -2, -3]})

print("W : %s\nb : %s\nloss : %s"%(curr_W, curr_b, curr_loss))
print(sess.run(W), sess.run(b))

