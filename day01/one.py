import tensorflow as tf
import numpy as np
import os, sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)

sess = tf.Session()
print(sess.run([node1, node2]))

node3 = tf.add(node1, node2)

print('node3:', node3)
print('sess.run(node3):', sess.run(node3))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b # + is short for tf.add(a, b)

print(sess.run(adder_node, {a:4, b:3.5}))
print(sess.run(adder_node, {a:[1, 2, 3, 4], b:[2, 3, 4, 5]}))
print(sess.run(adder_node, {a:[[1, 2, 3, 4], [5, 6, 7, 8]], b:[[2, 3, 4, 5], [6, 7, 8, 9]]}))

add_and_triple = adder_node * 3

print(sess.run(add_and_triple, {a:4, b:3.5}))
print(sess.run(add_and_triple, {a:[1, 2, 3, 4], b:[2, 3, 4, 5]}))
print(sess.run(add_and_triple, {a:[[1, 2, 3, 4], [5, 6, 7, 8]], b:[[2, 3, 4, 5], [6, 7, 8, 9]]}))


W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(dtype=tf.float32)

linear_model = W*x + b

init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(linear_model, {x:[1, 2, 3, 4, 5]}))

y = tf.placeholder(dtype=tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1, 2, 3, 4], y:[0, -1, -2, -3]}))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init)
for i in range(1000):
    sess.run(train,{ x:[1, 2, 3, 4], y:[0, -1, -2, -3]})
    print(sess.run(loss,{ x:[1, 2, 3, 4], y:[0, -1, -2, -3]}))

print(sess.run([W, b]))

