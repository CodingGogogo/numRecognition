import numpy
import tensorflow as tf

# ---------eg1-------------
matrix12 = tf.constant([[3, 3]])
matrix21 = tf.constant([[3], [3]])

product = tf.matmul(matrix21, matrix12)

# with..as can close the Session
with tf.Session() as sess:
    result = sess.run(product)
    print(result)


# -------eg2---------------
state = tf.Variable(0, name="counter")

one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# init_op = tf.initialize_all_variables()
init_op = tf.global_variables_initializer()


# start graph
with tf.Session() as sess:
    # sess.run(init_op)
    state.initializer.run()
    print(sess.run(state))
    for i in range(3):
        # sess.run(update)
        sess.run(update)
        print(sess.run(state))

# --------- fetch -------------------

input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2, input3)
mul_result = tf.multiply(input1, intermed)

with tf.Session() as sess:
    # fetch
    result = sess.run([mul_result, intermed])
    print(result)

# -------feed --------------
# Placeholder
feed_input1 = tf.placeholder(tf.float32)
feed_input2 = tf.placeholder(tf.float32)
feed_output = tf.multiply(feed_input1, feed_input2)

with tf.Session() as sess:
    print(sess.run(feed_output, feed_dict={feed_input1:7., feed_input2:3.}))










