import tensorflow as tf
import ssl
import datetime
import tensorflow.examples.tutorials.mnist.input_data as input_data
ssl._create_default_https_context = ssl._create_unverified_context

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# def model
xs = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# neurons in next layer
y = tf.nn.softmax(tf.matmul(xs, W) + b)

# label
y_ = tf.placeholder("float", [None, 10])
# cross_entropy  ?
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# train model
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    begin = datetime.datetime.now()
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xs: batch_xs, y_: batch_ys})
    end = datetime.datetime.now()

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(sess.run(accuracy, feed_dict={xs: mnist.test.images, y_: mnist.test.labels}))
k = end - begin
print("train cost:", k.total_seconds())

