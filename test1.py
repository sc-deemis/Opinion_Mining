import tensorflow as tf


t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
tf.concat([t1, t2], 0)  # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
tf.concat([t1, t2], 1)  # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]


x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = tf.concat(values=[t1, t2], axis=0)

with tf.Session() as sess:
    print(sess.run(z, feed_dict={x: t1,y:t2}))
