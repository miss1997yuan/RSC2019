import tensorflow as tf
from utils import karate
import matplotlib.pyplot as plt

NODE_SIZE = 34
NODE_FEATURE_DIM = 34
HIDDEN_DIM1 = 10
num_classes = 2
training_epochs = 100
step = 10
lr=0.03

X = tf.placeholder(tf.float32, shape=[NODE_SIZE, NODE_FEATURE_DIM])

Y = tf.placeholder(tf.int32, shape=[NODE_SIZE])
label = tf.one_hot(Y, num_classes)
Y_enc = tf.one_hot(Y, 2)

adj = tf.placeholder(tf.float32, shape=[NODE_SIZE, NODE_SIZE])

weights = {"hidden1": tf.Variable(tf.random_normal(dtype=tf.float32, shape=[NODE_FEATURE_DIM, HIDDEN_DIM1]), name='w1'),
           "hidden2": tf.Variable(tf.random_normal(dtype=tf.float32, shape=[HIDDEN_DIM1, num_classes]), 'w1')}

D_hat = tf.matrix_inverse(tf.matrix_diag(tf.reduce_sum(adj, axis=0)))

l1 = tf.matmul(tf.matmul(tf.matmul(D_hat, adj), X), weights['hidden1'])
output = tf.matmul(tf.matmul(tf.matmul(D_hat, adj), l1), weights['hidden2'])
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_enc, logits=output))

train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# tf.losses.b


init_op = tf.global_variables_initializer()

##karate data
g = karate()

feed_dict = {adj: g.A, X: g.I, Y: g.node_label}
with tf.Session() as sess:
        sess.run(init_op)
        plt.ion()
        for epoch in range(training_epochs):
                c, _ = sess.run([loss, train_op], feed_dict)
                if epoch % step == 0:
                        print(f'Epoch:{epoch} Loss {c}')

                represent = sess.run(output, feed_dict)
                plt.scatter(represent[:, 0], represent[:, 1], s=200, c=g.node_label)
                plt.title(f"$Epoch:{epoch}$")
                # plt.savefig(f"./tmp/{epoch}.png")
                plt.pause(0.1)
                plt.cla()
        # plt.ioff()
        # plt.show()
