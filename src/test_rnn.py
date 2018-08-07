__author__ = 'Andrew Mackenzie'
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell

class RecurrentNerualNetwork:

    def __init__(self,
                 hm_epochs: int=3,
                 n_classes: int=10,
                 batch_size: int=128,
                 chunk_size: int=28,
                 n_chunks: int=28,
                 rnn_size: int=128):

        self.mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

        self.hm_epochs = hm_epochs
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.n_chunks = n_chunks
        self.rnn_size = rnn_size

    def recurrent_neural_network(self, x):

        layer = {'weights':tf.Variable(tf.random_normal([self.rnn_size, self.n_classes])),
                 'biases':tf.Variable(tf.random_normal([self.n_classes]))}

        x = tf.transpose(x, [1,0,2])
        x = tf.reshape(x, [-1, self.chunk_size])
        x = tf.split(x, self.n_chunks, 0)

        lstm_cell = rnn_cell.BasicLSTMCell(self.rnn_size, state_is_tuple=True)
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']

        return output

    def train(self, x):

        prediction = self.recurrent_neural_network(x)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))
        optimizer = tf.train.AdamOptimizer().minimize(cost)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # sess.run(tf.initialize_all_variables())

            for epoch in range(self.hm_epochs):
                epoch_loss = 0
                for _ in range(int(mnist.train.num_examples / self.batch_size)):
                    epoch_x, epoch_y = self.mnist.train.next_batch(self.batch_size)
                    epoch_x = epoch_x.reshape((self.batch_size, self.n_chunks, self.chunk_size))

                    _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                    epoch_loss += c

                print('Epoch', epoch, 'completed out of', self.hm_epochs, 'loss:', epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy:',
                  accuracy.eval({x: self.mnist.test.images.reshape((-1, self.n_chunks, self.chunk_size)),
                                 y: self.mnist.test.labels}))


if __name__ == '__main__':

    # sess = tf.Session()
    # print(sess)

    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    hm_epochs = 3
    n_classes = 10
    batch_size = 64
    chunk_size = 28
    n_chunks = 28
    rnn_size = 64

    x = tf.placeholder('float', [None, n_chunks, chunk_size])
    y = tf.placeholder('float')

    rnnet = RecurrentNerualNetwork(hm_epochs, n_classes, batch_size, chunk_size, n_chunks, rnn_size)
    rnnet.train(x)
