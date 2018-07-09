from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

class RecursuveNeuralNetwork:

    def __init__(self,
                 num_epochs: int=20,
                 truncated_backprop_length: int=15,
                 state_size: int=4,
                 num_classes: int=2,
                 batch_size: int=5,
                 num_layers: int=4
                 ):

        self.num_epochs = num_epochs
        self.truncated_backprop_length = truncated_backprop_length
        self.state_size = state_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_layers = num_layers

    def fit(self,
            x: np.ndarray,
            y: np.ndarray
            ):

        num_batches = x.shape[1] // self.batch_size // self.truncated_backprop_length


        batchX_placeholder = tf.placeholder(tf.float32, [self.batch_size, self.truncated_backprop_length, x.shape[2]])
        batchY_placeholder = tf.placeholder(tf.int32, [self.batch_size, self.truncated_backprop_length])

        init_state = tf.placeholder(tf.float32, [self.num_layers, 2, self.batch_size, self.state_size])
        state_per_layer_list = tf.unstack(init_state, axis=0)
        rnn_tuple_state = tuple(
            [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
             for idx in range(self.num_layers)]
        )

        # W = tf.Variable(np.random.rand(self.state_size + 1, self.state_size), dtype=tf.float32)
        # b = tf.Variable(np.zeros((1, self.state_size)), dtype=tf.float32)

        W2 = tf.Variable(np.random.rand(self.state_size, self.num_classes), dtype=tf.float32)
        b2 = tf.Variable(np.zeros((1, self.num_classes)), dtype=tf.float32)

        # Forward passes
        cells = []
        for _ in range(self.num_layers):
            cell = tf.nn.rnn_cell.LSTMCell(self.state_size, state_is_tuple=True)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)
            cells.append(cell)
        cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
        states_series, current_state = tf.nn.dynamic_rnn(cell, batchX_placeholder, initial_state=rnn_tuple_state)
        states_series = tf.reshape(states_series, [-1, self.state_size])

        logits = tf.matmul(states_series, W2) + b2  # Broadcasted addition
        labels = tf.reshape(batchY_placeholder, [-1])

        logits_series = tf.unstack(tf.reshape(logits, [self.batch_size, self.truncated_backprop_length, 2]), axis=1)
        predictions_series = [tf.nn.softmax(logit) for logit in logits_series]

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        total_loss = tf.reduce_mean(losses)

        # Could also use AdamOptimizer?
        train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            plt.ion()
            plt.figure()
            plt.show()
            loss_list = []

            for epoch_idx in range(self.num_epochs):

                _current_state = np.zeros((self.num_layers, 2, self.batch_size, self.state_size))

                print("New data, epoch", epoch_idx)

                for batch_idx in range(num_batches):
                    start_idx = batch_idx * self.truncated_backprop_length
                    end_idx = start_idx + self.truncated_backprop_length

                    batchX = x[:, start_idx:end_idx]
                    batchY = y[:, start_idx:end_idx]

                    _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                        [total_loss, train_step, current_state, predictions_series],
                        feed_dict={
                            batchX_placeholder: batchX,
                            batchY_placeholder: batchY,
                            init_state: _current_state
                        })

                    loss_list.append(_total_loss)

                    if batch_idx % 100 == 0:
                        print("Step", batch_idx, "Batch loss", _total_loss)
                        self.plot(loss_list, _predictions_series, batchX, batchY)

        plt.ioff()
        plt.show()

        return _predictions_series

    def plot(self,
             loss_list,
             predictions_series,
             batchX,
             batchY):

        plt.subplot(2, 3, 1)
        plt.cla()
        plt.plot(loss_list)

        for batch_series_idx in range(5):
            one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
            single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])

            plt.subplot(2, 3, batch_series_idx + 2)
            plt.cla()
            plt.axis([0, self.truncated_backprop_length, 0, 2])
            left_offset = range(self.truncated_backprop_length)
            plt.bar(left_offset, batchX[batch_series_idx, :, 0], width=1, color="blue")
            plt.bar(left_offset, batchX[batch_series_idx, :, 1], width=1, color="orange")
            plt.bar(left_offset, batchX[batch_series_idx, :, 2], width=1, color="yellow")
            plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
            plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")

        plt.draw()
        plt.pause(0.0001)


    def generateData(self):
        echo_step = 3
        total_series_length = 50000

        x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
        x2 = np.roll(x, -1*echo_step)
        x3 = np.roll(x, -2*echo_step)
        y = np.roll(x, echo_step)
        y[0:echo_step] = 0

        x = x.reshape((self.batch_size, -1))  # The first index changing slowest, subseries as rows
        x2 = x2.reshape((self.batch_size, -1))  # The first index changing slowest, subseries as rows
        x3 = x3.reshape((self.batch_size, -1))  # The first index changing slowest, subseries as rows
        x = np.dstack((x, x2, x3))
        y = y.reshape((self.batch_size, -1))

        return x, y


if __name__ == '__main__':
    rnn = RecursuveNeuralNetwork(truncated_backprop_length=10)
    x, y = rnn.generateData()

    preds = rnn.fit(x,y)

    print(y)
    print(preds)
    lol=1