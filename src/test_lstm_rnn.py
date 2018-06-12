from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class lstm_rnn:
    def __init__(self,
                 num_epochs: int=10,
                 truncated_backprop_length: int=15,
                 state_size: int=4,
                 num_classes: int=1,
                 echo_step: int=3,
                 batch_size: int=5,
                 num_layers: int=3):

        self.num_epochs = num_epochs
        self.truncated_backprop_length = truncated_backprop_length
        self.state_size = state_size
        self.num_classes = num_classes
        self.echo_step = echo_step
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.total_series_length = 50000
        self.num_variables = 2
        self.num_batches = self.total_series_length//batch_size//truncated_backprop_length

    def generate_data(self):
        x = np.array(np.random.choice(2, self.total_series_length, p=[0.5, 0.5]))
        y = np.roll(x, self.echo_step)
        # y[0:self.echo_step] = 0

        for v in range(self.num_variables-1):
            x2 = np.roll(x, -self.echo_step * (v+1))
            x = np.dstack((x, x2))

        x = x.reshape((self.batch_size, -1, self.num_variables))  # The first index changing slowest, subseries as rows
        y = y.reshape((self.batch_size, -1))
        # y = np.remainder(np.sum(x, axis=1), 2).reshape(-1, 1)

        return x, y

    def train(self):

        batchX_placeholder = tf.placeholder(tf.float32, [self.batch_size, self.truncated_backprop_length, self.num_variables])
        if self.num_classes > 1:
            y_dtype = tf.int32
        else:
            y_dtype = tf.float32
        batchY_placeholder = tf.placeholder(y_dtype, [self.batch_size, self.truncated_backprop_length])

        init_state = tf.placeholder(tf.float32, [self.num_layers, 2, self.batch_size, self.state_size])

        state_per_layer_list = tf.unstack(init_state, axis=0)
        rnn_tuple_state = tuple(
            [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
             for idx in range(self.num_layers)]
        )

        # W = tf.Variable(np.random.rand(self.state_size+1, self.state_size), dtype=tf.float32)
        # b = tf.Variable(np.zeros((1, self.state_size)), dtype=tf.float32)

        # Forward passes
        cells = []
        for _ in range(self.num_layers):
            cell = tf.nn.rnn_cell.LSTMCell(self.state_size, state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.5)
            cells.append(cell)
        cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        # cell = tf.nn.rnn_cell.LSTMCell(self.state_size, state_is_tuple=True)
        # cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)
        # cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.num_layers, state_is_tuple=True)
        if len(batchX_placeholder.shape) == 2:
            batchX_placeholder = tf.expand_dims(batchX_placeholder, -1)

        states_series, current_state = tf.nn.dynamic_rnn(cell, batchX_placeholder, initial_state=rnn_tuple_state)
        states_series = tf.reshape(states_series, [-1, self.state_size])

        W = tf.Variable(np.random.rand(self.state_size, self.num_classes), dtype=tf.float32)
        b = tf.Variable(np.zeros((1, self.num_classes)), dtype=tf.float32)

        logits = tf.matmul(states_series, W) + b  # Broadcasted addition
        labels = tf.reshape(batchY_placeholder, [-1])

        if self.num_classes > 1:
            print('Setting up for a CLASSIFICATION model.')
            logits_series = tf.unstack(tf.reshape(logits, [self.batch_size, self.truncated_backprop_length, self.num_classes]), axis=1)

            predictions_series = [tf.nn.softmax(logit) for logit in logits_series]
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            total_loss = tf.reduce_mean(losses)
            train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)
        else:
            print('Setting up for a REGRESSION model.')
            predictions_series = logits
            total_loss = tf.reduce_mean(tf.abs(labels - logits))
            train_step = tf.train.GradientDescentOptimizer(0.3).minimize(total_loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            plt.ion()
            plt.figure()
            plt.show()
            loss_list = []

            for epoch_idx in range(self.num_epochs):
                x, y = self.generate_data()

                _current_state = np.zeros((self.num_layers, self.num_classes, self.batch_size, self.state_size))

                print("New data, epoch", epoch_idx)

                for batch_idx in range(self.num_batches):
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

    def plot(self, loss_list, predictions_series, batchX, batchY):
        plt.subplot(2, 3, 1)
        plt.cla()
        plt.plot(loss_list)

        for batch_series_idx in range(self.batch_size):
            one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
            single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])

            plt.subplot(2, 3, batch_series_idx + 2)
            plt.cla()
            plt.axis([0, self.truncated_backprop_length, 0, 2])
            left_offset = [x+0.5 for x in range(self.truncated_backprop_length)]
            # for v in range(self.num_variables):
            #     left = v/self.num_variables
            #     right = (v+1)/self.num_variables
            #     plt.bar(left_offset, batchX[batch_series_idx, :], width=(v+1)/self.num_variables, color="blue", edgecolor='black')
            # plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="green")
            # plt.bar(left_offset, single_output_series * 0.3, width=1, color="red")

            plt.bar(left_offset, [1] * self.truncated_backprop_length, width=1, facecolor="None", edgecolor='black', zorder=2)
            bar_width = (1 / self.num_variables)
            bar_buffer = bar_width * 0.02
            bar_width = bar_width - (bar_buffer*2)

            for v in range(self.truncated_backprop_length):
                actual = batchY[batch_series_idx, v]
                pred = single_output_series[v]
                if actual == pred:
                    pred_color = 'green'
                else:
                    pred_color = 'red'
                # plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="white", edgecolor='black')
                for var_num in range(self.num_variables):
                    left = v+0.25 + (var_num / self.num_variables) + bar_buffer
                    plt.bar(left, batchX[batch_series_idx,  v, var_num], width=bar_width, color="lightblue", zorder=1)
                plt.annotate(batchY[batch_series_idx, v], xy=(v+0.25, 0.7), color='green', zorder=2)
                plt.annotate(single_output_series[v], xy=(v+0.25, 0.3), color=pred_color, zorder=2)
            # plt.scatter(left_offset, [0.7]*self.truncated_backprop_length, data=batchY[batch_series_idx, :], color='green')
            # plt.scatter(left_offset, [0.3]*self.truncated_backprop_length, data=single_output_series, color='red')

        plt.draw()
        plt.pause(0.0001)


if __name__ == '__main__':

    rnn = lstm_rnn()
    rnn.train()
