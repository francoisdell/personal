from tensorflow.contrib.estimator.python.estimator.rnn import RNNClassifier, RNNEstimator
from tensorflow.contrib.estimator import regression_head
from tensorflow import parse_example
from tensorflow.feature_column import *
from tensorflow.contrib.feature_column import *
import tensorflow as tf

columns = [[1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,24,28,32,36,40]]
columns_feature = tf.feature_column.numeric_column(key='temperature')

features = make_parse_example_spec([columns_feature])

temperature = sequence_numeric_column('temperature')

estimator = RNNEstimator(head=regression_head(), sequence_feature_columns=features)

print(estimator)