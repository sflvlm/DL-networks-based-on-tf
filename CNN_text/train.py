import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn

# params

# data loading params
tf.flags.DEFINE_float('dev_sample_percentage', 1, 'percentage of the training data to use for validation')
tf.flags.DEFINE_string('positive_data_file', './data/....', 'data source for the positive')
tf.flags.DEFINE_string('negative_data_file', './data/....', 'data source for the negative')

# model_Hparams 超参数
tf.flags.DEFINE_integer('embedding', 128, 'Dimension of character embedding')
tf.flags.DEFINE_string('filter_size', '3,4,5', 'filter size')
tf.flags.DEFINE_integer('num_filters', 128, 'Number of filters')
tf.flags.DEFINE_float('dropout', 0.5, 'Dropout')
tf.flags.DEFINE_float('L2_reg_lambda', 0.0, 'L2')

# Training params
tf.flags.DEFINE_integer('batch_size', 64, 'Batch Size')
tf.flags.DEFINE_integer('num_epochs', 200, 'Number of epochs')
tf.flags.DEFINE_integer('evaluate_every', 100, 'evaluate every')
tf.flags.DEFINE_integer('checkpoint_every', 100, 'Saving models')

FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()
print('\nParameters:')
for attr, value in sorted(FLAGS._flags.items()):
    print('{}={}').format(attr.upper(),value)

# load data
x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

max_document_length = max(len(x.split(' ')) for x in x_text)
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array((vocab_processor.fit_transform(x_text)))

np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

dev_sample_index = -1*int(FLAGS.dev_sample_percentage*float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index]
y_train, y_dev = y_shuffled[:dev_sample_index], x_shuffled[dev_sample_index]
print('max_document_length:{:d}').format(len(vocab_processor.vocabulary_))
print('train/dev_split:{:d}-{:d}').format(len(y_train),len(y_dev))
