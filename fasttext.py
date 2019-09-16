# -*- coding: utf-8 -*-


import tensorflow as tf


class FastText():
    def __init__(self, seq_length, num_class, vocab_size, embedding_size):
        self.seq_length = seq_length
        self.num_class = num_class
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        # placeholder
        self.input_x = tf.placeholder(tf.int32, [None, self.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.num_class], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # embedding layer
        with tf.name_scope('embedding'):
            self.W = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1, 1),
                                 name='W')

            self.embedding_chars = tf.nn.embedding_lookup(self.W, self.input_x)

            # 取均值作为文本表达
            self.embedded_chars_mean = tf.reduce_mean(self.embedding_chars, axis=1)

        with tf.name_scope("output"):
            W = tf.get_variable("W", shape=[self.embedding_size, self.num_class],
                                initializer=tf.contrib.layers.xavier_initializer())

            b = tf.Variable(tf.constant(0.1, shape=[self.num_class]), name='b')

            self.logits = tf.nn.xw_plus_b(self.embedded_chars_mean, W, b, name='logits')

            self.scores = tf.sigmoid(self.logits, name='scores')

            self.predictions = tf.round(self.scores, name='predictions')

        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y,
                                                             logits=self.logits)

            self.loss = tf.reduce_mean(tf.reduce_sum(losses, axis=1), name='sigmoid_loss')

        with tf.name_scope('performance'):
            self.precision = tf.metrics.precision(self.input_y, self.predictions, name='precision-micro')[1]
            self.recall = tf.metrics.recall(self.input_y, self.predictions, name='recall-micro')[1]






