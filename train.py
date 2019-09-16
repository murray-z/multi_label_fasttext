# -*- coding: utf-8 -*-

import os
import tensorflow as tf
from fasttext import FastText
from config import config
import data_helper
import datetime


def train():
    X_train, y_train, all_words = data_helper.preprocess_data('./mini_data/train.txt')
    word_to_idx, idx_to_word = data_helper.generator_vocab(X_train, './mini_data')
    X_train_digit = data_helper.padding(X_train, word_to_idx)

    with tf.Graph().as_default():
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        with tf.Session(config=sess_config) as sess:
            fasttext = FastText(seq_length=config["seq_lenght"],
                                num_class=config["num_class"],
                                vocab_size=config["vocab_size"],
                                embedding_size=config["embedding_size"])

            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=config["learning_rate"])
            train_op = optimizer.minimize(fasttext.loss, global_step=global_step)

            loss_summary = tf.summary.scalar('loss', fasttext.loss)
            acc_summary = tf.summary.scalar('precision', fasttext.predictions)

            time_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            out_dir = os.path.join("runs", time_stamp)

            # train summary
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, 'summary', 'train')
            train_summary_write = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # dev summary
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, 'summary', 'dev')
            dev_summary_write = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # checkpoint
            checkpoint_dir = os.path.join(out_dir, 'model')

            if not os.path.exists(checkpoint_dir):
                os.mkdir(checkpoint_dir)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=config["max_to_keep"])

            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                feed_dict = {
                    fasttext.input_x: x_batch,
                    fasttext.input_y: y_batch
                }

                _, step, summaries, loss = sess.run([train_op, global_step, train_summary_op, fasttext.loss],
                                                    feed_dict=feed_dict)
                train_summary_write.add_summary(summaries, global_step=step)

                print("train_step: {}, loss: {}".format(step, loss))

            def dev_step(x_batch, y_batch, write=None):
                feed_dict = {
                    fasttext.input_x: x_batch,
                    fasttext.input_y: y_batch,
                    fasttext.dropout_keep_prob: 1.0
                }

                step, summaries, loss = sess.run([global_step, dev_summary_op, fasttext.loss],
                                                 feed_dict=feed_dict)

                print("dev_step: {}, loss: {}".format(step, loss))

                if write:
                    write.write(summaries, step)

            # generate batches
            batches = data_helper.generate_batchs(X_train_digit, y_train)
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % config["evaluate_every"] == 0:
                    dev_step(x_dev, y_dev, write=dev_summary_write)

                if current_step % config["checkpoint_every"] == 0:
                    path = saver.save(sess, checkpoint_dir, global_step=current_step)
                    print('save model checkpoint to {}'.format(path))

            # test
            feed_dict = {
                fasttext.input_x: x_test,
                fasttext.input_y: y_test,
                fasttext.dropout_keep_prob: 1.0
            }
            test_precision, test_recall = sess.run([fasttext.precision, fasttext.recall], feed_dict=feed_dict)
            print('test_precision: {}, test_recall: {}'.format(test_precision, test_recall))








