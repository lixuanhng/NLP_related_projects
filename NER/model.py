# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2020/10/27
# @Author: Bruce


import tensorflow as tf
import tensorflow_addons as tf_ad


class NerModel(tf.keras.Model):
    def __init__(self, hidden_num, vocab_size, label_size, embedding_size, embedding_matrix):
        """
        模型初始化
        """
        super(NerModel, self).__init__()
        # self.num_hidden = hidden_num  # 神经元个数，用于LSTM层（当前直接在lstm层中设置了神经元个数，这个暂时不用了）
        self.vocab_size = vocab_size  # vocab大小
        self.label_size = label_size  # 标签集大小

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size,
                                                   embeddings_initializer=tf.constant_initializer(value=embedding_matrix))

        # test
        # self.biLSTM_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_num, return_sequences=True))
        self.biLSTM_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))
        self.biLSTM_2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))

        # self.dense = tf.keras.layers.Dense(label_size, activation='relu')  # 这里新加入relu激活函数
        self.transition_params = tf.Variable(tf.random.uniform(shape=(label_size, label_size)))  # 11.5 状态转移矩阵，待训练参数
        self.dropout = tf.keras.layers.Dropout(0.5)
        # 新增
        self.time_distributed = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(label_size), input_shape=(64, 300))

    # @tf.function
    def call(self, text, labels=None, training=None):
        # 查看text中每个数组内有多少位不为0
        text_lens = tf.math.reduce_sum(tf.cast(tf.math.not_equal(text, 0), dtype=tf.int32), axis=-1)
        # -1 change 0
        inputs = self.embedding(text)  # embedding层

        # test
        inputs = self.biLSTM_1(inputs)
        inputs = self.dropout(inputs, training)  # 11.5 将dropout层加入到BiLSTM后面
        inputs = self.biLSTM_2(inputs)
        inputs = self.dropout(inputs, training)

        # logits = self.dense(inputs)
        logits = self.time_distributed(inputs)

        if labels is not None:
            label_sequences = tf.convert_to_tensor(labels, dtype=tf.int32)
            # tf_ad.text.crf_log_likelihood是addons中crf的损失函数，得到的loss_likelihoon就是损失值
            log_likelihood, self.transition_params = tf_ad.text.crf_log_likelihood(logits,  # A [batch_size, max_seq_len, num_tags] tensor of unary potentials to use as input to the CRF layer
                                                                                   label_sequences,  # A [batch_size, max_seq_len] matrix of tag indices for which we compute the log-likelihood
                                                                                   text_lens,  # A [batch_size] vector of true sequence lengths
                                                                                   transition_params=self.transition_params)  # transition_params: A [num_tags, num_tags] transition matrix
            return logits, text_lens, log_likelihood
        else:
            return logits, text_lens