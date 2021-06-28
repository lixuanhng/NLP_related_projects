"""
script_name: re_model.py
******************************************************
Purpose: 继承keras.Model，并继承keras.Layer自定义attention层
注：详细信息见：
Desktop/boohee_projects/base_KG/Information-Extraction-Chinese/RE_BGRU_2ATT/model.ipynb
******************************************************
Author: xuanhong li
******************************************************
Date: 2020-12-4
******************************************************
update:
"""

import tensorflow as tf
import numpy as np
from re_args_help import re_args


def arrayToTensor(text_batch):
    """将包含list的数据转化为adarray数据"""
    text_list = []
    for i in range(len(text_batch)):
        text_list.append(text_batch[i][0])
    text_array = np.array(text_list)
    return text_array


class wordAttention(tf.keras.layers.Layer):
    """
    word-level attention layer
    """
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.attention_w = self.add_weight(name='attention_w', shape=[input_shape[-1], self.units],
                                           initializer=tf.random_normal_initializer())

    def call(self, inputs):
        output_after_att = tf.matmul(inputs, self.attention_w)
        return output_after_att


class sentenceAttention(tf.keras.layers.Layer):
    """sentence-level attention layer
    """

    def __init__(self, units, num_classes, batch_size, gru_units):
        super().__init__()
        self.units = units
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.gru_units = gru_units

    def build(self, input_shape):
        self.sen_a = self.add_weight(name='sen_a', shape=[input_shape[-1]],
                                     initializer=tf.random_normal_initializer())
        self.sen_r = self.add_weight(name='sen_r', shape=[input_shape[-1], self.units],
                                     initializer=tf.random_normal_initializer())
        self.sen_d = self.add_weight(name='sen_d', shape=[self.num_classes],
                                     initializer=tf.random_normal_initializer())
        self.sen_s = self.add_weight(name='sen_s',
                                     shape=[self.num_classes, input_shape[-1]],
                                     initializer=tf.random_normal_initializer())

    def call(self, inputs):  # inputs.shape = (batch_size, gru_units)
        """
        句子attention的处理步骤全部在这里进行，涉及到tensor的分开和合并
        首先对tensor进行拆分，从一个batch中每次取出一条向量，对这个向量求解attention的alpha参数，
        最后将对应的alpha参数与每条向量依次相乘，得到的值再通过softmax计算，则
        """
        alpha_e_col = []
        probs = []
        for j in range(inputs.shape[0]):
            att_repre_r = tf.tanh(inputs[j:j + 1])  # att_repre_r.shape=(1, gru_units)
            att_alpha_e = tf.matmul(tf.math.multiply(att_repre_r, self.sen_a), self.sen_r)  # att_alpha_e.shape=(1,1)
            alpha_e_col.append(tf.reshape(att_alpha_e, [1]))
        alpha_e_col_tensor = tf.reshape(tf.stack(alpha_e_col, axis=0), [self.batch_size])  # tensor.shape=(batch_size)
        att_alpha = tf.reshape(tf.nn.softmax(alpha_e_col_tensor), [self.batch_size, 1])  # att_alpha.shape=(batch_size,1)
        for k in range(inputs.shape[0]):
            att_repre_r_ = tf.reshape(tf.tanh(inputs[k:k + 1]), [self.gru_units, 1])  # att_repre_r.shape=(1, gru_units)
            sin_att_alpha = tf.reshape(att_alpha[k:k + 1], [1])
            att_s_c = tf.math.multiply(sin_att_alpha, att_repre_r_)  # att_s_c.shape=(gru_units, 1)
            att_out = tf.add(tf.reshape(tf.matmul(self.sen_s, att_s_c), [self.num_classes]), self.sen_d)  # att_out.shape=(num_classes)
            prob = tf.nn.softmax(att_out)
            probs.append(prob)
        return probs


class REModel(tf.keras.Model):
    def __init__(self, batch_size, vocab_size, embedding_size, num_classes, pos_num, pos_size, gru_units, embedding_matrix):
        """
        模型初始化，embedding_matrix.shape=(100, 1, 70)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.pos_num = pos_num
        self.pos_size = pos_size
        self.gru_units = gru_units
        self.batch_size = batch_size
        self.max_len = re_args.fix_len  # 文本长度，这里的数值是对应与处理过程中的fix_len的，等于70

        self.pos_embedding = tf.keras.layers.Embedding(pos_num, pos_size)
        self.word_embedding = tf.keras.layers.Embedding(vocab_size, embedding_size,
                                                        embeddings_initializer=tf.constant_initializer(value=embedding_matrix))
        self.BiGRU = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru_units, return_sequences=True),
                                                   merge_mode='sum')
        self.dropout = tf.keras.layers.Dropout(re_args.keep_prob)
        self.word_attention_layer = wordAttention(units=1)
        self.sen_attention_layer = sentenceAttention(units=1, num_classes=self.num_classes,
                                                     batch_size=self.batch_size,
                                                     gru_units=self.gru_units)
        self.full_con = tf.keras.layers.Dense(self.num_classes)

    def call(self, inputs_list, training=None):
        """
        the shape of input_word, input_pos1 and input_pos2 is（50，70）
        """
        input_word = inputs_list[0]
        input_pos1 = inputs_list[1]
        input_pos2 = inputs_list[2]

        word_embedding = self.word_embedding(input_word)  # shape=(50, 70, 100)
        pos1_embedding = self.pos_embedding(input_pos1)  # shape=(50, 70, 5)
        pos2_embedding = self.pos_embedding(input_pos2)  # shape=(50, 70, 5)

        # reshape order but no shape changed
        word_embedding_re = self.word_embedding(tf.reverse(input_word, [1]))
        pos1_embedding_re = self.pos_embedding(tf.reverse(input_pos1, [1]))
        pos2_embedding_re = self.pos_embedding(tf.reverse(input_pos2, [1]))

        # shape of inputs_forward and inputs_backward is (50, 70, 110)
        inputs_forward = tf.concat(axis=2, values=[word_embedding, pos1_embedding, pos2_embedding])
        inputs_backward = tf.concat(axis=2, values=[word_embedding_re, pos1_embedding_re, pos2_embedding_re])

        # shape changed to (50, 70, 230)
        inputs_forward = self.BiGRU(inputs_forward)
        inputs_backward = self.BiGRU(inputs_backward)

        inputs_forward = self.dropout(inputs_forward, training)
        inputs_backward = self.dropout(inputs_backward, training)

        inputs_h_concat = tf.add(inputs_forward, inputs_backward)
        # shape reshaped to (50 * 70, 230)
        inputs_h = tf.reshape(tf.tanh(inputs_h_concat), [self.batch_size * self.max_len, self.gru_units])

        # 通过字attention层得到word_output，shape=(3500, 1)
        word_inputs_h = self.word_attention_layer(inputs_h)
        w_attention_r = tf.reshape(tf.matmul(tf.reshape(tf.nn.softmax(tf.reshape(word_inputs_h, [self.batch_size, self.max_len])),
                                                        [self.batch_size, 1, self.max_len]), inputs_h_concat),
                                   [self.batch_size, self.gru_units])  # shape=(50, 230)，长度为50
        probs = self.sen_attention_layer(w_attention_r)  # add to sentence attention layer
        return probs