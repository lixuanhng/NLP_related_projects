"""
本代码用来构建 ESIM 模型
"""


import tensorflow as tf
from args import args
import tensorflow.keras.layers as LS
import numpy as np
from utils import load_word_vocab, w2v
from gensim.models.word2vec import Word2Vec

# tensorflow2.0 要求keras中的每一层数据类型为float64
tf.keras.backend.set_floatx('float32')


def arrayToTensor(text_batch):
    """将包含list的数据转化为adarray数据"""
    text_list = []
    for i in range(len(text_batch)):
        text_list.append(text_batch[i][0])
    text_array = np.array(text_list)
    return text_array


class ESIM(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.keep_prob = args.keep_prob
        self.embedding_hidden_size = args.embedding_hidden_size
        self.context_hidden_size = args.context_hidden_size
        self.char_embedding_size = args.char_embedding_size
        self.word_vocab_size = args.word_vocab_size
        self.units_num = args.units_num

        self.W2V = Word2Vec.load(args.w2v_model_path)  # 获取训练好的word2vec模型
        self.word2idx, self.idx2word = load_word_vocab()  # 获取词表

        self.dropout = LS.Dropout(args.keep_prob)
        self.ful_con_layer_1 = LS.Dense(self.units_num, activation='tanh')
        self.ful_con_layer_2 = LS.Dense(2, activation='tanh')

        # 将已经训练好的词向量作为词向量的初始值
        self.embedding_matrix = self.W2V.wv.vectors
        self.W_emb = tf.Variable(initial_value=self.embedding_matrix,
                                 trainable=True,
                                 validate_shape=True,
                                 name='Word_Embedding',
                                 dtype=tf.float32)
        self.word_embedding = tf.keras.layers.Embedding(args.word_vocab_size,
                                                        args.word_embedding_size,
                                                        embeddings_initializer=tf.constant_initializer(value=self.embedding_matrix))


    def BiLSTM(self, x, hidden_size):
        forward_layer = LS.LSTM(hidden_size, return_sequences=True)
        backward_layer = LS.LSTM(hidden_size, activation='relu', return_sequences=True,
                                 go_backwards=True)
        bilstm = LS.Bidirectional(forward_layer, merge_mode='concat',
                                  backward_layer=backward_layer)
        hidden_state = bilstm(x)
        return hidden_state


    def call(self, inputs):
        # 输入数据由 premise 和 hypothesis 的索引形式组成
        # 分别获取 premise 和 hypothesis 向量
        p_index, h_index = inputs  # p_index.shape = (batch_size, max_len)
        # print(p_index)

        # print(self.W_emb)
        prem_emb = tf.nn.embedding_lookup(self.W_emb, p_index)
        hypo_emb = tf.nn.embedding_lookup(self.W_emb, h_index)
        # print('perm_emb.shape-->{}'.format(prem_emb.shape))
        # print(prem_emb)

        """
        # 根据传入的词的index，根据生成好的word2vec模型，将索引转化为词向量
        p_words = [list(map(lambda x: self.idx2word[x], sen)) for sen in p_index.numpy()]
        prem_emb = np.array([list(map(lambda x: w2v(x, self.W2V), sen)) for sen in p_words])
        # print('perm_emb.shape-->{}'.format(prem_emb.shape))
        # perm_emb.shape = (batch_size, max_len, word_embedding)

        h_words = [list(map(lambda x: self.idx2word[x], sen)) for sen in h_index.numpy()]
        hypo_emb = np.array([list(map(lambda x: w2v(x, self.W2V), sen)) for sen in h_words])
        """

        # 将输入向量中位置等于0的地方表示出来，这部分需要mask掉，防止全0的情况影响预测结果
        # p_mask, h_mask 应该在 p 和 h 转为向量之前就把他们的mask矩阵做出来，而不是现在
        p_mask = tf.cast(tf.math.equal(p_index, 0), dtype=tf.double)
        h_mask = tf.cast(tf.math.equal(h_index, 0), dtype=tf.double)


        """1. first BiLSTM"""
        # 维度应该是 [batch_size, max_len, embedding_hidden_size]
        p = self.BiLSTM(prem_emb, self.embedding_hidden_size)
        h = self.BiLSTM(hypo_emb, self.embedding_hidden_size)
        # print('p.shape-->{}'.format(p.shape))
        # p.shape = (batch_size, max_len, 2 * hidden_size)

        self.dropout(p)
        self.dropout(h)

        """2. Local Inference Modeling"""
        e = tf.matmul(p, tf.transpose(h, perm=[0, 2, 1]))
        # e.shape = [batch_size, max_len, max_len]
        # calculate weights from a and b

        """
        a_weights = tf.nn.softmax(e)
        b_weights = tf.transpose(tf.nn.softmax(tf.transpose(e, perm=[0, 2, 1])), perm=[0, 2, 1])
        """
        # tf.tile 表示对tensor进行平铺，输出tensor的第i维的结果是 tensor,dims(i) * multiples[i] 个元素
        h_masked = tf.cast(tf.tile(tf.expand_dims(h_mask*(-2**32 + 1),1), [1, tf.shape(e)[1],1]), dtype=tf.float32)
        p_masked = tf.cast(tf.tile(tf.expand_dims(p_mask*(-2**32 + 1),1), [1, tf.shape(tf.transpose(e, perm=[0, 2, 1]))[1], 1]), dtype=tf.float32)
        a_weights = tf.nn.softmax(e + h_masked)  # batch_size seq_len seq_len
        b_weights = tf.nn.softmax(tf.transpose(e, perm=[0, 2, 1]) + p_masked)#

        # 加权按位相乘，产生新向量
        a = tf.matmul(a_weights, h)
        b = tf.matmul(b_weights, p)
        # 获得交互层的输出层
        # 这4个元素的最后一维都是512，在axis拼接完成后得到2048
        m_a = tf.concat((a, p, a - p, tf.multiply(a, p)), axis=2)
        m_b = tf.concat((b, h, b - h, tf.multiply(b, h)), axis=2)
        # print('m_a.shape-->{}'.format(m_a.shape))
        # m_a.shape = [batch_size, max_len, 4 * 2 * hidden_size]

        """3. second BILSTM"""
        a = self.BiLSTM(m_a, self.context_hidden_size)
        b = self.BiLSTM(m_b, self.context_hidden_size)
        # print('a.shape-->{}'.format(a.shape))
        # 再次通过BiLSTM（维度为256），然后两个结果进行拼接，得到 a.shape = [batch_size, max_len, 2 * hidden_size]

        self.dropout(a)
        self.dropout(b)

        # getting mean pooling, reduce aixs=1
        a_avg = tf.math.reduce_mean(a, axis=1)
        b_avg = tf.math.reduce_mean(b, axis=1)
        # a_avg.shape = (batch_size, 2 * hidden_size)
        # getting max pooling, reduce aixs=1
        a_max = tf.math.reduce_max(a, axis=1)
        b_max = tf.math.reduce_max(b, axis=1)
        # a_max.shape = (batch_size, 2 * hidden_size)

        # concat at axis = 1, where the result is 2 * hidden_size * 4 !
        v = tf.concat((a_avg, a_max, b_avg, b_max), axis=1)
        # print('v.shape-->{}'.format(v.shape))
        # v.shape = (batch_size, 2 * 4 * hidden_size)

        """4. fully-connected"""
        v = self.ful_con_layer_1(v)  # v.shape = (batch_size, args.units_num)
        self.dropout(v)

        logits = self.ful_con_layer_2(v)  # logit.shape = (batch_size, 2) 因为是二分类
        # print('logits.shape-->{}'.format(logits.shape))


        # logits = tf.nn.softmax(logits)
        # logits = tf.math.argmax(logits, axis=1)  # 输出结果为 shape = (32,)
        return logits