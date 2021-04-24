"""
Name: model.py
Purpose: build R-GCN model including encoder and decoder
Data:   2021.3.19 encoder
        2021.4.19 decoder
Author: lixh

model framework：
    first layer：output is node embedding, output shape is (node nums, embedding dimension)
    second layer: define relation embedding and calculate score of a triple, output shape is (batch_size, 1)

@Nodes
1、featureless：如果外部传入的featureless的值为False，即表示使用"feature"的值，
需要将feature依次乘以邻接矩阵；否则直接使用邻接矩阵的结果，也就是说传入的feature（input[0]）没有用到
2、定义变量时，在变量中加入了初始化
"""

import tensorflow as tf
import numpy as np
from args_help import args
tf.random.set_seed(0)


def acc(logits, labels):
    # 选出mask对应的logits和label
    labels = tf.cast(labels, dtype=tf.int32)
    indices = tf.cast(tf.math.argmax(logits, axis=1), dtype=tf.int32)
    res = tf.cast((indices == labels), dtype=tf.int32)
    res_arr = np.bincount(res.numpy())
    accuracy = res_arr[1] / res_arr.sum()
    return accuracy, res


class Encoder(tf.keras.layers.Layer):
    """
    获取node embedding
    num_bases 意义不明
    """
    def __init__(self, embedding_dim=None, support=1, featureless=False, num_bases=-1):
        super(Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.support = support  # 邻接矩阵的个数，值为【19】
        self.featureless = featureless  # 是否不使用node本身的特征，值为【True】
        self.num_bases = num_bases  # 意义不明，值为【-1】

        self.bias = False
        self.keep_prob = 0.8
        # 在build中定义如下变量
        self.input_dim = None
        self.W = None
        self.W_comp = None
        self.b = None
        self.num_nodes = None

    def build(self, input_shape):
        # input_shapep[0] 对应的是应该是输入数据的维度(num_nodes, nums_nodes)，即（15851，15851）
        features_shape = input_shape[0]
        if self.featureless:
            self.num_nodes = features_shape[1]  # 从外部输入参数中，featureless这里指定为True
        self.input_dim = features_shape[1]
        if self.num_bases > 0:
            """注意：这里没有加入 W 的正则化方法，待模型训练跑通之后再加入"""
            # 若 num_bases 大于 0，则根据 num_bases 确定添加的 weight 数量
            self.W = tf.keras.layers.concatenate([self.add_weight(shape = (self.input_dim, self.embedding_dim),
                                                                  trainable=True,
                                                                  initializer=tf.random_normal_initializer(),
                                                                  name='W', regularizer='l2'
                                                                  ) for _ in range(self.num_bases)], axis=0)
            # self.W_comp.shape = (self.support, self.num_bases)
            self.W_comp = self.add_weight((self.support, self.num_bases),
                                          initializer=tf.random_normal_initializer(),
                                          name='W_comp', regularizer='l2')
        else:
            # 根据 support 长度确定添加的 weight 数量，shape=(19*15851, 6)
            self.W = tf.keras.layers.concatenate([self.add_weight(shape = (self.input_dim, self.embedding_dim),
                                                                  trainable=True,
                                                                  initializer=tf.random_normal_initializer(),
                                                                  name='W', regularizer='l2'
                                                                  ) for _ in range(self.support)], axis=0)

        if self.bias:
            self.b = self.add_weight((self.embedding_dim,),
                                     initializer=tf.random_normal_initializer(), name='b')

    def call(self, inputs):
        # list of basis functions，这里传入的input是【features】和【邻接矩阵】
        features, A = inputs[0], inputs[1:]

        # convolution part
        supports = list()
        for i in range(self.support):
            if not self.featureless:
                supports.append(tf.matmul(A[i], features))
            else:
                # 中间隐藏层使用 featureless = True，将每个邻接矩阵取出
                supports.append(A[i])
        supports_array = [supports[i].A for i in range(len(supports))]
        # supports_.shape = (15851, 15851*19)
        supports_ = tf.keras.layers.Concatenate(axis=1, dtype='float32')(supports_array)  # 将邻接矩阵在axis=1处进行拼接

        # step 1. 邻接矩阵 * self.W
        if self.num_bases > 0:
            self.W = tf.reshape(self.W, (self.num_bases, self.input_dim, self.embedding_dim))
            self.W = tf.transpose(self.W, perm=[1, 0 ,2])
            V = tf.matmul(self.W_comp, self.W)
            V = tf.reshape(V, (self.support * self.input_dim, self.embedding_dim))
            output = tf.matmul(supports_, V)
        else:
            # 将support与self.W进行矩阵乘法, output.shape = (15851, 6)
            output = tf.matmul(supports_, self.W)

        # step 3. 加入偏置值
        if self.bias:
            output += self.b  # self.b.shape = (6, )
        # output.shape = (15851, 6)
        return output


class Decoder_backup(tf.keras.layers.Layer):
    """
    本层目的在于创建【关系参数矩阵】，作为score函数中的乘数因子
    关系对应的参数矩阵 W 为数据维度为【edge count = 9，embedding dimension】，每行为一个关系向量
    需要传入的数据为一个batch的三元组，上一层输出节点embedding
    则输入数据的维度就是【batch_size, encoder embedding】
    所有triples（包含正负样本，每个正样本对应9个负样本）分batch进行输入到模型进行训练，一个批次下训练完所有数据
    """
    def __init__(self, embedding_dim, edge_count, batch_size):
        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.edge_count = edge_count
        self.batch_size = batch_size

    def build(self, input_shape):
        embedding_dim = input_shape[0]
        self.W_r = self.add_weight(shape=(self.edge_count, self.embedding_dim),
                                   trainable=True,
                                   initializer=tf.random_normal_initializer(),
                                   name='W_r', regularizer='l2')

    def call(self, inputs):
        # batch的train_triples (shape = [batch_size, embedding dimension])
        sbj_embs, obj_embs, rel_ids = inputs[0], inputs[1], inputs[2]
        # 获取关系向量
        rel_embs = tf.nn.embedding_lookup(self.W_r, rel_ids, name='rel_id2vector')
        mul_res = sbj_embs * rel_embs * rel_embs  # 计算score
        mul_res = tf.reduce_sum(mul_res, 1)  # axis=1上进行求和，shape=(batch_size, 1)
        return mul_res


class Decoder(tf.keras.layers.Layer):
    """
    本模型使用创建对角矩阵的方式训练关系矩阵，并使用矩阵乘法【e_s，r，T(e_o)】
    """
    def __init__(self, embedding_dim, edge_count, batch_size):
        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.edge_count = edge_count
        self.batch_size = batch_size
        rel_initial_value = tf.linalg.diag(np.random.uniform(size=(self.edge_count,
                                                                   self.embedding_dim)))
        self.W_r = tf.Variable(rel_initial_value, trainable=True, name='W_r')

    def call(self, inputs):
        # batch的train_triples (shape = [batch_size, embedding dimension])
        sbj_embs, obj_embs, rel_ids = inputs[0], inputs[1], inputs[2]
        rel_embs = tf.nn.embedding_lookup(self.W_r, rel_ids, name='rel_id2vector')
        # 计算 score，这里需要将【rel_embs】的数据类型转化为 tf.float32，
        sbj_rel_embs = tf.matmul(sbj_embs, tf.cast(rel_embs, dtype=tf.float32))
        mul_res = tf.matmul(sbj_rel_embs, tf.transpose(obj_embs, perm=[0, 2, 1]))
        mul_res = tf.reduce_sum(mul_res, 1)
        return mul_res


class RGCNNetwork(tf.keras.Model):
    def __init__(self, support, featureless, num_bases, edge_count=9,
                 batch_size=100, units=200, dropout_rate=0.1):
        """
        R-GCN模型，分别连接 encoder 模型和 decoder  模型
        :param units:          节点embedding的结果，default = 200
        :param support:        邻接矩阵的个数，default = 19
        :param featureless:    是否考虑节点特征，default = False
        :param num_bases:      邻接矩阵的个数的替代，default = -1
        :param sub_edge_count: 边子集的个数，default = 9
        :param batch_size:     decoder中一次输入的RDF数量，default = 100
        :param dropout_rate:   空置神经元的比例，default = 0.1
        """
        super(RGCNNetwork, self).__init__()
        self.units = units
        self.support = support
        self.featureless = featureless
        self.num_bases = num_bases
        self.edge_count = edge_count
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate

        self.h_layer = Encoder(embedding_dim=self.units,
                               support=self.support,
                               featureless=self.featureless,
                               num_bases=self.num_bases)
        self.act_h = tf.keras.layers.ReLU()
        self.act_o = tf.keras.layers.Softmax(axis=1)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.score = Decoder(embedding_dim=self.units,
                             edge_count=self.edge_count,
                             batch_size=self.batch_size)


    def call(self, inputs):
        adjs, triples = inputs[0], inputs[1]
        h = self.h_layer(adjs)
        h = self.dropout(h, training=True)  # 这一层结束后的维度为（15851，200）为节点的embedding
        # h = self.act_h(h)
        # h = self.act_o(h)

        # 完成数组的维度互换，分别取出【sbj， obj，rel】
        triples_used = tf.transpose(triples)
        sbj_ids, obj_ids, rel_ids = triples_used[0], triples_used[1], triples_used[2]
        sbj_embs = tf.nn.embedding_lookup(h, sbj_ids, name='sbj_id2vector')
        sbj_embs = tf.expand_dims(sbj_embs, 1)  # 增加一个维度
        obj_embs = tf.nn.embedding_lookup(h, obj_ids, name='obj_id2vector')
        obj_embs = tf.expand_dims(obj_embs, 1)
        decoder_inputs = [sbj_embs, obj_embs, rel_ids]
        output = self.score(decoder_inputs)
        output = tf.cast(output, dtype=tf.float64)  # 类型转化为float
        output = tf.reshape(output, shape=[self.batch_size])
        return output

