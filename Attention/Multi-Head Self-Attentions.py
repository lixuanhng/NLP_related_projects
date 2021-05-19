"""
practice：使用Attention完成对文本分类任务
"""


import tensorflow as tf
import warnings
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import numpy as np

warnings.filterwarnings('ignore')


class PositionEmbedding(layers.Layer):
    """
    在阅读理解中，Q就是篇章的向量序列，K和 V为问题的向量序列
    但如果，K和 V按行打乱（相当于句子中的词序打乱），那么 Attention 的结果还是一样的，说明 Attention 模型
    本质上还是一个“词袋模型”，但是对于NLP任务中，词之间的顺序很重要
    所有这里加入了 Position Embedding 位置向量，每个位置对应一个向量，并结合词向量（位置向量大小与词向量一样，两者相加而非拼接）
    在传统的 RNN和 CNN 中位置信息编码是辅助手段，但是在 Attention 中是核心成分
    """
    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size  # 必须为偶数，这里的size指的是词向量的维度
        self.mode = mode
        super(PositionEmbedding, self).__init__(**kwargs)

    def call(self, x):
        """调用此类时，并没有从外部传入 size 和 mode"""
        if (self.size == None) or (self.mode == 'sum'):
            # embedding层的第三个参数，也就是指定 size = embedding dim
            self.size = int(x.shape[-1])
        # 将【batch_size】和【序列长度】取出来了
        batch_size, seq_len = x.shape[0], x.shape[1]

        '''计算sin和cos函数中的分母'''
        # tf.math.pow(x,a) 元素级的指数运算操作，x为tensor底数，a为指数
        # np.arange(start=0, stop, step=1, dtype) 创建一个包含整数序列的1D tensor
        # 默认step=1，则产生的数组长度就等于stop的值
        position_j = 1. / tf.math.pow(10000., 2 * np.arange(self.size / 2, dtype='float32') / self.size)
        position_j = tf.expand_dims(position_j, 0)  # 扩展维度，shape=[1, self.size/2]

        '''计算sin和cos函数中的分子'''
        # tf.ones_like(x)  实例化与另一个张量相同形状的全1变量，这里只选取了x的前两个维度
        # np.cumsum(x, axis=1)  在某一指定轴，计算张量中的值的累加值，每个位置都是前面所有相同位置的和
        position_i = tf.math.cumsum(tf.ones_like(x[:, :, 0]), axis=1) - 1
        position_i = tf.expand_dims(position_i, 2)  # 扩展维度，shape=[batch_size, seq_len, 1]

        '''分子分母合起来，得到sin和cos函数中的值'''
        position_ij = tf.matmul(position_i, position_j)  # shape=[batch_size, seq_len, self.size/2]

        '''将两个向量合并，获得位置编码向量'''
        # tf.concat(tensors, axis=-1)  基于指定的轴，连接张量的列表，shape=[batch_size, seq_len, self.size]
        position_ij = tf.concat([tf.math.cos(position_ij), tf.math.sin(position_ij)], 2)
        if self.mode == 'sum':
            # 在论文中推荐使用向量相加的方式，而非向量拼接，相加后的shape=[batch_size, seq_len, self.size]
            return position_ij + x
        elif self.mode == 'concat':
            return tf.concat([position_ij, x], 2)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)


class Attention(layers.Layer):
    """
    Attention层的好处就是一步捕捉到全局的联系，因为它直接把序列两两比较
    主要是【多头自注意力模型】
    """
    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head  # 多头的个数，外部传值为8
        self.size_per_head = size_per_head  # 每个头的大小，外部传值为16，就是论文中使用的dk的值，需要开根号
        self.output_dim = nb_head * size_per_head  # 输出维度为 8*16，即所有注意力头拼接起来之后的尺寸
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        在Attention层中初始化 WQ, WK, WV 三个参数向量，并确定三个参数向量都是可训练的
        其中，WQ, WK, WV 的第一维形状是，input_shape[0][-1]，
        也就是 embedding 的最后一个维度，即 word_embedding_dim，此处设置为128
        由于【查询向量，键向量，值向量】传入的对象是同一个，
        所以三个参数矩阵的维度都是 shape=[word_embedding_dim, output_dim]
        :param input_shape:
        :return:
        """
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer=tf.random_uniform_initializer(),
                                  trainable=True,
                                  regularizer='l2')
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer=tf.random_uniform_initializer(),
                                  trainable=True,
                                  regularizer='l2')
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer=tf.random_uniform_initializer(),
                                  trainable=True,
                                  regularizer='l2')
        super(Attention, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode='mul'):
        """
        Mask的意义在于，用 0补全长度不足最大长度的向量后，在进行反向传播时，为了避免影响模型本身的效果，因此提出了在训练时将补全的位置mask掉
        这一步需要放在softmax之前
        具体内容见：https://blog.csdn.net/yeziyezi210/article/details/103864518
        mask.shape = [batch_size, seq_len] 或 [batch_size, seq_len, 1]
        :param inputs:
        :param seq_len: 是一个二维矩阵
        :param mode:
        :return:
        """
        if seq_len == None:
            return inputs
        else:
            # tf.shape(inputs)[1] 是每个 head 的维度
            # 生成的 one_hot 矩阵的维度是 (len(seq_len[:,0]), tf.shape(inputs)[1])
            # 这里的 len(seq_len[:,0]) 实际上就是文本长度 seq_len
            mask = tf.one_hot(seq_len[:, 0], tf.shape(inputs)[1])
            # 首先将每行为1的位置之后的所有位置全部置1，然后对角变换，1变0，0变1
            # 当前 mask 是一个下三角矩阵，一个角全1，另一个角全0
            mask = 1 - tf.math.cumsum(mask, 1)
            # inputs.shape 表示inputs的维度总数，还要注意减2
            # 当前mask是二维的，要根据input的维度情况，扩增mask的维度
            for _ in range(len(inputs.shape) - 2):
                mask = tf.expand_dims(mask, 2)
            if mode == 'mul':
                # 按位相乘，乘以0的元素就变为0
                return inputs * mask
            if mode == 'add':
                # 乘以0的元素就变为0，乘以1的被变为无穷大，然后元素减去这个无穷大就变成了一个负数
                return inputs - (1 - mask) * 1e12

    def call(self, x):
        """
        如果只传入Q_seq,K_seq,V_seq，那么就不做Mask
        如果同时传入Q_seq,K_seq,V_seq,Q_len,V_len，那么对多余部分做Mask
        传入的x是3个embedding
        :param x:
        :return:
        """
        if len(x) == 3:
            Q_seq, K_seq, V_seq = x  # Q_seq, K_seq, V_seq 都等于 embedding
            Q_len, V_len = None, None  # Q_len 为查询向量长度，V_len 为值向量的维度
        elif len(x) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = x

        # 对Q、K、V做线性变换
        Q_seq = tf.matmul(Q_seq, self.WQ)  # Q_seq.shape = (batch_size, seq_len, output_dim)
        Q_seq = tf.reshape(Q_seq, (-1, tf.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = tf.transpose(Q_seq, perm=[0, 2, 1, 3])  # 重新排列张量的轴
        # 重排后的 Q_seq.shape = (batch_size, self.nb_head, seq_len, self.size_per_head)
        # 输出的结果是：Q_seq.shape = (None, 8, None, 16)

        K_seq = tf.matmul(K_seq, self.WK)
        K_seq = tf.reshape(K_seq, (-1, tf.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = tf.transpose(K_seq, perm=[0, 2, 1, 3])

        V_seq = tf.matmul(V_seq, self.WV)
        V_seq = tf.reshape(V_seq, (-1, tf.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = tf.transpose(V_seq, perm=[0, 2, 1, 3])

        # 【Attention过程】计算内积，然后mask，然后softmax
        # keras.backend.batch_dot(x, y, axes)，批量化的点积，axes表示目标维度的整数或列表
        # 其功能相当于先将矩阵（针对某一位置）进行对位乘法，然后按照axes轴进行聚合加法（就是都加到一边，axes=0 表示行的维度不变，每一个列相加）
        A = tf.multiply(Q_seq, K_seq) / self.size_per_head ** 0.5  # 计算查询向量和键向量的积再对键向量维度开方
        A = tf.transpose(A, perm=[0, 3, 2, 1])  # A.shape = (batch_size, self.size_per_head, self.nb_head, seq_len)
        A = self.Mask(A, V_len, 'add')
        A = tf.transpose(A, perm=[0, 3, 2, 1])  # 重新reshape一次
        A = tf.nn.softmax(A)  # 进行softmax得到归一化分数
        # 输出的结果是：A.shape = (None, 8, None, 16)
        # 输出并mask
        O_seq = tf.multiply(A, V_seq)  # softmax的值乘以值向量
        O_seq = tf.transpose(O_seq, (0, 2, 1, 3))  # O_seq.shape = (batch_size, self.nb_head, seq_len, self.size_per_head)
        O_seq = tf.reshape(O_seq, (-1, tf.shape(O_seq)[1], self.output_dim))
        # 输出的结果是：A.shape = (None, None, 128)
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], self.output_dim


if __name__ == '__main__':
    vocab_size = 20000  # 文本中单词词频在前max_features个的词
    maxlen = 80  # 最大文本长度
    batch_size = 32

    print('Loading IMDB data...')
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)

    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

    print('x_train shape: {}'.format(x_train.shape))
    print('x_test shape: {}'.format(x_test.shape))

    S_inputs = layers.Input(shape=(None,), dtype='int32')
    embeddings = layers.Embedding(vocab_size, 128)(S_inputs)
    # embedding result.shape = [batch_size, input_length, embedding_dim]

    # 对文本embedding后结果进行位置编码
    embeddings = PositionEmbedding()(embeddings)  # shape = [None, None, 128]

    # 进行attention计算
    O_seq = Attention(8,16)([embeddings,embeddings,embeddings])

    # 全局平均池化层，将3D数据降低到2D，输入数据的维度是 (batch_size, max_len, features)
    # 平均池化的结果是：维度变为 (batch_size, features)，可以理解为在文本长度上进行池化操作
    O_seq = layers.GlobalAveragePooling1D()(O_seq)
    O_seq = layers.Dropout(0.5)(O_seq)
    outputs = layers.Dense(1, activation='sigmoid')(O_seq)  # 全连接，sigmoid完成二分类

    model = Model(inputs=S_inputs, outputs=outputs)
    print(model.summary())
    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=5,
              validation_data=(x_test, y_test))

