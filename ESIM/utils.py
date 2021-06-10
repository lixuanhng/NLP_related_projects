"""
主要面向数据处理的过程
本代码将data_utils与load_utils进行融合
"""

import pandas as pd
import os
from args import args
import jieba
import re
from gensim.models.word2vec import Word2Vec
import numpy as np



def shuffle(*arrs):
    """
    打乱数据
    Arguments:
        *arrs: 数组数据
    Returns:
        shuffle后的数据
    """
    arrs = list(arrs)
    for i, arr in enumerate(arrs):
        assert len(arrs[0]) == len(arrs[i])
        # 转换为numpy数组
        arrs[i] = np.array(arr)
    # 将数据打乱
    p = np.random.permutation(len(arrs[0]))
    # 三个数据进行融合
    return tuple(arr[p] for arr in arrs)


def load_char_vocab():
    """
    加载字典，生成 {word: id} 和 {id: word}
    :return:
    """
    # path = os.path.join(os.path.dirname(__file__), args.vocab_file)
    vocab = [line.strip() for line in open(args.vocab_file, encoding='utf-8').readlines()]
    word2idx = {word: index for index, word in enumerate(vocab)}
    idx2word = {index: word for index, word in enumerate(vocab)}
    return word2idx, idx2word


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='post',
                  truncating='post', value=0.):
    """ pad_sequences
    把序列长度转变为一样长的，如果设置了maxlen则长度统一为maxlen，如果没有设置则默认取
    最大的长度。填充和截取包括两种方法，post与pre，post指从尾部开始处理，pre指从头部
    开始处理，默认都是从尾部开始。

    Arguments:
        sequences: 序列
        maxlen: int 最大长度
        dtype: 转变后的数据类型
        padding: 填充方法'pre' or 'post'
        truncating: 截取方法'pre' or 'post'
        value: float 填充的值

    Returns:
        x: numpy array 填充后的序列维度为 (number_of_sequences, maxlen)
    """
    # 获取每个序列的长度
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)  # 样本长度
    if maxlen is None:
        maxlen = np.max(lengths)

    # 产生维度为 (nb_samples, maxlen) 填充矩阵，填充元素为 value
    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            # 返回【截取方法不明确】
            raise ValueError("Truncating type '%s' not understood" % padding)

        # 将得到的填充结果trunc替换为
        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            # 返回【填充方法不明确】
            raise ValueError("Padding type '%s' not understood" % padding)
    return x


def char_index(p_sentences, h_sentences):
    """
    将字转化为index
    :param p_sentences: primise 句子列表
    :param h_sentences: hypothesis 句子列表
    :return:
    """
    word2idx, idx2word = load_char_vocab()

    p_list, h_list = [], []
    # 每次取一个句子对，将句子中的所有字转化为id后，存入列表
    for p_sentence, h_sentence in zip(p_sentences, h_sentences):
        p = [word2idx[word.lower()] for word in p_sentence if len(word.strip()) > 0 and word.lower() in word2idx.keys()]
        h = [word2idx[word.lower()] for word in h_sentence if len(word.strip()) > 0 and word.lower() in word2idx.keys()]

        p_list.append(p)
        h_list.append(h)
    # 对文本长度进行统一
    p_list = pad_sequences(p_list, maxlen=args.max_len)
    h_list = pad_sequences(h_list, maxlen=args.max_len)

    return p_list, h_list


def load_all_data(path, data_size=None):
    # 载入训练好的词向量
    model = Word2Vec.load("./word2vec_model/word2vec.model")
    # 输入数据的格式为csv，sentence1,sentence2,label
    # 参考代码中的 vocab.txt 及所有文本数据都应该利用起来
    df = pd.read_csv(path)

    # 分别从原始数据中获取 data_size 组数组
    p = df['sentence1'].values[0:data_size]  # premise
    h = df['sentence2'].values[0:data_size]  # hypothesis
    label = df['label'].values[0:data_size]  # label

    # 打乱顺序
    p, h, label = shuffle(p, h, label)

    # 序列中词转为索引，并将序列转化为等长的索引序列
    p_c_index, h_c_index = char_index(p, h)

    return



if __name__ == '__main__':
    load_all_data(args.train_path, data_size=100)