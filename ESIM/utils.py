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
    # path = os.path.join(os.path.dirname(__file__), args.char_vocab_file)
    vocab = [line.strip() for line in open(args.char_vocab_file, encoding='utf-8').readlines()]
    char2idx = {char: index for index, char in enumerate(vocab)}
    idx2char = {index: char for index, char in enumerate(vocab)}
    return char2idx, idx2char


def load_word_vocab():
    # path = os.path.join(os.path.dirname(__file__), args.word_vocab_file)
    vocab = [line.strip() for line in open(args.word_vocab_file, encoding='utf-8').readlines()]
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
    将【字】转化为index
    :param p_sentences: primise 句子列表（按【字】分开）
    :param h_sentences: hypothesis 句子列表（按【字】分开）
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
    p_list = pad_sequences(p_list, maxlen=args.max_char_len)
    h_list = pad_sequences(h_list, maxlen=args.max_char_len)

    return p_list, h_list


def word_index(p_sentences, h_sentences):
    """
    将【词】转化为index，具体的操作步骤与字转化为index一样
    :param p_sentences: primise 句子列表（按【词】分开）
    :param h_sentences: hypothesis 句子列表（按【词】分开）
    :return:
    """
    word2idx, idx2word = load_word_vocab()

    p_list, h_list = [], []
    for p_sentence, h_sentence in zip(p_sentences, h_sentences):
        p = [word2idx[word.lower()] for word in p_sentence if len(word.strip()) > 0 and word.lower() in word2idx.keys()]
        h = [word2idx[word.lower()] for word in h_sentence if len(word.strip()) > 0 and word.lower() in word2idx.keys()]

        p_list.append(p)
        h_list.append(h)

    p_list = pad_sequences(p_list, maxlen=args.max_word_len)
    h_list = pad_sequences(h_list, maxlen=args.max_word_len)

    return p_list, h_list


def w2v(word, model):
    """
    静态w2v，如果这个词存在就返回其编码结果，不存在就返回其全0的向量
    :param word:
    :param model:
    :return:
    """
    try:
        return model.wv[word]
    except:
        return np.zeros(args.word_embedding_size)


def w2v_process(vec):
    if len(vec) > args.max_word_len:
        # 如果一个句子当中的词长度大于最大词长度，那么对于超出的部分进行截断
        vec = vec[0:args.max_word_len]
    elif len(vec) < args.max_word_len:
        # 如果小于，就针对每个词生成一个全0的向量，对缺失的位置进行补全
        zero = np.zeros(args.word_embedding_size)
        length = args.max_word_len - len(vec)
        for i in range(length):
            vec = np.vstack((vec, zero))
    return vec


def load_all_data(path, data_size=None):
    # 载入训练好的词向量
    model = Word2Vec.load(args.w2v_model_path)
    # 输入数据的格式为csv，sentence1,sentence2,label
    # 参考代码中的 vocab.txt 及所有文本数据都应该利用起来
    df = pd.read_csv(path)

    # 分别从原始数据中获取 data_size 组数组
    p = df['sentence1'].values[0:data_size]  # premise
    h = df['sentence2'].values[0:data_size]  # hypothesis
    label = df['label'].values[0:data_size]  # label

    # 打乱顺序
    p, h, label = shuffle(p, h, label)

    # 序列中【字】转为索引，并将序列转化为等长的索引序列，也就是padding
    p_c_index, h_c_index = char_index(p, h)

    # 首先替换 primise 和 hypothesis 中的特殊符号，然后对句子进行分词，然后转化为二维数组
    p_seg = list(map(lambda x: list(jieba.cut(re.sub("[！，。？、~@#￥%&*（）.,:：|/`()_;+；…\\\\\\-\\s]", "", x))), p))
    h_seg = list(map(lambda x: list(jieba.cut(re.sub("[！，。？、~@#￥%&*（）.,:：|/`()_;+；…\\\\\\-\\s]", "", x))), h))

    # 序列【分词结果】转为索引，并将序列转化为等长的索引序列，也就是padding
    p_w_index, h_w_index = word_index(p_seg, h_seg)

    # 重新生成map文件
    p_seg = map(lambda x: list(jieba.cut(x)), p)
    h_seg = map(lambda x: list(jieba.cut(x)), h)

    # 通过静态词向量将词转化为词向量，其中每个元素为每个句子中的每个词的词向量，也就是嵌套列表
    p_w_vec = list(map(lambda x: w2v(x, model), p_seg))
    h_w_vec = list(map(lambda x: w2v(x, model), h_seg))

    # 在上述得到词向量基础上，根据最大词长度对向量化的句子进行补全或截断
    p_w_vec = list(map(lambda x: w2v_process(x), p_w_vec))
    h_w_vec = list(map(lambda x: w2v_process(x), h_w_vec))

    # 判断是否有相同的词
    same_word = []
    for p_i, h_i in zip(p_w_index, h_w_index):
        dic = {}  # 统计每个词索引的频率
        for i in p_i:
            if i == 0:
                break
            dic[i] = dic.get(i, 0) + 1
        for index, i in enumerate(h_i):
            if i == 0:
                # 索引等于0的位置，要加入相同词列表中
                same_word.append(0)
                break
            # 将词索引字典中对应索引的频率-1
            dic[i] = dic.get(i, 0) - 1
            if dic[i] == 0:
                # 如果这个值就是0，那就直接添加0
                same_word.append(1)
                break
            if index == len(h_i) - 1:
                # 遍历完毕后，再次添加0
                same_word.append(0)
    # 这里传回的是p和h的索引文本
    return p_w_index, h_w_index, label
    # return p_c_index, h_c_index, p_w_index, h_w_index, p_w_vec, h_w_vec, same_word, label


class evaluationMetrics:
    """
    This function focuses on Precision, Recall and F1 score calculation
    """

    def __init__(self, prediction_list, true_list):
        self.array_pred = prediction_list  # prediction multi-label list
        self.array_true = true_list  # true multi-label ilst

    def F1_valuation(self):
        '''
        Precision, Recall and F1 score calculation
        :param self:
        :return: Precision, Recall, F1
        '''
        Precision = 0  # define Precision
        Recall = 0  # define Recall
        F1 = 0  # calculate F1 score by Macro Average rule
        n = 0  # n is the number of the elements(list) in true and prediction multi-label list
        num_vector = len(self.array_true)  # the length of array
        while n < num_vector:
            FP = 0  # define the count of scenario that judge is WORNG in POSITIVE example
            TP = 0  # define the count of scenario that judge is TRUE in POSITIVE example
            FN = 0  # define the count of scenario that judge is WORNG in NEGATIVE example
            if self.array_pred[n] == self.array_true[n]:
                # If they are exactly same, then P，R，F1 + 1
                Precision += 1
                Recall += 1
                F1 += 1
            else:
                # if not, compare each elements.
                m = 0  # m is the number of the elements in both binary vector
                num_ele = len(self.array_true[n])  # the length of binary vector
                while m < num_ele:
                    if self.array_pred[n][m] == 1:
                        if self.array_true[n][m] == 1:
                            TP += 1
                        else:
                            FP += 1
                    else:
                        if self.array_true[n][m] == 1:
                            FN += 1
                    m += 1

                # After comparison between 2 binary vectors, use TP, FN, FP calculate P, R and F1
                # Pay attention when denominator equals 0
                if TP == 0 & (FP == 0 | FN == 0):
                    Precision_step = 0
                    Recall_step = 0
                    F1_step = 0
                else:
                    Precision_step = TP / (TP + FP)
                    Recall_step = TP / (TP + FN)
                    F1_step = 2 * Precision_step * Recall_step / (Precision_step + Recall_step)

                Precision += Precision_step
                Recall += Recall_step
                F1 += F1_step
            # Then move to the next list
            n += 1

        # calculate the average of Precision, Recall and F1 score calculation separately
        Precision = Precision / num_vector
        Recall = Recall / num_vector
        F1 = F1 / num_vector
        return Precision, Recall, F1


if __name__ == '__main__':
    load_all_data(args.train_path, data_size=100)