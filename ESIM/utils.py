"""
主要面向数据处理的过程
本代码将data_utils与load_utils进行融合
"""

import pandas as pd
import os
import args
import jieba
import re
from gensim.models import Word2Vec
import numpy as np


def load_all_data(path, data_size=None):
    # 载入训练好的词向量
    model = Word2Vec.load(args.word2vec_model)
    # 输入数据的格式为csv，sentence1,sentence2,label
    # 参考代码中的 vocab.txt 比较好，应该利用起来

    # 分别从原始数据中获取 data_size 组数组
    p = df['sentence1'].values[0:data_size]  # premise
    h = df['sentence2'].values[0:data_size]  # hypothesis
    label = df['label'].values[0:data_size]  # label





if __name__ == '__main__':
    load_all_data('../input/train.csv', data_size=100)