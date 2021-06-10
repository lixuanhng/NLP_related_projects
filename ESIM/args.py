# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2020/10/27
# @Author: Bruce

import argparse

parser = argparse.ArgumentParser(description="train")
parser.add_argument("--w2v_model_path", type=str, default="./word2vec_model/word2vec.model",help="word2vec model")
parser.add_argument("--train_path", type=str, default="./inputs/train.csv",help="train file")
parser.add_argument("--vocab_file", type=str, default="./inputs/vocab.txt",help="vocab file")
parser.add_argument("--max_len", type=str, default=20, help="max length")


parser.add_argument("--test_path", type=str, default="./dataset_0202/test.txt",help="test file")  # 测试数据位置
parser.add_argument("--output_dir", type=str, default="./checkpoints_0202_test1/",help="output_dir")  # 模型保存位置（待写入）
parser.add_argument("--tag_file", type=str, default="./dataset_0202/tags.txt",help="tag_file")  #  标签位置（待写入）
parser.add_argument("--batch_size", type=int, default=32,help="batch_size")
parser.add_argument("--hidden_num", type=int, default=512,help="hidden_num")  # 神经网络层神经元个数（虽然传入了模型，但是目前没有使用）
parser.add_argument("--embedding_size", type=int, default=300,help="embedding_size")  # 词向量维度
parser.add_argument("--pretrain_embedding_vec", type=str, default='./pretrain_vec/token_vec_300.bin',help="pretrain_embedding_vec")  # 词向量文件
parser.add_argument("--epoch", type=int, default=8,help="epoch")
parser.add_argument("--lr", type=float, default=0.01,help="lr")
parser.add_argument("--require_improvement", type=int, default=100, help="require_improvement")
args = parser.parse_args()


import numpy as np

x = (np.ones((2, 5)) * 0.).astype('int32')
trunc = [1,1,1,0,0]
x[0, :len(trunc)] = trunc
print(x)