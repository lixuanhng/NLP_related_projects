# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2020/10/27
# @Author: Bruce

import argparse

parser = argparse.ArgumentParser(description="train")
parser.add_argument("--w2v_model_path", type=str, default="./word2vec_model/word2vec.model",help="word2vec model")
parser.add_argument("--train_path", type=str, default="./inputs/train.csv",help="train file")
parser.add_argument("--dev_path", type=str, default="./inputs/dev.csv",help="train file")
parser.add_argument("--char_vocab_file", type=str, default="./inputs/vocab.txt",help="char vocab file")
parser.add_argument("--word_vocab_file", type=str, default="./inputs/word_vocab.tsv",help="word vocab file")
parser.add_argument("--max_char_len", type=str, default=15, help="max char length in a sentence")
parser.add_argument("--max_word_len", type=str, default=8, help="max word length in a sentence")
parser.add_argument("--word_embedding_size", type=str, default=100, help="word embedding size")
parser.add_argument("--char_embedding_size", type=str, default=100, help="char embedding size")
parser.add_argument("--embedding_hidden_size", type=str, default=256, help="embedding hidden size")
parser.add_argument("--context_hidden_size", type=str, default=256, help="context hidden size")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--keep_prob", type=float, default=0.5, help="dropout rate")
parser.add_argument("--units_num", type=int, default=512, help="units num")  # 全连接层神经元个数
parser.add_argument("--lr", type=float, default=0.0005,help="lr")
parser.add_argument("--model_path", type=str, default="./checkpoints/", help="model saved path")
parser.add_argument("--epoch", type=int, default=5,help="epoch")
parser.add_argument("--word_vocab_size", type=int, default=7230, help="vocab_size")



parser.add_argument("--test_path", type=str, default="./dataset_0202/test.txt",help="test file")  # 测试数据位置
parser.add_argument("--tag_file", type=str, default="./dataset_0202/tags.txt",help="tag_file")  #  标签位置（待写入）
parser.add_argument("--pretrain_embedding_vec", type=str, default='./pretrain_vec/token_vec_300.bin',help="pretrain_embedding_vec")  # 词向量文件
parser.add_argument("--require_improvement", type=int, default=100, help="require_improvement")
args = parser.parse_args()