# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2020/10/27
# @Author: Bruce


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from model import NerModel
from utils import tokenize,read_vocab,format_result,build_embedding_matrix
import tensorflow_addons as tf_ad
from args_help import args
import json
import numpy as np


# 针对测试集完成词表字典，标签字典，文本序列长度和初始化词向量
vocab2id, id2vocab = read_vocab(args.vocab_file)
tag2id, id2tag = read_vocab(args.tag_file)
print(id2tag)
text_sequences, label_sequences, text_origin, label_origin = tokenize(args.test_path, vocab2id, tag2id)
# text_sequences 的维度是（159，110）
embedded_matrix = build_embedding_matrix(args.pretrain_embedding_vec, vocab2id)

# print('查看 text_sequences 的值和维度:')
# print(text_sequences.shape)
# print(type(text_sequences))


# 载入模型
optimizer = tf.keras.optimizers.Adam(args.lr)
model = NerModel(hidden_num = args.hidden_num, vocab_size =len(vocab2id), label_size = len(tag2id),
                 embedding_size = args.embedding_size, embedding_matrix=embedded_matrix)
# restore model
ckpt = tf.train.Checkpoint(optimizer=optimizer,model=model)
ckpt.restore(tf.train.latest_checkpoint(args.output_dir))


def evaluationMetrics(id2tag, logits_batch, labels_batch):
    """
    （待加入模型）添加 presicion和 recall 作为测试集的评估方式
    logits_batch 表示预测值（单位为batch）
    labels_batch 表示真实值（单位为batch）
    """
    entity = []  # 真实的实体
    pre_entity=[]  # 预测的实体
    for logit, label in zip(logits_batch, labels_batch):
        # 获取每一对预测值和真实值，记录标签类型和位置
        enti =[]
        pre =[]
        for i in range(len(label)):
            if id2tag[label[i]] != 'O':
                enti.append((id2tag[label[i]], i))
        for i in range(len(logit)):
            if id2tag[logit[i]] != "O":
                pre.append((id2tag[logit[i]], i))
        entity.append(enti)
        pre_entity.append(pre)
    pre_all = sum([len(i) for i in pre_entity])  # 预测是实体的总数
    true_all = sum([len(i) for i in entity])  # 真实是实体的总数
    correct = 0  # 预测正确的
    for pre, true in zip(pre_entity, entity):
        inter = list(set(pre) & set(true))
        correct += len(inter)
    print('正确预测的实体数量为：' + str(correct))
    print('预测为实体的数量为：' + str(pre_all))
    print('实体的数量为：' + str(true_all))
    precision = correct / (pre_all+1)
    recall = correct / (true_all + 1)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


"""
1. 批量测试文本
"""

logits, text_lens = model.predict(text_sequences)
paths = []
for logit, text_len in zip(logits, text_lens):
    viterbi_path, _ = tf_ad.text.viterbi_decode(logit[:text_len], model.transition_params)
    paths.append(viterbi_path)  # 返回的 viterbi_path 是标签预测值
Precision, Recall, F1 = evaluationMetrics(id2tag, paths, label_origin)
print('Precision的值为：' + str(Precision))
print('Recall的值为：' + str(Recall))
print('F1的值为：' + str(F1))

"""
2. 测试单个输入文本（不用时需要注释掉）
"""
"""
while True:
    text = input("input:")
    dataset = tf.keras.preprocessing.sequence.pad_sequences([[vocab2id.get(char,0) for char in text]], maxlen=275, padding='post')
    logits, text_lens = model.predict(dataset)  # logits和text_lens都是<class 'numpy.ndarray'>，shape分别为：(1, 39, 14)和(1, )
    paths = []
    for logit, text_len in zip(logits, text_lens):
        viterbi_path, _ = tf_ad.text.viterbi_decode(logit[:text_len], model.transition_params)
        paths.append(viterbi_path)  # paths[0]返回的 viterbi_path 是标签预测值
    # print(paths[0])
    tags = [id2tag[id] for id in paths[0]]
    # print(tags)
    # 输出每个识别到的实体，及其实体类型
    flag = False
    pred_result = dict()
    entity = ''
    for i in range(len(tags)):
        if tags[i] != 'O':
            if not flag:  # 这个条件只在第一次识别到实体头才执行
                flag = True
                cate_name = entity_mapping[tags[i].split('-')[0]]  # 获取类别名称
            entity += text[i]
        elif flag and tags[i] == 'O':
            pred_result.update({entity: cate_name})
            entity = ''
            flag = False
            cate_name = ''
    
    print('预测结果如下：')
    print(pred_result)

    # 格式化输出实体及位置（不重要）
    # entities_result = format_result(list(text), [id2tag[id] for id in paths[0]])
    # print(json.dumps(entities_result, indent=4, ensure_ascii=False))
"""