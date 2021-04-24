"""
Name: predict.py
Purpose: predict a triplet and check out how likely the provided triplet exits in our graph
Data:   2021.4.20
Author: lixh

模型预测部分，给输入的三元组进行打分
1、为了测试，定义1个正样本对应于30个负样本
2、
"""

from args_help import args
import pickle as pkl
import os
from model import RGCNNetwork
import tensorflow as tf
import scipy.sparse as sp
import numpy as np
from collections import defaultdict
import pandas as pd
import random
from utils import *


# hyper-parameters
VALIDATION = args['validation']
LR = args['learnrate']
L2 = args['l2norm']
EMB_DIM = args['hidden']
BASES = args['bases']
DROPOUT = args['dropout']
EDGE_NUMS = args['edge_nums']
CHECKPOINT_PTAH = args['model_save_path']


print('开始导入邻接矩阵完毕')
with open('./datasets/food' + '.pickle', 'rb') as f:
    data = pkl.load(f)

A = data['A']
num_nodes = A[0].shape[0]
support = len(A)

X = sp.csr_matrix(A[0].shape)

# 邻接矩阵做归一化
for i in range(len(A)):
    d = np.array(A[i].sum(axis=1)).flatten()
    d[np.where(d==0)] = 10
    d_inv = 1. / d
    d_inv[np.where(d_inv==0.1)] = 0.
    D_inv = sp.diags(d_inv)
    A[i] = D_inv.dot(A[i]).tocsr()

input_data = [X] + A


print('开始导入测试数据！')
# 获取正样本，关系-实体-词频字典，正样本数量
pos_tripelts, rel_ent_freq, pos_rdf_num, entity_dict, relation_dict = mainProcess()

"""
生成负样本，需要指定一个正样本对应的负样本个数，这里定为30个
随机选取10个正样本的数据，对应的负样本选择300
"""
part_pos_num = 10
related_neg_num = 30
neg_triplets, pos_spl_idxes = negativeSampling(pos_tripelts, part_pos_num,
                                               rel_ent_freq, pos_rdf_num, related_neg_num)

# 正负样本进行融合
inputs_rdf = combinePosNeg(pos_tripelts, neg_triplets, pos_spl_idxes,
                           part_pos_num, related_neg_num)
inputs_triplets = inputs_rdf.values  # 原始数据的数组格式，便于后面直接连接预测结果，用于展示结果

# 获取输入数据的id数组格式
inputs_ids = EntRelToID(inputs_rdf, entity_dict, relation_dict)
print('转换后的维度为：')
print(inputs_ids.shape)
print('其中，正样本的个数为{}，对应的负样本的个数为{}'.format(part_pos_num, part_pos_num*related_neg_num))

# 获取输入数据于标签，其中【inputs_rdfs】为要输入的三元组数据
inputs_rdfs, inputs_labels = tripletLabelSplit(inputs_ids)
inputs_data = tf.data.Dataset.from_tensor_slices((inputs_rdfs, inputs_labels))
# 将同一组数据分batch
inputs_data = inputs_data.batch(related_neg_num+1, drop_remainder=True)


print("模型实例化完毕!")
model = RGCNNetwork(support, True, BASES, EDGE_NUMS, related_neg_num+1, EMB_DIM, DROPOUT)
ckpt = tf.train.Checkpoint(model=model)
ckpt.restore(tf.train.latest_checkpoint(CHECKPOINT_PTAH))

print('模型预测结果如下：')
i = 0
for _, (rdf_batch, labels_batch) in enumerate(inputs_data):
    pred = model([input_data, rdf_batch])
    scores = tf.math.sigmoid(pred)  # 预测结果，将模型预测结果转化为(0,1)区间
    refer = inputs_triplets[i:i+related_neg_num+1]  # 展示原始三元组
    res_arr = scores.numpy()

    # 将数据进行整合，然后对最后的结果进行降序排列
    results = []
    for j in range(len(refer)):
        results.append((refer[j][0], refer[j][1], refer[j][2], refer[j][4], res_arr[j]))  # (原始三元组， score)
        results.sort(key=lambda x:x[4], reverse=True)
    for k in range(len(results)):
        if 1 in results[k]:
            rank = k
        print('triplets: {}-{}-{} | label: {} ｜ score: {:.5f}'.format(results[k][0], results[k][1], results[k][2], results[k][3], results[k][4]))
    print('正样本在 {} 个结果排序中，排在第 {} 位'.format(related_neg_num+1, rank+1))
    i += related_neg_num+1
    print('\n')

