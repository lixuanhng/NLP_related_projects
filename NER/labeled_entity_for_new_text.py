"""
本代码尝试使用训练好的实体识别模型完成陌生文本的实体识别
调用实体识别的模型，依次传入每个句子，然后输出得到其实体，并按照一定关键词规律确定其关系名称，并按照一定格式输出
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from model import NerModel
from utils import tokenize,read_vocab,format_result,build_embedding_matrix
import tensorflow_addons as tf_ad
from args_help import args
import json
import numpy as np

# 载入词表和标签字典，词向量矩阵
vocab2id, id2vocab = read_vocab(args.vocab_file)
tag2id, id2tag = read_vocab(args.tag_file)
embedded_matrix = build_embedding_matrix(args.pretrain_embedding_vec, vocab2id)
# print(id2tag)

# 实体类型映射字典
entity_mapping = {'NUTRIENT': '营养素','CROWD': '人群',
                  'FOOD_CATE': '食物类别','DISEASE': '疾病','FOOD': '食物'}

# 载入模型
optimizer = tf.keras.optimizers.Adam(args.lr)
model = NerModel(hidden_num = args.hidden_num, vocab_size = len(vocab2id), label_size = len(tag2id),
                 embedding_size = args.embedding_size, embedding_matrix=embedded_matrix)
ckpt = tf.train.Checkpoint(optimizer=optimizer,model=model)
ckpt.restore(tf.train.latest_checkpoint(args.output_dir))

# 用例1：备孕妇女膳食指南在一般人群膳食指南基础上特别补充以下3条内容：调整孕前体质量至适宜水平；常吃含铁丰富的食物，选用碘盐，孕前3个月开始补充叶酸；禁烟酒，保持健康生活方式。
# 用例2：芋头低脂肪，而且升糖指数也比红薯、土豆要低，适合糖尿病人群，可以替代部分的精制米面，增加每天粗粮的摄入量。

### 这里加入句法分析和角色标注的方式
### 每次从很多文本中选取一个文本，对这个文本进行句法分析和角色标注，查看修饰这些实体的词是哪些词
text = "提到橙⼦⼤家的第⼀反应就是补充VC，确实，橙⼦中的维⽣素C有33mg，⽐柠檬多了（22mg/100g）三分之⼀，也远⾼于苹果（3mg/100g)、梨之类的（5mg/100g）。"  # 输入句子
print('待分析的句子为：' + '\n' + text)

dataset = tf.keras.preprocessing.sequence.pad_sequences([[vocab2id.get(char,0) for char in text]], maxlen=275, padding='post')
logits, text_lens = model.predict(dataset)
paths = []
for logit, text_len in zip(logits, text_lens):
    viterbi_path, _ = tf_ad.text.viterbi_decode(logit[:text_len], model.transition_params)
    paths.append(viterbi_path)
tags = [id2tag[id] for id in paths[0]]  # 预测标签值
print(tags)

# 输出每个识别到的实体，及其实体类型
flag = False
pred_result = dict()
entity = ''
for i in range(len(tags)):
    if tags[i] != 'O':
        if not flag:
            flag = True
            cate_name = entity_mapping[tags[i].split('-')[0]]
        entity += text[i]
    elif flag and tags[i] == 'O':
        pred_result.update({entity: cate_name})
        entity = ''
        flag = False
        cate_name = ''

print('预测结果如下：')
print(pred_result)