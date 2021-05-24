# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2020/10/27
# @Author: Bruce

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from utils import tokenize, build_vocab, read_vocab, build_embedding_matrix
import tensorflow as tf
from model import NerModel
import tensorflow_addons as tf_ad
import numpy as np
from args_help import args
from my_log import logger


# 检查词表文件是否存在
if not (os.path.exists(args.vocab_file) and os.path.exists(args.tag_file)):
    logger.info("building vocab file")
    build_vocab([args.train_path], args.vocab_file, args.tag_file)
else:
    logger.info("vocab file exits!!")


# 针对对训练集
# 获取词表字典，标签字典
vocab2id, id2vocab = read_vocab(args.vocab_file)
tag2id, id2tag = read_vocab(args.tag_file)
text_sequences, label_sequences, _, _ = tokenize(args.train_path, vocab2id, tag2id)
# print(type(text_sequences))
# print(text_sequences.shape)
# print(text_sequences)
# 获取预训练的词向量
embedded_matrix = build_embedding_matrix(args.pretrain_embedding_vec, vocab2id)

# 获取训练数据
train_dataset = tf.data.Dataset.from_tensor_slices((text_sequences, label_sequences))
train_dataset = train_dataset.shuffle(len(text_sequences)).batch(args.batch_size, drop_remainder=True)  # Tensor序列中最后少于一个batch数量的不要了

# # 测试代码，用后即删
# for _, (text_batch, labels_batch) in enumerate(train_dataset):
#     print(type(text_batch))
#     print(text_batch.shape)
#     print(text_batch)
#     break


# 构建模型
logger.info("hidden_num:{}, vocab_size:{}, label_size:{}".format(args.hidden_num, len(vocab2id), len(tag2id)))
model = NerModel(hidden_num = args.hidden_num, vocab_size = len(vocab2id), label_size= len(tag2id),
                 embedding_size = args.embedding_size, embedding_matrix=embedded_matrix)
optimizer = tf.keras.optimizers.Adam(args.lr)

ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
ckpt.restore(tf.train.latest_checkpoint(args.output_dir))
ckpt_manager = tf.train.CheckpointManager(ckpt,
                                          args.output_dir,
                                          checkpoint_name='model.ckpt',
                                          max_to_keep=3)


# @tf.function
def train_one_step(text_batch, labels_batch):
    with tf.GradientTape() as tape:
        logits, text_lens, log_likelihood = model(text_batch, labels_batch, training=True)  # 此时调用model的call方法
        # log_likelihood 即表示loss结果
        loss = - tf.reduce_mean(log_likelihood)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, logits, text_lens


def get_acc_one_step(logits, text_lens, labels_batch):
    """
    每几步计算一次准确率，precision 和 recall 的值
    """
    paths = []
    accuracy = 0
    y_real = []
    for logit, text_len, labels in zip(logits, text_lens, labels_batch):
        viterbi_path, _ = tf_ad.text.viterbi_decode(logit[:text_len], model.transition_params)
        paths.append(viterbi_path)

        if type(labels) is np.ndarray:
            y_real.append(list(labels[:text_len]))
        else:
            y_real.append(list(labels.numpy()[:text_len]))

        correct_prediction = tf.equal(
            tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences([viterbi_path], padding='post'),
                                 dtype=tf.int32),
            tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences([labels[:text_len]], padding='post'),
                                 dtype=tf.int32)
        )
        accuracy = accuracy + tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # print(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))
    accuracy = accuracy / len(paths)
    pre, recall = evaluationMetrics(paths, y_real)
    return accuracy, pre, recall


def evaluationMetrics(logits,labels_batch):
    """
    （待加入模型）添加 presicion和 recall 作为训练结果的评估方式
    logits 表示预测值
    labels_batch 表示真实值
    """
    entity = []
    pre_entity=[]
    for logit, label in zip(logits, labels_batch):
        enti =[]
        pre =[]
        for i in range(len(label)):
            if id2tag[label[i]] != 'O':
                enti.append((id2tag[label[i]],i))
        for i in range(len(logit)):
            if id2tag[logit[i]] != "O":
                pre.append((id2tag[logit[i]],i))
        entity.append(enti)
        pre_entity.append(pre)
    pre_all = sum([len(i) for i in pre_entity])  # 预测的实体总数
    true_all = sum([len(i) for i in entity])  # 真实的实体总数
    correct = 0  # 预测正确的
    for pre, true in zip(pre_entity, entity):
        inter = list(set(pre) & set(true))
        correct += len(inter)
    precision = correct / (pre_all+1)
    recall = correct / (true_all + 1)
    return precision, recall


# 训练模型
best_presicion = 0
best_recall = 0
step = 0
accuracy_col = []
precision_col = []
recall_col = []
print('开始训练模型！')
for epoch in range(args.epoch):
    for _, (text_batch, labels_batch) in enumerate(train_dataset):
        step = step + 1
        loss, logits, text_lens = train_one_step(text_batch, labels_batch)
        if step % 5 == 0:
            accuracy, precision, recall = get_acc_one_step(logits, text_lens, labels_batch)
            # 将每次得到的结果记录
            accuracy_col.append(accuracy)
            precision_col.append(precision)
            recall_col.append(recall)
            logger.info('epoch %d, step %d, loss %.4f , accuracy %.4f, presicion %.4f, recall %.4f' % (epoch, step, loss, accuracy, precision, recall))
            if recall > best_recall and precision > best_presicion and epoch > 1:
                # best_presicion = precision
                best_recall = recall
                best_presicion = precision
                ckpt_manager.save()
                logger.info("model saved")

logger.info("finished")

# 展示accuracy, precision and recall
print(accuracy_col)
print(precision_col)
print(recall_col)