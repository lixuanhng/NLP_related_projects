"""
Name: main_train.py
Purpose: train the link prediction model and track the loss variation
Data: 2021.3.18
Author: lixh
"""

from args_help import args
import pickle as pkl
import os
import sys
import time
from utils import *
from model import RGCNNetwork
import tensorflow as tf
import scipy.sparse as sp

# hyper-parameters
NB_EPOCH = args['epochs']
VALIDATION = args['validation']
LR = args['learnrate']
L2 = args['l2norm']
EMB_DIM = args['hidden']
BASES = args['bases']
DROPOUT = args['dropout']
BATCH_SIZE = args['batch_size']
EDGE_NUMS = args['edge_nums']

dirname = os.path.dirname(os.path.realpath(sys.argv[0]))  # 获取当前脚本所在目录


"""step1、导入邻接矩阵"""
with open('./datasets/food' + '.pickle', 'rb') as f:
    data = pkl.load(f)

A = data['A']  # (list)邻接矩阵，共有19个

num_nodes = A[0].shape[0]  # 获取第一个邻接矩阵的第一个维度，作为【顶点数量】
support = len(A)  # 邻接矩阵的个数（包含一个单位矩阵，support = 19）

X = sp.csr_matrix(A[0].shape)  # 构建输入数据 X，shape=(15851, 15851)

# Normalize adjacency matrices individually
# 对每个输入的邻接矩阵做归一化
for i in range(len(A)):
    d = np.array(A[i].sum(axis=1)).flatten()
    d[np.where(d==0)] = 10
    d_inv = 1. / d
    d_inv[np.where(d_inv==0.1)] = 0.
    D_inv = sp.diags(d_inv)
    A[i] = D_inv.dot(A[i]).tocsr()

# 传入模型的数据，shape=[19, ]
input_data = [X] + A
# A_in = [InputAdj(sparse=True) for _ in range(support)]  # 做19次InputADJ
# X_in = Input(shape=(X.shape[1],), sparse=True)

"""step2、导入样本数据"""
RDF_data_ids = pkl.load(open('./datasets/triplets.pkl', 'rb'))
# 将数据划分为训练集，验证集，测试集
num_rdf = RDF_data_ids.shape[0]  # 所有RDF的长度
cut_1 = num_rdf // 5
train_triples = RDF_data_ids[:cut_1*3]
test_triples = RDF_data_ids[cut_1*3: cut_1*4]
valid_triples = RDF_data_ids[cut_1*4:]


# 获取数据的三元组及样本标签
train_rdfs, trian_labels = tripletLabelSplit(train_triples)
train_dataset = tf.data.Dataset.from_tensor_slices((train_rdfs, trian_labels))
# 将数据分batch
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)

test_rdfs, test_labels = tripletLabelSplit(test_triples)
valid_rdfs, valid_labels = tripletLabelSplit(valid_triples)


"""step3、模型传参"""
"""如果传参时带有参数名，则会报错 TypeError: ('Keyword argument not understood:', 'output_dim')"""
model = RGCNNetwork(support, True, BASES, EDGE_NUMS, BATCH_SIZE, EMB_DIM, DROPOUT)
# model = RGCNNetwork(19, True, -1, 7, 100, 200, 0.1)  # model实例化

optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
# train_precision = tf.keras.metrics.Precision(name='train_precision')
# train_recall = tf.keras.metrics.Recall(name='train_recall')

# 实例化checkpoint
ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
# ckpt.restore(tf.train.latest_checkpoint(save_path))
ckpt_manager = tf.train.CheckpointManager(ckpt, args['model_save_path'], checkpoint_name='model.ckpt', max_to_keep=3)

# training loop
# loss_fcn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
for epoch in range(NB_EPOCH):
    for _, (rdf_batch, labels_batch) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            # logits 返回的结果是tensor，shape=(batch_size, 1)
            logits = model([input_data, rdf_batch])
            labels_batch = tf.cast(labels_batch, dtype=tf.float64)  # 将 labels_batch 转化为float型
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels_batch, logits))
            # Manually Weight Decay  手动添加参数衰变
            for weight in model.trainable_weights:
                # loss = loss + tf.nn.l2_loss(weight)
                # 这里的 weight 是一个 shape=(15851, 6) 的 tensor
                loss = loss + tf.cast(L2 * tf.nn.l2_loss(weight), dtype=tf.float64)

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        # 这里不选择 model.variables 是因为自定义模型中并没有定义 variables，而是使用add_weights定义了 weights
        # grads = tape.gradient(loss, model.variables)
        # optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

        train_loss.update_state(loss)
        # train_metric.update_state(train_true, train_pred)

    # train_acc = train_metric.result().numpy()
    train_loss_res = train_loss.result().numpy()
    print("Epoch {:04d} | Train loss {:.4f}". format(epoch, train_loss_res))
    ckpt_manager.save()
    # train_acc, train_res = acc(train_pred, train_true)  # 计算训练集上的准确率
    train_loss.reset_states()
    # train_metric.reset_states()


"""
    # 验证集上的表现
    val_pred = tf.gather(logits, idx_val)
    val_true = tf.cast(tf.math.argmax(tf.gather(y_val, idx_val), axis=1), tf.int32)
    val_loss = loss_fcn(val_true, val_pred)
    val_acc, val_res = acc(val_pred, val_true)
    print("Train Accuracy: {:.4f} | Train Loss: {:.4f} | Validation Accuracy: {:.4f} | Validation loss: {:.4f}".
          format(train_acc, loss.numpy().item(), val_acc, val_loss.numpy().item()))
print()

logits = model(input_data)
test_pred = tf.gather(logits, test_idx)
test_true = tf.cast(tf.math.argmax(tf.gather(y_test, test_idx), axis=1), tf.int32)
test_loss = loss_fcn(test_true, test_pred)
test_acc, test_res = acc(test_pred, test_true)
print("Test Accuracy: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss.numpy().item()))
print()

# print("Mean forward time: {:4f}".format(np.mean(forward_time[len(forward_time) // 4:])))
# print("Mean backward time: {:4f}".format(np.mean(backward_time[len(backward_time) // 4:])))
"""
