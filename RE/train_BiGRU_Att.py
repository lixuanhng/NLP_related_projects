"""
script_name: train_BiGRU_Att.py
******************************************************
Purpose: 训练模型
注：详细信息见：
Desktop/boohee_projects/base_KG/Information-Extraction-Chinese/RE_BGRU_2ATT/model.ipynb
******************************************************
Author: xuanhong li
******************************************************
Date: 2020-12-8
******************************************************
update: 2020-12-10 使用自己数据集
"""


import tensorflow as tf
import numpy as np
from args_help import args
from re_model import arrayToTensor, REModel


save_path = args.checkpoints_path
wordembedding = np.load('./datasets_RE/vec.npy')

train_y = np.load('./datasets_RE/train_y.npy')
class_idx = []
for m in range(len(train_y)):
    idx = np.argmax(train_y[m], 0)
    class_idx.append(idx)
train_y_array = np.array(class_idx)

train_word = np.load('./datasets_RE/train_word.npy', allow_pickle=True)
train_pos1 = np.load('./datasets_RE/train_pos1.npy', allow_pickle=True)
train_pos2 = np.load('./datasets_RE/train_pos2.npy', allow_pickle=True)
# 将array数据转化为tensor，以下三者的shape=(967, 70)
train_word_a = arrayToTensor(train_word)
train_pos1_a = arrayToTensor(train_pos1)
train_pos2_a = arrayToTensor(train_pos2)

train_dataset = tf.data.Dataset.from_tensor_slices((train_word_a, train_pos1_a, train_pos2_a, train_y_array))
train_dataset = train_dataset.shuffle(len(train_word)).batch(args.batch_size, drop_remainder=True)

model = REModel(args.batch_size, vocab_size=args.vocab_size, embedding_size=args.embedding_size,
                num_classes=args.num_classes, pos_num=args.pos_num, pos_size=args.pos_size,
                gru_units=args.gru_units, embedding_matrix=wordembedding)
optimizer = tf.keras.optimizers.Adam(args.lr)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
ckpt.restore(tf.train.latest_checkpoint(save_path))
ckpt_manager = tf.train.CheckpointManager(ckpt, save_path, checkpoint_name='model.ckpt', max_to_keep=3)

accuracy = 0
for epoch in range(args.epoch):
    for _, (word_batch, pos1_batch, pos2_batch, label_batch) in enumerate(train_dataset):
        train_inputs = [word_batch, pos1_batch, pos2_batch]
        with tf.GradientTape() as tape:
            class_probs = model(train_inputs, training=True)
            logits = tf.stack(class_probs, axis=0)  # 所有预测结果的tensor堆叠
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=label_batch, y_pred=logits)
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

        train_loss.update_state(loss)
        train_metric.update_state(label_batch, logits)

    train_accuracy_v, train_loss_v = train_metric.result(), train_loss.result()
    if epoch % 1 == 0:
        print('第' + str(epoch) + '轮训练结果为：')
        print("train accuracy: %f" % train_accuracy_v)
        print("train loss: %f" % train_loss_v)
        print('-' * 50)
        if train_accuracy_v > accuracy:
            accuracy = train_accuracy_v
            ckpt_manager.save()

    train_loss.reset_states()
    train_metric.reset_states()