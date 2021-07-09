"""
author: lixh
mission: text similarity
date: train
model: ESIM
"""

from utils import load_all_data, evaluationMetrics
from model import ESIM, arrayToTensor
import tensorflow as tf
from args import args
import numpy as np


# 获取已经处理好的文本数据，共有10万条文本，
# 最大文本长度为20，<list>p_train.shape = [data_size, max_len]，返回的是包含字索引的文本数据
p_train, h_trian, y_trian = load_all_data(args.train_path, data_size=3200)
p_train, h_train = np.array(p_train), np.array(h_trian)

p_eval, h_eval, y_eval = load_all_data(args.dev_path, data_size=100)
p_eval, h_eval = np.array(p_eval), np.array(h_eval)

# primise，hypothesis，label三类数据转化为tensor
# train_prem = arrayToTensor(p_train)
# train_hypo = arrayToTensor(h_trian)
# eval_prem = arrayToTensor(p_eval)
# eval_hypo = arrayToTensor(p_eval)

# 生成数据集
train_dataset = tf.data.Dataset.from_tensor_slices((p_train, h_trian, y_trian))
eval_dataset = tf.data.Dataset.from_tensor_slices((p_eval, h_eval, y_eval))

# 分成多个batch
train_dataset = train_dataset.shuffle(len(p_train)).batch(args.batch_size, drop_remainder=True)
eval_dataset = eval_dataset.shuffle(len(p_eval)).batch(args.batch_size, drop_remainder=True)

# 载入模型
model = ESIM()

# 初始化优化器
optimizer = tf.keras.optimizers.Adam(args.lr)

# 对于文本匹配的模型，使用二元交叉熵和二元准确率评价函数
train_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
# loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
train_loss = tf.keras.metrics.SparseCategoricalCrossentropy(name='train_loss')

# 初始化模型保存机制
ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
ckpt.restore(tf.train.latest_checkpoint(args.model_path))
ckpt_manager = tf.train.CheckpointManager(ckpt, args.model_path, checkpoint_name='model.ckpt', max_to_keep=3)

accuracy = 0
for epoch in range(args.epoch):
    # y_ts, y_ps = list(), list()
    for _, (train_prem, train_hypo, y_true) in enumerate(train_dataset):
        # print(train_prem.shape)
        # 分布之后的结果是 train_prem.shape = (batch_size, max_len)
        train_inputs = [train_prem, train_hypo]
        with tf.GradientTape() as tape:
            logits = model(train_inputs, training=True)
            # 新加入的结果
            logits = tf.cast(logits, dtype=tf.float64)
            y_true = tf.reshape(tf.cast(y_true, dtype=tf.float64), [args.batch_size, 1])
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
            # loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y_true, y_pred=logits, from_logits=True)
            loss = tf.math.reduce_mean(loss)  # 对一个batch上的所有loss求平均
            # for weight in model.trainable_weights:
            #     # loss = loss + tf.nn.l2_loss(weight)
            #     # 这里的 weight 是一个 shape=(15851, 6) 的 tensor
            #     loss = loss + tf.cast(0.01 * tf.nn.l2_loss(weight), dtype=tf.float64)

        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

        train_loss.update_state(y_true, logits)
        train_metric.update_state(y_true, logits)

        pred = tf.math.argmax(tf.nn.softmax(logits), axis=1)
        labels = tf.cast(tf.reshape(y_true, [args.batch_size]), dtype=tf.int64)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, labels), tf.float32))

        # 将预测值和真实值全部转化为 [0, 1] 列表，计算precision和recall
        # y_t = tf.reshape(tf.cast(y_true, dtype=tf.int64), [args.batch_size]).numpy().tolist()
        # y_p = tf.math.argmax(tf.nn.softmax(logits), axis=1).numpy().tolist()
        # y_ts.append(y_t)
        # y_ps.append(y_ps)

    # train_accuracy_v = train_metric.result()
    # train_loss_v = train_loss.result()
    print('-' * 100)
    print("epoch {} --- loss: {}, metric: {}, acc: {}".format(epoch, train_loss.result(), train_metric.result(), accuracy))
    print("labels {}".format(labels))
    print("predic {}".format(pred))
    # print('y_pred-->{}'.format(tf.math.argmax(logits, axis=1).numpy().tolist()))
    # print('y_true-->{}'.format(tf.cast(np.reshape(y_true, [args.batch_size, ]), dtype=tf.int64).numpy().tolist()))

    # train_loss.reset_states()
    # train_metric.reset_states()

    # if epoch % 1 == 0:
    #     print('第' + str(epoch) + '轮训练结果为：')
    #     print("train accuracy: %f" % train_accuracy_v)
    #
    #     print('-' * 50)
    #     if train_accuracy_v > accuracy:
    #         accuracy = train_accuracy_v
    #         ckpt_manager.save()