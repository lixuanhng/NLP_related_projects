"""
script_name: test_BiGRU_Att.py
******************************************************
Purpose: 对模型进行测试
注：详细信息见：
Desktop/boohee_projects/base_KG/Information-Extraction-Chinese/RE_BGRU_2ATT/model.ipynb
******************************************************
Author: xuanhong li
******************************************************
Date: 2020-12-9
******************************************************
update: 2020-12-10 使用自己数据集
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from re_args_help import args
from re_model import arrayToTensor, REModel


# 这里设定测试集上的batch_num
batch_num = 50

save_path = args.checkpoints_path
wordembedding = np.load('./datasets_RE/vec.npy')

test_y = np.load('./datasets_RE/testall_y.npy')  # shape=(100, 12)
class_idx_test = []
for m in range(len(test_y)):
    idx = np.argmax(test_y[m], 0)
    class_idx_test.append(idx)
test_y_array = np.array(class_idx_test)  # shape=(100, 0)

test_word = np.load('./datasets_RE/testall_word.npy', allow_pickle=True)
test_pos1 = np.load('./datasets_RE/testall_pos1.npy', allow_pickle=True)
test_pos2 = np.load('./datasets_RE/testall_pos2.npy', allow_pickle=True)
# 将array数据转化为tensor，以下三者的shape=(967, 70)
test_word_a = arrayToTensor(test_word)
test_pos1_a = arrayToTensor(test_pos1)
test_pos2_a = arrayToTensor(test_pos2)

test_dataset = tf.data.Dataset.from_tensor_slices((test_word_a, test_pos1_a, test_pos2_a, test_y_array))
test_dataset = test_dataset.shuffle(len(test_word_a)).batch(batch_num, drop_remainder=True)

model = REModel(batch_size=batch_num, vocab_size=args.vocab_size, embedding_size=args.embedding_size,
                num_classes=args.num_classes, pos_num=args.pos_num, pos_size=args.pos_size,
                gru_units=args.gru_units, embedding_matrix=wordembedding)
optimizer = tf.keras.optimizers.Adam(args.lr)

ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
ckpt.restore(tf.train.latest_checkpoint(save_path)).expect_partial()

test_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

total = 0  # 每个batch中准确率的值
count = 0  # 统计batch个数
for _, (t_word_batch, t_pos1_batch, t_pos2_batch, t_label_batch) in enumerate(test_dataset):
    test_inputs = [t_word_batch, t_pos1_batch, t_pos2_batch]  # 对输入进行组合
    predictions = model(test_inputs)
    pred_results = tf.stack(predictions, axis=0)
    test_metric.update_state(t_label_batch, pred_results)
    total += test_metric.result().numpy()
    count += 1

    print("第{:d}批结果（先输出预测值，后输出真实值）：".format(count))
    print(tf.math.argmax(pred_results, axis=1))
    print(t_label_batch)
    print("准确率为{:.3f}".format(test_metric.result().numpy()))
    print('-' * 50)

print("测试集上总的准确率为：{:.3f}".format(total / count))
