"""
script_name: re_args_help.py
******************************************************
Purpose: 构建超参脚本
注：详细信息见：
Desktop/boohee_projects/base_KG/Information-Extraction-Chinese/RE_BGRU_2ATT/RE_args_help.ipynb
******************************************************
Author: xuanhong li
******************************************************
Date: 2020-11-30
******************************************************
update:
"""

import argparse

parser = argparse.ArgumentParser(description='train')
parser.add_argument("--vec_path", type=str, default="./datasets_RE/vec.npy", help="vec file")  # 字向量数据位置
parser.add_argument("--train_y_path", type=str, default="./datasets_RE/train_y.npy", help="train label file")  # 训练数据标签位置
parser.add_argument("--train_word_path", type=str, default="./datasets_RE/train_word.npy", help="train word file")  # 训练数据标签位置
parser.add_argument("--train_pos1_path", type=str, default="./datasets_RE/train_pos1.npy", help="train pos1 file")  # 训练数据实体距离1位置
parser.add_argument("--train_pos2_path", type=str, default="./datasets_RE/train_pos2.npy", help="train pos2 file")  # 训练数据实体距离2位置
parser.add_argument("--test_y_path", type=str, default="./datasets_RE/testall_y.txt", help="test label file")  # 测试数据标签位置
parser.add_argument("--test_word_path", type=str, default="./datasets_RE/test_word.npy", help="test word file")  # 测试数据标签位置
parser.add_argument("--test_pos1_path", type=str, default="./datasets_RE/test_pos1.npy", help="test pos1 file")  # 测试数据实体距离1位置
parser.add_argument("--test_pos2_path", type=str, default="./datasets_RE/test_pos2.npy", help="test pos2 file")  # 测试数据实体距离2位置
parser.add_argument("--checkpoints_path", type=str, default="./checkpoints_food_RE/",help="checkpoints path")  # 模型保存位置（待写入）
parser.add_argument("--vocab_size", type=int, default=16117, help="vocab_size")
parser.add_argument("--embedding_size", type=int, default=100, help="embedding_size")
parser.add_argument("--epoch", type=int, default=8, help="epoch")  # epoch大小
parser.add_argument("--fix_len", type=int, default=70, help="fixlen")   # 最大文本长度，实体距离60 + 2(两个实体) *  5(位置向量维度)
parser.add_argument("--max_len", type=int, default=60, help="max len")  # 位置向量的最大范围，(-60~+60)
parser.add_argument("--num_steps", type=int, default=70, help="num steps")
parser.add_argument("--num_classes", type=int, default=5, help="num classes")  # 类别个数
parser.add_argument("--gru_units", type=int, default=230, help="gru units")
parser.add_argument("--keep_prob", type=float, default=0.5, help="keep prob")
parser.add_argument("--num_layers", type=int, default=1, help="num layers")  # 网络层，暂时不用
parser.add_argument("--pos_size", type=int, default=5, help="pos size")  # 位置向量维度，类似 embedding_size
parser.add_argument("--pos_num", type=int, default=123, help="pos num")  # 位置种类，类似 vocab_size
parser.add_argument("--batch_size", type=int, default=50, help="batch_size")
parser.add_argument("--lr", type=float, default=0.01,help="lr")
re_args = parser.parse_args()