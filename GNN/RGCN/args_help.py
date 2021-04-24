"""
Name: data_preprocess.py
Purpose: mark the hyperparameters for unified management
Data: 2021.3.20
Author: lixh
"""

import argparse

ap = argparse.ArgumentParser(description="train")
ap.add_argument("--epochs", type=int, default=5, help="Number training epochs")
ap.add_argument("--Negative_samples", type=int, default=5, help="number of nagetive samples for each triple")
ap.add_argument("--hidden", type=int, default=200, help="Number hidden units")
ap.add_argument("--edge_nums", type=int, default=9, help="the total num of edge type")
ap.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
ap.add_argument("--batch_size", type=int, default=12000, help="input triple's batch size")
ap.add_argument("--bases", type=int, default=-1, help="Number of bases used (-1: all)")  # number of filter weight matrices，默认为-1
ap.add_argument("--learnrate", type=float, default=0.01, help="Learning rate")
ap.add_argument("--l2norm", type=float, default=0.01, help="L2 normalization of input weights")  # 归一化参数
ap.add_argument("--belong_rdf", type=str, default="./data/train.txt", help="train file")  # 训练数据位置
ap.add_argument("--model_save_path", type=str, default="./checkpoints_matmul_decoder/", help="model file")  # 模型保存路径
ap.add_argument("--validation", dest='validation', action='store_true')
ap.set_defaults(validation=True)
args = vars(ap.parse_args())
# print(args)