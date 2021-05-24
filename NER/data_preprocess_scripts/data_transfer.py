#!/usr/bin/env python3
# coding: utf-8
"""
Purpose: 本代码根据已知实体位置及类型的辅助文本，对原始文本打标签，承接 entity_position_locating.py
File: data_transfer.py
Author: Bruce Li
Date: 20-10-12
Update: 1. 实体种类：label_dict
        2. 标注名称：cate_dict
        3. 待标注的数据路径：origin_path
        4. 标注好数据后保存的路径：train_filepath
"""


import os
from collections import Counter


class TransferData:
    def __init__(self):
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.label_dict = {'食物': 'FOOD',
                           '营养元素': 'NUTRIENT',
                           '疾病': 'DISEASE',
                           '人群': 'CROWD',
                           '食物类型': 'FOOD_CATE',
                           '原生食物': 'RAW_FOOD',
                           '症状': 'SYMPTOM'}
        self.cate_dict = {'O': 0,
                          'FOOD-I': 1, 'FOOD-B': 2,
                          'NUTRIENT-I': 3, 'NUTRIENT-B': 4,
                          'DISEASE-I': 5, 'DISEASE-B': 6,
                          'CROWD-I': 7, 'CROWD-B': 8,
                          'FOOD_CATE-I': 9, 'FOOD_CATE-B': 10,
                          'RAW_FOOD-I': 11, 'RAW_FOOD-B': 12,
                          'SYMPTOM-I': 13, 'SYMPTOM-B': 14}
        # 注意：在下面两个语句中需要修改train和test的路径
        self.origin_path = os.path.join(cur, 'labeled_data_0202/test')  # 待标注的文本数据所在的目录
        self.train_filepath = os.path.join(cur, 'dataset_0202/test.txt')  # 标注后的文本路径
        return

    def transfer(self):
        f = open(self.train_filepath, 'w+', encoding='utf-8')
        # count = 0
        for root, dirs, files in os.walk(self.origin_path):
            for file in files:
                filepath = os.path.join(root, file)  # 原始文本
                if 'original' not in filepath:
                    continue
                label_filepath = filepath.replace('.txtoriginal', '')  # 标注文本
                print(filepath, '\t\t', label_filepath)
                content = open(filepath, 'r', encoding='utf-8').read().strip()
                res_dict = {}
                # 给每一个文本中的每一个字符打标签
                for line in open(label_filepath, 'r', encoding='utf-8'):
                    res = line.strip().split('	')
                    start = int(res[1])
                    end = int(res[2])
                    label = res[3]
                    label_id = self.label_dict.get(label)
                    for i in range(start, end+1):
                        if i == start:
                            label_cate = label_id + '-B'
                        else:
                            label_cate = label_id + '-I'
                        res_dict[i] = label_cate

                # 获取每一个字符和其BIO标签，写入文件
                for indx, char in enumerate(content):
                    char_label = res_dict.get(indx, 'O')  # 键值不存在时，返回'O'
                    f.write(char + '\t' + char_label + '\n')
                f.write('end' + '\n')
        f.close()
        return


if __name__ == '__main__':
    handler = TransferData()
    train_datas = handler.transfer()