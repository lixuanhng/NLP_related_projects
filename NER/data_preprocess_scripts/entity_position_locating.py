#!/usr/bin/env python3
# coding: utf-8
"""
Purpose: 本短代码的目的是根据已知的每类实体列表，将原始文本中涉及到的实体标注位置和实体类型
            代码承接：data_preprocess.py
File: data_transfer.py
Author: Bruce Li
Date: 20-10-12
Update: 1. 实体列表
        2. 待处理的原始文本数据及位置
"""

import re
import os

"""
载入关键词列表
"""
# 载入食物实体词表
# 1、读入原始食材
with open('./vocabularies_0202/vocab_raw_foods.txt', 'r', encoding='utf-8') as f:
    raw_foods_l = [item.strip('\n') for item in f.readlines()]

# 2、读入营养素
with open('./vocabularies_0202/vocab_nutrients.txt', 'r', encoding='utf-8') as f:
    nutrients_l = [item.strip('\n') for item in f.readlines()]

# 3、读入人群
with open('./vocabularies_0202/vocab_crowds.txt', 'r', encoding='utf-8') as f:
    crowds_l = [item.strip('\n') for item in f.readlines()]

# 4、读入疾病
with open('./vocabularies_0202/vocab_diseases.txt', 'r', encoding='utf-8') as f:
    diseases_l = [item.strip('\n') for item in f.readlines()]

# 5、读入食物类别
with open('./vocabularies_0202/vocab_food_cates.txt', 'r', encoding='utf-8') as f:
    food_cates_l = [item.strip('\n') for item in f.readlines()]

# 6、读入食物
with open('./vocabularies_0202/vocab_foods.txt', 'r', encoding='utf-8') as f:
    foods_l = [item.strip('\n') for item in f.readlines()]

# 7、疾病
with open('./vocabularies_0202/vocab_symptoms.txt', 'r', encoding='utf-8') as f:
    symptoms_l = [item.strip('\n') for item in f.readlines()]


class PositionLocating:
    def __init__(self):
        pass

    @staticmethod
    def positionFilter(locations, position_inside):
        """
        本代码旨在将实体范围包含关系的最短范围的实体去掉，举例来说：
        慢性病	38	40	疾病
        性病	39	40	疾病
        修改原因：（1）能识别到【慢性病】（2）两个实体的范围是包含关系（38-40包含39-40），就说明【性病】的识别是多余的，需要去掉
        param locations: 标注结果
        param position_inside: 范围元组集合
        """
        delete_list = []  # 记录被包含实体
        for i in range(len(position_inside)):
            left_i, right_i = position_inside[i][0], position_inside[i][1]  # 当前研究对象的范围
            rest = position_inside[:i] + position_inside[i + 1:]  # 除对象外的剩余元组集合
            for item in rest:
                left_item, right_item = item[0], item[1]
                if set(list(range(left_i, right_i))) < set(list(range(left_item, right_item))):
                    delete_list.append((left_i, right_i))
        # 需要将含有这类实体的数据去掉
        for location in locations:
            part = location.split('	')
            if (int(part[1]), int(part[2])) in delete_list:
                locations.remove(location)
        return locations

    @staticmethod
    def locatingLabel(data_path, raw_foods, nutrients, crowds, diseases, food_cates, foods, symptoms):
        for root, dirs, files in os.walk(data_path):
            for file in files:
                filepath = os.path.join(root, file)  # 原始文本
                # 非原始文本的文件都不处理
                if 'original' not in filepath:
                    continue
                with open(filepath, 'r', encoding='utf-8') as test_file:
                    # 载入每个原始文本
                    text = test_file.readlines()[0]
                    position_inside = []  # 记录命中的实体词的位置元组
                    # 对每一个实体标注其在原始文本中位置和类别
                    locations = []
                    for food in foods:
                        if food in text:
                            idx_food = text.index(food)
                            distance1 = len(food) - 1
                            line1 = food + '	' + str(idx_food) + '	' + str(idx_food + distance1) + '	' + '食物'
                            locations.append(line1)  # 保存标签
                            position_inside.append((idx_food, idx_food + distance1))  # 保存实体位置元组
                    for cate in food_cates:
                        if cate in text:
                            idx_cate = text.index(cate)
                            distance5 = len(cate) - 1
                            line5 = cate + '	' + str(idx_cate) + '	' + str(idx_cate + distance5) + '	' + '食物类型'
                            locations.append(line5)  # 保存标签
                            position_inside.append((idx_cate, idx_cate + distance5))
                    for nutrient in nutrients:
                        if nutrient in text:
                            idx_nutrients = text.index(nutrient)
                            distance2 = len(nutrient) - 1
                            line2 = nutrient + '	' + str(idx_nutrients) + '	' + str(idx_nutrients + distance2) + '	' + '营养元素'
                            locations.append(line2)
                            position_inside.append((idx_nutrients, idx_nutrients + distance2))
                    for disease in diseases:
                        if disease in text:
                            idx_disease = text.index(disease)
                            distance3 = len(disease) - 1
                            line3 = disease + '	' + str(idx_disease) + '	' + str(idx_disease + distance3) + '	' + '疾病'
                            locations.append(line3)
                            position_inside.append((idx_disease, idx_disease + distance3))
                    for crowd in crowds:
                        if crowd in text:
                            idx_crowd = text.index(crowd)
                            distance4 = len(crowd) - 1
                            line4 = crowd + '	' + str(idx_crowd) + '	' + str(idx_crowd + distance4) + '	' + '人群'
                            locations.append(line4)
                            position_inside.append((idx_crowd, idx_crowd + distance4))
                    for raw_food in raw_foods:
                        if raw_food in text:
                            idx_raw_food = text.index(raw_food)
                            distance6 = len(raw_food) - 1
                            line6 = raw_food + '	' + str(idx_raw_food) + '	' + str(idx_raw_food + distance6) + '	' + '原生食物'
                            locations.append(line6)
                            position_inside.append((idx_raw_food, idx_raw_food + distance6))
                    for symptom in symptoms:
                        if symptom in text:
                            idx_symptom = text.index(symptom)
                            distance7 = len(symptom) - 1
                            line7 = symptom + '	' + str(idx_symptom) + '	' + str(idx_symptom + distance7) + '	' + '症状'
                            locations.append(line7)
                            position_inside.append((idx_symptom, idx_symptom + distance7))
                # 这里需要调用去掉最长公共子串的模块
                locations = PositionLocating.positionFilter(locations, position_inside)
                save_path = filepath.replace('.txtoriginal', '')
                # save_path = re.sub('.txtoriginal', '', data_path)
                with open(save_path, 'w', encoding='utf-8') as save_file:
                    for label in locations:
                        save_file.write(label + '\n')
        return


if __name__ == '__main__':
    path = './labeled_data_0202'  # 这里传入的数据路径，是全部待处理数据
    locator = PositionLocating()
    result = locator.locatingLabel(path, raw_foods_l, nutrients_l, crowds_l, diseases_l, food_cates_l, foods_l, symptoms_l)