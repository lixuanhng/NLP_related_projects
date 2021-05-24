"""
Date: 2020.10.21
Author: lixh
Purpose: 本代码旨在构建用于实体标注的文本数据
Steps:  1. 首先需要明确共有哪些实体类型
        2. 载入待处理的原始文本数据
        3. 采用分词的方式，检查 raw_context_data_for_NER 中 是否存在不包含任何实体词的句子，如果有就去掉
        4. 准备训练数据，测试数据
Updates:    2020-12-21  重新整理词表，确定各类词表位置，及生成自定义实体词典文本
            2021-02-01  确定实体类型，当前为6类实体
                        重新整理jieba自定义字典，将问答模型中的自定义词典和运营标注的词进行融合

"""

import random
import re
import jieba

jieba.load_userdict(r"./vocabularies_0202/custom_dict_0201.txt")

"""
1. 首先需要明确共有哪些实体类型

当前实体共有6类，分别是原始食材，食物，食物类别，营养元素，疾病，人群，症状
"""


def removeDuplicateAndEnglishSign(path, word_list):
    """
    本段代码用来将每类实体集合中的实体进行以下操作：
    1. 去重
    2. 英文符号转化为中文符号
    3. 将每个实体集合重新保存一次
    """
    pattern1 = r"\("
    pattern2 = r"\)"
    entities = []  # 所有实体的集合
    for w in word_list:
        w = re.sub(pattern1, '（', w)
        w = re.sub(pattern2, '）', w)
        entities.append(w)
    entities = list(set(entities))
    with open(path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(entities))
    return entities


# 载入食物实体词表
# 1、读入原始食材
with open('./vocabularies_0202/vocab_raw_foods.txt', 'r', encoding='utf-8') as f:
    raw_foods = [item.strip('\n') for item in f.readlines()]

# 2、读入营养素
with open('./vocabularies_0202/vocab_nutrients.txt', 'r', encoding='utf-8') as f:
    nutrients = [item.strip('\n') for item in f.readlines()]

# 3、读入人群
with open('./vocabularies_0202/vocab_crowds.txt', 'r', encoding='utf-8') as f:
    crowds = [item.strip('\n') for item in f.readlines()]

# 4、读入疾病
with open('./vocabularies_0202/vocab_diseases.txt', 'r', encoding='utf-8') as f:
    diseases = [item.strip('\n') for item in f.readlines()]

# 5、读入食物类别
with open('./vocabularies_0202/vocab_food_cates.txt', 'r', encoding='utf-8') as f:
    food_cates = [item.strip('\n') for item in f.readlines()]

# 6、读入食物
with open('./vocabularies_0202/vocab_foods.txt', 'r', encoding='utf-8') as f:
    foods = [item.strip('\n') for item in f.readlines()]

# 7、疾病
with open('./vocabularies_0202/vocab_symptoms.txt', 'r', encoding='utf-8') as f:
    symptoms = [item.strip('\n') for item in f.readlines()]

entities = raw_foods + nutrients + crowds + diseases + food_cates + foods + symptoms
print('词库中实体词总数为：' + str(len(entities)))


"""
2. 载入待处理的原始文本数据
"""
# 载入原始文本数据
# raw_context_data_for_NER.txt 这个文件中文本数据是经过批量处理和手动加入之后产生的
context_path = './context_data_for_NER_0201.txt'
with open(context_path, 'r', encoding='utf-8') as context_file:
    raw_context_data_for_NER = context_file.readlines()
raw_context_data_for_NER = [sentence.strip('\n') for sentence in raw_context_data_for_NER]
# 需要将这里面的每一个文本按【。；;】分开（即分句），并打乱顺序
pattern = r'。|；|;'
raw_context_sentences = []  # 记录每一个分句
for para in raw_context_data_for_NER:
    re.sub(' ', '', para)  # 将每个句子中可能存在的空格去掉
    sentences = re.split(pattern, para)
    for sen in sentences:
        if sen and sen not in raw_context_sentences:
            raw_context_sentences.append(sen+'。')
# raw_context_data_for_NER = [re.sub(' ', '', para) for para in raw_context_data_for_NER]
random.shuffle(raw_context_sentences)
print('待处理的文本有' + str(len(raw_context_sentences)) + '个')  # 每段文本不分句时，文本个数有???；分句时，文本数量有1395个


"""
3. 采用分词的方式，检查 context_data_for_NER_0201 中 是否存在不包含任何实体词的句子，如果有看一下怎么处理
这种做法还有一个好处就是，我们已经总结好的实体词将来可以作为自定义用户字典添加到jieba中，词性都是名词
"""
no_entity_sentences = []  # 统计不包含任何实体词的句子
for sentence in raw_context_sentences:
    words = jieba.lcut(sentence)
    flag = False
    for w in words:
        if w in foods:
            flag = True
            break
        elif w in nutrients:
            flag = True
            break
        elif w in diseases:
            flag = True
            break
        elif w in crowds:
            flag = True
            break
        elif w in food_cates:
            flag = True
            break
        elif w in raw_foods:
            flag = True
            break
        elif w in symptoms:
            flag = True
            break
    if not flag:
        # 收集不包含任何实体的句子
        no_entity_sentences.append(sentence)

print('不包含任何实体词的句子个数：' + str(len(no_entity_sentences)))  # 116
# 对于（1）没有实体词的句子，当前的做法是将他们从原始数据中删除
content_used_for_NER = [sen for sen in raw_context_sentences if sen not in no_entity_sentences]
print('筛选后原始文本的个数有' + str(len(content_used_for_NER)))  # 1279


"""
4. 准备训练数据，测试数据
"""
# 保存1/8的文本用来做测试数据
cut_p = int(len(content_used_for_NER) / 8)
# with open('./test_data/predict_context.txt', 'w', encoding='utf-8') as file1:
#     file1.write('\n'.join(content_used_for_NER[-cut_p:]))
content_train_used_for_NER = content_used_for_NER[:-cut_p]  # 训练数据
content_test_used_for_NER = content_used_for_NER[-cut_p:]  # 测试数据


def dataLabeled(former_path, context_list):
    """
    将整理好的文本数据分文件写入
    """
    later_path = '.txtoriginal.txt'
    count = 1
    for sentence in context_list:
        save_path = former_path + str(count) + later_path
        with open(save_path, 'w', encoding='utf-8') as save_file:
            save_file.write(sentence)
        count += 1
    return


# 保存训练数据
train_name_former = './labeled_data_0202/train/text_'
dataLabeled(train_name_former, content_train_used_for_NER)
# 保存测试数据
test_name_former = './labeled_data_0202/test/text_'
dataLabeled(test_name_former, content_test_used_for_NER)




# 查看原始文本数据的文本长度超过300的个数
# count = 0
# for sentence in content_used_for_NER:
#     if len(sentence) > 300:
#         count += 1
# print('文本数据中最大文本长度为：' + str(count / len(content_used_for_NER)))