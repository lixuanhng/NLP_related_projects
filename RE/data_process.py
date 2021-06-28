"""
script_name: re_args_help.py
******************************************************
Purpose: 做初步的数据处理，并完成数据划分
注：详细信息见：
Desktop/boohee_projects/base_KG/Information-Extraction-Chinese/RE_BGRU_2ATT/RE_data_preprocess.ipynb
******************************************************
Author: xuanhong li
******************************************************
Date: 2020-11-30
******************************************************
update:
"""


import re
import numpy as np
import os
import random
from args_help import args


def pos_embed(x, maxlen):
    """
    针对每一个句子，以每一个字相对实体的位置做 embedding
    """
    if x < -maxlen:
        return 0
    elif -maxlen <= x <= maxlen:
        return x + maxlen + 1
    elif x > maxlen:
        return 1 + maxlen + maxlen + 1


def find_index(x, y):
    """
    找到实体词在字符串中的index；找不到就返回-1
    """
    flag = -1
    for i in range(len(y)):
        if x != y[i]:
            continue
        else:
            return i
    return flag


def readWordEmbedding(path):
    """
    读取字向量文本，生成字典和字向量数组
    """
    print('reading word embedding data')
    vec = []
    word2id = {}

    # 读取中文字向量中的字和向量
    with open(path, encoding='utf-8') as f:
        content = f.readline()

        while True:
            content = f.readline()
            if content == '':
                break
            content = content.strip().split()
            dim = len(content[1:])
            word2id[content[0]] = len(word2id)
            content = content[1:]
            content = [(float)(i) for i in content]
            vec.append(content)

    word2id['UNK'] = len(word2id)
    word2id['BLANK'] = len(word2id)
    vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
    vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
    vec = np.array(vec, dtype=np.float32)

    np.save('./datasets_RE/vec.npy', vec)
    return vec, dim, word2id


def readRealationToId(path):
    """
    读取关系数据，生成关系字典
    """
    print('reading relation to id')
    relation2id = {}
    with open(path, 'r', encoding='utf-8') as f:
        content = f.readlines()
        for sin_content in content:
            if sin_content:
                pair = sin_content.strip('\n').split()
                relation2id[pair[0]] = int(pair[1])
    return relation2id


def readTrainData(path, word2id, relation2id):
    """
    读取训练数据，生成实体对-标签句子字典
    """
    fixlen = args.fix_len  # 70
    maxlen = args.max_len  # max length of position embedding is 60 (-60~+60)

    train_sen = {}  # the label is one-hot vector
    train_ans = {}

    print('reading train data')
    with open(path, 'r', encoding='utf-8') as f:
        while True:
            content = f.readline()
            if content == '':
                break
            content = content.strip().split('\t')
            en1, en2 = content[0], content[1]

            if content[2] not in relation2id:
                relation = relation2id['NA']
            else:
                relation = relation2id[content[2]]  # 获取关系

            # put the same entity pair sentences into a dict
            tup = (en1, en2)
            label_tag = 0
            y_id = relation
            label = [0 for i in range(len(relation2id))]
            label[y_id] = 1  # 这里的label实际上变成了一个one-hot向量

            if tup not in train_sen:
                train_sen[tup] = []
                train_sen[tup].append([])
                train_ans[tup] = []
                train_ans[tup].append(label)  # 用来标记一个tup的关系，label表示一个one-hot向量
            else:
                temp = find_index(label, train_ans[tup])
                if temp == -1:
                    # label列表没有出现在已知label列表集合中，说明这是同实体对的新label列表，
                    # 需要在句子中新加入一个空列表，label列表集合中新加入一个新label列表
                    train_ans[tup].append(label)
                    train_sen[tup].append([])
                    # label_tag：同一实体对下对label列表进行标号
                    label_tag = len(train_ans[tup]) - 1
                else:
                    label_tag = temp
            sentence = content[3]

            en1pos = sentence.find(en1)
            if en1pos == -1:
                en1pos = 0
            en2pos = sentence.find(en2)
            if en2pos == -1:
                en2pos = 0

            output = []
            # Embeding the position
            for i in range(fixlen):
                word = word2id['BLANK']
                rel_e1 = pos_embed(i - en1pos, maxlen)
                rel_e2 = pos_embed(i - en2pos, maxlen)
                output.append([word, rel_e1, rel_e2])

            for i in range(min(fixlen, len(sentence))):
                if sentence[i] not in word2id:
                    word = word2id['UNK']
                else:
                    word = word2id[sentence[i]]
                output[i][0] = word

            train_sen[tup][label_tag].append(output)
    return train_sen, train_ans


def readTestData(path, word2id, relation2id):
    """
    读取测试数据，生成实体对-标签句子字典
    """
    fixlen = args.fix_len
    maxlen = args.max_len  # max length of position embedding is 60 (-60~+60)
    print('reading test data')

    test_sen = {}
    test_ans = {}

    with open(path, 'r', encoding='utf-8') as f:
        while True:
            content = f.readline()
            if content == '':
                break

            content = content.strip().split('\t')
            en1 = content[0]
            en2 = content[1]
            if content[2] not in relation2id:
                print(content[2])
                relation = relation2id['NA']
            else:
                relation = relation2id[content[2]]
            relation2id_dict
            tup = (en1, en2)
            y_id = relation

            if tup not in test_sen:
                test_sen[tup] = []
                label_tag = 0
                label = [0 for i in range(len(relation2id))]
                label[y_id] = 1
                test_ans[tup] = label
            else:
                test_ans[tup][y_id] = 1

            sentence = content[3]

            en1pos = sentence.find(en1)
            if en1pos == -1:
                en1pos = 0
            en2pos = sentence.find(en2)
            if en2pos == -1:
                en2pos = 0

            output = []

            for i in range(fixlen):
                word = word2id['BLANK']
                rel_e1 = pos_embed(i - en1pos, maxlen)
                rel_e2 = pos_embed(i - en2pos, maxlen)
                output.append([word, rel_e1, rel_e2])

            for i in range(min(fixlen, len(sentence))):
                word = 0
                if sentence[i] not in word2id:
                    word = word2id['UNK']
                else:
                    word = word2id[sentence[i]]
                output[i][0] = word

            test_sen[tup].append(output)
    return test_sen, test_ans


def generateTrainData(train_sen, train_ans):
    """
    生成训练数据
    :param train_sen:
    :param train_ans:
    :return:
    """
    print('organizing train data')
    train_x = []
    train_y = []
    with open('./datasets_RE/train_q&a.txt', 'w', encoding='utf-8') as f:
        temp = 0
        for tup in train_sen:
            if len(train_ans[tup]) != len(train_sen[tup]):
                print('ERROR')
            lenth = len(train_ans[tup])
            for j in range(lenth):
                train_x.append(train_sen[tup][j])
                train_y.append(train_ans[tup][j])
                # 在文件中写入【序号，实体1，实体2，标签位置】
                f.write(str(temp) + '\t' + tup[0] + '\t' + tup[1] + '\t' + str(np.argmax(train_ans[tup][j])) + '\n')
                temp += 1
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    np.save('./datasets_RE/train_x.npy', train_x)
    np.save('./datasets_RE/train_y.npy', train_y)
    return train_x, train_y


def generateTestData(test_sen, test_ans):
    """
    生成测试数据
    :param test_sen:
    :param test_ans:
    :return:
    """
    print('organizing test data')
    test_x = []
    test_y = []
    with open('./datasets_RE/test_q&a.txt', 'w', encoding='utf-8') as f:
        temp = 0
        for tup in test_sen:
            test_x.append(test_sen[tup])
            test_y.append(test_ans[tup])
            tempstr = ''
            for j in range(len(test_ans[tup])):
                if test_ans[tup][j] != 0:
                    tempstr = tempstr + str(j) + '\t'
            f.write(str(temp) + '\t' + tup[0] + '\t' + tup[1] + '\t' + tempstr + '\n')
            temp += 1
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    np.save('./datasets_RE/testall_x.npy', test_x)
    np.save('./datasets_RE/testall_y.npy', test_y)
    return test_x, test_y


def seperate():
    """
    将字id，实体距离1，实体距离2的数据分离
    """
    print('reading training data')
    x_train = np.load('./datasets_RE/train_x.npy', allow_pickle=True)

    train_word = []
    train_pos1 = []
    train_pos2 = []

    print('seprating train data')
    for i in range(len(x_train)):
        word = []
        pos1 = []
        pos2 = []
        for j in x_train[i]:
            temp_word = []
            temp_pos1 = []
            temp_pos2 = []
            for k in j:  # 获取字id，实体距离1，实体距离2
                temp_word.append(k[0])
                temp_pos1.append(k[1])
                temp_pos2.append(k[2])
            word.append(temp_word)
            pos1.append(temp_pos1)
            pos2.append(temp_pos2)
        train_word.append(word)
        train_pos1.append(pos1)
        train_pos2.append(pos2)

    # 将字id，实体距离1，实体距离2分开
    train_word = np.array(train_word)
    train_pos1 = np.array(train_pos1)
    train_pos2 = np.array(train_pos2)
    np.save('./datasets_RE/train_word.npy', train_word)
    np.save('./datasets_RE/train_pos1.npy', train_pos1)
    np.save('./datasets_RE/train_pos2.npy', train_pos2)

    print('seperating test all data')
    x_test = np.load('./datasets_RE/testall_x.npy', allow_pickle=True)
    test_word = []
    test_pos1 = []
    test_pos2 = []

    for i in range(len(x_test)):
        word = []
        pos1 = []
        pos2 = []
        for j in x_test[i]:
            temp_word = []
            temp_pos1 = []
            temp_pos2 = []
            for k in j:
                temp_word.append(k[0])
                temp_pos1.append(k[1])
                temp_pos2.append(k[2])
            word.append(temp_word)
            pos1.append(temp_pos1)
            pos2.append(temp_pos2)
        test_word.append(word)
        test_pos1.append(pos1)
        test_pos2.append(pos2)

    test_word = np.array(test_word)
    test_pos1 = np.array(test_pos1)
    test_pos2 = np.array(test_pos2)

    np.save('./datasets_RE/testall_word.npy', test_word)
    np.save('./datasets_RE/testall_pos1.npy', test_pos1)
    np.save('./datasets_RE/testall_pos2.npy', test_pos2)


def get_metadata():
    """
    保存vec中的字
    """
    with open('./datasets_RE/metadata.tsv', 'w', encoding='utf-8') as fwrite:
        with open('./origin_data/vec.txt', encoding='utf-8') as f:
            f.readline()
            while True:
                content = f.readline().strip()
                if content == '':
                    break
                name = content.split()[0]
                fwrite.write(name + '\n')


if __name__ == '__main__':
    data_path1 = './raw_boohee_RE_dataset_updated.txt'
    with open(data_path1, 'r', encoding='utf-8') as data_file1:
        raw_data_update = data_file1.readlines()

    # 富含类数据
    fullof_data = [sin for sin in raw_data_update if '\t富含\t' in sin]
    random.shuffle(fullof_data)
    fullof_train = random.sample(fullof_data, 630)
    fullof_rest = list(set(fullof_data) ^ set(fullof_train))
    fullof_test = random.sample(fullof_rest, 92)

    # 适合类数据
    suit_data = [sin for sin in raw_data_update if '\t适合\t' in sin]
    random.shuffle(suit_data)
    suit_train = random.sample(suit_data, 600)
    suit_rest = list(set(suit_data) ^ set(suit_train))
    suit_test = random.sample(suit_rest, 34)

    # 较少类数据
    lack_data = [sin for sin in raw_data_update if '\t较少\t' in sin]
    random.shuffle(lack_data)
    lack_train = random.sample(lack_data, 120)
    lack_rest = list(set(lack_data) ^ set(lack_train))
    lack_test = random.sample(lack_rest, 19)

    # 适量类数据
    few_data = [sin for sin in raw_data_update if '\t适量\t' in sin]
    random.shuffle(few_data)
    few_train = random.sample(few_data, 650)
    few_rest = list(set(few_data) ^ set(few_train))
    few_test = random.sample(few_rest, 55)

    train_data = fullof_train + suit_train + lack_train + few_train  # 获取训练数据集
    test_data = fullof_test + suit_test + lack_test + few_test  # 获取测试数据集

    with open('./datasets_RE/train_data.txt', 'w', encoding='utf-8') as train_file:
        for line in train_data:
            train_file.write(line)

    with open('./datasets_RE/test_data.txt', 'w', encoding='utf-8') as test_file:
        for line in test_data:
            test_file.write(line)

    # 读取字向量
    vec_path = './origin_data/vec.txt'
    vector, dimen, word2id_dict = readWordEmbedding(vec_path)

    # 读取关系2id数据
    re_path = './datasets_RE/relation2id.txt'
    relation2id_dict = readRealationToId(re_path)  # 长度为5

    # 读取训练数据
    train_path = './datasets_RE/train_data.txt'
    train_sentences, train_labels = readTrainData(train_path, word2id_dict, relation2id_dict)

    # 读取测试数据
    test_path = './datasets_RE/test_data.txt'
    test_sentences, test_labels = readTestData(test_path, word2id_dict, relation2id_dict)

    # 生成训练数据
    train_data, train_label = generateTrainData(train_sentences, train_labels)

    # 生成测试数据
    test_data, test_label = generateTestData(test_sentences, test_labels)

    # 将字id，实体距离1，实体距离2的数据分离
    seperate()

    # 保存vec.txt中的字
    get_metadata()