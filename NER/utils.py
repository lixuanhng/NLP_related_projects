# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2020/10/27
# @Author: Bruce


import tensorflow as tf
import json, os
import numpy as np

def build_vocab(corpus_file_list, vocab_file, tag_file):
    # 构建词列表，标签列表（已去重）
    words = set()
    tags = set()
    for file in corpus_file_list:
        # words = words.union(set([line.strip().split()[0]  for line in open(file, "r", encoding='utf-8').readlines()]))
        # tags = tags.union(set([line.strip().split()[-1] for line in open(file, "r", encoding='utf-8').readlines()]))
        for line in open(file, "r", encoding='utf-8').readlines():
            line = line.strip()
            if line == "end":
                continue
            try:
                w, t = line.split('\t')
                words.add(w)
                tags.add(t)
            except Exception as e:
                print(line.split())
                # raise e

    # 根据 words 构建词表 vocab
    if not os.path.exists(vocab_file):
        with open(vocab_file,"w",encoding='utf-8') as f:
            for index,word in enumerate(["<UKN>"]+list(words) ):
                f.write(word+"\n")

    # 根据 tags 构建 标签字典
    tag_sort = {
        "O": 0,
        "B": 1,
        "I": 2,
        "E": 3,
    }

    tags = sorted(list(tags),
           key=lambda x: (len(x.split("-")), x.split("-")[-1], tag_sort.get(x.split("-")[0], 100))
           )
    if not os.path.exists(tag_file):
        with open(tag_file,"w",encoding='utf-8') as f:
            for index,tag in enumerate(["<UKN>"]+tags):
                f.write(tag+"\n")

# build_vocab(["./data/train.utf8","./data/test.utf8"])


def read_vocab(vocab_file):
    """
    获取词表和 id的关系
    """
    vocab2id = {}
    id2vocab = {}
    for index,line in enumerate([line.strip('\n') for line in open(vocab_file,"r",encoding='utf-8').readlines()]):
        vocab2id[line] = index
        id2vocab[index] = line
    return vocab2id, id2vocab

# print(read_vocab("./data/tags.txt"))



def tokenize(filename,vocab2id,tag2id):
    contents = []  # 遇到end，将content添加进来
    labels = []  # 遇到end，将label添加进来
    content = []  # 以end截止作为一段文本，记录每个字对应的id
    label = []  # 以end截止作为一段文本，记录每个标签对应的id
    with open(filename, 'r', encoding='utf-8') as fr:
        for line in [elem.strip('\n') for elem in fr.readlines()][:50000]:
            try:
                if line != "end":
                    w,t = line.split()
                    content.append(vocab2id.get(w,0))
                    label.append(tag2id.get(t,0))
                else:
                    if content and label:
                        contents.append(content)
                        labels.append(label)
                    content = []
                    label = []
            except Exception as e:
                content = []
                label = []

    contents_pad = tf.keras.preprocessing.sequence.pad_sequences(contents, padding='post')  # pad完成后的content就变成了一个数据集
    labels_pad = tf.keras.preprocessing.sequence.pad_sequences(labels, padding='post')
    # 返回编码后 padding 和 no-padding 的 constents, labels
    return contents_pad, labels_pad, contents, labels


tag_check = {
    "I":["B","I"],
    "E":["B","I"],
}


# 获取预训练的词向量
def load_pretrained_embedding(embedded_file):
    """
    读取token_vec_300.bin中训练好的词向量
    返回为字典格式：，key为字，value为其300维的向量（array类型，每个数字为float32），
    """
    embeddings_dict = {}
    with open(embedded_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split(' ')
            if len(values) < 300:
                continue
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_dict[word] = coefs
    print('Found %s word vectors.' % len(embeddings_dict))
    return embeddings_dict


def build_embedding_matrix(embedded_file, vocab2id):
    embedding_dict = load_pretrained_embedding(embedded_file)
    vocab_size = len(vocab2id)  # 词表大小
    embedded_matrix = np.zeros((vocab_size, 300))  # 预先生成（词表长度，词向量维度）的 embedded_matrix
    for word, i in vocab2id.items():
        embedding_vector = embedding_dict.get(word)
        if embedding_vector is not None:
            # 如果词表中的词在embedding_dict中是有出现的，则将这个词的对应的词向量添加进embedded_matrix；不出现则表示为0
            embedded_matrix[i] = embedding_vector
    return embedded_matrix


def check_label(front_label,follow_label):
    if not follow_label:
        raise Exception("follow label should not both None")

    if not front_label:
        return True

    if follow_label.startswith("B-"):
        return False

    if (follow_label.startswith("I-") or follow_label.startswith("E-")) and \
        front_label.endswith(follow_label.split("-")[1]) and \
        front_label.split("-")[0] in tag_check[follow_label.split("-")[0]]:
        return True
    return False


def format_result(chars, tags):
    entities = []
    entity = []
    for index, (char, tag) in enumerate(zip(chars, tags)):
        entity_continue = check_label(tags[index - 1] if index > 0 else None, tag)
        if not entity_continue and entity:
            entities.append(entity)
            entity = []
        entity.append([index, char, tag, entity_continue])
    if entity:
        entities.append(entity)

    entities_result = []
    for entity in entities:
        if entity[0][2].startswith("B-"):
            entities_result.append(
                {"begin": entity[0][0] + 1,
                 "end": entity[-1][0] + 1,
                 "words": "".join([char for _, char, _, _ in entity]),
                 "type": entity[0][2].split("-")[1]
                 }
            )

    return entities_result


if __name__ == "__main__":
    text = ['国','家','发','展','计','划','委','员','会','副','主','任','王','春','正']
    tags =  ['B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'E-ORG', 'O', 'O', 'O', 'B-PER', 'I-PER', 'E-PER']
    entities_result= format_result(text,tags)
    print(json.dumps(entities_result, indent=4, ensure_ascii=False))