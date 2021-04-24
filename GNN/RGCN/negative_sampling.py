"""
Name: negative_sampling.py
Purpose: generate regative samples and combine samples all togather
Data:   2021.4.19
Author: lixh

1、  选取所有边的一个子集：这里不选择有关【疾病】的RDF
2、  统计在【选定RDF】中所有实体出现的频率，按照由高到低进行排序，生成【实体频率序列】
3、  根据【实体频率序列】以及需要的正负样本比例，随机选择 sub 或 obj 进行替换，
    替换的实体是在【实体频率序列】中比被替换实体排名靠前的，并生成新的负样本RDF
4、  需要保存实体字典和关系字典
"""
import os
import pandas as pd
from collections import defaultdict
import random
import pickle as pkl
import numpy as np
from data_preprocess import rel_zh_en, rel_en_zh
from args_help import args

num_neg_sample = args['Negative_samples']


def readChosenRDF(data_path):
    """
    1、选定关系边的子集，这里不选择和疾病有关的节点数据和关系，也就是从数据集中直接删掉这部分的数据
    2、在每种关系下，统计每个实体出现的频率，和所属的类别（数字表示），由高到底排序
    :param data_path:
    :return:
    """
    ent_freq_dict = defaultdict()  # 存放每个关系边中的sub词频和obj词频
    chosen_RDFs = pd.DataFrame(columns=['source_name', 'relation', 'target_name', 'rdf_type', 'label'])
    rdf_count = 0
    for root, dirs, files in os.walk(data_path, topdown=False):
        for name in files:
            if "cure" not in name:  # 不选择与疾病相关的三元组
                cur_df = pd.read_csv(os.path.join(root, name))  # 获取每一种dataframe

                # 针对每种关系，统计每个词的出现频率
                # 针对 subject 和 object 分别生成两个新的字典
                subs, objs, subs_list, objs_list = countEntityFreq(cur_df)
                rel_cate = name.split('_rdf_')[0]
                # 举例 {'benefit': (subject list, object list)}
                # ent_freq_dict 中每个关系对应的元组中的sub_list和obj_list本身就是排好续的
                # 当需要进行负采样时，可以直接找到被替换词，在sub_list（obj_list）中的位置，
                # 然后取出这个位置之前的列表，从中随机选取10个词作为负采样词
                ent_freq_dict[rel_cate] = (subs_list, objs_list)

                # 在每个dataframe中添加关系类型
                cur_df_len = len(cur_df)
                cur_df['rdf_type'] = [rel_cate for _ in range(cur_df_len)]
                cur_df['label'] = [1 for _ in range(cur_df_len)]
                chosen_RDFs = pd.concat([chosen_RDFs, cur_df], ignore_index=True)
                rdf_count += cur_df_len
    return chosen_RDFs, ent_freq_dict, rdf_count


def countEntityFreq(cur_df):
    subs = defaultdict()
    objs = defaultdict()
    for i in range(len(cur_df)):

        # subject 统计
        if cur_df.iloc[i]['source_name'] not in subs.keys():
            subs[cur_df.iloc[i]['source_name']] = 0
        subs[cur_df.iloc[i]['source_name']] += 1

        # object 统计
        if cur_df.iloc[i]['target_name'] not in objs.keys():
            objs[cur_df.iloc[i]['target_name']] = 0
        objs[cur_df.iloc[i]['target_name']] += 1

    # 频率由高到低进行排序
    subs = sorted(subs.items(), key=lambda x: x[1], reverse=True)
    objs = sorted(objs.items(), key=lambda x: x[1], reverse=True)
    # 去掉频率值，按频率顺序展示实体的列表
    subs_list = [pair[0] for pair in subs]
    objs_list = [pair[0] for pair in objs]
    return subs, objs, subs_list, objs_list


def negativeSampling(chosen_RDFs, ent_freq_dict, rdf_count):
    """
    创建负采样的数据，每个正样本产生一个负样本，在实体出现频率排序中，负样本排在正样本的前几个
    4/16日更新，每个样本产生10个负样本
    :param chosen_RDFs: 所有正样本的dataframe数据
    :param ent_freq_dict: 实体频率字典--{rel: (sbj_list, obj_list)}
    :param rdf_count: 正样本个数
    :return:
    """
    RDFs_with_neg = pd.DataFrame(columns=['source_name', 'relation', 'target_name', 'rdf_type', 'label'])
    for j in range(rdf_count):
        if j % 2000 == 0: print("第{}个正样本数据已经产生负样本".format(j))
        replace_entity = random.choice(['subject', 'object'])
        rel_name_zh = chosen_RDFs.iloc[j]['relation']  # 取出关系名称
        raw_sbj = chosen_RDFs.iloc[j]['source_name']
        raw_obj = chosen_RDFs.iloc[j]['target_name']
        rel_name = rel_zh_en[rel_name_zh]  # 关系中文转化为英文

        if replace_entity == 'subject':
            """替换subject的情况"""
            ent_re = chosen_RDFs.iloc[j]['source_name']  # 待替换的实体
            neg_candidate_list = negativeWordLookup(ent_freq_dict[rel_name][0], ent_re)  # 获取【备选词表】
            for _ in range(num_neg_sample):
                new_ent = random.choice(neg_candidate_list)  # 替换后的词
                new_df = pd.DataFrame({'source_name': new_ent,
                                       'relation': rel_name_zh,
                                       'target_name': raw_obj,
                                       'rdf_type': chosen_RDFs.iloc[j]['rdf_type'],
                                       'label': 0}, index=[1])
                RDFs_with_neg = RDFs_with_neg.append(new_df, ignore_index=True)
        else:
            """替换object的情况"""
            ent_re = chosen_RDFs.iloc[j]['target_name']  # 待替换的实体
            neg_candidate_list = negativeWordLookup(ent_freq_dict[rel_name][1], ent_re)  # 获取【备选词表】
            for _ in range(num_neg_sample):
                new_ent = random.choice(neg_candidate_list)  # 替换后的词
                new_df = pd.DataFrame({'source_name': raw_sbj,
                                       'relation': rel_name_zh,
                                       'target_name': new_ent,
                                       'rdf_type': chosen_RDFs.iloc[j]['rdf_type'],
                                       'label': 0}, index=[1])
                RDFs_with_neg = RDFs_with_neg.append(new_df, ignore_index=True)
    return RDFs_with_neg


def negativeWordLookup(freq_type_words, ent_re):
    """
    :param freq_vocab: 某个关系下同位置的词频列表，由高到低进行排序
    :param ent_re: 实体名称
    :return: 【备选同类词表】包含比当前词词频高的词
    """
    ent_idx = freq_type_words.index(ent_re)
    # 词频排序在替换词前面的词表，这里称之为【被选同类词表】
    neg_candidate_list = freq_type_words[:ent_idx]
    # 如果【备选同类词表】为空，则指定整个【同类词表】为【备选同类词表】
    if not neg_candidate_list:
        neg_candidate_list = freq_type_words[1:]
    return neg_candidate_list


def EntRelToID(RDFs, entity_dict, relation_dict):
    """将RDF中的【实体】和【关系】转化为ID，输出为numpy数组"""
    triplets = []
    for i in RDFs.index:
        triplet = RDFs.loc[i].values.tolist()
        triplet_id = [int(entity_dict[triplet[0]]),
                     int(entity_dict[triplet[2]]),
                     int(relation_dict[triplet[1]]),  # 【关系】英文转为中文
                     int(triplet[4])]
        triplets.append(triplet_id)
    random.shuffle(triplets)  # 打乱顺序
    triplets_id = np.array(triplets)
    return triplets_id


def mainProcess():
    # 载入【实体字典】，format:{entity: id, ...}
    entity_dict = defaultdict()
    with open('./datasets/entity_index_dict.txt', 'r', encoding='utf-8') as ent_dict_file:
        ent_dict_list = ent_dict_file.readlines()
    for line in ent_dict_list:
        pair = line.strip('\n').split('\t')
        entity_dict[pair[0]] = pair[1]

    relation_dict = pkl.load(open('./datasets/rel_dict.pkl', 'rb'))
    # print("载入关系字典")
    # print(relation_dict)

    data_path = "./data"
    # 正样本数据
    chosen_RDFs, ent_freq_dict, rdf_count = readChosenRDF(data_path)
    chosen_RDFs.to_csv('./datasets/positive_triplets.csv', header=True, index=False)
    # print('所有RDF正样本数据的个数为{}：'.format(str(rdf_count)))

    """产生一个实体对应着的所有的其他实体"""
    # save ent_freq_dict as relation-sbj-obj-frequency-dict
    pkl.dump(ent_freq_dict, open('./datasets/ent_freq_dict.pkl', 'wb'))

    # 负样本（对应于正样本）
    neg_RDFs = negativeSampling(chosen_RDFs, ent_freq_dict, rdf_count)
    # 正负样本合并
    RDF_data = pd.concat([chosen_RDFs, neg_RDFs], ignore_index=True)
    RDF_data.reset_index(drop=True, inplace=True)
    # 保存样本的数据
    RDF_data.to_csv('./datasets/positive_triplets.csv', header=True, index=False)

    # word2id in array form: [[sbj_id, rel_id, obj_id], ...]
    RDF_data_ids = EntRelToID(RDF_data, entity_dict, relation_dict)
    # print('转化后的结果：')
    # print(RDF_data_ids)
    # print("形状为：")
    # print(RDF_data_ids.shape)

    pkl.dump(RDF_data_ids, open('./datasets/triplets.pkl', 'wb'))


if __name__ == '__main__':
    mainProcess()