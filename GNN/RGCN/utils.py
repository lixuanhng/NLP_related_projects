import numpy as np
import random
import pandas as pd
from data_preprocess import rel_zh_en, rel_en_zh
from negative_sampling import negativeWordLookup, countEntityFreq
from collections import defaultdict
import pickle as pkl
import os


def tripletLabelSplit(triplets):
    """
    从三元组数据中获取三元组数据和样本标签
    :param triplets:
    :return:
    """
    labels = triplets.transpose()[-1]
    datas = triplets.transpose()[:-1]
    rdf = datas.transpose()
    return rdf, labels


def EntRelToID(RDFs, ent_dict, rel_dict):
    """将RDF中的【实体】和【关系】转化为ID，输出为numpy数组"""
    triplets = []
    for i in RDFs.index:
        triplet = RDFs.loc[i].values.tolist()
        triplet_id = [int(ent_dict[triplet[0]]),
                     int(ent_dict[triplet[2]]),
                     int(rel_dict[triplet[1]]),  # 【关系】英文转为中文
                     int(triplet[4])]
        triplets.append(triplet_id)
    # random.shuffle(triplets)  # 这里不使用打乱顺序，因为要对正负样本进行同时输入
    triplets_id = np.array(triplets)
    return triplets_id


def readChosenRDF(data_path, drop_rel_type=None):
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
            if drop_rel_type != None and drop_rel_type in name:
                # 如果有drop掉的关系类型，并且类型在文件名称中
                pass
            else:
                # 不选择与【指定关系类型】相关的三元组
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


def mainProcess(drop_rel_type=None):
    # ent_dict = defaultdict()
    # with open('./datasets/entity_index_dict.txt', 'r', encoding='utf-8') as ent_dict_file:
    #     ent_dict_list = ent_dict_file.readlines()
    # for line in ent_dict_list:
    #     pair = line.strip('\n').split('\t')
    #     ent_dict[pair[0]] = pair[1]
    # pkl.dump(ent_dict, open('./datasets/ent_dict.pkl', 'wb'))
    # 载入【诗词字典】
    ent_dict = pkl.load(open('./datasets/ent_dict.pkl', 'rb'))

    # 载入【关系字典】
    rel_dict = pkl.load(open('./datasets/rel_dict.pkl', 'rb'))

    data_path = "./data"

    # 获取正样本数据，及词频排序
    chosen_RDFs, ent_freq_dict, rdf_count = readChosenRDF(data_path, drop_rel_type)
    # 保存正样本的数据
    chosen_RDFs.to_csv('./datasets/positive_triplets.csv', header=True, index=False)
    return chosen_RDFs, ent_freq_dict, rdf_count, ent_dict, rel_dict


def negativeSampling(chosen_RDFs, pos_sample_num, ent_freq_dict, rdf_count, num_neg_sample):
    """
    产生负样本数据
    :param chosen_RDFs: 正样本
    :param pos_sample_num: 选择的正样本个数，这里一般是一个不大于10的数，只是为了验证模型
    :param ent_freq_dict: 关系-实体-词频 字典
    :param rdf_count: 正样本个数
    :param num_neg_sample: 一个正样本对应的负样本个数
    :return: 正样本的行索引列表
    """
    print('开始负采样！')
    pos_sample_idxes = []
    RDFs_neg = pd.DataFrame(columns=['source_name', 'relation', 'target_name', 'rdf_type', 'label'])
    for _ in range(pos_sample_num):
        j = random.choice(range(rdf_count))  # j 为正样本下的行号
        pos_sample_idxes.append(j)  # 记录将正样本的行索引
        rel_name_zh = chosen_RDFs.iloc[j]['relation']  # 取出关系名称
        raw_sbj = chosen_RDFs.iloc[j]['source_name']
        raw_obj = chosen_RDFs.iloc[j]['target_name']
        rel_name = rel_zh_en[rel_name_zh]  # 关系中文转化为英文
        for _ in range(num_neg_sample):
            replace_entity = random.choice(['subject', 'object'])  # pick up the corrupted entity randomly
            if replace_entity == 'subject':
                ent_re = chosen_RDFs.iloc[j]['source_name']  # 待替换的实体
                neg_candidate_list = negativeWordLookup(ent_freq_dict[rel_name][0], ent_re)  # 获取【备选词表】
                new_ent = random.choice(neg_candidate_list)  # 替换后的词
                new_df = pd.DataFrame({'source_name': new_ent,
                                       'relation': rel_name_zh,
                                       'target_name': raw_obj,
                                       'rdf_type': chosen_RDFs.iloc[j]['rdf_type'],
                                       'label': 0}, index=[1])
            else:
                ent_re = chosen_RDFs.iloc[j]['target_name']  # 待替换的实体
                neg_candidate_list = negativeWordLookup(ent_freq_dict[rel_name][1], ent_re)  # 获取【备选词表】
                new_ent = random.choice(neg_candidate_list)  # 替换后的词
                new_df = pd.DataFrame({'source_name': raw_sbj,
                                       'relation': rel_name_zh,
                                       'target_name': new_ent,
                                       'rdf_type': chosen_RDFs.iloc[j]['rdf_type'],
                                       'label': 0}, index=[1])
            RDFs_neg = RDFs_neg.append(new_df, ignore_index=True)
    return RDFs_neg, pos_sample_idxes


def samplingUse(chosen_RDFs, sbj_cdts, obj_cdts, pos_start, pos_end, sbj_num, obj_num, chosen_rel_name):
    """
    根据选中的正样本，及正样本数据量，这类样本中头节点和尾节点所属于的词表
    :param chosen_RDFs: 选中的正样本
    :param pos_sample_num: 选择的正样本个数
    :param ent_freq_dict: 关系-实体-词频 字典
    :param num_neg_sample: 一个正样本对应的负样本个数
    :param chosen_rel_name: 要产生的样本的关系类型9+
    :return: 正负样本的集合，dataframe
    """
    print('开始负采样！')
    sampled_RDFs = pd.DataFrame(columns=['source_name', 'relation', 'target_name', 'rdf_type', 'label'])
    rel_name_zh = rel_en_zh[chosen_rel_name]
    sbj_end = min(len(sbj_cdts), sbj_num)  # 选取【指定样本数量】和【提供备选实体数量】较小的那个
    obj_end = min(len(obj_cdts), obj_num)
    for i in range(pos_start, pos_end):
        # 选取正样本中的一个三元组，将其添加到结果中
        sampled_RDFs = sampled_RDFs.append(chosen_RDFs.iloc[i], ignore_index=True)
        raw_sbj = chosen_RDFs.iloc[i]['source_name']
        raw_obj = chosen_RDFs.iloc[i]['target_name']
        for replace_entity in ['subject', 'object']:  # 选取头节点或者尾节点
            if replace_entity == 'subject':
                for j in range(sbj_end):
                    new_df = pd.DataFrame({'source_name': sbj_cdts[j],  # 替换后的sbj实体
                                           'relation': rel_name_zh,
                                           'target_name': raw_obj,
                                           'rdf_type': chosen_RDFs.iloc[i]['rdf_type'],
                                           'label': 0}, index=[1])
                    sampled_RDFs = sampled_RDFs.append(new_df, ignore_index=True)
            else:
                for j in range(obj_end):
                    new_df = pd.DataFrame({'source_name': raw_sbj,
                                           'relation': rel_name_zh,
                                           'target_name': obj_cdts[j],  # 替换后的obj实体
                                           'rdf_type': chosen_RDFs.iloc[j]['rdf_type'],
                                           'label': 0}, index=[1])
                    sampled_RDFs = sampled_RDFs.append(new_df, ignore_index=True)
    return sampled_RDFs


def combinePosNeg(chosen_RDFs, RDFs_neg, pos_sample_idxes, pos_sample_num, num_neg_sample):
    """
    将正负样本进行穿插合并
    :param chosen_RDFs:         正样本
    :param RDFs_neg:            负样本
    :param pos_sample_idxes:    随机抽取的正样本的行索引
    :param pos_sample_num:      选取的样本组数，即一共选取多少组正负样本
    :param num_neg_sample:      一个正样本对应的负样本的个数
    :return:            正负样本结合后的dataframe
    """
    triplets = pd.DataFrame(columns=['source_name', 'relation', 'target_name', 'rdf_type', 'label'])
    # 根据选取正样本的行索引生辰对应的正样本RDF数据，并重建行索引
    RDFs_pos = chosen_RDFs.iloc[pos_sample_idxes]
    RDFs_pos.reset_index(drop=True, inplace=True)

    for i in range(pos_sample_num):
        pos_rdf = RDFs_pos[i:i+1]  # 每次取1个
        neg_rdf = RDFs_neg[i*num_neg_sample: (i+1)*num_neg_sample]  # 每次取30个
        whole_rdf = pd.concat([pos_rdf, neg_rdf], ignore_index=True)
        triplets = pd.concat([triplets, whole_rdf], ignore_index=True)
    return triplets  # 返回的RDF数量应为310