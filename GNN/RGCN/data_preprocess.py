"""
Name: data_preprocess.py
Purpose: transform raw RDF data into vertex and edge data
Data: 2021.3.18
Author: lixh
"""
import pandas as pd
import random
import numpy as np
import pickle as pkl
from utils import *
import time
import scipy.sparse as sp


"""
先拿旧数据测一下，需要确定所有的edges数据加起来，共有多少种edge类型，构造关系-id、实体-id字典
step1、准备数据集数据，然后将数据集划分为训练数据和测试数据


补充：
    1、
"""

# 载入所有边关系
compose_path = './data/compose_rdf_1.csv'
belong_path = './data/belong_rdf_2.csv'
relevent_path = './data/relevent_rdf_3.csv'
benefit_path = './data/benefit_rdf_4.csv'
not_benefit_path = './data/not_benefit_rdf_5.csv'
fullof_path = './data/fullof_rdf_6.csv'
not_fullof_path = './data/not_fullof_rdf_7.csv'
cure_path = './data/cure_rdf_8_update.csv'
not_cure_path = './data/not_cure_rdf_9_update.csv'


# 指定rdf路径与类型
raw_rdf = [(compose_path, 'raw_food-compose-food'), (belong_path, 'food-belong-food_cate'),
           (relevent_path, 'food-relevent-food'),
           (benefit_path, 'food-benefit-crowd'), (not_benefit_path, 'food-not_benefit-crowd'),
           (fullof_path, 'food-fullof-nutrient'), (not_fullof_path, 'food-not_fullof-nutrient'),
           (cure_path, 'food-cure-disease'), (not_cure_path, 'food-not_cure-disease')]

rel_zh_en = {'组成': 'compose', '属于': 'belong', '相关': 'relevent', '适合':'benefit', '适量': 'not_benefit',
             '富含': 'fullof', '较少': 'not_fullof', '适宜': 'cure', '不适宜': 'not_cure'}
rel_en_zh = {'compose': '组成', 'belong': '属于', 'relevent': '相关', 'benefit':'适合', 'not_benefit': '适量',
             'fullof': '富含', 'not_fullof': '较少', 'cure': '适宜', 'not_cure': '不适宜'}


def readData(raw_data):
    """
    读取所有关系数据，并整合在一起
    :param raw_data:
    :return:    RDFs 所有数据 dataframe
                node_type_dict 实体-实体类型 dict
                freq_rel 关系-出现频率 dict
    """
    RDFs = pd.DataFrame(columns=['source_name', 'relation', 'target_name'])
    node_type_dict = dict()  # 构建实体名称和实体类型的字典
    freq_rel = dict()
    for pair in raw_data:
        loaded_df = pd.read_csv(pair[0])
        # 获取实体-类型字典
        node_type_dict = generateNodeTypeDict(node_type_dict, loaded_df, pair[1])
        # 在每个dataframe中添加rdf类型，即【实体1-关系-实体2】的类型
        loaded_df_len = len(loaded_df)
        # 统计每种关系类型出现的频率，key为关系名称，value为出现频率
        freq_rel[pair[1].split('-')[1]] = loaded_df_len
        loaded_df['rdf_type'] = [pair[1] for _ in range(loaded_df_len)]
        RDFs = pd.concat([RDFs, loaded_df], ignore_index=True)
    return RDFs, node_type_dict, freq_rel


def generateNodeTypeDict(node_type_dict, loaded_df, rdf_type_name):
    """
    返回键值对分别为实体名称和实体类型的字典
    :param node_type_dict: 融合字典
    :param loaded_rdf:
    :param type_name: 诸如 raw_food-compose-food
    :return:
    """
    part_dict = dict()
    source_type, target_type = rdf_type_name.split('-')[0], rdf_type_name.split('-')[2]
    # 构建【出边顶点】的【顶点-顶点类型】字典
    source_nodes = list(set(loaded_df['source_name'].tolist()))
    for node in source_nodes:
        if node not in part_dict.keys():
            part_dict[node] = source_type
    # 构建【入边顶点】的【顶点-顶点类型】字典
    target_nodes = list(set(loaded_df['target_name'].tolist()))
    for node in target_nodes:
        if node not in part_dict.keys():
            part_dict[node] = target_type
    # 字典融合
    node_type_dict.update(part_dict)
    return node_type_dict


def generateNodeIDDict(data_list, id_lookup=True):
    """
    返回键值对分别为实体（关系）名称和id的字典
    :param data_dict:
    :param data_list:
    :return: {relation_name(entity_name): id}
    """
    data_dict = dict()
    data_list = [(i, item) for i, item in enumerate(data_list)]
    for pair in data_list:
        if id_lookup:
            data_dict[pair[1]] = pair[0]
        else:
            data_dict[pair[0]] = pair[1]
    return data_dict


def readTripletsAsList(RDFs, node_type_dict, freq_rel):
    """
    读取RDF，将所有字符转化为序号，并生成train，valid，test数据
    :param RDFs:
    :return:
    """
    """生成关系，实体字典
    step 1
    """
    # 分别获取【关系列表】和【实体列表】
    relation_list = list(set(RDFs['relation'].tolist()))  # 关系列表
    entity_list = list(set(RDFs['source_name'].tolist() + RDFs['target_name'].tolist()))  # 实体列表
    rdf_list = list(set(RDFs['rdf_type'].tolist()))  # rdf类型列表
    # 构建【关系字典】和【实体列表】
    relation_dict = generateNodeIDDict(relation_list, id_lookup=True)
    entity_dict = generateNodeIDDict(entity_list, id_lookup=True)
    rdf_type_dict = generateNodeIDDict(rdf_list, id_lookup=True)
    # print('关系类型数量：' + str(len(relation_dict)))  # 9
    # print('实体数量：' + str(len(entity_dict)))  # 15851
    # print('RDF类型数量：' + str(len(rdf_type_dict)))  # 9

    # 将实体词字典转化为列表并写入文件
    with open('./datasets/entity_index_dict.txt', 'w', encoding='utf-8') as f:
        for ent_name, ent_idx in entity_dict.items():
            f.write(ent_name + '\t' + str(ent_idx) + '\n')

    label_header = 'type'  # RDFs中【实体类型】的列名
    nodes_header = 'nodes'  # RDFs中【实体名称】的列名

    """创建邻接矩阵
    step 2
    """
    # 确定邻接矩阵的维度
    adj_shape = (len(entity_list), len(entity_list))
    adjacencies = adjGeneration(relation_list, RDFs, freq_rel, entity_dict, adj_shape)
    # 将nodes_dict中的key转化为unicorn，以 encoding 指定的编码格式解码字符串。默认编码为字符串编码。
    entity_u_dict = {np.unicode(to_unicode(key)): val for key, val in entity_dict.items()}

    """构建数据集
    step 3. 使用构建好的实体字典，选取300个实体，并查询其id，和实体类型
    """
    nodes_dataset = []  # 收集用于训练的顶点数据集
    # count_r, count_f, count_d = 0, 0, 0  # 食物，原始食材，疾病需要定量收集，其他类型不需要
    for k, v in node_type_dict.items():
        # if v == 'raw_food':
        #     if count_r < 2000:
        #         nodes_dataset.append(k)
        #         count_r += 1
        # elif v == 'food':
        #     if count_f < 3000:
        #         nodes_dataset.append(k)
        #         count_f += 1
        # elif v == 'disease':
        #     if count_d < 1000:
        #         nodes_dataset.append(k)
        #         count_d += 1
        # else:
        #     nodes_dataset.append(k)
        if k not in nodes_dataset:
            nodes_dataset.append(k)
    random.shuffle(nodes_dataset)

    # 几个常见疾病需要添加进去
    # nodes_dataset += ['糖尿病', '高血压', '血脂异常', '痛风']
    # nodes_dataset = list(set(nodes_dataset))

    # 根据收集好的顶点转化为labels_df
    labels_df = pd.DataFrame(columns=('nodes', 'id', 'type'))
    for name in nodes_dataset:
        new = pd.DataFrame({'nodes': name,
                            'id': entity_dict[name],
                            'type': node_type_dict[name]}, index=[1])
        labels_df = labels_df.append(new, ignore_index=True)
    # print('数据集的长度为：' + str(len(labels_df)))  # 326, or 327
    # print(labels_df)
    # 划分数据集
    cut = int(len(labels_df) // 5)
    labels_train_df = labels_df[cut:]  # 训练数据
    labels_test_df = labels_df[:cut]  # 测试数据

    """构造数据集
    step 4. 使用labels_df, labels_train_df, labels_test_df
    """
    # 将nodes_dict中的key转化为unicorn，以 encoding 指定的编码格式解码字符串。默认编码为字符串编码。
    entity_u_dict = {np.unicode(to_unicode(key)): val for key, val in entity_dict.items()}
    # 取出列名为【type】的数据，构造标签集
    labels_set = set(labels_df[label_header].values.tolist())
    # 形成标签（顶点类型）字典：{'raw_food': 0, 'food': 1, 'disease': 2, 'food_cate': 3, 'nutrient': 4, 'crowd': 5}
    labels_dict = {lab: i for i, lab in enumerate(list(labels_set))}
    # print('{} classes: {}'.format(len(labels_set), labels_set))  # 共有6个类型的实体

    # 生成全0的稀疏矩阵
    labels = sp.lil_matrix((adj_shape[0], len(labels_set)))  # labels稀疏矩阵的 shape=（total_node_nums, total_node_type_nums）
    labeled_nodes_idx = []

    print('Loading training set')
    train_idx = []  # 记录训练数据集中的顶点id
    train_names = []  # 记录训练数据集中的顶点
    for nod, lab in zip(labels_train_df[nodes_header].values, labels_train_df[label_header].values):
        # 取出顶点和标签
        nod = np.unicode(to_unicode(nod))  # type转为unicode
        if nod in entity_u_dict:
            labeled_nodes_idx.append(entity_u_dict[nod])  # 添加训练、测试数据顶点id
            label_idx = labels_dict[lab]  # 取出标签id
            # 根据【顶点id】和【顶点类型id】确定labels中对应位置，给这个位置赋值1
            labels[labeled_nodes_idx[-1], label_idx] = 1
            train_idx.append(entity_u_dict[nod])  # 添加顶点id
            train_names.append(nod)  # 添加顶点名称
        else:
            print(u'Node not in dictionary, skipped: ', nod.encode('utf-8', errors='replace'))

    print('Loading test set')  # 与上面处理训练集是一样的步骤
    test_idx = []
    test_names = []
    for nod, lab in zip(labels_test_df[nodes_header].values, labels_test_df[label_header].values):
        nod = np.unicode(to_unicode(nod))
        if nod in entity_u_dict:
            labeled_nodes_idx.append(entity_u_dict[nod])
            label_idx = labels_dict[lab]
            labels[labeled_nodes_idx[-1], label_idx] = 1
            test_idx.append(entity_u_dict[nod])
            test_names.append(nod)
        else:
            print(u'Node not in dictionary, skipped: ', nod.encode('utf-8', errors='replace'))

    # 对列表进行排序
    labeled_nodes_idx = sorted(labeled_nodes_idx)
    # 保存【标签】稀疏矩阵
    labels = labels.tocsr()
    save_sparse_csr('./labels/labels.npz', labels)
    # 保存所有train，test的idx和names数据
    np.save('./datasets/train_idx.npy', train_idx)
    np.save('./datasets/train_names.npy', train_names)
    np.save('./datasets/test_idx.npy', test_idx)
    np.save('./datasets/test_names.npy', test_names)
    # 保存【关系】和【实体】字典
    pkl.dump(relation_dict, open('./datasets/rel_dict.pkl', 'wb'))
    pkl.dump(entity_list, open('./datasets/nodes.pkl', 'wb'))
    # 创建单位矩阵
    features = sp.identity(adj_shape[0], format='csr')  # 构建单位矩阵

    # 将字符转化为id
    # datasets = []
    # for i in range(len(RDFs)):
    #     # 将训练数据取出，每次取一行，分别将每行的实体和关系都转化为id
    #     entity_1 = entity_dict[RDFs.iloc[i]['source_name']]
    #     relation = relation_dict[RDFs.iloc[i]['relation']]
    #     entity_2 = entity_dict[RDFs.iloc[i]['target_name']]
    #     datasets.append([entity_1, relation, entity_2])  # 将每行数据处理后的结果添加到列表
    # # 打乱数据顺序，然后切分train，valid，test数据
    # random.shuffle(datasets)
    # cut_1 = int(len(datasets) / 5 * 3)
    # cut_2 = int(len(datasets) / 5) + cut_1
    # train_triplets = np.array(datasets[: cut_1])
    # valid_triplets = np.array(datasets[cut_1: cut_2])
    # test_triplets = np.array(datasets[cut_2 :])
    return adjacencies, features, labels, labeled_nodes_idx, train_idx, test_idx, relation_dict, train_names, test_names


def adjGeneration(relation_list, RDFs, freq_rel, entity_dict, adj_shape):
    # 创建邻接矩阵
    adjacencies = []
    for i, rel in enumerate(relation_list):
        # 针对每一种关系，输出序号，关系，出现频率
        print(u'Creating adjacency matrix for relation {}: {}, frequency {}'.format(i, rel, freq_rel[rel_zh_en[rel]]))
        # 创建 shape = (freq(rel), 2) 的空数组
        edges = np.empty((freq_rel[rel_zh_en[rel]], 2), dtype=np.int32)
        # 记录edges的大小
        size = 0
        # 输出在【rel】关系下的三元组
        chosen_df = RDFs[RDFs['relation'] == rel]
        for j in range(len(chosen_df)):
            s = chosen_df.iloc[j]['source_name']
            o = chosen_df.iloc[j]['target_name']
            # 在【rel】的关系下，[entity_dict[s], entity_dict[o]]位置上的值为1
            edges[j] = np.array([entity_dict[s], entity_dict[o]])
            size += 1
        print('{} edges added'.format(size))

        row, col = np.transpose(edges)  # 取出的row就是s坐标，col就是o坐标
        data = np.ones(len(row), dtype=np.int32)  # 生成全1向量data
        # 根据行列坐标及全1向量生成邻接矩阵和邻接矩阵转置
        adj = sp.csr_matrix((data, (row, col)), shape=adj_shape, dtype=np.int8)
        # 这里能够取到 adj.data, adj.indices, adj.indptr, adj.shape
        adjacencies.append(adj)
        adj_transp = sp.csr_matrix((data, (col, row)), shape=adj_shape, dtype=np.int8)
        adjacencies.append(adj_transp)
        # 保存两个邻接矩阵，即adj， adj_transp
        save_sparse_csr('./adjacencies/' + '%d.npz' % (i * 2), adj)
        save_sparse_csr('./adjacencies/' + '%d.npz' % (i * 2 + 1), adj_transp)
    return adjacencies


def to_unicode(input):
    input = input.encode('utf-8', errors='replace')
    if isinstance(input, str):
        return input.decode('utf-8', errors='replace')
    else:
        return input
    return str(input).decode('utf-8', errors='replace')


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def adj_generation(train_triplets, entity_dict):
    """
    根据数据创建邻接矩阵和度矩阵（记录节点位置）
    一个训练集对应者一个邻接矩阵和度矩阵
    :param train_triplets:
    :param entity_dict:
    :return:
    """
    # 有多少个实体，邻接矩阵的行就有多少个
    adj_list = [[] for _ in entity_dict]
    for i, triplet in enumerate(train_triplets):
        # adj_list[triplet[]]为【中心节点】，将如果存在相关实体，则将这个实体的位置记录在adj_list对应实体中
        # 对于同一条关系边，要记两条，主次顺序相反，最终生成的adj_list是有个三层嵌套的数组
        adj_list[triplet[0]].append([i, triplet[2]])
        adj_list[triplet[2]].append([i, triplet[0]])

    # 生成度矩阵，adj_list的一个元素长度为多少，与【中心顶点】相关的顶点就有多少
    degrees = np.array([len(a) for a in adj_list])
    adj_list = [np.array(a) for a in adj_list]  # 将邻接矩阵的每行进行数组化
    return degrees, adj_list


def dataProcess(raw_data):
    """
    数据处理主函数
    :param raw_data_paths:
    :return:
    """
    RDFs, node_type_dict, freq_rel = readData(raw_data)  # 所有RDF，shape=(119506, 4)
    adjacencies, features, labels, labeled_nodes_idx, train_idx, test_idx, relation_dict, train_names, test_names = readTripletsAsList(RDFs,
                                                                                                                                       node_type_dict,
                                                                                                                                       freq_rel)
    # rel_list = range(len(adjacencies))
    # # 每个关系产生两个邻接矩阵，因此【adjacencies】的维度是9*2=18；rel_list 为 range(18)
    # for key, value in relation_dict.items():
    #     rel_list[value * 2] = key
    #     rel_list[value * 2 + 1] = key + '_INV'

    num_nodes = adjacencies[0].shape[0]
    identity_matrix = sp.identity(num_nodes, format='csr')  # 构建单位矩阵
    adjacencies.append(identity_matrix)  # add identity matrix

    support = len(adjacencies)  # 邻接矩阵的个数（包含一个单位矩阵，support = 19）
    # a.sum() 实际上统计的是一个邻接矩阵中位置是1的个数，以次来计算度矩阵
    print("Relations used and their frequencies" + str([a.sum() for a in adjacencies]))

    print("Calculating level sets...")
    t = time.time()
    # Get level sets (used for memory optimization)
    bfs_generator = bfs_relational(adjacencies, labeled_nodes_idx)
    lvls = list()
    lvls.append(set(labeled_nodes_idx))
    lvls.append(set.union(*bfs_generator.__next__()))
    print("Done! Elapsed time " + str(time.time() - t))

    # Delete unnecessary rows in adjacencies for memory efficiency
    todel = list(set(range(num_nodes)) - set.union(lvls[0], lvls[1]))
    for i in range(len(adjacencies)):
        csr_zero_rows(adjacencies[i], todel)

    data = {'A': adjacencies,
            'y': labels,
            'train_idx': train_idx,
            'test_idx': test_idx}

    with open('./datasets/food' + '.pickle', 'wb') as f:
        pkl.dump(data, f, pkl.HIGHEST_PROTOCOL)
    return relation_dict,


if __name__ == '__main__':
    dataProcess(raw_rdf)
