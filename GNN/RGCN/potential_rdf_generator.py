"""
本代码用来生成所有潜在的RDF数据，用来作为新的数据
"""

from utils import *
import tensorflow as tf
from args_help import args
from model import RGCNNetwork
import pandas as pd
import scipy.sparse as sp
import numpy as np


def generating(drop_rel_type, pos_start, pos_end, sbj_num, obj_num, chosen_rel_name):
    # 获取正样本
    pos_RDFs, ent_freq_dict, rdf_count, ent_dict, rel_dict = mainProcess(drop_rel_type)

    # 将所有实体进行拆分，每个实体类别进行聚类
    compose_df = pos_RDFs.query('rdf_type == "compose"')
    relevent_df = pos_RDFs.query('rdf_type == "relevent"')
    belong_df = pos_RDFs.query('rdf_type == "belong"')
    fullof_df = pos_RDFs.query('rdf_type == "fullof"')
    not_fullof_df = pos_RDFs.query('rdf_type == "not_fullof"')
    benefit_df = pos_RDFs.query('rdf_type == "benefit"')
    not_benefit_df = pos_RDFs.query('rdf_type == "not_benefit"')
    cure_df = pos_RDFs.query('rdf_type == "cure"')
    not_cure_df = pos_RDFs.query('rdf_type == "not_cure"')

    print('【compose】关系个数为{}'.format(len(compose_df)))
    print('【relevent】关系个数为{}'.format(len(relevent_df)))
    print('【belong】关系个数为{}'.format(len(belong_df)))
    print('【fullof】关系个数为{}'.format(len(fullof_df)))
    print('【not_fullof】关系个数为{}'.format(len(not_fullof_df)))
    print('【benefit】关系个数为{}'.format(len(benefit_df)))
    print('【not_benefit】关系个数为{}'.format(len(not_benefit_df)))
    print('【cure】关系个数为{}'.format(len(cure_df)))
    print('【not_cure】关系个数为{}'.format(len(not_cure_df)))
    print('\n')

    raw_foods = compose_df['source_name'].values.tolist()
    nutrients = fullof_df['target_name'].values.tolist() + not_fullof_df['target_name'].values.tolist()
    food_cates = belong_df['target_name'].values.tolist()
    foods = compose_df['target_name'].values.tolist() + relevent_df['source_name'].values.tolist() \
            + relevent_df['target_name'].values.tolist() + belong_df['source_name'].values.tolist() \
            + fullof_df['source_name'].values.tolist() + not_fullof_df['source_name'].values.tolist() \
            + benefit_df['source_name'].values.tolist() + not_benefit_df['source_name'].values.tolist() \
            + cure_df['source_name'].values.tolist() + not_cure_df['source_name'].values.tolist()
    crowds = benefit_df['target_name'].values.tolist() + not_benefit_df['target_name'].values.tolist()
    diseases = cure_df['target_name'].values.tolist() + not_cure_df['target_name'].values.tolist()

    # 各实体词表
    raw_foods = list(set(raw_foods))
    nutrients = list(set(nutrients))
    food_cates = list(set(food_cates))
    foods = list(set(foods))
    crowds = list(set(crowds))
    diseases = list(set(diseases))

    print('【基础食材】个数为{}'.format(len(raw_foods)))
    print('【营养素】个数为{}'.format(len(nutrients)))
    print('【食物类别】个数为{}'.format(len(food_cates)))
    print('【食物】个数为{}'.format(len(foods)))
    print('【人群】个数为{}'.format(len(crowds)))
    print('【疾病】个数为{}'.format(len(diseases)))
    print('\n')

    # 根据每一条正样本，替换正样本中头节点和尾节点中的任意一个，换成被替换的那个所在的列表中的其他实体词
    # 这里的 chosen_RDFs 就是根据 chosen_rel_name 确定的三元组
    if chosen_rel_name == 'compose':
        chosen_RDFs = compose_df
        sbj_cdts, obj_cdts = raw_foods, foods
    elif chosen_rel_name == 'relevent':
        chosen_RDFs = relevent_df
        sbj_cdts, obj_cdts = foods, foods
    elif chosen_rel_name == 'belong':
        chosen_RDFs = belong_df
        sbj_cdts, obj_cdts = foods, food_cates
    elif chosen_rel_name == 'fullof':
        chosen_RDFs = fullof_df
        sbj_cdts, obj_cdts = foods, nutrients
    elif chosen_rel_name == 'not_fullof':
        chosen_RDFs = not_fullof_df
        sbj_cdts, obj_cdts = foods, nutrients
    elif chosen_rel_name == 'benefit':
        chosen_RDFs = benefit_df
        sbj_cdts, obj_cdts = foods, crowds
    elif chosen_rel_name == 'not_benefit':
        chosen_RDFs = not_benefit_df
        sbj_cdts, obj_cdts = foods, crowds
    elif chosen_rel_name == 'cure':
        chosen_RDFs = cure_df
        sbj_cdts, obj_cdts = foods, diseases
    elif chonen_rel_name == 'not_cure':
        chosen_RDFs = not_cure_df
        sbj_cdts, obj_cdts = foods, diseases

    sampled_RDFs = samplingUse(chosen_RDFs, sbj_cdts, obj_cdts,
                               pos_start, pos_end, sbj_num, obj_num, chosen_rel_name)

    return sampled_RDFs, ent_dict, rel_dict, fullof_df, not_fullof_df, benefit_df, \
           not_benefit_df, cure_df, not_cure_df


if __name__ == '__main__':
    drop_rel = None
    pos_num_start = 20
    pos_num_end = 40  # 正样本个数
    sbj_neg_num = 3000  # 需要替换的sbj的个数（根据【实际实体类型】确定）
    obj_neg_num = 12  # 需要替换的obj的个数（根据【实际实体类型】确定）
    rel_object = 'benefit'  # 通过模型进行预测时，值参考benefit，fullof和cure数据
    output_path = './rdf_output/output_RDF_0429.csv'

    # hyper-parameters
    VALIDATION = args['validation']
    LR = args['learnrate']
    L2 = args['l2norm']
    EMB_DIM = args['hidden']
    BASES = args['bases']
    DROPOUT = args['dropout']
    EDGE_NUMS = args['edge_nums']
    CHECKPOINT_PTAH = args['model_save_path']

    print('开始导入邻接矩阵完毕')
    with open('./datasets/food' + '.pickle', 'rb') as f:
        data = pkl.load(f)

    A = data['A']
    num_nodes = A[0].shape[0]
    support = len(A)

    X = sp.csr_matrix(A[0].shape)

    # 邻接矩阵做归一化
    for i in range(len(A)):
        d = np.array(A[i].sum(axis=1)).flatten()
        d[np.where(d == 0)] = 10
        d_inv = 1. / d
        d_inv[np.where(d_inv == 0.1)] = 0.
        D_inv = sp.diags(d_inv)
        A[i] = D_inv.dot(A[i]).tocsr()

    input_data = [X] + A

    # 一类正样本产生的数量为 1 + sbj_neg_num + obj_neg_num，即 batch_size
    pos_RDFs, entity_dict, relation_dict, fullof_rdf, not_fullof_rdf, benefit_rdf, \
    not_benefit_rdf, cure_rdf, not_cure_rdf = generating(drop_rel, pos_num_start, pos_num_end,
                                                         sbj_neg_num, obj_neg_num, rel_object)
    print(pos_RDFs)
    print('\n')

    inputs_triplets = pos_RDFs.values
    # 获取输入数据的id数组格式
    inputs_ids = EntRelToID(pos_RDFs, entity_dict, relation_dict)
    print('转换后的维度为：')
    print(inputs_ids.shape)
    print('其中，正样本的个数为{}，对应的负样本的个数为{}'.format(pos_num_end - pos_num_start, (sbj_neg_num + obj_neg_num) * (pos_num_end - pos_num_start)))
    batch_size = 1 + sbj_neg_num + obj_neg_num
    print('batch_size = {}'.format(str(batch_size)))
    print('\n')

    # 获取输入数据于标签，其中【inputs_rdfs】为要输入的三元组数据
    inputs_rdfs, inputs_labels = tripletLabelSplit(inputs_ids)
    inputs_data = tf.data.Dataset.from_tensor_slices((inputs_rdfs, inputs_labels))
    inputs_data = inputs_data.batch(batch_size, drop_remainder=True)

    model = RGCNNetwork(support, True, BASES, EDGE_NUMS, batch_size, EMB_DIM, DROPOUT)
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(tf.train.latest_checkpoint(CHECKPOINT_PTAH))
    print("模型实例化完毕!")

    print('模型预测结果如下：')
    i = 0
    rdf_check = list()  # 记录排位在正样本之前的所有三元组
    for _, (rdf_batch, labels_batch) in enumerate(inputs_data):
        pred = model([input_data, rdf_batch])
        scores = tf.math.sigmoid(pred)  # 预测结果，将模型预测结果转化为(0,1)区间
        refer = inputs_triplets[i:i + batch_size]  # 展示原始三元组
        res_arr = scores.numpy()

        # 将数据进行整合，然后对最后的结果进行降序排列
        results = list()
        for j in range(len(refer)):
            results.append((refer[j][0], refer[j][1], refer[j][2], refer[j][4], res_arr[j]))  # (原始三元组， score)
            results.sort(key=lambda x: x[4], reverse=True)
        for k in range(len(results)):
            rdf_check.append([results[k][0], results[k][1], results[k][2]])
            if 1 in results[k]:
                break
        i += batch_size

    # 将列表数据转化为dataframe
    res_rdf = pd.DataFrame(rdf_check, columns=['subject', 'relation', 'object'])
    # rdf结果进行去重
    res_rdf = res_rdf.drop_duplicates(subset=None, keep='first', inplace=False)
    # 从结果中去掉实际存在于图谱中的数据
    print('产生数据为: {}'.format(str(len(res_rdf))))

    res_rdf.to_csv(output_path, header=True, index=False)