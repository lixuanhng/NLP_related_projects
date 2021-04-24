import numpy as np
"""
读取labels中的数据，按理来说labels是一个稀疏矩阵，shape=（15851，6），且只有train和test中的数据在labels中存在one-hot
别数据都为零，所以也需要读取train_idx, test_idx, train_names, test_names
"""

# 首先尝试读取 labels
# labels = np.load('./labels/labels.npz')
# print(labels['data'].shape)
# print(labels['indices'])  # 这是实体类型
# print(labels['indptr'])  # 所有节点都对应上了327个值


"""
从checkpoint中恢复模型，然后载入测试数据，获取每个节点的embedding，
"""
import tensorflow as tf
from model import RGCNNetwork, HiddenLayer, acc
from utils import *
import pickle as pkl
from args_help import args
VALIDATION = args['validation']


"""构建输入数据"""
with open('./datasets/food' + '.pickle', 'rb') as f:
    data = pkl.load(f)

A = data['A']  # (list)邻接矩阵，共有19个
y = data['y']  # (稀疏矩阵)标签，shape=（所有顶点个数， 所有顶点类型）
train_idx = data['train_idx']  # (list)训练数据集中的顶点id，4/5
test_idx = data['test_idx']  # (list)训练测试集中的顶点id，1/5

# Get dataset splits 数据划分
y_train, y_val, y_test, idx_train, idx_val, idx_test = get_splits(y, train_idx,
                                                                  test_idx,
                                                                  VALIDATION)
# train_mask = sample_mask(idx_train, y.shape[0])  # bool型数组，长度为【顶点总数】

num_nodes = A[0].shape[0]  # 获取第一个邻接矩阵的第一个维度，作为【顶点数量】，其实就是15851
support = len(A)  # 邻接矩阵的个数（包含一个单位矩阵，support = 19）

X = sp.csr_matrix(A[0].shape)  # 构建输入数据 X，shape=(15851, 15851) 目前这个特征值是全0的

# Normalize adjacency matrices individually
# 对每个输入的邻接矩阵做归一化
for i in range(len(A)):
    d = np.array(A[i].sum(axis=1)).flatten()
    if d.sum() > 0:
        d_inv = 1. / d  # 这里出现的问题暂时不予修改
        d_inv[np.isinf(d_inv)] = 0.
        D_inv = sp.diags(d_inv)
        A[i] = D_inv.dot(A[i]).tocsr()

# 传入模型的数据
input_data = [X] + A

"""读入【节点-编号】字典"""
node_dict = dict()
with open('./datasets/entity_index_dict.txt', 'r', encoding='utf-8') as f:
    pair_list = f.readlines()
    for line in pair_list:
        pair = line.strip('\n').split('\t')
        node_dict[pair[0]] = int(pair[1])


"""模型实例化"""
save_path = './checkpoints'
model = RGCNNetwork(6, 19, True, -1)
optimizer = tf.keras.optimizers.Adam(0.01)
ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
ckpt.restore(tf.train.latest_checkpoint(save_path))


"""获取模型中的参数"""
# 调用模型
logits = model(input_data)
# 自定义层（关键层）中的所有参数，list，length=19，每个元素是 (15851, 6) 的参数矩阵
weights_0 = model.get_layer(index=0).get_weights()
# print(weights_0)
# uid_embedding = weights_0[0]
# print(uid_embedding)


"""选择临近点和较远点进行weights的测试
1、【柚子】，最近点【虾米】，较远点【海参】，更远点【冬瓜粥】

"""
centre_idx = node_dict['柚子']
close_1_idx = node_dict['虾米']
close_2_idx = node_dict['海参']
close_3_idx = node_dict['冬瓜粥']


def meanPooling(idx, weights):
    """取出某个节点对应的向量的平均池化"""
    arr_sum = np.zeros((6))
    for arr in weights:
        arr_sum += arr[idx]
    return arr_sum / support


def simlarityCal(vector1, vector2):
    """余弦相似度"""
    vector1Mod=np.sqrt(vector1.dot(vector1))
    vector2Mod=np.sqrt(vector2.dot(vector2))
    if vector2Mod!=0 and vector1Mod!=0:
        simlarity=(vector1.dot(vector2))/(vector1Mod*vector2Mod)
    else:
        simlarity=0
    return simlarity


def requireVector(weights, close_1, close_2, close_3):
    # 获取节点对应的向量表示
    # v_centre = meanPooling(node_dict[centre], weights)
    v_close_1 = meanPooling(node_dict[close_1], weights)
    v_close_2 = meanPooling(node_dict[close_2], weights)
    v_close_3 = meanPooling(node_dict[close_3], weights)
    # v_close_4 = meanPooling(node_dict[close_4], weights)
    # 计算余弦相似度
    sim_c_1 = simlarityCal(v_close_2, v_close_1)
    sim_c_2 = simlarityCal(v_close_3, v_close_2)
    sim_c_3 = simlarityCal(v_close_1, v_close_3)
    # sim_c_4 = simlarityCal(v_centre, v_close_4)

    print("{} 与 {} 的相似度为 {:.4f}".format(close_2, close_1, sim_c_1))
    print("{} 与 {} 的相似度为 {:.4f}".format(close_3, close_2, sim_c_2))
    print("{} 与 {} 的相似度为 {:.4f}".format(close_1, close_3, sim_c_3))
    # print("{} 与 {} 的相似度为 {:.4f}".format(centre, close_4, sim_c_4))
    # return sim_c_1, sim_c_2, sim_c_3


requireVector(weights_0, '乳鸽', '腊鹅', '扒鸡')
# 从图上看，【腊鹅】和【乳鸽】应该是更像的