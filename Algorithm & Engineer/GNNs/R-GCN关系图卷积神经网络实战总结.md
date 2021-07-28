# R-GCN 关系图卷积神经网络链路预测任务论文复现总结

今天给大家带来的是R-GCN关系图卷积神经网络实战总结。这篇总结是参考于图神经网络的经典论文 [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103) 及作者源码的基础上完成的，使用的脚本语言为python，深度学习框架TensorFlow2.1。



## 开始前的一些说明

本论文包含的实际任务有两个，一个[**实体分类任务**](https://github.com/tkipf/relational-gcn)（Entity Classification），另一个是[**链路预测任务**](https://github.com/MichSchli/RelationPrediction)（Link Prediction）。这两个任务分别由两位作者大佬独立完成。实体分类任务中，作者代码采用了继承keras的方法编写，相对容易懂一些；而在链路预测任务中，作者写了一个很深的pipeline，同时融合了encoder和decoder模型，说实话看起来特别费劲（可能是能力有限）。由于本次主要复现的是链路预测的任务，加上该任务作者代码几乎没有注释，所以在模型的encoder方面主要采用实体分类的encoder写法，而decoder方面尽可能避开原作者代码，但又尽可能按照论文意愿构建decoder的部分。在讲解期间，可能还会穿插一些关于原作者代码的讨论。当然，由于能力有限，如果在通读期间有什么问题，或是出现明显错误，欢迎大家给我留言。

代码详情请见我的[github项目](https://github.com/lixuanhng/NLP_related_projects/tree/master/GNN/RGCN)。

这篇总结不会包含太多的论文翻译内容，这类的博客很多，大家如果有需要可以随便找一篇，应该都差不多，建议在熟悉论文内容的基础上参考我的复现总结。由于在GNN方面，我还是新手，所以在实践过程中也遇到了很多问题，这些也会与大家一一分享与探讨。



## 模型解释

在论文链路预测任务中，R-GCN的模型框架如下图所示：

![WX20210424-183317@2x](/Users/lixuanhong/Desktop/WX20210424-183317@2x.png)

在`encoder`部分，模型的主要贡献是给顶点进行编码，将顶点转化为embedding vector。在`decoder`部分，模型使用一个打分函数`score function`，对多个正负样本$(e_s, r, e_o)$进行打分，然后评估正样本在所有样本打分结果排序中的位置，如果大多数正样本排名比较靠前，就认为这个模型效果不错。

关于关系图卷积，我认为其特点表现在，一个顶点的更新是由不同类型的边连接的顶点来决定的，在同一类型边下有分为进边和出边，还有假设指向自己的边类型。其实不同于普通GNN，或者GCN，模型结构不会有太大的改变，真正改变的是邻接矩阵的类型和个数。

![WX20210424-185810@2x](/Users/lixuanhong/Desktop/WX20210424-185810@2x.png)



## 数据准备

首先需要明确输入模型的数据是什么类型。在encoder中，我们需要传入的数据是图的拓扑结构，这里指的就是邻接矩阵；在decoder中，我们需要传入的是正负样本的三元组数据，也就是多个`(s, r, o)`。

### 准备邻接矩阵

对于顶点的邻接矩阵，这里需要注意的是由于数据中我们使用的关系类型总共有9种，如果认为出边和入边的拓扑关系不一样，那么总共应该生成的邻接矩阵应该有9*2等于18种；还需要加上self-loop的部分（一个单位矩阵），那么需要传入模型的邻接矩阵共有19种。具体代码大致如下：

```python
import numpy as np
import scipy.sparse as sp
import pandas as pd

def adjGeneration(relation_list, RDFs, freq_rel, entity_dict, adj_shape):
    """
    relation_list: 三元组中所有关系的列表
    RDFs:          所有三元组数据，按照dataframe组织
    freq_rel:      字典，关系-频率
    entity_dict:   字典，顶点-id
    adj_shape:	   邻接矩阵长度
    """
    adjacencies = []
    for i, rel in enumerate(relation_list):
        print(u'Creating adjacency matrix for relation {}: {}, frequency {}'.format(i, rel, freq_rel[rel_zh_en[rel]]))
        edges = np.empty((freq_rel[rel_zh_en[rel]], 2), dtype=np.int32)
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
        
        row, col = np.transpose(edges)
        data = np.ones(len(row), dtype=np.int32)
        
        adj = sp.csr_matrix((data, (row, col)), shape=adj_shape, dtype=np.int8)
        adjacencies.append(adj)
        
        adj_transp = sp.csr_matrix((data, (col, row)), shape=adj_shape, dtype=np.int8)
        adjacencies.append(adj_transp)
	
    num_nodes = adjacencies[0].shape[0]
    identity_matrix = sp.identity(num_nodes, format='csr')  
    adjacencies.append(identity_matrix)  # add identity matrix
    return adjacencies
```

这样，邻接矩阵就准备好了，这段代码源自实体分类作者源码，具体处理方式可跳转查看。在送入模型之前，也需要对邻接矩阵做归一化。

### 准备正负样本数据

由于模型在decoder部分是同时对正负样本进行打分的，所以需要同时准备正负样本数据。对于已经在图谱中的样本，我们称之为正样本。至于向模型中输入多少正样本数据，论文中提到：

Rather than the full set of edges $E$, we are given only an incomplete subset $\hat E$. The task is to assign scores $f(s,r,o)$ to possible edges $(s,r,o)$ in order to determine how likely those edges are belong to $\hat E$.

所以我们去掉两类边类型组成的三元组，将剩下的数据送入模型。

论文中对负样本的表述为：

For each observed example we sample $\omega$ negative ones. We sample by randomly corrupting either the subject or the object of each positive example.

其中 $\omega$ 表示负样本的数量，这里我们选择1个正样本对应10个负样本。论文思想是我们随机从所有实体中选取一个实体，将其替换为一个三元组中的subject或是object（这里也是随机选取），作为负样本。这里为了验证模型的有效性，我们在编写负采样时，指定了替换后的实体类型与被替换实体类型是一致的。负采样中，我们认为某个词被选中的概率和它出现的次数有关，并且有3/4采样公式。所以这里我们首先对同一个关系类型下subject和object进行排序，选择出现频率大于被替换实体的其他实体作为替换实体，这样可以保证新生成的三元组首先是符合逻辑的。

这种情况下，可能会产生很多重复的负样本（例如被替换实体在频率排序中已经是比较靠前了，在它前面的实体数量不多，而我们又需要10个负样本的情况），或是已经在图谱中存在的三元组（修改的三元组中，没有被修改的那个实体本来就和替换后的实体有关系），这些情况我都没有处理。当然，也可以通过完全随机采样的方式替换实体，也许这样在最后的打分结果中能够更明显的看到一个符合逻辑三元组和随机三元组的分值区别。负采样部分的代码如下：

```python
import pandas as pd
import random


def negativeSampling(chosen_RDFs, ent_freq_dict, rdf_count):
    """
    创建负采样的数据，每个正样本产生10个负样本，在实体出现频率排序中，负样本排在正样本的前10个
    :param chosen_RDFs:   所有正样本的dataframe数据
    :param ent_freq_dict: 实体频率字典--{rel: (sbj_list, obj_list)}
    :param rdf_count:     正样本个数
    :return:
    """
    RDFs_with_neg = pd.DataFrame(columns=['source_name', 'relation', 'target_name', 'rdf_type', 'label'])
    for j in range(rdf_count):
        if j % 2000 == 0: print("第{}个正样本数据已经产生负样本".format(j))
        replace_entity = random.choice(['subject', 'object'])
        rel_name_zh = chosen_RDFs.iloc[j]['relation']
        raw_sbj = chosen_RDFs.iloc[j]['source_name']
        raw_obj = chosen_RDFs.iloc[j]['target_name']
        rel_name = rel_zh_en[rel_name_zh]

        if replace_entity == 'subject':
            """替换subject的情况"""
            ent_re = chosen_RDFs.iloc[j]['source_name']  
            neg_candidate_list = negativeWordLookup(ent_freq_dict[rel_name][0], ent_re)  
            for _ in range(num_neg_sample):
                new_ent = random.choice(neg_candidate_list)  
                new_df = pd.DataFrame({'source_name': new_ent,
                                       'relation': rel_name_zh,
                                       'target_name': raw_obj,
                                       'rdf_type': chosen_RDFs.iloc[j]['rdf_type'],
                                       'label': 0}, index=[1])
                RDFs_with_neg = RDFs_with_neg.append(new_df, ignore_index=True)
        else:
            """替换object的情况"""
            ent_re = chosen_RDFs.iloc[j]['target_name']  
            neg_candidate_list = negativeWordLookup(ent_freq_dict[rel_name][1], ent_re)  
            for _ in range(num_neg_sample):
                new_ent = random.choice(neg_candidate_list)  
                new_df = pd.DataFrame({'source_name': raw_sbj,
                                       'relation': rel_name_zh,
                                       'target_name': new_ent,
                                       'rdf_type': chosen_RDFs.iloc[j]['rdf_type'],
                                       'label': 0}, index=[1])
                RDFs_with_neg = RDFs_with_neg.append(new_df, ignore_index=True)
    return RDFs_with_neg


def negativeWordLookup(freq_type_words, ent_re):
    """
    :param freq_vocab: 某个关系下同位置（头顶点或尾顶点）的词频列表，由高到低进行排序
    :param ent_re:     实体名称
    :return:          【备选同类词表】包含比当前词词频高的词
    """
    ent_idx = freq_type_words.index(ent_re)
    # 词频排序在替换词前面的词表，这里称之为【被选同类词表】
    neg_candidate_list = freq_type_words[:ent_idx]
    # 如果【备选同类词表】为空，则指定整个【同类词表】为【备选同类词表】
    if not neg_candidate_list:
        neg_candidate_list = freq_type_words[1:]
    return neg_candidate_list
```



## 模型构建

### encoder部分

R-GCN模型中encoder的核心公式为: $h_i^{l+1} = \sigma(\sum_{r\in R}\sum_{j\in N_i^T} \frac {1}{c_{i,r}} W_r^lh_j^l+W_0^lh_i^l)$

激活函数 $\sigma$ 内其实表达了前向传播时（上一层到下一层），针对每一种关系下图结构（邻接矩阵）与参数矩阵的乘法（当然还要算上self-loop的情况）的求和。这里我们效仿原作者代码，对输入的邻接矩阵只使用一个变量矩阵作为可训练的参数。由于我们输入的邻接矩阵的维度是`[node_num, node_num, abj_num]`，其中`node_num`表示图谱中的顶点数量，`adj_num`表示邻接矩阵的个数（如上文所述，这里为19），于是参数矩阵表达为`[adj_num, node_num, node_embedding_dim]`的矩阵，其中`node_embedding_dim`为顶点编码后的向量维度，通过encoder后的特征表示为`[node_num, node_embedding_dim]`的矩阵。

在原作者代码中，还设定了是否使用顶点本身的属性特征。如果顶点存在属性特征，那么会在数据处理阶段进行提取并做归一化。而且在训练阶段，会先将顶点特征与邻接矩阵进行矩阵相乘统一作为模型输入。由于我的数据中顶点属性大多是文本，且结构并不统一，这里就不做处理了，模型训练阶段直接使用邻接矩阵做为输入特征。

此外，作者还使用了一个超参数`num_bases`。在注释中写明这个应该是采样器的个数，类似于卷积神经网络中的多卷积核采样。但在作者源码中，他将这个`num_bases`设置为-1，也就是不需要考虑多采样的情况（我的模型也没有考虑这个问题，相对简单好理解）。如果要考虑，则需要创建更多的参数矩阵，encoder模型会更加复杂，具体请看下面的代码。

```python
class Encoder(tf.keras.layers.Layer):
    def __init__(self, embedding_dim=None, support=1, featureless=False, num_bases=-1):
        super(Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.support = support              # 邻接矩阵的个数
        self.featureless = featureless      # 是否不使用node本身的特征，值为【True】
        self.num_bases = num_bases          # 采样器个数

        self.bias = False
        self.keep_prob = 0.8
        self.input_dim = None
        self.W = None
        self.W_comp = None
        self.b = None
        self.num_nodes = None

    def build(self, input_shape):
        features_shape = input_shape[0]
        if self.featureless:
            self.num_nodes = features_shape[1] 
        self.input_dim = features_shape[1]
        if self.num_bases > 0:
            self.W = tf.keras.layers.concatenate([self.add_weight(shape = (self.input_dim,self.embedding_dim), 
                                                                  trainable=True, initializer=tf.random_normal_initializer(),
                                                                  name='W', regularizer='l2'
                                                                  ) for _ in range(self.num_bases)], axis=0)
            self.W_comp = self.add_weight((self.support, self.num_bases),
                                          initializer=tf.random_normal_initializer(),
                                          name='W_comp', regularizer='l2')
        else:
            self.W = tf.keras.layers.concatenate([self.add_weight(shape = (self.input_dim, self.embedding_dim),
                                                                  trainable=True,
                                                                  initializer=tf.random_normal_initializer(),
                                                                  name='W', regularizer='l2'
                                                                  ) for _ in range(self.support)], axis=0)

        if self.bias:
            self.b = self.add_weight((self.embedding_dim,),
                                     initializer=tf.random_normal_initializer(), name='b')

    def call(self, inputs):
        features, A = inputs[0], inputs[1:]

        supports = list()
        for i in range(self.support):
            if not self.featureless:  # 考虑顶点本身特征
                supports.append(tf.matmul(A[i], features))
            else:
                supports.append(A[i])
        supports_array = [supports[i].A for i in range(len(supports))]
        supports_ = tf.keras.layers.Concatenate(axis=1, dtype='float32')(supports_array)  

        if self.num_bases > 0:  # 当有多个采样器时
            self.W = tf.reshape(self.W, (self.num_bases, self.input_dim, self.embedding_dim))
            self.W = tf.transpose(self.W, perm=[1, 0 ,2])
            V = tf.matmul(self.W_comp, self.W)
            V = tf.reshape(V, (self.support * self.input_dim, self.embedding_dim))
            output = tf.matmul(supports_, V)
        else:   # 不考虑多个采样器的情况
            output = tf.matmul(supports_, self.W)

        if self.bias:
            output += self.b
        return output
```



### decoder部分

decoder部分也是表述相对模糊的，论文是这么叙述的：

In our experiments, we use the DistMult factorization (Yang et al. 2014) as the scoring function, which is known to perform well on standard link prediction benchmarks when used on its own. In DistMult, every relation *r* is associated with a diagonal matrix $R_r \in R^{d*d}$ and a triple (*s, r, o*) is scored as $f(s,r,o) = e_s^TR_re_o$

这里的 score function 使用的是 DistMult 模型，因为 DistMult 在链路预测上表现相对较好。score function 中的 $e_s$ 和 $e_o$ 就是一个三元组中的头顶点和尾顶点的向量表示。对于关系 $r$，论文中说，每一种关系矩阵对应于一个对角矩阵 $R_r$，但是论文中并没有提到是如何对关系进行编码的。

联想到在Attention中同样也有类似打分函数，在计算 encoder 的 `hidden state` 和 decoder 的 `hidden state` 时，他们的分数有三种计算方法，其中一种就是在二者之间添加一个可训练的参数矩阵 $W_a$，用公式表示为 $score(h_t, h_s) = h_t^T W_a h_s^T$，在训练Attention的过程中训练 $W_a$。或许这里的关系矩阵，我们也可以使用参数来表达，关系作为模型参数会随模型进行训练。

在查阅作者源码时发现，作者在decoder的部分设置了三种不同的处理方式，分别是

+ bilinear-diag: 
  + $Score = e_s r e_o$ 矩阵内积，也就是三元组按位相乘。
  + 这里的关系作者还是使用了变量进行了表达，在模型中进行训练，和我们猜测的方向是一致的。
  + 但是关系矩阵并不是一个对角矩阵，与论文有出入。
+ complex:
  + $Score = e_sre_o + n(e_s)rn(e_o) + e_sn(r)n(e_o) - n(e_s)n(r)e_o$ , 其中n()表示负样本，也就是将正负样本进行了整合计算
+ Nonlinear-transform
  + $Score = matmul(e_s, W_{s}) + matmul(r, W_{r}) + matmul(e_o, W{o})$  ，分别将三元组各与一个参数矩阵进行矩阵乘法后再求和

无论采用上述的哪种打分函数，作者在源码或论文都没有给出原因或这么做的出发点。这里只好选择自己尝试一下。能够明确的是关系矩阵一定是一个能够训练参数矩阵，我的decoder中采用了两种方式：

+ $Score = e_s W_r e_o$ 矩阵内积，也就是三元组按位相乘
  + 关系矩阵的维度为`[edge_num, node_embedding_dim]`，这里`edge_num`表示边类型的个数。数据中总共有9种边类型的三元组，这里的`edge_num=9`
  + 每个三元组输入进来，通过encoder的编码器将顶点的id转化为向量 $e_s$ 和 $e_o$。进入decoder后，根据边的id取出$W_r$ 对应行中的向量，然后将这三个向量按位相乘，得到一个`[1, node_embedding_dim]`矩阵，然后在 `axis=1`的轴上进行求和，得到这一个向量的打分结果
+ $Score = matmul(e_s, W_r, e_o^T)$ 矩阵乘法，
  + 按照论文中提到的，每个关系矩阵都是一个对角矩阵，其维度为`[edge_num, node_embedding_dim, node_embedding_dim]`，也就是共有 `edge_num` 个对角矩阵
  + 虽然模型同样使用继承`keras.layers.Layer`的自定义层方式进行编写，但是这里我们使用`tf.Varaible`作为关系的变量，因为其包含一个`initial_value`参数以供我们可以初始化一个对角矩阵
  + 对于每一个三元组，顶点的embedding过程不变，每一个关系id对应于一个对角矩阵，矩阵乘法的结果也是得到了一个`[1, node_embedding_dim]`矩阵，同样在 `axis=1`的轴上进行求和，得到这一个向量的打分结果

decoder模型构建如下，为了简化，这里省去了第一种计算打分的情况；

```python
class Decoder(tf.keras.layers.Layer):
    """
    本模型使用创建对角矩阵的方式训练关系矩阵，并使用矩阵乘法【e_s，r，T(e_o)】
    """
    def __init__(self, embedding_dim, edge_count, batch_size):
        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.edge_count = edge_count
        self.batch_size = batch_size
        rel_initial_value = tf.linalg.diag(np.random.uniform(size=(self.edge_count,
                                                                   self.embedding_dim)))
        self.W_r = tf.Variable(rel_initial_value, trainable=True, name='W_r')

    def call(self, inputs):
        # batch的train_triples (shape = [batch_size, embedding dimension])
        sbj_embs, obj_embs, rel_ids = inputs[0], inputs[1], inputs[2]
        rel_embs = tf.nn.embedding_lookup(self.W_r, rel_ids, name='rel_id2vector')
        # 计算 score，这里需要将【rel_embs】的数据类型转化为 tf.float32，
        sbj_rel_embs = tf.matmul(sbj_embs, tf.cast(rel_embs, dtype=tf.float32))
        mul_res = tf.matmul(sbj_rel_embs, tf.transpose(obj_embs, perm=[0, 2, 1]))
        mul_res = tf.reduce_sum(mul_res, 1)
        return mul_res
```

encoder和decoder的模型构建完毕后，需要继承`keras.model.Model`将上述两层拼接起来。注意，为了保证计算时维度正确，需要将顶点向量在第1维度出进行扩展。计算结束后，重新`reshape`为需要的结果格式。



### 训练及测试

使用cross-entropy loss function，除去部分系数有变化外，主要的公式与常见没有区别。损失函数的公式如下：

$L = - \frac {1} {(1+\omega)|\hat E|} \sum_{(s,r,o,y)\in T} y logl(f(s,r,o)) + (1-y)log(1-l(f(s,r,o)))$

如果是正样本，则`y=1`，如果是负样本，则`y=0`。需要注意的是打分结果之后需要先接一个激活函数，然后再放入loss function。对于这个问题，TensorFlow给出了一个[现成的方法](https://tensorflow.google.cn/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits?hl=en)：

```python
tf.nn.sigmoid_cross_entropy_with_logits(labels_batch, logits)
```

直接将打分结果和样本标签放入就行了，在外面还对结果取了平均。

如果是在预测阶段，打分结果并不是最终的结果，还需要再外面加一个`sigmoid`函数，将打分结果转化到`(0, 1)`的范围内。

在训练阶段，我的正负样本数据共有19万左右，其中正样本与负样本的比例是1:5，一共训练5轮，batch_size = 12000。使用CPU训练，大概花费了1个小时。训练结果如下（这里只展示了损失下降的情况）

![训练结果](/Users/lixuanhong/Desktop/训练结果.png)



### 模型评估

作者提出了使用 MRR (mean reciprocal rank) 和 H@n (Hits at n) 方法进行评估，大致意思是：对于每一个三元组样本，我们通过随机替换掉它的头顶点或是尾顶点的方式，产生很多个负样本。将对所有样本进行打分，并对这些打分结果进行升序排序。

Hits@10，表示在所有样本结果中，计算正样本的得分排名前10的个数有多少，除以样本总数，就等于这个三元组的Hits@10的值。Hits@1，Hits@5同理。MRR则表示所有正样本在各自打分结果中的排名的平均值。我的理解是，最终打分的具体分值不重要，关键是正样本的排名要尽可能靠前，满足这个条件，模型的效果就可以认为是不错的。打分结果比正样本还高的负样本，可以作为新的三元组关系添加到图谱中。返回的结果如下图：

![正负样本的测试结果](/Users/lixuanhong/Desktop/正负样本的测试结果.jpg)

从预测结果上看，效果一般，10组样本下的Hits@10基本能排在80%左右，Hits@5应该只有30%

具体情况来看，模型中没有使用上述两个评估方式的原因有如下几个：

+ 由于数据本身具有比较高的专业性，对结果的评估应多参考一些专家意见
+ 网络上并没有找到合适的计算Hits@n和MRR的计算方法，好像使用频率并不是特别高
+ 计算资源不够，一个正样本对应5个负样本的计算量已经很大了，在训练阶段想要完全实现论文的效果还是比较难的



### 总结：

复现这篇论文最大的感受就是，很多地方论文和作者源码有些对不上，而且作者源码没有多少注释，论文是看了很多遍，但是复现出来问题依旧比较多。继续努力吧！如果有什么我没有说明白的地方，欢迎大家给我留言！

代码详情请见我的[github项目](https://github.com/lixuanhng/NLP_related_projects/tree/master/GNN/RGCN)。