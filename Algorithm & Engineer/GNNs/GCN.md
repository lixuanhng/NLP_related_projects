# GCN 相关

## 综述

### 图 *G=（V， E)* 

+ *V* 是图中节点的集合
+ *E* 为边的集合
+ *N*为节点的个数

### 三个重要矩阵

+ 特征矩阵 *X*：维度为 $N\times D$，表示图中有N个节点，每个节点的特征个数是D
+ 邻居矩阵 *A*：维度为 $N \times N$ ，表示图中N个节点之间的连接关系
+ 度矩阵 *D*：维度为 $N\times N$ ，是一个对角矩阵，即只有对角线上不为零，其他位置元素都是 0 ，表示图中N个节点与其他节点相连的边的个数。对于无权图而言，$D_{ii} = \sum_{j}{A_{ij}}$
+ 邻接矩阵
+ ![1](/Users/lixuanhong/Desktop/boohee_projects/base_KG/KGC/images/1.png)

+ 说明：在标记图中，<font color=red>顶点1</font>与<font color=red>顶点2</font>和<font color=red>顶点5</font>连接，于是在邻接矩阵中，<font color=red>只有第二个和第五个位置的值为1</font>，其余位置为0，于是在度矩阵中第一个顶点的位置为11，且这个位置上的值为2，因为顶点1只与两个顶点有关
+ 不同于image数据，graph数据形状不规则，不具有平移不变性。对于GCN，其目标就是<font color=red>设计一个特征提取器，进而完成节点分类，边预测等任务，还可以得到每个节点的embedding</font>

### 图的特征提取思路一

+ ![2](/Users/lixuanhong/Desktop/boohee_projects/base_KG/KGC/images/2.png)

+ 对于节点 $i$，我们可以用其邻接节点加权求和的结果来表示当前节点，这个操作我们称为“聚合(aggregate)”：

  $agg(X_i) = \sum_{j \in neighbor(i)} {A_{ij}X_j}$

+ 考虑到与节点 $i$ 没有边连接的节点 $j$ ，对应的权重 $A_{ij}$ 为 0 ，因此上面的公式又可以改写为：

  $agg(X_i) = \sum_{j \in N} {A_{ij}X_j}$

+ 对于所有节点，聚合结果为

  $agg(X) = AX$

+ 上面主要考虑的是节点 $i$ 与邻居节点的关系，往往节点 $i$ 本身也含有关键信息，需要将自身特征添加进来

  $agg(X_i) = \sum_{j \in N} {A_{ij}X_j + X_i}$

  $agg(X) = AX + X = (A + I) X$， 其中 $I$ 是单位矩阵

+ 令 $\widetilde A=A+I$，于是

  $agg(X) = \widetilde AX$

### 图的特征提取思路二

+ 在某些情况下，我们更关注节点之间的差值，因此可以对差值进行加权求和：

  $agg(X_i) = \sum_{j\in N}{A_{ij}(X_i-X_j)} = D_{ii}X_i - \sum_{j\in N}A_{ij}X_j$，其中$D_{ii} = \sum_{j}{A_{ij}}$

+ 对于整个图的节点而言，上面的公式矩阵化为：

  $agg(X) = DX-AX = (D-A)X$

+ 上面的公式中的 $D-A$ 是拉普拉斯矩阵，用 $L$ 表示

  $agg(X) = LX$

### 归一化

+ 无论是思路一的 $ \widetilde A$ 还是思路二的 *L*，<font color=red>与CNN的卷积相似之处都是局部数据的聚合操作，只不过CNN 中卷积的局部连接数是固定的</font>。<font color=red>但是在Graph中每个节点的邻居个数都可能不同</font>，进行聚合操作后，对于度较大的节点，得到的特征比较大，度较少的节点得到的特征就比较小，因此还需要进行归一化的处理。
+ 算术平均：$L^{rw}=D^{-1}L$
+ 几何平均：$L^{sym}=D^{-\frac {1}{2}}LD^{-\frac {1}{2}}$，几何平均值受极端值影响较小，因此是GCN中比较常用的归一化方法
+ 对于思路一：$agg(X)=L^{sym}X=D^{-\frac {1}{2}}LD^{-\frac {1}{2}}X=D^{-\frac {1}{2}}(D-A)D^{-\frac {1}{2}}X$

+ 对于思路二：$agg(X)=D^{-\frac {1}{2}}\widetilde AD^{-\frac {1}{2}}X=D^{-\frac {1}{2}}(A+I)D^{-\frac {1}{2}}X$

+ 在GCN中，应对聚合结果进行变换，于是从 $l$ 层到 $l+1$ 层的传播方式为：

  $H^{(l+1)}=\sigma (\widetilde D^{-\frac {1}{2}}\widetilde A \widetilde D^{-\frac {1}{2}}H^{(l)}W^{(l)})$

+ 其中

  + $\widetilde A = A + I$，或者 $\widetilde A = D - A$

  + $\widetilde D$是$\widetilde A$的度矩阵，每个元素为：$\widetilde D_{ii} = \sum_{j}{\widetilde A_{ij}}$

  + $H$ 是每一层的特征，对于输入层而言，$H$ 就是 $X$

  + $\sigma$ 是 sigmoid 函数 
  + 由于 D 是在矩阵 A 的基础上得到的，因此在给定矩阵 A 之后，$\widetilde D^{-\frac {1}{2}}\widetilde A \widetilde D^{-\frac {1}{2}}$ 就可以事先计算好

## 总结

+ 优点：捕捉graph的全部信息，很好地表示node特征；
+ 缺点：直推式的学习方式，模型学习的权重W与图的邻接矩阵A和度矩阵D息息相关，一旦图的结构发生变化，那么A与D也就变化了，模型就得重新训练；需要把所有节点都参与训练才能得到node embedding，当图的节点很多，图的结构很复杂时，训练成本非常高，难以快速适应图结构的变化。

## 实验

+ 数据处理
  + 

# RGCN

***

## 综述

+ 论文：modeling relational data with graph convolutional networks
+ 超强资源：https://docs.dgl.ai/guide_cn/index.html
+ 能够解决的问题：
  + Link completion（recovery of missing facts）
  + Entity classification（recovery of missing entity attributes）
+ 介绍：
  + 再大规模的知识库也存在缺失的情况，缺失情况也会影响下游任务
  + 预测知识库中的缺失信息是统计关系学习（statistical relational learning，以下简称 <font color=red>SRL</font>）的主要内容
+ 模型组成：
  + 编码器：数据embedding的过程
  + 解码器：
  + 前向传播模型：$h_i^{l+1} = \sigma(\sum_{r\in R}\sum_{j\in N_i^T} \frac {1}{c_{i,r}} W_r^lh_j^l+W_0^lh_i^l)$
    + $h_i^l$ 表示隐藏层$l$ 的节点 $v_i$ , $N_i^r$ 表示节点 $i$ 在关系$r$ 下的邻居节点的集合，$c_{i,r}$ 是一个标准化常量，可以学习
    + R-GCN 的每层节点特征都是由上一层<font color=red>节点特征和节点的关系（边）得到</font>
    + R-GCN 对节点的<font color=red>邻居节点特征和自身特征进行加权求和</font>得到新的特征
    + R-GCN 为了保留节点自身的信息，会考虑自环。
  + 与 GCN 的区别：
    + <font color=red>R-GCN 会考虑边的类型和方向，某些边是不会被选择作为聚合依据的</font>
    + 利用稀疏矩阵乘法实现前向传播，将多层堆叠起来，以便跨多个关系步骤实现依赖关系
    + 针对某一被更新节点，<font color=red>叠加与其有关的 rel_1_in + rel_1_out + ... + rel_n_in + rel_n_out + self-loop</font>
  + 正则化方法：
    + <font color=red> 基函数分解（basis decomposition）</font>：$W_r^l = \sum_{b=1}^B a_{rb}^l V_b^l$
      + $r$ 只与系数 $a_{rb}^l$ 有关
      + 基函数分解可以看作是不同关系类型之间权重共享的一种方式
    + <font color=red> 块分解（block diagonal decomposition）</font>：$W_r^l = ⨁_{b=1}^B Q_{br}^l = diag(Q_{1r}^l,...,Q_{Br}^l)$ 
      + $W_r^l$ 为对角矩阵
      + 而块分解可以看作是对每个关系类型的权值矩阵的稀疏约束，其核心在于潜在的特征可以被分解成一组变量，这些变量在组内的耦合比在组间的耦合更紧密
    + 减少参数数量，同时，参数化也可以缓解对稀有关系的过度拟合，因为稀有关系和常见关系之间共享参数更新。
  + 链路预测（link prediction）：
    + 预测一个三元组（subject, relation, object），通过一个打分函数 $f(s,r,o)$ 来判断 $(s,r,o)$ 是否符合要求
    + 考虑使用DistMult分解作为评分函数，每个关系 r 都和一个对角矩阵有关：$f(s,r,o)=e_s^TR_re_o$
    + 考虑负采样的训练方式，对于观测样本，考虑 $\omega$ 个负样本，并利用交叉熵损失进行优化：
      + $loss = - \frac {1}{(1+\omega)\vert E \vert}\sum_{(s,r,o,y) \in T}ylog\sigma(f(s,r,o))+(1-y)log(1-\sigma(f(s,r,o)))$
  + 节点分类（entity classfication）：
    + 数据准备
      + 
  + 模型结构：
    + <font color=red>Input -- R-GCN（encoder）-- DistMult（decoder）-- Edge loss</font>
    + $w_r$ -- self.weight -- shape=(num_bases, in_feat, out_feat)
    + $a_{rb}$ -- self.w_comp -- shape=(num_rels, num_bases)
    +  -- self.h_bias -- shape=(out_feat)
    +  -- self.loop_weight -- shape=(in_feat, out_feat)

