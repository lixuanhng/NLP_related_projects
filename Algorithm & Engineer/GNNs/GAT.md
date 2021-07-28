# GAT 相关

## 综述

### 每层状态

+ input：$\acute h = \{\vec h_1, \vec h_2, ..., \vec h_N\}, \vec h_i \in R^F$

+ output：$\acute h = (\acute {\vec h_1}, \acute {\vec h_2},..., \acute {\vec h_N)}, \acute {\vec h_i} \in R^F$

### 计算注意力系数

+ 计算顶点 $i$ 与周围邻居节点 $j \in N_i$ 的相似度

  $e_{ij} = a(W \vec h_i, W \vec h_j)$

+ 公式中的$(W \vec h_i, W \vec h_j)$可以看出特征 $h_i$ $h_j$ 共享了参数矩阵 *W*，都使用 *W* 对特征进行线性变换， (⋅,⋅) 在公式中表示横向拼接。

+ 而公式最外层的 *a* 表示单层前馈神经网络（使用*LeakyReLU*作为激活函数），输出为一个数值。

+ 数据归一化，使用softmax：

  $\alpha = softmax_j(e_{ij}) $ 

  $\alpha = \frac {exp(e_{ij})} {\sum_{k\in N_i} exp(e_{ik})}$

  $e_{ij} = LeakyReLU(\vec a^T [W \vec h_i || W \vec h_j])$

  $e_{ik} = LeakyReLU(\vec a^T [W \vec h_i || W \vec h_k])$

+ 其中 $||$ 表示向量拼接，以上的计算过程如下

![1](/Users/lixuanhong/Desktop/boohee_projects/base_KG/KGC/images/4.png)

### 加权求和

+ 得到注意力系数 $\alpha$ 之后，就是对邻居节点的特征进行加权求和：

  $\acute {\vec h_1} = \sigma(\sum_{j\in N_i} \alpha_{ij} W \vec h_j)$， 对节点 $i$，将所有邻居节点 $j$ 的特征进行加权求和

+ 为了更好的学习效果，作者使用了 “multi-head attention”，也就是使用K个注意力。对于K个注意力又可以使用两种方法对邻居节点进行聚合

+ 横向拼接（聚合的特征维度就是原来的K倍）：$\acute {\vec h_1} = \mid\mid_{k=1}^K \sigma(\sum_{j\in N_i} \alpha_{ij}^k W^k \vec h_j)$

+ 将K个注意力得到的结果取平均值：$\acute {\vec h_1} = \sigma(\frac {1}{K} \sum_{k=1}^K \sum_{j\in N_i} \alpha_{ij}^k W^k \vec h_j)$

+ ![3](/Users/lixuanhong/Desktop/boohee_projects/base_KG/KGC/images/3.png)

### 总结

+ 为邻接节点分配不同的权重，考虑到节点特征之间的相关性。
+ 不需要事先得到整个图结构或所有顶点的特征(只需访问目标节点的临接节点)。
+ 能够运用于inductive任务中。