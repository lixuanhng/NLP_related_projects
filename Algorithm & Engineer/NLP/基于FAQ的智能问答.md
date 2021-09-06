## 基于FAQ的智能问答

#### 参考

+ https://github.com/WenRichard/Customer-Chatbot/tree/master/xiaotian-chatbot1.0
  + ifidf 完成召回（可以使用recall@k方式进行评估）<font color=red> 完成 </font>
  + LSTM 完成精排
    + 问题：损失在下降，准确率不提升，检查了评价metric方法，是没有问题的
    + 问题：epoch=10，训练损失在下降，验证损失几乎不变或者上升，应该是过拟合了
      + 感觉应该是数据本身的问题，数据的初始化方法，数据的处理，是否存在问题
      + 损失函数应该怎么去修改，使用rank loss
      + 最大词长度是否有必要设置大一些 <font color=red> 完成 </font>
+ 尝试转化为tf2，下周就开始使用这个编写模型
+ 模型中没有提供的包括负样本的构建（是否可以直接使用蚂蚁金服的数据）
  + 负样本的快速选择
  + 输入的数据是q-d对儿，我们认为其中的d都是正样本，那么，当选定一个q时，除了它本来对应的d之外，这个batch中所有d都可以作为当前这个q的负样本
+ 精排的损失函数：
  + binary cross entropy，其实就是二分类（尝试之后感觉不对）
  + pair-wise 的rank loss



### 基于相关度的标注

+ 人工标注相关度等级
+ 人工标注 pairwise preference，即一个 doc 是否相对另一个 doc 与该 query 更相关。
+ 最 costly 的方式是，人工标注 docs 与 query 的整体相关度排序。



### 排序模型评估指标

+ **MAP（Mean Average Presicion）**

  + AP （Average Presicion）

    + 针对一个Query匹配文本排序的结果

    + AP为所有真实标签为正的文本在<font color=red>正例结果中的位置</font>与<font color=red>在排序结果中的位置</font>的比值的求和平均

    + | 召回的候选子集的排序 | 是否在ground truth中 | 该位置的AP（比值） |
      | -------------------- | -------------------- | ------------------ |
      | 1                    | 1                    | 1/1 = 1            |
      | 2                    | 0                    | 0                  |
      | 3                    | 1                    | （1+1）/ 3 = 0.66  |
      | 4                    | 0                    | 0                  |
      | 5                    | 0                    | 0                  |
      | 6                    | 1                    | （2+1）/ 6 = 0.5   |
      
      最后的得分就是 （1+0.66+0.5）/ 3

  + MAP 就是对所有Query都做AP，然后求平均
    + 参考 https://makarandtapaswi.wordpress.com/2012/07/02/intuition-behind-average-precision-and-map/

  ```python
  def AP(ground_truth):
      """
      Compute the average precision (AP) of a list of ranked items
      """
      n = 6     # 一组输入的候选总数（本质上来讲，AP本身和一组样本的个数无关）
      hits = 0  # 表示 score （预测结果分值）降序排序后，从第0个到当前个的累计预测正确样本数
      Sum = 0   # 表示每个 ground_truth = 1 的位置的 precision 的累加
      ground_truth = set(ground_truth)
      for i in range(n):
          if i in ground_truth:
              hits += 1
              Sum += hits / (i + 1.0)
      return Sum / len(ground_truth) if hits > 0 else 0
      
  # 这里假设召回子集的大小为6，其中包含的正样本个数为3
  ground1 = [0, 2, 5]  # 表示前n个结果中正样本出现的位置列表
  ground2 = [0, 1, 2]
  ground3 = [0, 1, 4]
  
  res = (AP(ground1) + AP(ground2) + AP(ground3)) / 3
  print(res)
  ```

+ **NDCG**

  + **CG** 累积增益

    + 在推荐系统中，CG即将每个推荐结果相关性 (relevance) 的分值累加后作为整个推荐列表 (list) 的得分。
    + $CG_k = \sum_{i=1}^k rel_i$
    + $rel_i$ 表示处于位置 ii 的推荐结果的相关性， $k$ 表示所要考察的推荐列表的大小

  + **DCG** 折损累积增益

    + <font color=red>加入位置信息</font>，希望相关性高的结果应排在前面，对于排名靠后推荐结果的推荐效果进行【打折处理】
    + $DCG_k = \sum_{i=1}^k \frac{2^{rel_i}-1}{log_2(i+1)}$
    + 分子部分：相关性越高，推荐效果越好，DGC越大；分母部分，i 表示推荐结果的位置。i 越大，则推荐结果在推荐列表中排名越靠后，推荐效果越差，DCG越小。

  + **NDCG** 归一化折损累积增益

    + <font color=red>不同的query的匹配分数的结果应该做归一化</font>
    + IDCG，即Idea DCG，指推荐系统为某一用户返回的最好推荐结果列表， 即假设返回结果按照相关性排序， 最相关的结果放在最前面， 此序列的DCG为IDCG。
    + 因此DCG的值介于 (0, IDCG] ，故NDCG的值介于(0, 1]
    + 对于query u的NDCG@k定义为：$NDCG_u@k = \frac{DCG_u@k}{IDCG_u}$，这里的 k 表示推荐列表的大小
    + 对于所有query，做法就是对每个query做NDCG，然后求平均

  + 可以事先确定推荐目标和推荐结果的相关性分级

    + 可以使用 0，1分别表示相关或不相关，比如此处我们用 $ref_i=δ(i∈I_u^{te})$ , 在这里如果 x 为true, 则δ(x)=1，否则δ(x)=0；或是使用 0~5 分别表示严重不相关到非常相关, 也即相当于确定了rel 值的范围
    + 之后对于每一个推荐目标的返回结果给定 rel 值，然后使用 DCG 的计算公式计算出返回结果的DCG值。使用根据排序后的 rel 值序列计算IDCG值， 即可计算NDCG.

  + 测试代码如下

    ```python
    import math
    
    def DCG(rank_list, ideal_rank):
        # 计算每一次的排序结果
        DCG_res = 0
        n = len(rank_list)
        for i in range(n):
            score1 = (2**rank_list[i] - 1) / math.log((i+1)+1, 2)
            DCG_res += score1
        # 计算最优的排序结果：IDCG
        IDCG = 0
        for j in range(n):
            score2 = (2**ideal_rank[j] - 1) / math.log((j+1)+1, 2)
            IDCG += score2
        NDCG = DCG_res / IDCG
        return NDCG
        
        
    # 假设召回的候选子集大小为6
    # 对于一个用户，排序的结果为：6个结果的排序结果为 ranks
    ideal_rank = [1, 1, 1, 0, 0, 0]
    ranks1 = [1, 0, 1, 0, 0, 1]
    ranks2 = [1, 0, 0, 0, 0, 1]
    ranks3 = [0, 1, 1, 1, 0, 0]
    
    res1 = DCG(ranks1, ideal_rank)
    res2 = DCG(ranks2, ideal_rank)
    res3 = DCG(ranks3, ideal_rank)
    # 在三次Query中取平均，获得模型效果
    ave = (res1 + res2 + res3) / 3
    ave
    ```

    

### 常见的Learning2Rank方法

+ **point-wise 类**

  + <font color=red>含义</font>：
    + 将排序问题转化为分类问题或者回归问题。考虑**单一文档**作为训练数据，**不考虑文档间的关系**。以分类问题来说，即对于查询 $q$ ,文档集 $D={d_1, d_2,...,d_n}$ 这样的训练样例 $n$ 个，对于二分类问题学习目标可以分为相关 (y=1) 和不相关 (y=0)，然后训练模型，对未知的样本做相关性预测。
    + 输入数据为 【q-d】 对
    + ==如果最终预测目标是一个实数值，就是回归问题，如果目标是概率预测，就是一个分类问题，例如CTR预估。== 
  + L2R 框架特征
    + 输入空间中样本是单个 doc（和对应 query）构成的特征向量；
    + 输出空间中样本是单个 doc（和对应 query）的相关度；
    + 假设空间中样本是打分函数；
    + 损失函数评估单个 doc 的预测得分和真实得分之间差异
  + 人工标签转换到pointwise的输出空间的方法
    + 如果标注直接是相关度 $s_j$，则 doc x_j 的真实标签定义为 $y_j = s_j$
    + 如果标注是 pairwise preference $s_{u,v}$，则 doc $x_j$ 的真实标签可以利用该 doc 击败了其他 docs 的频次
    + 如果标注是整体排序 π，则 doc $x_j$ 的真实标签可以利用映射函数，如将 doc 的排序位置序号当作真实标签
  + 能够使用的模型
    + 基于回归的算法：输出空间包含的是实值相关度得分，采用回归模型
    + **基于分类的算法**：输出空间包含的是无序类别，对于二分类，SVM、LR 等均可；对于多分类，提升树等均可。
    + 基于有序回归的方法：输出空间包含的是有序类别，通常是找到一个打分函数，然后用一系列阈值对得分进行分割，得到有序类别
  + 缺陷：
    + ranking 追求的是排序结果，并不要求精确打分，只要有相对打分即可。
    + pointwise 类方法并没有考虑同一个 query 对应的 docs 间的内部依赖性。
    + 当不同 query 对应不同数量的 docs 时，整体 loss 将会被对应 docs 数量大的 query 组所支配，前面说过应该每组 query 都是等价的。
    + 损失函数也没有 model 到预测排序中的位置信息。因此，损失函数可能无意的过多强调那些不重要的 docs，即那些排序在后面对用户体验影响小的 doc。
  + 改进：
    + 在 loss 中引入基于 query 的正则化因子的 RankCosine 方法

+ pair-wise 类

  + <font color=red>含义</font>：
    + **Pairwise**是目前比较流行的方法，相对**pointwise**他将重点转向**文档顺序关系**。
    + ==输入数据为 【q，d+，d-】 对，学习一对有序对==
    + 对于同一query的相关文档集中，对任何两个不同**label**的文档，都可以得到一个训练实例 ![[公式]](https://www.zhihu.com/equation?tex=%28d_%7Bi%7D%2Cd_%7Bj%7D%29) ，如果 ![[公式]](https://www.zhihu.com/equation?tex=d_%7Bi%7D%3Ed_%7Bj%7D) ，则赋值+1，反之为-1。于是我们就得到了二元分类器训练所需的训练样本。**预测时可以得到所有文档的一个偏序关系**，**从而实现排序**。
  + L2R 框架特征
    + 输入空间中样本是（同一 query 对应的）两个 doc（和对应 query）构成的两个特征向量
    + <font color=red>输出空间中样本是 pairwise preference，也就是q-d（+或-）的打分结果</font>
    + 假设空间中样本是二变量函数；
    + 损失函数评估 doc pair 的预测 preference 和真实 preference 之间差异。
  + 人工标签转换到pointwise的输出空间的方法
    + 如果标注直接是相关度 $s_j$，则 doc pair $(x_u,x_v)$ 的真实标签定义为 $y_{u,v}=2*I_{s_u>s_v}-1$
    + 如果标注是 pairwise preference $s_{u,v}$，则 doc pair $(x_u,x_v)$ 的真实标签定义为 $y_{u,v}=s_{u,v}$
    + 如果标注是整体排序 π，则 doc pair $(x_u,x_v)$ 的真实标签定义为 $y_{u,v}=2*I_{π_u,π_v}-1$
  + 可使用的模型
    + 二分类算法
    + 基于 NN 的 SortNet，基于 NN 的 RankNet，基于 fidelity loss 的 FRank，基于 AdaBoost 的 RankBoost，基于 SVM 的 RankingSVM，基于提升树的 GBRank
  + 缺陷
    + 如果人工标注给定的是第一种和第三种，即已包含多有序类别，那么转化成 pairwise preference 时必定会损失掉一些更细粒度的相关度标注信息。
    + doc pair 的数量将是 doc 数量的二次，从而 pointwise 类方法就存在的 query 间 doc 数量的不平衡性将在 pairwise 类方法中进一步放大。
    + pairwise 类方法相对 pointwise 类方法对噪声标注更敏感，即一个错误标注会引起多个 doc pair 标注错误
    + pairwise 类方法仅考虑了 doc pair 的相对位置，损失函数还是没有 model 到预测排序中的位置信息。
    + pairwise 类方法也没有考虑同一个 query 对应的 doc pair 间的内部依赖性，即输入空间内的样本并不是 IID 的，违反了 ML 的基本假设，并且也没有充分利用这种样本间的结构性。
  + 改进：
    + Multiple hyperplane ranker，主要针对前述第一个缺陷
    + magnitude-preserving ranking，主要针对前述第一个缺陷
    + IRSVM，主要针对前述第二个缺陷
    + 采用 Sigmoid 进行改进的 pairwise 方法，主要针对前述第三个缺陷
    + P-norm push，主要针对前述第四个缺陷
    + Ordered weighted average ranking，主要针对前述第四个缺陷
    + LambdaRank，主要针对前述第四个缺陷
    + Sparse ranker，主要针对前述第四个缺陷

+ list-wise 类

  + 含义：

    + 学习一个有序序列的样本特征
  
  + L2R 框架特征
  
    + 输入空间中样本是（同一 query 对应的）所有 doc（与对应的 query）构成的多个特征向量（列表）；
    + 输出空间中样本是这些 doc（和对应 query）的相关度排序列表或者排列；
    + 假设空间中样本是多变量函数，对于 docs 得到其排列，实践中，<font color=red>通常是一个打分函数</font>，根据打分函数对所有docs的排列打分，得到的打分结果对docs序列进行排序；
    + 损失函数分成两类，一类是直接和评价指标相关的，还有一类不是直接相关的。
  
  + 人工标签转换到pointwise的输出空间的方法

    + 如果标注直接是相关度 $s_j$，则 doc set 的真实标签可以利用相关度 $s_j$ 进行比较构造出排列
    + 如果标注是 pairwise preference $s_{u,v}$，则 doc set 的真实标签也可以利用所有 $s_{u,v}$  进行比较构造出排列
    + 如果标注是整体排序 π，则 doc set 则可以直接得到真实标签
  
  + 直接基于评价指标的算法
  
    + 直接取优化 ranking 的评价指标，也算是 listwise 中最直观的方法。但这并不简单，因为前面说过评价指标都是离散不可微的
    + 有时虽然使用信息量更少的指标来评估模型，但仍然可以使用更富信息量的指标来作为 loss 进行模型训练

  + 间接基于评价指标的算法

    + 设计能衡量模型输出与真实排列之间差异的 loss，如此获得的模型在评价指标上也能获得不错的性能。

  + 缺陷：

    + listwise 类相较 pointwise、pairwise 对 ranking 的 model 更自然，解决了 ranking 应该基于 query 和 position 问题。

    + listwise 类存在的主要缺陷是：一些 ranking 算法需要基于排列来计算 loss，从而使得训练复杂度较高
  
    + 位置信息并没有在 loss 中得到充分利用，可以考虑在 ListNet 和 ListMLE 的 loss 中引入位置折扣因子。
  
      



### 召回

#### 背景：

+ 本质是一个信息检索的问题
+ 可以简单划分为：召回+精排
+ 召回目标：是从知识库中快速的召回一小批与query相关的候选集。所以召回模型的评价方法，主要侧重于 **响应时间** 和 **recall@n的召回率** 两个方面。
+ 召回模型的迭代：ES字面召回，到ES字面召回和向量召回的双路召回

#### 基于ES的简单召回

+ 实质是基于BM25的召回
+ BM25原理
  + 
+ 缺点
  + 没有语义上的召回，比如知识库内的问题：可以免运费吗？query是：包邮吗？

#### 基于语义的召回

+ 实质是基于embedding的召回

+ 首先训练sentence embedding模型，然后将知识库中的问题都预先计算出embedding向量。在线上预测阶段，对于每个query同样先计算出embedding，再到知识库中检索出相近的embedding所属的问题。

+ 参考论文：facebook最新的论文: Embedding-based Retrieval in Facebook Search

  + 我们采用的基本结构是albert获取embedding，然后通过pair-loss function进行fine-tuning。

  ![Embedding-based Retrieval in Facebook Search](https://tva1.sinaimg.cn/large/008i3skNgy1gt6v4z86fgj317i0e6jsm.jpg)

  + 这里的 loss function： $L = max(0, D(q, d+)-D(q, d-) + m)$，其中 $D(q, d) = 1 - cos(q, d) \in [0, 1]$
    + 损失函数：rank loss，预测输入之间的相对距离，也就是度量学习
    + 只需要得到数据点之间的相似性得分就可以使用它们
      + 这个度量可以是二值的，相似或不相似
      + 也是可以连续的，也就是余弦相似度
    + 样本选择：
      + Easy Triplets: 相对于嵌入空间中的正样本，负样本已经足够远离锚定样本。损失是0并且网络参数不会更新。
      + Hard Triplets: 负样本比正样本更接近锚点，损失是正的。
      + Semi-Hard Triplets:负样本比正样本离锚的距离远，但距离不大于margin，所以损失仍然是正的。
    + 这里margin为超参数，对于“简单”的样本，正样本和负样本之间的距离大于margin，L即为0。所以margin的存在，让模型更加关注比较“难”的样本。
  
+ **召回模型的评估方法**：

  + 响应时间
  + top@n：n个样本中正例样本所占的比例（<font color=red>正例是需要标注的，但是基于TF-IDF的算法是无监督的，不需要标注</font>）



#### recall@k

+ 模型对候选的reponse排序后，前k个候选中存在正例数据（正确的那个）的占比；
+ k值越大，该指标会越高，对模型性能的要求越松





### 排序

+ 排序模型
  + 使用Albert，对每一个q-d对文本进行打分，得到每个文本对的分值
  + 其中每个q对应一个d，将二者以[sep]符号拼接起来
  + 通过Albert获取词向量，在整个序列上做pooling获得句子向量，进而得到句子的分值
+ 损失函数
  + lambda loss，来自 list-wise
  + https://blog.csdn.net/baroque123/article/details/87887401?utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-2.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-2.control
  + $L = \sum_{r(i) > r(j)}|\Delta NDCG|log_2(1+ e^{-\sigma * (S_i - S_j)})$
    + $r(i) > r(j) => i > j$，打分越高，排序的位置应该越靠前
    + $|\Delta NDCG|$ 表示交换位置 i， j 前后NGCD的差值，表明排错的代价程度
    + $log_2(1+ e^{-\sigma * (S_i - S_j)})$ 促使模型学习到排在前面的评分 $S_i$ 大于排在后面的评分 $S_j$
    + 该loss设计可以使得模型有效区分开 正样本(r=2)，推荐样本(r=1)，不相关的样本(r=0).