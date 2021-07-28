# ESIM 

用于文本匹配，效果较好，且速度较快

https://github.com/lukecq1231/nli

https://zhuanlan.zhihu.com/p/372484444

https://github.com/hsiaoyetgun/esim

https://github.com/terrifyzhao/text_matching/tree/master/esim （tracking）

https://github.com/wangle1218/deep_text_matching (updating)

## 模型架构：

![68747470733a2f2f70696373666f726461626c6f672e6f73732d636e2d6265696a696e672e616c6979756e63732e636f6d2f323032302d31322d31372d3033323635342e706e67](/Users/lixuanhong/Desktop/Materials/NLP/ESIM/68747470733a2f2f70696373666f726461626c6f672e6f73732d636e2d6265696a696e672e616c6979756e63732e636f6d2f323032302d31322d31372d3033323635342e706e67.png)

1. Input Encoding: BiLSTM

   + 假设当前有两个句子：$a=(a_{1},a_{2},...,a_{l_{a}})$，$b=(b_{1},b_{2},...,b_{l_{b}})$，0表示两句不相似，1表示相似

   +  tokens 的 embedding 输入进 BiLSTM

   + 两个句子输入进同一个 BiLSTM，而不是两个

     $\bar a_i = BiLSTM(a, i), \forall i \in [1, ..., l_a]$

     $\bar b_j = BiLSTM(b, j), \forall j \in [1, ..., l_b]$

     $ \bar{a_{i}} $是句子 `a` 在 `i` 时刻的是输出，$\bar{b_{j}}$是句子 `b` 在 `j` 时刻的输出

2. Local Inference Modeling

   + 使用这个公式计算两个单词之间的交互：$e_{ij} = \bar a_i^T \bar b_j$

   + 比如隐层维度256，那么$ \bar{a_{i}}=[1,256] $，$ \bar{b_{j}}=[1,256] $。那么相乘之后，就是维度[1,1]的值；

   + 假设句子a长度为10，句子b长度为20，那么最后产生的矩阵是是 [10, 20]，用来描述两个句子中单词之间的交互，根据下面的公式来对这个矩阵进行操作

     $\widetilde a_{i} = \sum_{j=1}^{l_b} \frac {exp(e_{ij})} {\sum_{k=1}^{l_b} exp(e_{ik})} \bar b_j \forall i \in [1, ..., l_a]$

     $\widetilde b_{j} = \sum_{i=1}^{l_a} \frac {exp(e_{ij})} {\sum_{k=1}^{l_a} exp(e_{kj})} \bar a_j \forall j \in [1, ..., l_b]$

     对第一个式子来说，对生成的 [10, 20] 矩阵每一行做softmax，然后乘以 $\bar b_{j}$，得到 $\widetilde a_{i}$；也就是说对矩阵做了按照行的相似度（<font color=red>实际上是句子a中的词和句子b中的词的相似度</font>），然后每个词的相似度分别乘以每个词的embedding得到加权和，得到 $\widetilde a_{i}$，这样就完成了句子间的交互模型

   + 最后对特征进行拼接，包括对位相减和对位相乘；<font color=red>注意这里的向量拼接，在一定程度上模仿了残差网络的思路，将上一层输出当前层的输出进行拼接，起到了扩增特征和加快训练的目的</font>

     $m_a = [\bar a, \widetilde a, \bar a-\widetilde a, \bar a \odot \widetilde a]$

     $m_b = [\bar b, \widetilde b, \bar b-\widetilde b, \bar b \odot \widetilde b]$

3. Inference Composition: BiLSTM

   + 上一层获取的 $m_a, m_b$ 作为输入送入 这层的 BiLSTM，把输出做最大池化和平均池化，然后进行特征拼接

     $V_{a,ave} = \sum_{i=1}^{l_a} \frac {V_{a,i}}{l_a}$, $V_{a,max} = max_{i=1}^{l_a} V_{a,i}$

     $V_{b,ave} = \sum_{j=1}^{l_b} \frac {V_{b,j}}{l_b}$, $V_{b,max} = max_{j=1}^{l_b} V_{b,j}$

     $V = [V_{a,ave}, V_{a,max}, V_{b,ave}, V_{b,max}]$

4. Prediction：全连接

   + 上一层获取的 $V$ 作为全连接的输入，接softmax

Premise 前提

Hypothesis 假设

NLI： Natural Language Inference



7.5

1. 问题训练准确率不提升的可能原因：
   1. 使用到的损失函数：
      + 结论：<font color=red> 损失函数的选择应该是正确的，问题不在损失函数</font>

   ```python
   """
   从形式上说，损失函数主要由以下两种方式组成
   """
   
   # RGCN
   tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels_batch, logits))
   
   # RE
   tf.keras.losses.sparse_categorical_crossentropy(y_true=label_batch, y_pred=logits)
   # 注意：真实的标签值y_true可以直接传入int类型的标签类别
   # 并且logits的结果是不经过softmax处理的原始结果
   # 知乎上查到，这个损失函数用于文本分类
   # The shape of y_true is [batch_size] and the shape of y_pred is [batch_size, num_classes] 这个要求是符合预期的
   # 从逻辑上说，采用这种方法是不应该有问题的
   ```

   2. 标签数据

      + 因为每次返回的准确率的改变值只存在于千分位，所以有理由怀疑预测结果与标签结果是否不变
      + 检查了预测结果，发现除第一轮训练外，其余各轮的训练结果在一个batch上全部为1，也就是所有结果预测为1，因为原始数据中正负样本的比例是1:1的，所以导致预测结果一直都在50%上下浮动，当然损失函数的优化也没有起到作用
      + 实践：为什么会出现预测结果全部为1的情况？首先检查一下模型搭建的过程是否出错
      + 模型搭建有问题。

   3. 解决方法：

      1. pooling的方向是有问题的，已经在模型上进行修改
      2. 需要对为0的位置进行mask，防止0位置影响模型判断，这部分没改，主要由于下面的问题
      3. 当前我的模型输入的是根据index找到了预训练好的词向量，但是需要对位置为0的进行mask操作的对象是文本的字序列，而不是字表示的向量；所以在原作者模型中，词向量是训练出来的，也就是说传入模型的是文本序列，根据文本序列，生成mask结果，然后再使用embedding_lookup的方法将文本序列转化为词向量，后续接着做就行了

      参考：https://github.com/terrifyzhao/text_matching/commit/002758625b16366fc4985c8dd6fddfb4ccfcf9ff



7.6

+ 截止当天，在y_pred的预测结果中，可以看到在一个batch上预测的结果并不是全1或者是全0的情况，并且每次的预测结果是不一样的，基本说明模型的结构应该是没有问题的；
+ 在本地训练时，能够看到损失是在下降，<font color=red>但是下降的速度比较慢，原因可能是又可能损失并没有合理化的计算出来，或者在该损失上梯度计算有问题</font>
+ 5000条训练数据下，batch_size=500，学习率为0.002，5轮之后的训练结果还是只有50%左右，没有明显的提升
+ 可优化的方向：
  + 将全部的参数设置修改为原作者使用的具体参数

  + 当前采取的方法是从头训练一个词向量的方法，最好能在model中直接将索引转化为已经训练好的词向量 <font color=red> 重点 </font>
    + 不使用字符向量，而使用词向量
    + 从load_all_data中返回的是p_w_index, h_w_index, label，（已经处理好的 p_w_index, h_w_index 已经能够保证最大文本长度是相同的）
    + 词表大小为7284
    + 通过idx2word，将每个索引转化为词，然后将这个词转化为词向量，将一个文本中的所有索引全部转化为词向量。（或者直接将w2v模型展开为（vocab_size, embedding_size）的矩阵，然后让索引进行embedding_lookup即可，这种方法不可取，对于「UNK」类的特殊字符不知道该怎么展开为向量）
    + 训练好的词向量的维度都是100
    + <font color=red>注意：</font>如果采取这种情况，测试数据进来后，也需要将文本转化为词向量，尤其要注意如果文本中存在不在既定词表中的情况
    + <font color=red> 7.8更新 </font> ：将词向量全部转为word2vec，即训练好的词向量，但是结果依旧不好；
      + 具体表现为：在每个batch上的loss能够逐步下降，但是下降速度较慢，准确率同样，在提升但是变化不明显，==参数：lr=0.0005，全连接层只有一层，训练数据500，batch_size=50==
      + 虽然每个batch的预测结果都不一样，且没有出现全1或是全0的情况，但是准确率基本保持在50%上下，不会有太大的改变
      + 如果加入两层full-connection，那么会在某一轮训练结果中全部预测为1，==所以只要保留最后一层full-connection就行==
      + 更新lr=0.001，训练数据1000，batch_size=100
    
  + 添加precision和recall，失败，怎么加都不对，就先不加了

  + 修改模型结构，根据别人优化的方法，包括mask方法和pooling优化 <font color=red> 已修改 </font>

  + 检查文本数据和标签是不是对不上： <font color=red> 已检查，能对上 </font>

  + 是否由于文本长度设定太大，一般也就是不到10个词，当前设定15个词

  + 将已经训练好的词向量作为词的初始化向量，也就是可以作为参数与模型一同训练

    + 使用tf.Variable将预训练好的词向量作为变量的初始化，然后每次传入的idx通过embedding_looup转化为词向量，这个词向量在不同的训练阶段会同步保持训练

    + 结果：目前来看，5000条训练数据，batch_size=50，lr=0.001，<font color=red>损失有明显下降</font>，准确率变化不明显

      ![截屏2021-07-08 下午7.11.45](/Users/lixuanhong/Library/Application Support/typora-user-images/截屏2021-07-08 下午7.11.45.png)

  + 猜测：这个结果，基本可以断定现在的损失函数基本是没有问题的，问题有可能出在评估函数上。<font color=red> 评估函数确实有问题，目前手动计算准确率 </font>

  + 最大文本长度原来为15，表示这个文本中最多只容纳15个词，实际上大多数分词结束后，一句话的词很少，远达不到15个，使得多数位置为0。<font color=red> 解决方法：减小最大文本长度，目前设置为10 </font>

  + 可能梯度的更新没有加入到各个变量中

  + 需要看一下loss输出的结果的维度
  
  + 可以尝试一下，将词向量转化为模型的参数，作为整个模型的第一层（即embedding层，在实体识别模型中曾经使用过的方法）
  
    + 已尝试：结果并没有变化
  
  + w2v.vec 究竟有没有使用
  
    + 作者代码中并没有使用训练好的word2vec向量，而是直接使用词的索引，在建模的部分，创建了一个变量，即形状为（vocab_size，char_embedding_size）的词向量矩阵，使用词向量矩阵和embedding_lookup 将传入的数据（字索引）映射为向量，即得到了输入数据的词向量
    + 上述方法已经尝试过了，但是效果不好。
    + 同样也尝试过将训练好的词向量作为词向量矩阵的初始值，然后词向量矩阵作为一个变量在模型中继续训练（原论文作者也使用这种方法，词向量由Glove训练出来，维度为300，所有LSTM隐藏层的维度都是300）效果依旧不好，不能保证准确率是递增的
  
  + 从知乎上看一下大家的实现方法：https://zhuanlan.zhihu.com/p/47580077
  
    > tf.nn.softmax_cross_entropy_with_logits(labels, logits, axis=-1, name=None)
    >
    > 1. 调用这个函数，传入的logits不能是softmax之后的结果
    > 2. logits和labels的形状都是[batch_size, num_classes]
    > 3. logits和labels必须具有相同的dtype
  
  + 明天来了之后，直接使用10w的数据量，其余各项参数使用deep_text_matching中的参数，完成训练进行查看
  
  + 7.16 当前结果：50000万条数据，训练5轮，batch_size为5，损失是在下降的（0.737），但是准确率不降
  
  + ![截屏2021-07-16 上午10.23.37](https://tva1.sinaimg.cn/large/008i3skNgy1gsikuue7mfj30en0ep75n.jpg)
  
    + 汇报：
      + 全面测试完毕，当前存在如下几个情况：
      + 训练速度比较慢，占用资源比较大
      + 训练数据是蚂蚁金服的数据，文本长度和词的编码方式和我们自己的问答数据有些差别
      + 单个的文本匹配可能效果不好，依次返回最相似的前3个问题，让用户去选
