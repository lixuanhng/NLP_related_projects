### 文本相关性在搜索，广告，推荐三大场景中的应用！

核心：通过对内容/商品的召回和排序，来优化Query-Doc的匹配结果

链接：https://mp.weixin.qq.com/s?__biz=MzAxMTk4NDkwNw==&mid=2247486703&idx=1&sn=73e22f9c349899572754726ddc7a4b3c&chksm=9bb9858bacce0c9db4e140062d4a40e899aafba1c4eb89bcc70bed57e0d2bc9e85d6b087a294&token=837504255&lang=zh_CN#rd

1. **Doc的理解**：现在的候选Doc/Item是各种模态的，比如视频、商品、图片、文本，但不管哪种形式，文本都是其中的重要一种，可以利用阅读理解、信息抽取、文本匹配打标签等技术加深对内容的理解
2. **Query的理解**：在搜索、广告中输入都是真实的Query，可以基于NLP进行意图、分词、NER等各种解析，而推荐中则是把User当作Query，这时可以把用户历史消费的内容作为用户兴趣，这件又回归到了Doc理解
3. **Query-Doc相关性**：通过用户行为、用户信息、时空信息、上述的Query和Doc理解等海量特征来找出符合用户需求的结果。搜索和广告是命题作文，其中文本层面的Query-Doc很重要，而推荐中内容信息则可以解决用户、物品冷启动问题，也起着不可或缺的作用
   + 冷启动：没有物品，没有用户，冷启动就是从0用户开始积累用户的过程。



#### Query-Doc相关性

+ DSSM

  + 论文：Learning deep structured semantic models for web search using clickthrough data

  + 目的与原因
    + 平衡搜索的关键词，和被点击的文本标题之间的相关性
    + 用DNN把Query和Document表达为低维语义向量，并通过余弦相识度来计算两个语义向量的距离，最终训练出语义相似度模型。
    + 该模型既可以用来预测两个句子的语义相似度，又可以获得某句子的低维语义Embedding向量。

  + 结构

  + ![8dc123e968e0fd19592d2724657da3d3](https://tva1.sinaimg.cn/large/008i3skNgy1gu01hma7esj60k0095js702.jpg)
    + Q代表Query信息，D表示Document信息

    + **Term Vector：**表示文本的Embedding向量，这个向量是由 bag-of-word 组成的，原始的维度很大；

    + **Word Hashing** 技术：为解决Term Vector太大问题，对bag-of-word向量降维（500k-30k），类似于 fast-text 中的子词分解

      + <font color=red>fast-text 中会取多个不同大小窗口对一个单词进行分解，比如2、3、4、5，词表是这些所有的子词构成的集合</font>
      + Word Hashing 只会取一个固定大小窗口对单词进行分解，词表是这个固定大小窗口子词的集合，比如letter-bigram，letter-trigram，比如输入词为【#good#】，被分解为【#go，goo，ood，od#】所有的子词作为词表
      + 对输入处理方法：
        + 首先将每个词进行Word Hashing分解
        + 获得每个词的表示，比如 [0,1,1,0,0,0...,0,1] , 维数为N，其中在词表中出现了的位置为1，否则为0
        + 将Query中所有的词的表示向量相加可以得到一个N维向量，**「其实就是bag-of-word表示」**（只考虑有没有出现，并不考虑出现顺序位置）
        + Doc端输入的处理也类似于上面Query端的处理

    + **Multi-layer nonlinear projection：**表示深度学习网络的隐层，就是全连接
      
      + Word Hashing 线性函数（这一层没有偏置值）
      + nonlinear projection 层是带有偏置值的线性函数，经 tanh 后得到的结果是 y
      
    + **Semantic feature** ：表示Query和Document 最终的Embedding向量

    + **Relevance measured by cosine similarity：**表示计算Query与Document之间的余弦相似度
      
      + $R(Q, D) = cosine(y_Q,y_D) = \frac{y_Q^Ty_D}{||y_Q||||y_D||}$
      
    + **Posterior probability computed by softmax：**表示通过 Softmax 函数把 Query 与<font color=red>正样本Document 的语义相似性转化为一个后验概率，就是极大似然估计</font>
      
      + $P(D^+|Q) = \frac{exp(\eth R(Q, D^+))}{\sum_{D'\in D}(exp(\eth R(Q,D')))}$
      + $\eth$ 为 softmax 的平滑因子，$D^+$ 为 Query下的正样本，$D^-$  为Query的随机采取的负样本，个数为k，D 为Query下的整体样本空间。 
      + 所以上述公式中的 softxmax 操作只需要在 $\hat D = \{D^+, D_1^-,...,D_k^-\}$ 这个集合上计算即可
      + 损失函数：为极大似然估计$L = -log \prod_{(Q, D+)}p(D^+|Q)$，就是说可以使用交叉熵计算损失
      
    + **Sampled Softmax 的两个做法**：

      + 输入数据中就已经采样好负样本，输入数据直接是正样本 + 负样本

        + 假设doc正样本的维度是 `[5, 10]`，其中batch_size = 5，向量维度等于10。负样本个数为4，则所有batch中，负样本的维度是 `[4*5, 10]` 。两个向量拼接在一起，得到 `[5+4*5, 10]=[25, 10]`的结果。

        + query向量的维度也是`[5, 10]`，使用 `tf.tile` 扩展为 `[25, 10]` ，与 doc向量做矩阵内积。同时分别处理query和doc内积开方，那么余弦相似度的分子和分母就完成了，乘以20，得到的结果作为 softmax logits 输入

        + 计算交叉熵损失即可，**因为只有一个正样本，而且其位置在第一个**，所以我们的标签one-hot编码为：`[1, 0, 0, 0, 0]`。计算交叉熵损失的时候，**只需要取第一列的概率值即可**

          ```python
          # 转化为softmax概率矩阵。
          prob = tf.nn.softmax(cos_sim)
          # 只取第一列，即正样本列概率。相当于one-hot标签为[1,0,0,0,.....,0]
          hit_prob = tf.slice(prob, [0, 0], [-1, 1])
          loss = -tf.reduce_sum(tf.log(hit_prob))
          ```

      + 输入数据batch均为正样本，负样本通过batch中其他Doc构造

        + 意思就是说，一个batch中的doc就是对应于query的，可以在batch中获取其他的doc作为当前query的负样本，可以减少计算量

    + **Sample Softmax 的思想**：

      + 如果不使用，那么就要在所有文档上计算logits，而Sampled Softmax表示从全集中采样出一个子集，在这个子集上做logits就行，并softmax归一化
      + 我们如果对每个类别`logits`加上一个与类别无关的常数，结果将不会变化。对应着softmax中的系数
      + 分母其实是一个归一化因子，分母**「与类别无关」**，因为分母中对整个类别集合进行了求和，给定输入后，分母归一化因子也就确定了。
      + <font color=red>经公式推导可以发现，在选择的一个子集上求logits时，模型对采样类别子集C中的类别分别计算logtis，得到结果需要减去log(采样函数)，之后，得到的才是送往softmax的输入。如果是随机采样，那么每个类型子集采样的概率都相同，log(采样函数)可以看作是一个常数，可忽略，所以可以用原始logtis替代采样后的logits </font>

  + 工业界DSSM双塔模型
    + ![双塔](https://tva1.sinaimg.cn/large/008i3skNgy1gtugvw03rqj60e70bpt9902.jpg)
    + X 为（用户，上下文）的特征， Y 为（物品)的特征；
    + $u(x)$ 表示（用户，上下文）最终的 Embedding 向量表示， $v(y)$ 表示（物品)最终的 Embedding 向量表示；
    + <$u(x)$, $v(y)$> 表示（用户，上下文）和（物品）的余弦相似度。

  + **传统DSSM局限**

    + 没有考虑到Query-Doc的交互信息，没有上下文的信息
    + 如果只是简单的交互，即Query-Doc之间词与词的交互，那么计算量过大，难以在线运行
    + **改进1: 弱交互**：在计算完毕Query和Doc表达之后，再进行交互，而是每层之间的词与词都做交互
      + 交互方式1-- ColBERT：$S_{q,d} = \sum_{i\in|E_q|}max_{j\in E_d}E_{q_{i}}E^T_{d_{j}}$  向量内积作为余弦相似度
        + <font color=red>Efficient and Effective Passage Search via Contextualized Late Interaction over BERT</font>
      + 交互方式2-- Poly-encoder：使用 Attention 进行交互
        + Query Encoder 中的隐层表示：学习m个上下文向量，作为Attention中的Q向量，每层的隐层表示为K和V向量，计算Attention得到隐层向量。其中m个向量随机初始化，然后在finetuning阶段继续训练
        + Q-D 交互表示：上面计算得到的隐层作为K和V向量，Doc向量作为Q向量，通过Attention计算得到最终的隐层表示
        + $E_{q}$ 与 $E_{d}$ 内积为score
        + <font color=red> Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring </font>
    + **改进2:使用bert时设计合适的预训练任务**：
      + Pre-train任务应该与下游任务相关
        + 同一段落之中的局部窗口两句话之间的语义关联（Inverse Cloze Task）
          + 一个段落中包含n个句子，随机采样一个句子作为Query，剩下的句子集合作为Doc句子
        + 一个Document中的全局一致的语义信息（归纳理解）（Body First Selection）
          + q 是总Wikipage 第一段中随机选出来的句子，这个句子通常是整个页面的描述或者总结。 d 则是同一页面中随机选取的一个passage
        + 两个Document之间的语义信息关联（Wiki Link Prediction）
          +  q 是总Wikipage 第一段中随机选出来的句子， d 是另外一个页面中的passage，而该句子中有超链接到 q 页面
      + Pre-train的数据应当容易获取，不需要额外的人工标注
      + <font color=red> Pre-training Tasks for Embedding-based Large-scale Retrieval </font>

+ C-DSSM

  + 将全连接层替换成卷积层
  + ![C-DSSM](https://tva1.sinaimg.cn/large/008i3skNgy1gtuh1eydnij60re0ex3zy02.jpg)

+ 直接使用DSSM：pair-wise loss会导致Q-D预测结果偏向正太分布
  + 使用二分模型，改为拟合伯努利分布，使预测结果成哑铃的形状，降低阈值对截断的影响。
  + 从 word level 开始进行信息交互（在通过BiLSTM之后），加入了Q-D的attention交互计算，获得更好的向量表示
  
+ 文本相关性融入到召回特征中
  + 分别给Query和商品建立encoder，除了二者的文本信息外，还要融合其他特征
  + 左侧的Query编码器对拼接和集成的向量采用了multi-head机制，用于获取不同的语义
  + ![WechatIMG66](https://tva1.sinaimg.cn/large/008i3skNgy1gtvczmxidoj60so0esaav02.jpg)
  
+ （阿里飞猪）更关注Doc理解
  + 提前对商品的目的地和类目id进行了编码，作为Doc侧的输入，更好的学习目的地和类目相关性
  + query和商品title输入的都是character字符，避免了分词不一致的问题
  + 特征融合没有使用concat，而是tensor fusion
  + 损失函数：Large margin loss（rank loss），直接建模正负样本距离，召回评价方法是AUC
    + $loss=max(0 ,margin - cosine_+ + cosine_-) $
    + 

+ 使用bert
  + 二轮精排中用交互式的小bert进行相关性计算
  
+ 召回策略
  + 构造训练数据中的负样本
  + 把用户点击的数据当作正例是无可后非的，**但一个很常见的错误就是把召回结果中未点击的内容作为负例**。
  + 在召回和排序这两个步骤中我们的视角是不一样的，召回阶段的视角是内容全集，同时未点击也不意味着就不是好结果，所以**更好的做法是在全局范围内进行随机负采样**
  
+ 随机负例也有问题
  + 可能会把相关商品变成负例，降低结果与Query的相关性。
  + **京东电商搜索**采用的方案是按一定比例将随机负例与batch negatives相融合，因为<font color=red>batch内的doc都有对应的batch query作为正例，打散的话更大概率与当前正例不相关，</font>之后再通过控制参数来平衡召回结果的多样性与相关性。
  + **知乎搜索**还提出了一种hard负例的挖掘方法，就是将query和doc映射到一个空间中，再对doc进行聚类，这样就可以找到与query相近的类别，从其中采样一些较难的负例。
  + 参考**阿里飞猪**的做法，在一个类目或者目的地下随机选择负例，提升训练数据的难度。
  
+ 更高效的问题样本发现机制
  + **阿里文娱**基于Q-Learning的思想，对于线上预测较差的样本，再用一层模型进行挖掘，把低置信度的样本、或者高置信度但和线上预测不一致的样本拿去给人工标注，这样就可以快速迭代出问题集并有针对性地优化：
  
+ 线上使用
  + 不是所有厂都用BERT，主要因为代价太大。即使Doc的表示都可以离线预测好存起来，但Query来了真心受不了。
  + 方法1：模型做小
    + **知乎搜索**：基于Roberta-large进行蒸馏，采用Patient-KD方式，将12层蒸馏到了6层。对BERT表示模型进行了维度压缩，在不影响线上效果的情况下，将768维压缩到了64维，减少了存储空间占用。
    + **阿里文娱**只用到基于表示的方案，因此他们蒸馏了一个非对称的双塔模型，并且在doc端采用multi-head方式来减少指标衰减。
  + 方法2：把数据尽可能存起来
    + **360搜索广告**在训练阶段用了16层transformer，因此在应用到线上时选择了离线的方式，先从日志中挖出一部分Q-D对，进行相关性预测，再通过CTR预测进行过滤，把最终结果存储到线上KV系统。
    + 这种方案只限于数据量可以承受的场景，也可以把两种方式进行融合，将一些高频Query提前存好，剩下的长尾Query用小模型预测。
  
+ CTR与Q-D相关性的融合
  + 百度风巢：将ctr预估引入到搜索召回阶段
  + 最单纯的想法是直接在召回阶段以ctr为目标进行训练，但由于ctr目标本身的问题，很可能将相关性不高的广告预测为正例。
  + MOBIUS提出了一个数据增强模块，先从日志里捞一批Q-D对，再用相关性模型进行打分，找出相关性低的pair作为badcase，让最终的模型预测。这样模型从二分类变成了三分类，就具备了过滤不相关case的能力，将相关性预测与ctr预测较好地融合在了一起。
  
+ 对UGC内容的利用
  + 利用用户提供的文本和图像内容训练了两个编码器，很好地把文本和图像映射到了一个空间内。
  + 利用用户发表的图文动态，训练一个双塔模型，分别对文本和图像编码，有三个loss：
    + 图文匹配
    + 文本语言模型
    + 图片内容一致性：有的动态包含多张图片，我们认为多张图片在表达一个意思



#### Query 理解

+ 地位
  + 搜索引擎中的必备模块，它的主要功能是对Query进行深入理解，保证召回的数量和最终排序精度
  + 被称之为 QU 或者 QP，有以下几个方面
    + 基础解析：包括预处理、分词、词性识别、实体识别、词权重等多个基础任务
    + Query改写：包括纠错、扩展、同义替换功能，可以进行扩召回
    + 意图识别：判断Query的具体类目，便于召回和最终排序
+ **基础解析**
  + 词法解析
    + 选择工具时，主要考虑效率和可控性，包括：
    + 分词和NER在业务数据的精度、速度
    + 粒度控制：用短语级别召回会比细粒度更准确，但短语结果不够时还是需要细粒度的结果补充
    + 自定义词典、模型迭代的支持（模型增量）
    + 新词发现：因为涉及建立倒排索引，Query和Doc需要用同一套分词。但互联网总是会出现各种新型词汇，这就需要分词模块能够发现新词，或者用更重的模型挖掘新词后加到词典里
  + 词权重
    + 比如“女士牙膏”这个Query，“牙膏”明显比“女士”要重要，即使无法召回女士牙膏类的内容，召回牙膏内容也是可以的。
    + 权重可以用分数或分类表达，在计算最终相似度结果时把term weight信息加入召回排序模型中
    + Term weighting
      + 基于统计+词表：比如根据doc统计出词的tfidf，直接存成词典就行了。
        + 问题：但这种方法无法解决OOV，**[知乎搜索](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s%3F__biz%3DMzU1NTMyOTI4Mw%3D%3D%26mid%3D2247496409%26idx%3D1%26sn%3D7b2f5984d71454e1a2812321f6018cf8%26chksm%3Dfbd740b5cca0c9a37723c8c05b4e1cf95fd8678bc54e9b4591c09a7af06f2acf79e28276a502%26scene%3D27%23wechat_redirect)**的解决方法是对ngram进行统计，不过ngram仍然无法捕获长距离依赖、不能根据上下文动态调整权重
      + 基于Embedding：针对上下文动态调整的问题，
        + **[知乎搜索](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s%3F__biz%3DMzU1NTMyOTI4Mw%3D%3D%26mid%3D2247496409%26idx%3D1%26sn%3D7b2f5984d71454e1a2812321f6018cf8%26chksm%3Dfbd740b5cca0c9a37723c8c05b4e1cf95fd8678bc54e9b4591c09a7af06f2acf79e28276a502%26scene%3D27%23wechat_redirect)**的迭代方案是用term的向量减去query整个的pooling向量来算term权重，diff越小词越重要；
        + **[腾讯搜索](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s%3F__biz%3DMzU1NTMyOTI4Mw%3D%3D%26mid%3D2247499470%26idx%3D1%26sn%3D6a6e80673353fb485a854ed2cffc5dcb%26chksm%3Dfbd74ca2cca0c5b4fb0622140eb9fa3cd15c860d06b24a4c7fc37e8b3db614af8c4ad9cbe0d3%26scene%3D27%23wechat_redirect)**则是用移除term之后的query和原始query的embedding做差值，diff越大词越重要
      + 基于统计模型：用速度更快的统计分类/回归模型同样可以解决上述问题，
        + **[腾讯搜索](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s%3F__biz%3DMzU1NTMyOTI4Mw%3D%3D%26mid%3D2247499470%26idx%3D1%26sn%3D6a6e80673353fb485a854ed2cffc5dcb%26chksm%3Dfbd74ca2cca0c5b4fb0622140eb9fa3cd15c860d06b24a4c7fc37e8b3db614af8c4ad9cbe0d3%26scene%3D27%23wechat_redirect)**采用了term 词性、长度信息、term 数目、位置信息、句法依存 tag、是否数字、是否英文、是否停用词、是否专名实体、是否重要行业词、embedding 模长、删词差异度、前后词互信息、左右邻熵、独立检索占比 ( term 单独作为 query 的 qv / 所有包含 term 的 query 的 qv 和）、iqf、文档 idf、统计概率等特征，来预测term权重。
        + 训练语料可以通过query和被点击doc的共现词来制作
      + 深度学习的模型：模型过于复杂，还是不要使用
+ **Query 改写**
  + 意义：Query改写是个很重要的模块，因为用户的输入变化太大了，有短的有长的，还有带错别字的，如果改写成更加规范的query可以很大地提升搜索质量。
  + 基础方法可以使用纠错，扩展，同义替换等多个功能。
  + 提前把高频Query都挖掘好，存储成pair对的形式，线上命中后直接替换就可以了，所以能上比较fancy的模型。
  + 纠错
    + **[腾讯搜索](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s%3F__biz%3DMzU1NTMyOTI4Mw%3D%3D%26mid%3D2247499470%26idx%3D1%26sn%3D6a6e80673353fb485a854ed2cffc5dcb%26chksm%3Dfbd74ca2cca0c5b4fb0622140eb9fa3cd15c860d06b24a4c7fc37e8b3db614af8c4ad9cbe0d3%26scene%3D27%23wechat_redirect)**把错误分成了Non-word（包括拼音，英文和数字）和Real-word（字出错）两类
    + 对于 Non-word，可以利用编辑距离挖掘出改写的pair；比如拼音汉字混合、漏字、颠倒等可以通过人工pattern生成一批pair。
    + 批量挖掘或生成，对用户session、点击同一个doc的不同query的行为日志进行统计，计算ngram语言模型的转移概率；也可以直接利用业务语料上预训练的BERT，mask一部分之后得到改写的词。
    + 当有了第一批pair语料后，就可以用seq2seq的方式来做了。
  + 扩展：
    + 用户的表述不一定总是精确的，扩展则能够起到【推荐】的作用，可以对搜索结果进行召回、在搜索时进行提示以及推荐相关搜索给用户
    + 目的主要是丰富短query的表达，更好捕捉用户意图。
    + Query扩展pair的挖掘方式和纠错差不多，可以建模为pair对判别或者seq2seq生成任务。
      + **[丁香园-搜索中的Query扩展技术](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s%3F__biz%3DMzU1NTMyOTI4Mw%3D%3D%26mid%3D2247500267%26idx%3D2%26sn%3D0a38ea1f7d4a96a632b129ffbabc2d2c%26chksm%3Dfbd77387cca0fa91abbd69ad1885c44d05ab578886102d36e9ea11d9c7c529ebc47dbfa36d61%26scene%3D27%23wechat_redirect)**
      + **[丁香园-再谈搜索中的Query扩展技术](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s%3F__biz%3DMzU1NTMyOTI4Mw%3D%3D%26mid%3D2247517070%26idx%3D2%26sn%3D367376c143ca8f5e526b33d93518ea16%26chksm%3Dfbd731e2cca0b8f4c97ea4f22464c6ee101e853f2a95b8159763c72e723d8902843a13970bf9%26scene%3D27%23wechat_redirect)**
    + 除了用模型发现之外，也可以利用知识图谱技术，将一些候选少的query扩展到上位词，或者某种条件下的相关词，比如把“能泡澡的酒店”改写成“带浴缸的酒店”
  + 同义替换
    + 主要解决query和doc表述不一致的问题，保证能召回到用户想找的item。
    + 但这个模块面临的困境是不同垂搜下的标准不一致，对于这个问题一方面可以针对不同领域训练不同模型，但每个领域一个模型不太优雅，所以也可以在语料上做文章，比如加一个统一的后缀
+ **意图识别**
  + 通常是一个分类任务，要识别用户要查询的类目，再输出给召回和排序模块
  + 很多query都是模糊的，可能有多个类别满足情况，意图模块主要是输出一个类目的概率分布，进行多路召回，让排序层进行汇总。
  + 通常会采用层级式的类目体系，模块先判断大类目，再去判断更细化的类目。
  + 这里一般会使用浅层的模型，比如统计方法或者是浅层神经网络，据说fasttext方法是比较好的（**[微信](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzU1NTMyOTI4Mw%3D%3D%26mid%3D2247512053%26idx%3D2%26sn%3D0ab4ade5ee6c83f1f53a0e03583dc822%26chksm%3Dfbd71d99cca0948f7fb6e29943d559214fb55373287ee66cd7c786986f673b9d587ef1d6a3f2%26scene%3D27%23wechat_redirect)**和**[第四范式](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s%3F__biz%3DMzU1NTMyOTI4Mw%3D%3D%26mid%3D2247511081%26idx%3D1%26sn%3Dfa3d25dff6d12bb298f51b2c317f0574%26chksm%3Dfbd71e45cca097533ae6397a6d9af024e39817ca9c48470c12a3e85d46e38f0082d0dc73fdca%26scene%3D27%23wechat_redirect)**）。在浅层模型下要想提升效果，可以增加更多的输入信息，比如**[微信](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzU1NTMyOTI4Mw%3D%3D%26mid%3D2247512053%26idx%3D2%26sn%3D0ab4ade5ee6c83f1f53a0e03583dc822%26chksm%3Dfbd71d99cca0948f7fb6e29943d559214fb55373287ee66cd7c786986f673b9d587ef1d6a3f2%26scene%3D27%23wechat_redirect)**就提供了很多用户画像的feature
  + 由于类目层级和数目的增加，光靠一两个模型是很难同时满足速度和精度的，在这个模块少不了词表和pattern的帮助
+ 问答中的query理解
  + 



https://zhuanlan.zhihu.com/p/393914267

https://zhuanlan.zhihu.com/p/125139937

https://zhuanlan.zhihu.com/p/152251002