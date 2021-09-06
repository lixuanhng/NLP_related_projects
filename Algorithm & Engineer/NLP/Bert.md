Bert相关知识：

### 论文解读：

+ bert 框架
  + pre-train：无监督训练，针对不同的任务
  + fine-tuned（下游任务）：模型初始化使用 pre-train 的 parameters，使用标注数据进行训练。每个任务有不同的fine-tuned模型，但是初始化都是一样的



1. Bert整体模型架构

   + ![bert模型](/Users/lixuanhong/Desktop/Materials/NLP/bert模型.jpeg)
   + 基础架构是：Transformer中的Encoder
     + 实际bert采用的是经过堆叠的Encoder，base使用12层堆叠结果，large采用24层堆叠结果
     + 输入部分（3 部分向量相加）
       + input = token embedding + segment embedding + position embedding
       + token embedding，使用 word2vec or random initialization
         + <font color=red> 重点 </font>bert输入是subword，bert自己自带一个词表，输入subword在词表中的id，然后将bert输出和你自己模型的随机初始化词向量拼接，
       + segment embedding，前面的句子中每个词都用0来表示，后面的句子中每个词都用1来表示
       + position embedding，采用随机初始化，让模型去学习这个位置的表示（不同的是，在 transformer中使用三角函数去表征）
     + 注意力机制
     + 前馈神经网络

2. 如何做预训练：MLM + NSP

   + MLM （mask language model）：
     + 需要解决的是无监督的任务，没有标注
     + （AR模型，自回归模型，值考虑单侧信息，典型代表GPT）
     + MLM是一种AE模型，即自编码模型，可以使用上下文信息。<font color=red>打破了文本，让模型重建文本</font>。P(我爱吃苹果) = P(我爱mask苹果)
     + 缺点：破坏的比较多，导致重建困难，mask掉的词之间存在关系，mask掉之后就丢失了两个词间的关系
     + 随机mask15%的词，其中，10%替换成其他，10%不动，80%替换成mask

     ```python
     # relative code shown as follow
     for index in mask_indices:
         # 80% of the timem, replace with [MASK]
         if random.random() < 0.8:
             masked_token = "[MASK]"
         else:
             # 10% of the time, keep original
             if random.random() < 0.5:
                 masked_token = tokens[index]
             # 10% of the time, replace with random word
             else:
                 masked_token = random.choice(vocab_list)
     ```
     
     
     
   + NSP：
     + 样本的构造：从同一文档中取出两个连续的段落作为正样本；从不同的文档中随机创建一对段落作为负样本。
     + 缺点：主题预测和连贯性预测合并为一个单项任务。主题预测相对简单，会使得这个任务学习的结果不好（ALBERT优化了这个问题）
     + 处理两个句子中的关系，使用[SEP]分割两个句子
     + 对于分类任务（或文本匹配），使用[CLS]的输出向量（不能代表整个句子的语义信息，在无监督的任务上表现未必好），后面接一个分类器

3. 如何微调Bert，提升bert在下游任务中的效果（性能最好）
   + 首先在大量通用语料上训练一个预训练模型（这一步中文bert已经完成了）
   + 在相同领域上使用大量数据继续训练这个模型（domain transfer 领域迁移）
     + 动态mask，每次epoch去训练时再开始mask，而不是使用同一个（处理好后保存，需要时再load in）
     + n-gram mask：
   + 在任务相关的小数据上继续训练模型（task transfer 任务迁移，去掉不属于相关任务的文本）
   + 任务相关数据上做具体任务（fine-tuning）
   + 参数说明
     + learning rate 较小
     + epoch 为 3，4
     + Weighted decay 修改后的Adam，使用warmup， 搭配线性衰减
     + 可以考虑数据增强，自蒸馏，外部知识的融入
4. 脱敏数据中使用bert
   + 语料很大，直接从零开始训练bert
   + 语料小，按照词频，把脱敏数据对照到中文或者其他语言，使用中文bert做初始化，也就是把映射对象的权重拿来作为自己的权重；可以将脱敏数据当成是另一种语言。
5. 如何使用NSP任务
   + 类NSP任务，测试数据使用MLM，训练数据可以使用MLM+类NSP任务



Bert相比于word2vec的优势：

1. 一词多义

   + word2vec的词是由前后频繁出现的词（有固定的窗口大小）决定的，学习到一个查询参数矩阵，每一个单词被映射到一个唯一的稠密向量。词表示是静态的，不考虑上下文的，所以也就不存在一词多义的情况。
   + ELMo，考虑了上下文，每个词的表示都应该是整个文本序列的函数。使用了两个堆叠的双向LSTM来为每个词提供前后两个方向的上下文信息。
   + Bert使用Transformer作为特征抽取器，抽取的就是上下文的特征，也不需要像BiLSTM那样双向堆叠。

2. 词的多层特征

   + 好的词表示还应该表示出词的复杂特性，包括语法，语义等
   + ELMo、Bert等预训练方法能够学习到一个深度的网络，在预训练之后可以在不同的网络层上的得到不同层次的特征。
   + 高层中产生的特征更多体现了抽象的、依赖于上下文的部分，而低层产生的特征更关注语法层面的部分

   + word2vec得到查询参数矩阵，很难适应所有的任务；而bert预训练产生的词表示则能够在上游任务中解决语法的问题，而在下游任务更好的适应各种场景。不同任务对不同层次特征的依赖不一样，bert可以选择性的利用所有层次的信息。

3. 对于句子建模的方式不同

   + word2vec采用平均所有词向量的和，但是这种方式和BOW模型很像，忽略了词之间的顺序；