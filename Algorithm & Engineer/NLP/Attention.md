# Attention相关资源

https://github.com/CyberZHG/keras-self-attention/blob/master/README.zh-CN.md

https://github.com/CyberZHG/keras-multi-head/blob/master/README.zh-CN.md



attention简介+代码：

https://kexue.fm/archives/4765



Attention的文本分类模型

https://blog.csdn.net/weixin_30802273/article/details/96889595



理论讲解

https://zhuanlan.zhihu.com/p/34781297



tensorflow实现

https://github.com/bojone/attention



+ 各种attention变种

https://github.com/monk1337/Various-Attention-mechanisms



## Transformer中的Attention

+ 优点：利用self-attention实现快速并行

+ 宏观：
  + 注意力会关注整个输入句子中的所有单词对某一个单词的影响
  + 与RNN的区别：RNN将前面处理过的单词与当前单词进行融合，而Attention则是将所有单词与当前单词进行融合
  + 注意tensor2tensor的模型，可以下载一个transformer的模型，并用交互式可视化的方式进行检验

+ 微观：
  + 第一步：
    + 从每个编码器的输入向量（需要传入编码器的词向量），生成<font color=red>查询向量q</font>，<font color=red>键向量k</font>和<font color=red>值向量v</font>
    + `q, k, v` 是词向量与是三个<font color=red>权重矩阵</font>$W_q, W_k, W_v$相乘后得到的，维度更低
  + 第二步：
    + 计算得分，计算句子中其他词对当前词的影响程度（使用其他词中的每个词与当前词计算打分）
    + 分数等于，其他单词的`k`与当前单词的`q`点积。当前词的第一个分数`q1*k1`，第二个分数是`p1*k2`，以此类推
  + 第三步：
    + 每个分数除以8（键向量梯度的平方根，会让梯度更稳定，也可以尝试其他值）
  + 第四步：
    + softmax归一化。自己与自己的softmax结果一定是高的，其余高分的结果对应的另些个单词就是与当前词关系最大的词。
  + 第五步：
    + 针对当前单词，将每个softmax的得分与每个单词的`v`进行乘积。直觉上是希望关注与当前单词有更强语义关系的单词（打分结果大的），而减小关系小的单词的关系
  + 第六步：
    + 乘积的结果求和，得到当前单词输出向量
  + 综述：==自注意力的另一种解释就是在编码某个单词时，就是将所有单词的表示（值向量）进行加权求和，而权重是通过该词的表示（键向量）与被编码词表示（查询向量）的点积并通过softmax得到。==
  + 公式表示：$Z = softmax(\frac {QK^T} {\sqrt d_k})V$
    + 这里的 $d_k$ 其实是查询向量，键向量和值向量的向量长度
    + Q和K进行点乘，得到一个标量，值大的那个说明更关注
    + 对 V 来说，也就是获得了一个加权平均的值
    + 当然，完成`softmax`后乘以 `V` 的方式，也是通过点积完成的
    + <font color=red>问题：</font>
      + 为什么`WQ`，`WK`，`WV`最后一个维度是考虑多头之后的结果？按理说其最后一维度应该是$d_k$，按照论文说法，多头的结果应该是将8个结果拼接到一起的，从而达到 $d_k$ * head_num 的效果；在实际代码中，则是直接将`Q, K, V`的最后一维直接变为 $d_k$ * head_num ，结果好像也是可以的。
  
+ Mask 操作：

  + padding mask

    + 在`encoder`和`decoder`端都是用的mask操作

    + 改进的原因：`padding`指的是将一个`batch`中所有的句子都补全到最长的长度，比如拿0进行填充；但是拿0进行填充没有实际意义，而且也不希望填充的位置，参与反向传播过程。在self-attention中，也不会希望有效词的注意力集中在无意义的位置上。

    + 改进的方式：因此需要在训练时将补全的位置给mask掉

    + `padding mask` 在 `attention` 的计算过程中处于`softmax` 之前，也是可选的，不加的话直接softmax

    + 通过 `padding mask` 操作，使得补全位置上的值成为一个非常大的负数，经过Softmax层后，这些位置上的概率就是0（此操作就相当于把补全位置的无用信息给遮蔽掉了）

      ```python
      # tf.sign：element-wise value mappping.
      # y = sign(x) = -1 if x < 0; 0 if x == 0; 1 if x > 0.
      # 这里的 keys 指的是 K 矩阵
      key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))
      ```

    + 这一步操作在主流`transformer`中没有被提及，很有可能是这个位置的mask可以被省略，因为论文中`this mask is optional`

  + sequence mask

    + 仅在`decoder`端使用的`mask`操作
    + 改进的目的：使得`decoder`不能看见未来的信息（<font color=red>这里和Transformer中提到的一样</font>）
    + 处理方式和`padding mask`比较相似

+ multi-head attention：

  + 扩展了模型专注于不同位置的能力。尽量避免找到的最大关系词就是自己
  + 多个表示子空间。<font color=red>照自己的理解，多头有点像是多个采样器，每一种采样结果都是不一样的，能够保证得到的结果在概率上更加符合语义关系。只采用单个注意力得到的结果随机性相对较大</font>
  
+ 8个注意力头得到8个结果矩阵，采用的做法是可以将这些矩阵进行拼接，然后再创建一个参数矩阵 $W_0$，通过拼接的结果矩阵与参数矩阵进行乘积得到最后结果，然后送往前馈网络
  + 多头结果的向量长度 = 多头个数 * $d_k$

+ 使用**<font color=red>位置编码</font>**表示序列的顺序

  + 向词向量中添加位置向量，有助于更好的表达词与词之间的距离（词在句子中的位置）

  + 输入编码器时，每个词的编码结果是词向量+位置编码（直接<font color=red>按位相加</font>，不是拼接）

  + 位置编码的维度与词向量维度相同，其中左边一半的值由`sin`函数生成，右边一半的值由`cos`函数生成，然后将它们拼接到一起作为一个位置编码向量

  + 这种编码方式的优点是，训练出来的模型需要处理比训练集句子更长的句子，由于三角函数值域在`(-1,1)`，所以其能够扩展足够长的序列长度

