# 多头自注意力在文本分类上的应用及几个细节

想来至今还没有从0完整实现过`Attention`，即使公式掌握的足够熟悉，但很多细节的问题，还是要亲自动手实践一次后，才能了解得更清楚。本次我们通过手动实现一个多头注意力的方式，对文本分类做一次实践，加深对`Attention`机制的理解。在实践过程中，穿插着会提到一些`Attention`的细节，争取做到代码与论文紧密结合。

实践部分，我们还是选择使用`TensorFlow2.1`进行建模，由于要尝试构建`Attention`的模型，所以在建模部分我们主要采用继承`keras.Layer`的方式分别构建各个模块，并且使用`Sequential API`的方式将各层连接起来。具体建模过程，请见[我的github仓库](https://github.com/lixuanhng/NLP_related_projects/tree/master/Attention)。本次实践主要基于[Shicoder](https://github.com/Shicoder/DeepLearning_Demo/tree/master/attention) 的代码进行修改。受限于作者能力，如有问题，欢迎在下方留言讨论。



## 数据准备：

本次文本分类任务，我们选取的`TensorFlow.keras.datasets`自带的IMDB数据集。由于数据集已经是经过处理后的数据，我们只需设定每条句子的最大文本长度后进行训练集和测试集的划分。这里针对数据准备过程中设置的超参数如下：

```python
vocab_size = 20000  # 文本中单词词频在前max_features个的词
maxlen = 80  		# 最大文本长度
batch_size = 32
```



## 位置编码

我们认为，对于句子中每个词，`Attention`机制更容易使得这个词关注到与其相关的其他词，在表现上是要强于传统的`CNN`和`RNN`的。但是`CNN`和`RNN`有一个优势，就是他们都会关注到句子中词与词之间的顺序问题。`RNN`自然不必说，如果`CNN`中的卷积核的长度大于1，那么在一次采样的时候，编码结果能够包含一定词顺序的信息。而`Attention`本身更多关注的是一个并行的结果，并没有考虑到词与词之间的顺序。对于这种情况，有必要在`embeddings`的基础上加入位置编码，`positional emcodings`。每个位置对应于一个编码，位置编码完成后与`embeddings`的结果直接相加。

假设我们定义`embedding`的向量维度为128，那么`positional emcodings`的维度也是128。（在论文中这两个向量维度定义为$d_{model}$，其大小为512）关于位置编码的方式，论文使用三角函数对不同频率的位置进行编码，具体编码公式为：

$PE_{(pos, 2i)} = sin(pos/10000^{2i/d_{model}})$

$PE_{(pos, 2i+1)} = cos(pos/10000^{2i/d_{model}})$

对于每个位置，也就是每一个词，使用`sin`函数或`cos`函数进行编码，得到的是一个依赖于具体位置的<font color=red>绝对位置</font>的编码。论文中还提到，选择这种编码方式的原因，在于其能够学习到一定程度的<font color=red>相对位置</font>信息。根据三角函数的和差化积公式，对于固定的偏移量$k$，$PE_{pos+k}$都可以被表示为$PE_{pos}$的线性组合。具体公式可以表示为：

$PE(pos+k, 2i) = PE(pos, 2i)PE(k, 2i+1) + PE(pos, 2i+1)PE(k, 2i)$

$PE(pos+k, 2i+1) = PE(pos, 2i+1)PE(k, 2i+1) - PE(pos, 2i)PE(k, 2i)$

从代码层面看：

```python
from tensorflow.keras import layers

# 首先构建输入层和embedding层
# 输入层中的shape=2D，第0位是batch_size，第1位是max_len，
# 在这种条件下embedding的维度就是 (batch_size, max_len, embedding_dim)，其中embedding_dim=128
S_inputs = layers.Input(shape=(None,), dtype='int32')
embeddings = layers.Embedding(vocab_size, 128)(S_inputs)
```

接下来我们开始构建`positional emcodings`

首先计算三角函数中的分母，根据位置编码公式

```python
# 使用np.arange()，设置长度为size大小的一半，其中size = d_model
# 默认step=1，np.arange() 会创建一个包含整数序列的一维张量
position_j = 1. / tf.math.pow(10000., 2 * np.arange(self.size / 2, dtype='float32') / self.size)
```

然后计算分子，分子中主要获取的是位置

```python
# 其中x是输入位置编码模型的embedding结果，这里选取每个向量的第一个位置，同时在batch_size和max_len的维度上保持不变
# 这个位置下初始化全1的矩阵，然后使用cumsum，在新位置存放之前所有位置的累加和，相当于对每个维度进行位置编号
position_i = tf.math.cumsum(tf.ones_like(x[:, :, 0]), axis=1) - 1
```

由于分子和分母可能存在维度不匹配的情况，如果有必要需要对两个编码结果在不同位置进行扩展维度。然后将分子与分母进行矩阵相乘，得到三角函数的输入部分。再将这个输入部分分别通过`sin`和`cos`函数，得到的结果在同一维度进行拼接，表示如下：

```python
position_ij = tf.matmul(position_i, position_j)
position_ij = tf.concat([tf.math.cos(position_ij), tf.math.sin(position_ij)], 2)
```

拼接之后的位置编码与原`x`，也就是`embedding`的形状是一摸一样的，之后根据论文中的说法，我们需要将位置编码与`embedding`进行相加，得到的结果就可以往Attention层送了。



## Attention机制

首先要明确一下各个变量之间的联系和向量的长度。我们知道，`Attention`中`Q`，`K`，`V`分别是输入数据，也就是上文的`embeddings + positional encodings`，乘以各自的参数矩阵得到的结果，并且这些参数矩阵是可以学习的。于是应该先定义`Q`，`K`，`V`对应的参数矩阵。

同时我们需要明确的是，论文中使用了多头注意力`multi-head self-Attention`，对于每个头，完成`Attention`的计算后，会将得到的8个头拼接在一起。原始文本的向量长度定义为512，那么每个头下产生的`Q`，`K`，`V`矩阵的输出维度就是 512 / 8 = 64，那么三个权重矩阵`W_Q`，`W_K`，`W_V`的维度就是`[512, 64]`，这样输入数据与参数矩阵的乘积，就会得到向量为 64 的`Q`，`K`，`V`。并且，多头对应的参数矩阵是不一样的，也就是说每个头都有独一无二的一组权重矩阵`W_Q`，`W_K`，`W_V`，正好对应了多头的使用意义，扩展不同的表达子空间（这里可以将不同的表达子空间理解为CNN中卷积核，不同权重的多个卷积核对数据进行多次采样，这样就能够获取更加丰富的表达结果）。

在本次实践中，输入数据的向量长度为128，那么每个头下，三个权重矩阵的输出维度是16。除了与论文中的维度差异外，多头下的权重矩阵维度也有改变。为了避免同时维护不同头下三个权重矩阵，这里选择将权重矩阵的输出维度就设置为输入数据的向量长度128，也就是将不同头下，同一类型的权重矩阵合并为一个，而不是分成8。这样做的好处是不需要同时定义8组权重矩阵，显得代码过于冗长，最后Attention的计算结果也不必进行拼接了，因为就是在8个头的条件下计算的。

```python
def build(self, input_shape):
    """
    input_shape 是输入数据的维度，这里我们定义输入的数据是上一层编码的结果
    input_shape[-1]表示输入数据最后一个维度的大小，也就是embedding_dim，此处设置为128
    output_dim 为输出维度，这里是 nb_head * size_per_head
    其中，nb_head = 8 表示8个头，size_per_head = 16 表示每个头下的向量维度
    三个参数矩阵的维度都是 shape=[embedding_dim, output_dim]，也就是 [128, 128]
    """
    self.WQ = self.add_weight(name='WQ',
                              shape=(input_shape[-1], self.output_dim),
                              initializer=tf.random_uniform_initializer(),
                              trainable=True,
                              regularizer='l2')
    self.WK = self.add_weight(name='WK',
                              shape=(input_shape[-1], self.output_dim),
                              initializer=tf.random_uniform_initializer(),
                              trainable=True,
                              regularizer='l2')
    self.WV = self.add_weight(name='WV',
                              shape=(input_shape[-1], self.output_dim),
                              initializer=tf.random_uniform_initializer(),
                              trainable=True,
                              regularizer='l2')
    super(Attention, self).build(input_shape)
```

参数矩阵完成初始化后，下一步就是通过输入数据乘以各自的参数矩阵得到查询向量`Q`，键向量`K`，和值向量`V`。具体如下（拿查询向量`Q`的获取为例）；

```python
Q_seq = tf.matmul(x, self.WQ)
Q_seq = tf.reshape(Q_seq, (-1, tf.shape(Q_seq)[1], self.nb_head, self.size_per_head))
Q_seq = tf.transpose(Q_seq, perm=[0, 2, 1, 3])  # 为了后续计算方便，这里重新排列张量的轴
# 重排后的 Q_seq.shape = (batch_size, self.nb_head, seq_len, self.size_per_head)
```

三个向量全部获取完成后，就是根据公式计算`Attention`了。 

$Z = softmax(\frac {QK^T} {\sqrt d_k})V$

从这个公式中不管怎么看，都感觉`Q`和`K`应该是矩阵相乘，但是论文中又明确提出了`Scaled Dot-Product Attention`这样的概念，于是这里还是选择矩阵内积的方式。在实践RGCN时发现，可能在某种条件下，矩阵乘法和内积得到的效果是一样的。

得到的结果需要经过$\frac {1} {\sqrt{d_k}}$的缩放。采取缩放的原因是作者怀疑过大的$d_k$值容易产生很大的乘积结果，使得`softmax`的梯度变得很小。为抵消这种情况，需要采取这种缩放方式。

```python
A = tf.multiply(Q_seq, K_seq) / self.size_per_head ** 0.5
A = tf.transpose(A, perm=[0, 3, 2, 1])
A = self.Mask(A, V_len, 'add')
A = tf.transpose(A, perm=[0, 3, 2, 1])
A = tf.nn.softmax(A)
O_seq = tf.multiply(A, V_seq)  # softmax的值乘以值向量
O_seq = tf.transpose(O_seq, (0, 2, 1, 3))  
# O_seq.shape = (batch_size, self.nb_head, seq_len, self.size_per_head)
O_seq = tf.reshape(O_seq, (-1, tf.shape(O_seq)[1], self.output_dim))
```



## mask操作

可以看到，在进行`softmax`之前，可能还需要加入一步`mask`操作。`mask`操作的意义有两个。

### padding mask

在很多NLP任务上，如果文本的原始长度小于我们指定的最大文本长度时，常用的做法是用0将缺失的位置补齐。但从文本语义关系的角度分析，这些位置其实没有意义，我们也不想要这些位置的信息参与到学习过程中。在`self-attention`中，也不会希望有效词的注意力集中在无意义的位置上。

因此我们希望在训练时将补全的位置给`mask`掉。通过 `padding mask` 操作，使得补全位置上的值成为一个非常大的负数，经过softmax层后，这些位置上的概率就是0。此操作就相当于把补全位置的无用信息给遮蔽掉了。

这一步要放在softmax之前，当然没有也是可以的。论文中描述`Scaled Dot-Product Attention`的图中mask，后面标注了optional。具体的做法可以参考：

```python
def Mask(self, inputs, seq_len, mode='mul'):
    """
        mask.shape = [batch_size, seq_len] 或 [batch_size, seq_len, 1]
        :param inputs:
        :param seq_len: 是一个二维矩阵
        :param mode:  可以指定两种不同的mask方式
        :return:
        """
    if seq_len == None:
        return inputs
    else:
        # tf.shape(inputs)[1] 是每个 head 的维度
        # 生成的 one_hot 矩阵的维度是 (len(seq_len[:,0]), tf.shape(inputs)[1])
        # 这里的 len(seq_len[:,0]) 实际上就是文本长度 seq_len
        mask = tf.one_hot(seq_len[:, 0], tf.shape(inputs)[1])
        # 首先将每行为1的位置之后的所有位置全部置1，然后对角变换，1变0，0变1
        # 当前 mask 是一个下三角矩阵，一个角全1，另一个角全0
        mask = 1 - tf.math.cumsum(mask, 1)
        for _ in range(len(inputs.shape) - 2):
            mask = tf.expand_dims(mask, 2)
            if mode == 'mul':
                # 按位相乘，乘以0的元素就变为0
                return inputs * mask
            if mode == 'add':
                # 乘以0就变为0，乘以1的被变为无穷大，然后元素减去这个无穷大就变成了一个负数
                return inputs - (1 - mask) * 1e12
```



### sequence mask

这里的`mask`是比较重要的，需要在`Transformer`的`Decoder`中被使用到。`Attention`本质上会关注到全局的信息，但在执行预测时，不能让其看到当前词后面的词信息，所有有必要将这些词`mask`掉。这部分属于`Transformer`的`Decoder`部分，这里就不细说了。



## 训练

Attention模型搭建完毕后，后续还要构建一个平均池化层和全连接层，然后就可以对模型进行训练了。模型结构如下：

| Model: "model"                         |                   |         |                                 |
| -------------------------------------- | ----------------- | ------- | ------------------------------- |
| Layer (type)                           | Output Shape      | Param # | Connected to                    |
| input_1 (InputLayer)                   | [(None, None)]    | 0       |                                 |
| embedding (Embedding)                  | (None, None, 128) | 2560000 | input_1\[0][0]                  |
| position_embedding (PositionEmbedding) | (None, None, 128) | 0       | embedding\[0][0]                |
| attention (Attention)                  | (None, None, 128) | 49152   | position_embedding\[0][0]       |
|                                        |                   |         | position_embedding\[0][0]       |
|                                        |                   |         | position_embedding\[0][0]       |
| global_average_pooling1d               | (None, 128)       | 0       | attention\[0][0]                |
| dropout (Dropout)                      | (None, 128)       | 0       | global_average_pooling1d\[0][0] |
| dense (Dense)                          | (None, 1)         | 129     | dropout\[0][0]                  |

Total params: 2,609,281
Trainable params: 2,609,281
Non-trainable params: 0
