# Transformer细节解读

## Encoder 部分

包含6个`encoder`层次，每个层次的参数不同，不共享

### 输入部分

+ embedding + 位置编码

+ 为什么需要位置编码：

  + RNN 共享的是一套参数，不止是`W`，包括输入层到隐藏层的参数`U`，隐藏层到输出层的参数`V`，也都是共享的

  + RNN的梯度消失和普通网络不太一样，它的梯度被近距离梯度主导，被远距离梯度忽略（注意对比于连乘效应导致梯度为0的区别）

  + transformer与RNN的区别：RNN是按顺序一个个单词进行处理的，而transformer可以并行化处理信息，能够提高效率，但是会忽略掉单词之间的顺序，所以才会需要对文本中的单词进行位置编码，也就是加入了单词的顺序关系

  + 位置编码公式：

    $PE_{(pos, 2i)} = sin(pos/10000^{2i/d_{model}})$
    
    $PE_{(pos, 2i+1)} = cos(pos/10000^{2i/d_{model}})$
    
    + Where `pos` is <font color=red>word position</font> and `i` is the dimension. That is, each dimension of the position encoding corresponds to a sinusoid (which means actul `sin` function or `cos` function).
    + 中文：对于一个`pos`，也就是一个词，其位置向量编码也是512维，偶数位为sin，奇数位为cos
    
    + The wavelengths form a geometric gropression from $2\pi$ to $10000*2\pi$
    + 对于每个词，最终embedding = position embedding + word embedding
    
  + 位置编码的作用：
  
    + 对于每个pos，每一个词，使用sin，cos进行编码，得到的是一个<font color=red>绝对位置</font>的编码
  
    + we hypothesized  it would allow the model to easily learn to attend by <font color>relative position </font>, since for any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$
  
      $sin(\alpha + \beta) = sin\alpha cos\beta + cos\alpha sin\beta$
  
      $cos(\alpha + \beta) = cos\alpha cos\beta - sin\alpha sin \beta$
  
      由上面两个式子可以推出
  
      $PE(pos+k, 2i) = PE(pos, 2i)PE(k, 2i+1) + PE(pos, 2i+1)PE(k, 2i)$
  
      $PE(pos+k, 2i+1) = PE(pos, 2i+1)PE(k, 2i+1) - PE(pos, 2i)PE(k, 2i)$
  
      这个式子说明：对于 `pos+k` 位置的位置向量的某一维 `2i` 或 `2i+1`，而言，可以表示为，`pos`和`k`位置的位置向量 `2i` 和 `2i+1` 维的线性组合，也就意味着位置向量中包含了<font color=red>相对位置</font>信息
  
      但是相对位置会在Attention中消失

### Attention

+ 具体见Attention那篇

### 前馈神经网络

+ 残差和`LayerNorm`
  + 这一层输入的数据为：传入encoder的数据（word_embedding + position embedding），和经过attention结果后的数据（多头结果）的按位相加
  + 残差
    + 什么是残差：在已经经过神经网络和激活函数的输出基础上，按位加上输入这个网络的数据（也就是前两层的输出），公式表示就是$F(x) + x$
    + 残差网络的目的：
      + 缓解梯度消失的情况，根据后向传播的链式法则，加入原始数据后求偏导，会在链式法则中加入1，即使后续连乘再多，也不会出现的梯度消失的情况
      + 可以用来解决<font color=red>梯度消失的情况</font>，将模型扩展得更深
  + layer normalization：
    + 为什么不使用batch normalization
      + batch normalization 效果差，所以不用
        + Batch normalization: <font color=red>针对一个batch中的样本在同一特征维度进行处理</font>
        + （有待商榷）优点1：可以解决内部协变量偏移
        + （有待商榷）优点2：缓解了梯度饱和问题，加快收敛
        + 缺点1：小样本的均值和方差，来模拟所有样本的均值和方法，如果batch_size比较小，那效果一定不好
        + （有待商榷）缺点2：在RNN中效果不好，RNN输入是动态的，长度不同的输入文本不能保证具有同一维度的特征，如果有，可能也是小样本
      + <font color=red>feature scaling</font>
    + 为什么使用layer normalization
      + <font color=red>针对同一个样本的所有单词做缩放</font>
+ 输入前馈网络的是每个词对应的残差结果，然后分别通过前馈网络和残差网络

## Decoder 部分

包含6个`decoder`层次，每个层次的参数不同，不共享

+ masked multi-head attention
  + This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position `i` can depend only on the known outputs at position less then `i`.  
  + 当前时刻的单词后面的单词要mask掉 
  + <font color=red>也就是做attention时，需要保证只有排在当前词前面的其他词与当前词的相关度，后面的词在实际应用中是无法预知的，所以不能使用</font>
+ encoder + decoder multi-head attention
  + 前一个encoder组中最后一个encoder的输出，和每一个（6个）decoder进行交互
  + 交互方式为：encoder产生的是k和v矩阵，decoder产生的是q矩阵，然后就是做attention



Transformer的tensorflow实现：

https://github.com/terrifyzhao/transformer