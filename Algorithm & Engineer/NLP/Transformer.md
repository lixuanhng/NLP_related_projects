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
          + 我们知道sigmoid激活函数和tanh激活函数存在梯度饱和的区域，其原因是激活函数的输入值过大或者过小，其得到的激活函数的梯度值会非常接近于0，使得网络的收敛速度减慢。传统的方法是使用不存在梯度饱和区域的激活函数，例如ReLU等。BN也可以缓解梯度饱和的问题，它的策略是在调用激活函数之前将 ![[公式]](https://www.zhihu.com/equation?tex=WX%2Bb) 的值归一化到梯度值比较大的区域。假设激活函数为 ![[公式]](https://www.zhihu.com/equation?tex=g) ，BN应在 ![[公式]](https://www.zhihu.com/equation?tex=g) 之前使用
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

### 实践

+ Transformer的tensorflow实现：
  + https://github.com/terrifyzhao/transformer
+ tensorflow库下面的包
  + tensor2tensor
+ 知乎上使用tf2.0手动实现transformer
  + https://zhuanlan.zhihu.com/p/64881836



#### Batch Normalization

+ 作用：

  + 网络更加稳定
  + 能够起到一定正则化的作用

+ 小批量随机梯度下降，其缺点是对参数比较敏感，较大的学习率和不适合的初始化值均有可能导致训练过程中发生梯度消失，或梯度爆炸。BN用来解决这个问题

+ 作者认为因为存在内部协变量偏移（ICS），模型需要学习在训练过程中会不断变化的隐层输入分布。

+ <font color=red>提出BN的动机是企图在训练过程中将每一层的隐层节点的输入分布固定下来，这样就可以避免ICS通过一定的规范化手段，把每层神经网络输入值的分布强行拉回到接近均值为0方差为1的标准正态分布</font>

+ 缓解梯度饱和

  + 缓解使用因存在梯度饱和区域的激活函数sigmoid或者tanh函数，使用BN的方法就是将输入通过激活函数之前，先通过BN

+ 训练过程

  + 以一个batch为单位，假设一个批量有 m 样本 $B = {x_1, x_2, ...,x_m}$ ，每个样本有 d 个特征，那么这个批次的每个样本第 k 个特征的归一化后的值为

    $\hat x = \frac{x^{(k)}-E[x^{(k)}]}{\sqrt Var[x^{(k)}]}$ ，其中 E 和 Var 分别表示第 k 个特征在这个批次中所有的样本的均值和方差。

  + 这种表示会对模型的收敛有帮助，但是也可能破坏已经学习到的特征。为了解决这个问题，==BN添加了两个可以学习的变量 $\beta$ 和 $\gamma$ 用于控制网络能够表达直接映射，也就是能够还原BN之前学习到的特征==。

    $y^{(k)} = \gamma^{(k)} \hat x^{(k)} + \beta^{(k)}$

  + <font color=red>BN可以看作是一个以 $\beta$ 和 $\gamma$ 为参数的，从 $x_{1...m}$ 到 $y_{1...m}$ 的一个映射</font>

  + 对 x 进行求导，$\frac {\partial l}{\partial \gamma} = \sum_{i=1}^m \frac{\partial l}{\partial y_i} \hat x_i$， $\frac{\partial l}{\partial \beta} = \sum_{i=1}^m \frac{\partial l}{\partial y_i}$

  + BN是处处可导的，因此可以直接作为层的形式加入到神经网络中。

+ BN应用于卷积网络

  + 卷积网络和MLP的不同点是卷积网络中每个样本的隐层节点的输出是三维（宽度，高度，维度）的
  + 假设一个批量有 m 个样本，Feature Map的尺寸是 p * q ，通道数是 d。在卷积网络的中，BN的操作是以Feature Map为单位的，因此一个BN要统计的数据个数为 m * p * q ，每个Feature Map使用一组 $\beta$ 和 $\gamma$ 。

+ BN效果好的真正原因

  + 平滑了损失平面，收敛较快
  + 残差网络也能起到这个作用

+ 几个场景下需要谨慎使用

  + 受制于硬件限制，每个Batch的尺寸比较小，这时候谨慎使用BN；
  + 在类似于RNN的动态网络中谨慎使用BN；
  + 训练数据集和测试数据集方差较大的时候。



#### Layer Normalization

+ 作用

  + 解决BN中出现的RNN动态网络和小batch时的效果不好的问题
  + BN 取不同样本的同一个通道的特征做归一化；LN则是如左侧所示，它取的是同一个样本的不同通道做归一化。
  + 文本信息，一个词的语义表达应该到其所在的句子中获取，所以特征上的norm（BN）不合适

+ BN效果不好的原因

  + 小样本的均值和方差便不能反映全局的统计分布息，所以基于少量样本的BN的效果会变得很差。
  + RNN网络中各个样本的长度都是不同的，当统计到比较靠后的时间片时，只有一个样本还有数据，基于这个样本的统计信息不能反映全局分布。

+ MLP中的LN

  + 独立于batch size，根据样本特征做归一化

  + 设 H 是一层中隐层节点的数量，其中 $a_i$ 为该层中每一个节点，$l$ 是MLP的层数，我们可以计算LN的归一化统计量 $\mu$ 和 $\sigma$

    $\mu^l = \frac{1}{H} \sum_{i=1}^H a_i^l$， $\sigma^l = \sqrt{\frac{1}{H} \sum_{i=1}^H(a_i^l-\mu^l)^2}$

    上面统计量的计算只取决于隐层节点的数量，所以只要隐层节点的数量足够多，我们就能保证LN的归一化统计量足够具有代表性

    归一化后：$\hat a^l = \frac{a^l-\mu^l}{\sqrt{(\sigma^l)^2+\epsilon}}$

  + <font color=red>LN中我们也需要一组参数来保证归一化操作不会破坏之前的信息，在LN中这组参数叫做增益（gain）g 和偏置（bias）b （等同于BN中 $\beta$ 和 $\gamma$）。</font>假设激活函数为 f ，最终LN的输出为：

    $h^l = f(g^l  * \hat a^l + b^l)$

    忽略参数 $l$，得到：$h = f(\frac{g}{\sqrt{\sigma^2+\epsilon}}*(a - \mu) + b)$

+ RNN中的LN

  + 可以表示为：$a^t = W_{hh}h^{t-1} + W_{xh}X^t$

  + 归一化过程是完全一样的表达，公式见上面

+ 同样能够起到平滑损失平面的作用



