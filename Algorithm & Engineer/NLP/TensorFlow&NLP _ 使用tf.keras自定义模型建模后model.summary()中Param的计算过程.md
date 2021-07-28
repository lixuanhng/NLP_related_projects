## 摘要
当我们使用TensorFlow2.0中keras.layers API进行自定义模型组网时，我们可以通过使用<font face="黑体" color=#CC3366 size=4> model.summary()</font>来输出模型中各层的一些信息。输出的图中包含了3列信息，第一列为各层的名称（层的类型，在tf.keras.layers中定义好了）；第二层为数据经过每层之后，输出的数据维度；第三列为当前层中共有多少个参数。

由于已经有一些讲得较为清楚的博客提到了这些内容，比如：
[详解keras的model.summary()输出参数Param计算过程](https://blog.csdn.net/ybdesire/article/details/85217688?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase)
该博客中主要讲述了 **基础神经网络** 和 **CNN（2维卷积）** 中的Param计算过程，这篇文章中就不再赘述了。我们重点来探讨一下当我们使用<font face="黑体" color=#CC3366 size=4> CNN（1维卷积）</font>模型对<font face="黑体" color=#CC3366 size=4> NLP任务 </font>进行建模时，model.summary() 的展示结果中Param的计算过程。

## 代码演示
以下是使用自定义模型方式完成的demo，仅供参考

```python
# to show the whole model.summary(), especially the part of output shape
from tensorflow import keras
from tensorflow.keras import layers as klayers

class MLP(keras.Model):
    def __init__(self, input_shape, **kwargs):
        super(MLP, self).__init__(**kwargs)
        # Add input layer
        self.input_layer = klayers.Input(input_shape)
        
        self.embedding = klayers.Embedding(10000, 7, input_length=input_shape)
        self.conv_1 = klayers.Conv1D(16, kernel_size=5, name = "conv_1", activation = "relu")
        self.pool_1 = klayers.MaxPool1D()
        self.conv_2 = klayers.Conv1D(128, kernel_size=2, name = "conv_2", activation = "relu")
        self.pool_2 = klayers.MaxPool1D()
        self.flatten = klayers.Flatten()
        self.dense = klayers.Dense(1,activation = "sigmoid")

        # Get output layer with `call` method
        self.out = self.call(self.input_layer)

        # Reinitial
        super(MLP, self).__init__(
            inputs=self.input_layer,
            outputs=self.out,
            **kwargs)
    
    def build(self):
        # Initialize the graph
        self._is_graph_network = True
        self._init_graph_network(
            inputs=self.input_layer,
            outputs=self.out)
    
    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.conv_1(x)  # x.shape(batch_size, length_text+1-kernel_size, filters)
        x = self.pool_1(x)
        x = self.conv_2(x)
        x = self.pool_2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


if __name__ == '__main__':
    mlp = MLP(200)  # input max_context_length
    mlp.summary()
```

运行上述代码后，可以得到 model.summary() 生成的模型图，如下图所示：

![model.summary()](https://img-blog.csdnimg.cn/20200608170401971.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwNjQyNTQ2,size_16,color_FFFFFF,t_70#pic_center)
## 参数计算详解

 1. 首先，网络的第1层<font face="黑体" color=#CC3366 size=4> klayers.Input() </font>为输入层，将外部传入的**文本最大长度**<font face="黑体" color=#CC3366 size=4> input_shape </font>接入模型，所以此时没有参数，对应的Param为0。假设我们处理过后的每个数据文本的词长度为200，则此时数据通过该层后的数据维度为<font face="黑体" color=#CC3366 size=4> (None, 200) </font>，这里的<font face="黑体" color=#CC3366 size=4> None </font>可以认为是数据输入的batch。
 2. 网络的第2层<font face="黑体" color=#CC3366 size=4> klayers.Embedding() </font>为词嵌入层，根据有多个文本决定的词表大小，将每个文本中的词 Embedding 为词向量。这里我们假设数据文本的词表为10000个，设定 Embedding 后的词向量维度为7。由于该层的参数也是需要学习的，因此该层的参数总数（Param）为:10000 * 7 = 70000。此时数据的输出维度为<font face="黑体" color=#CC3366 size=4> (None, 200, 7) </font>。
 3. 网络的第3层<font face="黑体" color=#CC3366 size=4> klayers.Conv1D() </font>为第一层卷积层，该层的卷积核大小为5，卷积核个数为16，于是Param = (卷积核大小 x 词向量维度 + 1) x 卷积核个数 = (5 x 7 + 1) x 16 = 576，其中+1为考虑到了偏置值。同时，用大小为5的卷积核对长度为200的文本进行采样时（默认步长为1，valid padding），采样结束后文本长度为 200 + 1 - 5 = 196，且共生成16个结果，所以数据的输出维度为<font face="黑体" color=#CC3366 size=4> (None, 196, 16) </font>。
 4. 网络的第4层<font face="黑体" color=#CC3366 size=4> klayers.MaxPool1D() </font>为第一层最大池化层，该层默认参数为池化尺寸为2，valid padding。池化作用为对数据进行降维，因此参数为0。数据经过池化层后，维度降为原来的一半，<font face="黑体" color=#CC3366 size=4> (None, 98, 16) </font>。
 5. 网络的第5层<font face="黑体" color=#CC3366 size=4> klayers.Conv1D() </font>为第二层卷积层，该层的卷积核大小为2，卷积核个数为128，于是Param = (卷积核大小 x 词向量维度 + 1) x 卷积核个数 = (2 x 16 + 1) x 128 = 4224。同时，用大小为2的卷积核对长度为98的文本进行采样时（默认步长为1，valid padding），采样结束后文本长度为 98 + 1 - 2 = 97，且共生成128个结果，所以数据的输出维度为<font face="黑体" color=#CC3366 size=4> (None, 98, 128) </font>。
 6. 网络的第6层<font face="黑体" color=#CC3366 size=4> klayers.MaxPool1D() </font>为第二层最大池化层，该层默认参数为池化尺寸为2，valid padding。池化作用为对数据进行降维，因此参数为0。数据经过池化层后，维度降为原来的一半，数据的输出维度为<font face="黑体" color=#CC3366 size=4> (None, 48, 128) </font>。
 7. 网络的第7层<font face="黑体" color=#CC3366 size=4> klayers.Flatten() </font>为展平层，不涉及新的参数。该层将output_shape中除第一维数字外的其他数字相乘，即将多维数据转化为一维，所以数据的输出维度为<font face="黑体" color=#CC3366 size=4> (None, 6144) </font>(48 x 128 = 6144)。
 8. 网络的第8层<font face="黑体" color=#CC3366 size=4> klayers.Dense() </font>为全连接层，因为只有一个神经元，所以参数个数为：6144（个w）+ 1 (偏置值) = 6145。通过一个神经元后，数据维度就变为<font face="黑体" color=#CC3366 size=4> (None, 1) </font>。


希望这篇文章对大家有帮助，也欢迎大家批评指正！
