今天向各位分享TensorFlow中张量Tensor的转置函数tf.transpose()的用法，重点介绍其参数perm及原理。

## Tensor 的阶
在开始介绍转置函数之前，我们先来看一下Tensor的阶

当张量Tensor为一个标量时，即不带方向的纯量，其阶为0；

```python
x0 = tf.constant(1)
print(x0)  
# 输出 tf.Tensor(1, shape=(), dtype=int32)
```
当Tensor为一个向量时，如[1, 2, 3]时，其阶为1；

```python
x1 = tf.constant([1, 2, 3])
print(x1)  
# 输出 tf.Tensor([1 2 3], shape=(3,), dtype=int32)
```
当Tensor为矩阵时，其阶为2，如
![矩阵举例](https://img-blog.csdnimg.cn/20200607141535121.png#pic_center)

```python
x2 = tf.constant([[1, 2], [3, 4]])
print(x2)  
# 输出 tf.Tensor([[1 2] 
# 				 [3 4]], shape=(2, 2), dtype=int32)
```
而3阶Tensor可以被认为是一个立方体的数字集合，由多个小立方体组成，每个小立方体上存放了一个数字，如下图所示：

![3阶张量示意图](https://img-blog.csdnimg.cn/20200607141833539.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwNjQyNTQ2,size_16,color_FFFFFF,t_70#pic_center)

```python
x3 = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(x3)  
# 输出 tf.Tensor([[[ 1  2  3]  
# 				  [ 4  5  6]] 
#   
# 				 [[ 7  8  9]  
# 				  [10 11 12]]], shape=(2, 2, 3), dtype=int32)
```

## Tensor 的转置
接下来我们对Tensor的转置进行讨论

0阶，1阶Tensor的转置，可以说没有意义；2阶Tensor转置就相当于矩阵转置，比如
![2阶矩阵举例](https://img-blog.csdnimg.cn/20200607142233807.png#pic_center)
转置为
![2阶矩阵转置举例](https://img-blog.csdnimg.cn/20200607142317949.png#pic_center)
属于大学线性代数部分，也无需过多介绍；

我们重点来讨论**3阶Tensor的转置**，这时就需要用到tf.transpose()函数了

tf.transpose()函数的官方文档中，介绍了该函数存在一个参数perm，通过指定perm的值，来完成的Tensor的转置。

perm表示张量阶的指定变化。假设Tensor是2阶的，且其shape=(x, y)，此状态下默认perm = [0, 1]。当对2阶Tensor进行转置时，如果指定tf.transpose(perm=[1, 0])，就直接完成了矩阵的转置，此时Tensor的shape=(y, x).

```python
x2_ = tf.transpose(x2)
print(x2_)  
# 输出 tf.Tensor([[1 3]  
#                [2 4]], shape=(2, 2), dtype=int32)
```
而处理对象为3阶Tensor时，在下方例子中，[官方文档](https://tensorflow.google.cn/api_docs/python/tf/transpose)中给出了这么一句话：
![文档举例](https://img-blog.csdnimg.cn/20200607143130668.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwNjQyNTQ2,size_16,color_FFFFFF,t_70#pic_center)

```python
# 'perm' is more useful for n-dimensional tensors, for n > 2
```
于是问题来了，为什么要设置perm=[0, 2, 1]？当参数perm=[0, 2, 1]设置完成后，为什么会得到这样的转置结果呢？

## tf.transpose()函数及perm参数详解
这就要和原Tensor本身的shape有关了。

首先看Tensor x3是如何组成的。该Tensor中，最外层1个中括号包含了2个中括号，这两个中括号又分别包含了2个中括号，这两个中括号又包含了3个int型数值，所以其shape值为(2, 2, 3)。当我们将这个3维Tensor画成立体图时，如下图所示。
![3阶张量举例](https://img-blog.csdnimg.cn/20200607143242516.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwNjQyNTQ2,size_16,color_FFFFFF,t_70#pic_center)

```python
x3 = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(x3)  
# 输出 tf.Tensor([[[ 1  2  3]  
#                 [ 4  5  6]]   
# 
# 				 [[ 7  8  9]  
# 				  [10 11 12]]], shape=(2, 2, 3), dtype=int32)
```
关键来了，这里我们可以将perm理解为**切割该立方体的切割顺序**。我们已知Tensor x3的shape是(2, 2, 3)，它对应着原perm的切割顺序。这个顺序就是，**先竖着与侧边平行切一次**，**再横着切一次**，**再竖着平行于横边切一次**，如下图所示，就得到了Tensor原本的形状。
![分割方式举例1](https://img-blog.csdnimg.cn/20200607143550904.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwNjQyNTQ2,size_16,color_FFFFFF,t_70#pic_center)
我们将这种切割顺序依次定义为0，1，2，于是perm=[0, 1, 2]，如下图所示：
![切割顺序举例](https://img-blog.csdnimg.cn/20200607143702100.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwNjQyNTQ2,size_16,color_FFFFFF,t_70#pic_center)
在搞懂这个对应关系后。再来看如果不通过代码结果，我们如何确定转置后的Tensor形状。

当我们对这个3维Tensor x3进行转置，并设定perm值为[0, 2, 1]时，则此时对应的shape形状就会转化为(2, 3, 2)。为什么呢？

perm=[0, 2, 1]就意味着，对立方体要按照如下顺序进行切割：**先竖着与侧边平行切一次**，**再竖着平行于横边切一次**，**再横着切一次**，如下图所示，就得到了转置后Tensor的形状
![分割方式举例](https://img-blog.csdnimg.cn/20200607143837282.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwNjQyNTQ2,size_16,color_FFFFFF,t_70#pic_center)
![切割顺序举例](https://img-blog.csdnimg.cn/2020060714385678.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwNjQyNTQ2,size_16,color_FFFFFF,t_70#pic_center)
这时，我们使用函数语句 tf.transpose(x3, perm = [0, 2, 1]) 进行验证，转置结果与推演结果一致。也就是说，shape=(2, 2, 3) 的Tensor经过perm=[0, 2, 1]转置后，变为shape=(2, 3, 2)的Tensor。

```python
x3_ = tf.transpose(x3, perm = [0, 2, 1])
print(x3_)  
# 输出 tf.Tensor([[[1  4]  
# 				  [2  5]  
# 				  [3  6]]   
# 
#  				 [[7 10]  
# 				  [8 11]  
# 				  [9 12]]], shape=(2, 3, 2), dtype=int32)
```
这也是为什么在TensorFlow2.0官网教程中，**官方推荐在Tensor维度大于2时**，**使用perm参数进行转置操作**，会更方便的达到效果。**当然前提是你要明确原Tensor shape及你想要的变形后的Tensor shape，根据后续需求确定参数perm的值。**

希望这篇文章对大家理解张量Tensor有帮助！
