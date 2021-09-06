### FastText

+ 文本分类：使用fastText进行文本分类的同时也会产生词的embedding，即embedding是fastText分类的产物

#### 字符级别的n-gram

+ word2vec把每个词当作是基本元素，为每个词生成词向量，忽略了单词内部的形态特征
+ fastText使用了字符级别的n-grams来表示一个单词
+ 采用tri-gram来表示apple：【apple】 -> 【<ap, app, ppl, ple, le>】,<表示前缀，>表示后缀，那么apple这个词的词向量就可以用5个trigram向量叠加得到
+ 优点：
  + 对于低频词生成的词向量效果会更好。因为它们的n-gram可以和其它词共享。
  + 对于训练词库之外的单词，仍然可以构建它们的词向量。我们可以叠加它们的字符级n-gram向量

#### 模型架构

![fasttext](https://tva1.sinaimg.cn/large/008i3skNgy1gu141p10xfj60hc0bwaam02.jpg)

+ 其结构与CBOW相似，隐含层都是对多个词向量的叠加平均
+ 不同点：
  + CBOW的输入是目标单词的上下文，fastText的输入是单个文档下多个单词及其n-gram特征，也就是文档特征，就是表达文档的向量
  + CBOW的输入单词被onehot编码过，fastText的输入特征是被embedding过
  + CBOW的输出是目标词汇，fastText的输出是文档对应的类别。
  + fastText在输入时，将单词的字符级别的n-gram向量作为额外的特征
  + fastText采用了分层Softmax，大大降低了模型训练时间。
+ 将整篇文档的词及n-gram向量叠加平均得到文档向量，然后使用文档向量做softmax多分类。
+ 效果好的原因：
  + **使用词embedding的叠加而非词本身作为特征**
  + **字符级n-gram特征的引入对分类效果会有一些提升** 。
+ 