@[TOC](BERT文本相似度实战)


# Bert 文本相似度实战（使用详解）

大家好！其实 BERT 的实战代码已经很多了，也被诸多大佬引用和实验过。这篇主要用来记录自己在使用与处理时注意到的点，私以为比较适合BERT小白快速上手。

当修改模型后，并没有跑通，或查看完整代码，欢迎大家查看我的github ([lixuanhng](https://github.com/lixuanhng/NLP_tricks/tree/master/Bert_sim))

## 参考资料
这篇笔记主要参考了一个智能客服的项目，项目地址请点击[这里](https://github.com/charlesXu86/Chatbot_CN)。其中主要借鉴了该项目的子项目chatbot_retrieval中的文本相似度计算代码。

另外还想说明的是在本篇笔记中，我不打算再详细说明如何下载代码和预训练模型了，请随机找一篇讲述bert的博客，就能找到这部分内容。总结来说，对于将要使用bert模型的使用者，首先应该下载bert代码，其中主要包含了

> modeling.py
> optimization.py
> tokenization.py
> run_classifier.py

等代码。当我们做文本相似度模型时，主要改造的是  `run_classifier.py`。

同时，我们还应当下载bert中文预训练模型， `chinese_L-12_H-768_A-12.zip` 解压后可以看到

> bert_config.json                                          是BERT在训练时可选调整参数
> bert_model.ckpt.meta                                 开头的文件是负责模型变量载入的
> bert_model.ckpt.data-00000-of-00001
> bert_model.ckpt.index
> vocab.txt                                                     中文词表文件

至此，我们的准备工作就算是完成了。

## 代码准备
能够完整的跑完bert的代码，需要准备以下几个文件。

#### 数据
首先，我们需要准备一份数据。对于文本相似度的任务，数据格式为每行是一组句子对，并使用0或1标注句子对中两个句子是否相似，0表示不相似，1表示相似。

如果在自己场景下的数据并没有那么多，可以寻找一些开源的数据集。这里我们以蚂蚁金服的智能客服对话数据集作为测试数据集。数据结构为句子1，句子2，标签，三项之间由制表符隔开。例如：

> 为什么我申请额度输入密码就一直是那个页面	为什么要输入支付密码来验证	0

数据相关检查与下载，请移步github  ([lixuanhng](https://github.com/lixuanhng/NLP_tricks/tree/master/Bert_sim))
#### 参数配置文件
在跑通代码之前，我们需要对代码的超参数进行配置。这里选择新建一个python文件，将超参数全部指定在该文件中。在  `run_classifier.py` 中只要将超参数python文件导入进去即可。在超参数python文件中，我们主要关注的是以下几个方面的参数值。

>  bert_config_file  bert配置文件bert_config.json的路径
>  vocab_file  词表文件vocab.txt的路径
>  data_dir  数据路径，要分成train, test, dev三个文件，类型为csv
>  out_put  模型输出地址
>  init_checkpoint  预训练模型bert_model.ckpt的路径
>  task_name  任务名称为 sim

其余参数不变，完整的参数设置代码可见下图（使用时需修改自己的各文件路径）：
```python
class Config():

    def __init__(self):
        self.bert_config_file = '/bert_dir/bert_config.json'
        self.vocab_file = 'bert_dir/vocab.txt'
        self.data_dir = os.path.join(basedir2, 'data/bert_sim/')
        self.output_dir = basedir2 + '/Bert_sim/results'
        self.predict_file = basedir2 + '/data/bert_sim/dev.txt'
        self.test_file = basedir2 + '/data/bert_sim/test.txt'
        self.init_checkpoint = '/bert_dir/bert_model.ckpt'
        self.train_checkpoint = basedir2 + '/Bert_sim/results'

        self.do_lower_case = True
        self.verbose_logging = False
        self.master = None
        self.version_2_with_negative = False
        self.null_score_diff_threshold = 0.0
        self.use_tpu = False
        self.tpu_name = None
        self.tpu_zone = None
        self.gcp_project = None
        self.num_tpu_cores = 8
        self.task_name = 'sim'  # 重要
        self.gpu_memory_fraction = 0.8

        self.max_seq_length = 128
        self.doc_stride = 128
        self.max_query_length = 64

        self.do_train = False
        self.do_predict = False
        self.batch_size = 20
        self.predict_batch_size = 8
        self.learning_rate = 5e-5
        self.num_train_epochs = 3.0
        self.warmup_proportion = 0.1
        self.save_checkpoints_steps = 1000
        self.iterations_per_loop = 1000
        self.n_best_size = 20
        self.max_answer_length = 30
        self.eval_batch_size = 16
```

#### 数据输入
在完成超参数的配置后，下一步，需要将`run_classifier.py`中关于数据输入的部分进行替换与再精确。

由于是文本相似度检查任务，我们首先将`run_classifier.py`改名为`run_similarity.py`。打开该文件后，我们可以看到有很多个`xxxProcessor`类，这些类都有同一个父类 `DataProcessor`，其中`DataProcessor`提供4个方法，分别是：

>  Processor for the XNLI data set
>  Processor for the MultiNLI data set (GLUE version)
>  Processor for the MRPC data set (GLUE version)
>  Processor for the CoLA data set (GLUE version)

Processor就是用来获取对应的训练集、验证集、测试集的数据与label的数据，并把这些数据喂给BERT的，而我们要做的就是自定义新的Processor并模仿上述4个方法，也就是说我们只需要提供我们自己场景对应的数据。

这里我们自定义了一个名叫SimProcessor的类，完成投喂训练数据，验证数据和测试数据。这部分代码如下：
```python
class SimProcessor(DataProcessor):
	# 载入训练数据，需要确定训练数据的路径及读取方式
	# 验证，测试数据集相同
    def get_train_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'train.csv')
        train_df = pd.read_csv(file_path, encoding='utf-8')
        train_data = []
        for index, train in enumerate(train_df.values):
            guid = 'train-%d' % index
            # 根据数据的实际结构确定text_a, text_b（两个待匹配的句子）, label（标签）
            # 验证，测试数据集相同
            text_a = tokenization.convert_to_unicode(str(train[0]))
            text_b = tokenization.convert_to_unicode(str(train[1]))
            label = str(train[2])
            train_data.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return train_data

    def get_dev_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'dev.csv')
        dev_df = pd.read_csv(file_path, encoding='utf-8')
        dev_data = []
        for index, dev in enumerate(dev_df.values):
            guid = 'test-%d' % index
            text_a = tokenization.convert_to_unicode(str(dev[0]))
            text_b = tokenization.convert_to_unicode(str(dev[1]))
            label = str(dev[2])
            dev_data.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return dev_data

    def get_test_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'test.csv')
        test_df = pd.read_csv(file_path, encoding='utf-8')
        test_data = []
        for index, test in enumerate(test_df.values):
            guid = 'test-%d' % index
            text_a = tokenization.convert_to_unicode(str(test[0]))
            text_b = tokenization.convert_to_unicode(str(test[1]))
            label = str(test[2])
            test_data.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return test_data

    def get_sentence_examples(self, questions):
        for index, data in enumerate(questions):
            guid = 'test-%d' % index
            text_a = tokenization.convert_to_unicode(str(data[0]))
            text_b = tokenization.convert_to_unicode(str(data[1]))
            label = str(0)
            yield InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)

    def get_labels(self):
        return ['0', '1']
```


## 开始实践
完成上述工作后，就可以开始运行`run_similarity.py`了。运行之前，请注意超参数配置python文件中是否`self.do_train = True`。只有当训练开关打开时，模型才会进行训练，否则模型不回有任何返回。`self.do_predict = True`相同。`run_similarity.py`中输出的两个句子的相似度，是一个浮点值的相似度，可以输出查看相似度的值。

```python
if __name__ == '__main__':
    sim = BertSim()
    if cf.do_train:
        sim.set_mode(tf.estimator.ModeKeys.TRAIN)
        sim.train()
        sim.set_mode(tf.estimator.ModeKeys.EVAL)
        sim.eval()

    if cf.do_predict:
        sim.set_mode(tf.estimator.ModeKeys.PREDICT)
        # while True:
        # sentence1 = input('sentence1: ')
        # sentence2 = input('sentence2: ')
        sentence1 = '十万预算买什么车好？'
        sentence2 = '当预算只有十万元时，买哪种车最好？'
        predict = sim.predict(sentence1, sentence2)
        # 输出值就是浮点值的相似度
        print(predict[0][1])
```

## 预测结果
```python
sentence1 = '十万预算买什么车好？'
sentence2 = '当预算只有十万元时，买哪种车最好？'
print(predict[0][1])
# 预测值为 0.9995735
```

```python
sentence1 = '今天晚上要不去吃西餐吧？'
sentence2 = '当预算只有十万元时，买哪种车最好？'
print(predict[0][1])
# 预测值为 0.0011827872
```
基本达到要求。

完整代码请查看： ([lixuanhng](https://github.com/lixuanhng/NLP_tricks/tree/master/Bert_sim))

希望这篇文章对大家有帮助，欢迎批评指正。
