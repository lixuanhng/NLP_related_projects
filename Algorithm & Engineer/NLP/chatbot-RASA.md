# chatbot-RASA

+ rasa 中国社区

+ chatbot对话机器人

  + 架构
    + 人--声音--语音识别（ASR）--转化文本--==NLU==（语言理解）--输出meaning&context（至此，完成一轮的信息）--对话管理（==DM==，如何引导用户，管理对话交互）--输出动作或指令（action）--==NLG==（语言生成，生成文本）--TTS（语音合成）--人
  + NLG：语义转文本，主要使用模板的方式
  + NLU
    + 功能：
      + 非结构化文本，转化为结构化的语义
      + 无限种可能（多种问法）转化为有限的组合（结构化结果一致）（近义词能同统一吗？）
    + 识别意图
    + 识别实体
  + DM：
    + 对话状态追踪（前几轮对话中的信息），DST
    + 对话策略（模拟人的判断），DP

+ 对话机器人工程

  + 数据获取
  + 数据扩充，数据不平衡
  + 数据标注：业务需求，长尾需求
  + 数据清洗：错标，漏标
  + 数据和模型版本控制
  + 模型部署：字典转化，序列解码
  + 模型效果评估：模型哪里不行？如何改进？
  + RASA能够管理模型的相关问题

+ RASA 用于生产，面向任务

  + 端到端（训练数据），工业级别（部署可扩展），任务型（主要）+FAQ型
  + RASA NLU：intent + entity => function(x1, x2, x3, ...) 参数为实体，func为意图
    + 训练数据：markdown 格式（给出意图，实体）
    + 组件配置：YAML格式
      + 使用 pipeline，组件方便替换
      + \- name，一个组件，对应具体实现的代码
      + 组件配置，传递给上面的组件
  + RASA core
    + DST 对话状体追踪
      + 工作方式：将多轮用户输入，action分别连续加入DS中
    + 训练数据：Story（markdown）
      + \* 表示用户说的话，\- 表示action，还有action的副作用
    + Domain：YAML格式
      + DP 存储 action 的数量
      + 管理意图和action
    + 组件配置：policy，YAML格式
      + 工作原理：5轮历史，纵向为一个DS
      + 基于规则：mapping policy
      + 基于记忆：Memorization policy
      + 基于DNN预测：keras policy，能够做到泛化，输入为tracker state
      + 基于编程：form policy
    + action sever，调用外部接口
    + 自定义 action：
  + RASA CLI
  + RPC 方式调用

  

