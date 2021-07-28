## python工程能力

### 由部署者指定参数

+ 代码中需要引入外部的url链接
  + 使用python自带的os.getenv()
  + 比如 url = os.getenv('HUGEGRAPH_URL')，其中url是代码中使用的变量名，而HUGEGRAPH_URL是部署代码中使用的变量名，部署者可以使用部署命令指定该变量的值。