# easy-nanoGPT

关于nanoGPT的学习和仿写

### 快速开始



### 文件介绍
train.py：模型训练文件

configurator.py：train.py调用的配置文件，用于根据控制台输入读取不同config路径下的已设定参数

config/：该路径下保存了一系列固定超参数配置，用于不同规模训练

data/：路径下保存了不同的训练数据集

### 架构思路

##### 1.整体架构
①全局变量配置

②数据集加载与词表创建

③网络设计

④网络训练与验证

##### 2.模型的参数传递

其中（1）-（4）为参数被参考设定的优先级，如（1）会将（2）重写；

**总结为：脚本设定参数 > 脚本指定参数配置文件 > 开头参数初始化 > class config网络默认参数**

①**train.py前几行** 会进行全体参数的初始化，定义为全局变量（3）

②**运行python脚本** python script.py config/traingpt.py --key1=value1 --key2=value2

此时首先执行config/路径下的参数文件进行 对应参数重写（2）

然后针对脚本中的指定的参数，在全局变量中进行 对应参数重写（1）

③**数据集加载与词表创建** 获取vocab_size

④**有关网络设计的参数** 创建model_args网络参数汇总字典，将class config中 对应参数重写（4）

⑤**class Config** 存储网络参数的类，若无重写则参考它运行网络



