# Easy-NanoGPT

这是关于nanoGPT的学习和改写项目，支持更多情形下的可调控训练场景。

通过该项目，你可以体验

1.自己准备数据集，并基于已有数据从头训练一个小规模的chatgpt

2.在已有模型基础上进行训练

3.调用已训练好的模型进行推理

n.还在开发更新中...

参考链接： https://github.com/karpathy/nanoGPT

### 快速开始
安装相关包

`pip install torch numpy transformers datasets tiktoken wandb tqdm`

#### 训练一个能跑就行、胡言乱语的gpt迷你版本
准备数据：

```sh
python data/shakespeare_char/prepare.py
```

训练小模型：

```sh
python train.py config/train_nanogpt_shakespearchar_cpu.py --max_iters=1000 --n_embd=64
```

试试效果：

```sh
python sample.py --file_name=ckpt_1000.pt --num_samples=3
```

(val loss ~ 3.20)

#### 训练一个稍微正规、能说人话的gpt进阶版本
准备数据：

```sh
python data/shakespeare_char/prepare.py
```

训练小模型：

```sh
python train.py config/train_nanogpt_shakespearchar_cpu.py
```


试试效果：

```sh
python sample.py --file_name=ckpt_5000.pt --num_samples=3
```

(val loss ~ 2.00)

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

**总结为：脚本设定参数(还有检查点处保存的模型参数) > 脚本指定参数配置文件 > 开头参数初始化 > class config网络默认参数**

①**train.py前几行** 会进行全体参数的初始化，定义为全局变量（3）

②**运行python脚本** python script.py config/traingpt.py --key1=value1 --key2=value2

此时首先执行config/路径下的参数文件进行 对应参数重写（2）

然后针对脚本中的指定的参数，在全局变量中进行 对应参数重写（1）

③**数据集加载与词表创建** 获取vocab_size

④**有关网络设计的参数** 创建model_args网络参数汇总字典，将class config中 对应参数重写（4）

⑤**class Config** 存储网络参数的类，若无重写则参考它运行网络



