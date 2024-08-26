
import os.path
import pickle
import math
import numpy as np
import torch
from model.nanogpt import Nanogpt_LM, NanogptConfig

# hyperparameters超参数
# 1.I/O
out_dir = 'out' #模型输出路径
init_from = 'scratch' #模型训练从头还是从检查点 'scratch' or 'resume' or 'gpt2*'

eval_iters = 200 #一轮验证所取的样本数量200
# 2.system
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 3.data
dataset = 'shakespare_char'
gradient_accumulation_steps = 5 #梯度累计步数5*8（几轮才更一次梯度）
batch_size = 16 #每轮批次维度：每次送入网络的独立序列数64
block_size = 16 #时间维度：序列输入的最大字符长度256
# 4.optimizer
max_iters = 5000 #训练迭代数5000
eval_interval = 100 #验证迭代数500
learning_rate = 1e-3 #学习率3e-4
weight_decay = 1e-1 #权重衰减
# 5.learning rate
warmup_iters = 2000 #热身
lr_decay_iters = 600000
min_lr = 6e-5
# 6.model
n_embd = 64 #嵌入层维数384
n_head = 4 #注意力头个数6
n_layer = 4 #注意力块的重复个数6
dropout = 0.0 #前-后向传播时，随机将神经元输出置零的比率0.2
bias = False #偏差
#--------------------

torch.manual_seed(1337)

print('1.全局变量配置')
#将全局变量的变量名保存后续做键；globals()是存储所有全局变量的字典！
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
#打开读取执行配置.py文件，实现对全局变量参数的修改
exec(open('configurator.py').read())
#构建超参数字典，保存的全局变量名：对应变量值
config = {k: globals()[k] for k in config_keys}

#打印每次迭代（参数更新一次）处理的tokens数量=梯度累计步数*bs*bz
tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
print(f'每次参数更新所迭代的词元数为：{tokens_per_iter}={gradient_accumulation_steps}*{batch_size}*{block_size}')

#I/O 设置输出目录out_dir和设备device_type
os.makedirs(out_dir, exist_ok=True)

print('2.数据集加载与词表创建')
data_dir = os.path.join('data', dataset)

def get_batch(split): #split决定输入的数据源自训练集还是测试集
    '''数据加载,[B,T]'''
    ##A.数据为数组格式的train_data和val_data
    #data = train_data if split == 'train' else val_data
    #B.数据存储为.bin文件，由memmap读取train.bin和val.bin
    #memmap:创建内存映射文件，可以使用数组索引访问文件内容，比读写更快
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    ix = torch.randint(len(data) - block_size, (batch_size,)) #随机抓4个数，在0到（数据长度-块长）间随机生成
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device) #数据放在设备上
    return x,y

#读取存储着数据集信息的元数据
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        #pickle: pkl文件是序列化 Python 对象的二进制文件
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f'数据路径为{meta_path}')
    print(f'数据集的元数据中词表大小vocab_size:{meta_vocab_size}')

print('3. 模型参数配置与模型实例化')
#将模型参数统一为model_args的字典 目的是后续方便重写其中的值，最后结果重写入model中的初始化类中
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)
#不同情况下的模型实例化和权重加载
if init_from == 'scratch':
    print('从头开始初始化模型')
    #如果词表空，说明数据集.pkl文件不存在！那就设成默认的
    if meta_vocab_size is None:
        print('meta.pkl文件不存在，因此将词表大小设定与GPT-2统一的50304')
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = NanogptConfig(**model_args)# **可解包类中的字典并将结果重写入对应模型的参数初始化类中
    print(gptconf)
    model = Nanogpt_LM(gptconf) #实例化模型
elif init_from == 'resume':
    print(f'从{out_dir}路径下的检查点恢复并继续训练模型')
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device) #map_location指定加载检查点时数据应该映射到的设备

    #从检查点中加载模型参数：和本次模型训练的参数必须一致（除了不改变模型结构的dropout）
    checkpoint_model_args = checkpoint['model_args']  # 检查点的模型参数
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = NanogptConfig(**model_args) # **可解包类中的字典并将结果重写入对应模型的参数初始化类中
    print(gptconf)
    model = Nanogpt_LM(gptconf) #实例化模型
    #从检查点中加载模型权重
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod' #检查点的模型参数可能有前缀，有的话需要去掉
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            # 更新字典取k后部分 = 字典取出键k的值
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    #从检查点取出迭代数和最小验证损失
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f'从OpenAI GPT-2权重初始化{init_from}')
    #初始化,直接调gpt2实例化模型和参数
    overrides_args = dict(dropout=dropout)
    model = Nanogpt_LM.from_pretrained(init_from, overrides_args)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

# #模型块的大小修改;设置的块大小 小于 模型配置的块大小时，要裁剪 ???????????????????
# print(f'block_size{block_size},model.config{model.config.block_size}')
# if block_size < model.config.block_size:
#     model.crop_block_size(block_size)
#     model_args['block_size'] = block_size

model.to(device) #模型放在设备上

print(f'模型当前运行在{device}')
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters参数量')

print('4.AdamW优化器与学习率')
#optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
optimizer = model.configure_optimizers(weight_decay, learning_rate)
#如果是检查点继续，则优化器需要用检查点的，然后至此检查点的内存就可以释放了
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None
print(f'初始学习率learning_rate:{learning_rate},衰减weight_decay:{weight_decay}')
print(f'warmup_iters:{warmup_iters},lr_decay_iters:{lr_decay_iters},min_lr:{lr_decay_iters}')
#⑨学习率衰减调度器
# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

#损失计算函数
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters-1:
        losses = estimate_loss()
        t_loss = losses['train']
        v_loss = losses['val']
        print(f'step {iter}: train loss {t_loss:.4f}, val loss {v_loss:.4f}')

    #每轮都随机抓取[B,T]数据进行训练
    xb, yb = get_batch('train')

    #正向传播、损失计算与反向更新
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#print(f'数据集长度:{vocab_size},嵌入层大小:{n_embd}\n学习率:{learning_rate},最大训练轮数:{max_iters},每轮训练序列样本数:{batch_size},损失计算间隔:{eval_interval}')
context = torch.zeros((1,1), dtype=torch.long, device=device)
#print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

'''
上述总结：
1.加载数据集
2.创建字符级词表
3.构建词-整数映射表
4.按照映射表编码数据集
5.网络设计
外1层：控制文本最大长度blocksize，每个样本为随机采样blocksize长输入idx，标签为blocksize长的targets（相比idx后移了一个字符）
2层：循环将编码文本idx按定长输入网络，输出长度为idx+1的文本继续输入网络
3层：idx文本作为输入，targets可作标签计算损失
'''



