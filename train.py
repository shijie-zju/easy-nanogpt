
import os.path
import pickle
import math
import time

import numpy as np
import torch
from model.nanogpt import Nanogpt_LM, NanogptConfig

# hyperparameters超参数
# 1.I/O
out_dir = 'out/shakespeare_char' #模型输出路径
init_from = 'scratch' #模型训练从头还是从检查点 'scratch' or 'resume' or 'gpt2*'
always_save_checkpoint = True #如果true则每次验证时不管loss是不是最小，都保存检查点

eval_iters = 200 #一轮验证所取的样本数量200
# 2.system
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 3.data
dataset = 'shakespeare_char'
gradient_accumulation_steps = 5 #梯度累计步数5*8（几轮才更一次梯度）
batch_size = 16 #每轮批次维度：每次送入网络的独立序列数64
block_size = 16 #时间维度：序列输入的最大字符长度256
# 4.optimizer
max_iters = 1000 #训练迭代数5000
eval_interval = 50 #验证迭代数(保存检查点)500
learning_rate = 1e-3 #学习率3e-4
weight_decay = 1e-1 #权重衰减
beta1 = 0.9
beta2 = 0.95

# 5.learning rate
whether_decay_lr = True #True则衰减学习率
warmup_iters = 2000 #热身
lr_decay_iters = max_iters# should be ~= max_iters per Chinchilla

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
    meta_data_size = meta['data_size']
    meta_vocab_size = meta['vocab_size']
    print(f'数据路径为{meta_path}')
    print(f'数据集的元数据中，规模为：{meta_data_size}词表大小vocab_size:{meta_vocab_size}')

iter_num = 0
best_val_loss = 1e9
print('3. 模型超参数配置与模型实例化')
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
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device)
#如果是检查点继续，则优化器需要用检查点的，然后至此检查点的内存就可以释放了
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None
print(f'初始学习率learning_rate:{learning_rate},衰减weight_decay:{weight_decay}')
print(f'warmup_iters:{warmup_iters},lr_decay_iters:{lr_decay_iters},min_lr:{lr_decay_iters}')
#⑨学习率衰减调度器(带预热的余弦)，获取learning_rate
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

print('5.训练training loop')
#损失计算函数(eval_iters)
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

X, Y = get_batch('train') #第一次数据抓取
t0 = time.time()

#一大轮是gradient_accumulation_steps次数据抓取利用、正反向传播，1次梯度更新
while True:
    #设置学习率，遍历优化器中的参数组，寻找lr参数并修正更新
    lr = get_lr(iter_num) if whether_decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    #每隔eval_interval，评估损失, 保存检查点
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        train_loss = losses['train']
        val_loss = losses['val']
        print(f'step {iter_num}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, lr {lr}')

        #如果验证时发现损失小于最佳损失 或 设置为总是保存检查点
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val'] #这里理解是，如果每次都保存检查点，每次都知道loss，best记录的是当前的也无所谓了
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f'保存检查点至 {out_dir}')
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt_' + str(iter_num) + '.pt'))

    #gradient_accumulation_steps次 前向后向传播，数据抓取
    for micro_step in range(gradient_accumulation_steps):
        logits, loss = model(X, Y)
        loss = loss / gradient_accumulation_steps
        loss.backward()
        # 下一轮数据准备
        X, Y = get_batch('train')

    #1次 梯度更新
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    #计时
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    iter_num += 1
    if iter_num > max_iters:
        break

print(f'6.结果记录')
import pandas as pd
from datetime import datetime

def read_dataframe_from_csv(filename):#读取csv文件
    try:
        df = pd.read_csv(filename)
        print(f"从 '{filename}' 成功读取DataFrame。")
        return df
    except Exception as e:
        print(f"读取文件 '{filename}' 时出错: {e}")
        return None
def save_dataframe_to_csv(df, filename):
    df.to_csv(filename, index=False)
    print(f"DataFrame已保存到 '{filename}'。")

csv_file = os.path.join('out', 'log_train.csv')

# 读取csv文件,检查文件是否存在没有就创建
if os.path.exists(csv_file):
    # 尝试读取CSV文件
    try:
        df_read = pd.read_csv(csv_file)
    except Exception as e:
        print(f"读取文件 '{csv_file}' 时出错: {e}")
        df_read = None  # 确保在发生错误时df_read为None
else:
    print(f"文件 '{csv_file}' 不存在。创建新文件。")
    df_read = pd.DataFrame()  # 创建一个空的DataFrame
# 如果df_read是None或者为空的DataFrame，初始化一个新的DataFrame
if df_read is None or df_read.empty:
    print("创建新的DataFrame。")
    df_read = pd.DataFrame(columns=['time', 'out_dir', 'init_from', 'gra_acc_steps', 'batch_size', 'block_size', 'max_iters', 'learning_rate', 'n_embd', 'n_head', 'n_layer', 'loss'])

now = datetime.now()
# 格式化时间输出为 年-月-日 时:分:秒
formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
# 构建DataFrame
data = {
    'time': formatted_now,
    'out_dir': config['out_dir'],
    'init_from': config['init_from'],
    'gra_acc_steps': config['gradient_accumulation_steps'],
    'batch_size': config['batch_size'],
    'block_size': config['block_size'],
    'max_iters': config['max_iters'],
    'learning_rate': config['learning_rate'],
    'n_embd': config['n_embd'],
    'n_head': config['n_head'],
    'n_layer': config['n_layer'],
    'loss': loss.item()
}
new_row = pd.DataFrame(data, index=[0])
df_read = pd.concat([df_read, new_row], ignore_index=True)
print(df_read.tail(5))
save_dataframe_to_csv(df_read, csv_file)
