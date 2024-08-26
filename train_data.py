
import os.path
import torch
from model.nanogpt import Nanogpt_LM, NanogptConfig

# hyperparameters超参数
# 1.data
dataset = 'shakespare_char'
gradient_accumulation_steps = 5 #梯度累计步数5*8（几轮才更一次梯度）
batch_size = 16 #每轮批次维度：每次送入网络的独立序列数64
block_size = 32 #时间维度：序列输入的最大字符长度256
# 2.optimizer
max_iters = 5000 #训练迭代数5000
eval_interval = 100 #验证迭代数500
learning_rate = 1e-3 #学习率3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200 #一轮验证所取的样本数量200
# 3.model
n_embd = 64 #嵌入层维数384
n_head = 4 #注意力头个数6
n_layer = 4 #注意力块的重复个数6
dropout = 0.0 #前-后向传播时，随机将神经元输出置零的比率0.2
#--------------------

torch.manual_seed(1337)

print('1.全局变量配置')
#将全局变量的变量名保存后续做键；globals()是存储所有全局变量的字典！
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
#打开读取执行配置.py文件，实现对全局变量参数的修改
exec(open('configurator.py').read())
#构建超参数字典，保存的全局变量名：对应变量值
config = {k: globals()[k] for k in config_keys}

#打印每次迭代（参数更新一次）处理的tokens数量
tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
print(f'每次参数更新所迭代的词元数为：{tokens_per_iter}={gradient_accumulation_steps}*{batch_size}*{block_size}')

print('2.数据集加载与词表创建')
##数据集下载
# #!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

#data_dir = os.path.join('data', dataset)
#数据集加载
with open('input.txt','r',encoding='utf-8') as f:
    text = f.read()
print(f'数据集前50字符是{text[:50]}')
print("数据集中字符的长度是：",len(text)) #1M字符量

#创建字符级词表
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"词表序列可简化为{''.join(chars)}") #join将序列连接成字符串
print(f'vocab_size为：{vocab_size}')

#创建 字符string-整数integers 映射表
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] #将s文本编码为数字格式
decode = lambda l: ''.join([itos[i] for i in l]) #将l数字解码为字符格式
print(f'编码后：{encode("hi there")}')
print(f'解码后：{decode(encode("hi there"))}')


print('-------用网络训练和测试切片-------')

print('----原数据集编码----')
data = torch.tensor(encode(text),dtype=torch.long)
print(f'数据格式：{data.shape},数据类型：{data.dtype}')
print(f'data前50项是：{data[:50]}')

print('----划分训练验证集----')
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
print(f'训练集序列长{len(train_data)},测试集序列长{len(val_data)}')


def get_batch(split): #split决定输入的数据源自训练集还是测试集
    '''数据加载,[B,T]'''
    data = train_data if split == 'train' else val_data #取数据
    ix = torch.randint(len(data) - block_size, (batch_size,)) #随机抓4个数，在0到（数据长度-块长）间随机生成
    x = torch.stack([data[i:i+block_size] for i in ix]) #ix是四个输入序列的起始索引
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) #y相对x移动一位
    x, y = x.to(device), y.to(device)
    return x,y

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


print(f'数据集长度:{vocab_size},嵌入层大小:{n_embd}\n学习率:{learning_rate},最大训练轮数:{max_iters},每轮训练序列样本数:{batch_size},损失计算间隔:{eval_interval}')

print('----网络设计----')
#将模型参数统一为model_args的字典目的是后续方便重写其中的值，最后结果重写入model中的初始化类中
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  vocab_size=vocab_size, dropout=dropout)
# **可解包类中的字典并将结果重写入对应模型的参数初始化类中
gptconf = NanogptConfig(**model_args)
print(gptconf)
model = Nanogpt_LM(gptconf)
m = model.to(device)
print(f'当前运行在{device}')
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters参数量')
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

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

print(f'数据集长度:{vocab_size},嵌入层大小:{n_embd}\n学习率:{learning_rate},最大训练轮数:{max_iters},每轮训练序列样本数:{batch_size},损失计算间隔:{eval_interval}')
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

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



