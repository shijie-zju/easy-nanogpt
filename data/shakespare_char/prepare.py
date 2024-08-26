
import os
import pickle
import requests
import numpy as np

print('2.1 读取当前目录下的input.txt文件')
input_file_path = os.path.join(os.path.dirname(__file__),'input.txt')
#如果当前没有.txt文件，再去网上下载,写入input.txt中
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

#读取txt文件并打印
with open(input_file_path,'r',encoding='utf-8') as f:
    data = f.read()
data_size = len(data)
print(f'数据集前50字符是{data[:50]}')
print(f'数据集的字符长度为：{data_size}')

print('2.2 创建字符级词表')
chars = sorted(list(set(data)))
vocab_size = len(chars)
print(f"词表序列可简化为{''.join(chars)}") #join将序列连接成字符串
print(f'vocab_size为：{vocab_size}')

print('2.3 创建 字符string-整数integers 映射表')
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] #将s文本编码为数字格式
decode = lambda l: ''.join([itos[i] for i in l]) #将l数字解码为字符格式
# print(f'编码后：{encode("hi there")}')
# print(f'解码后：{decode(encode("hi there"))}')

print('2.4 对原数据集编码')
data = encode(data)
print(f'data集前50项是：{data[:50]}')
print(f'数据编码后的长度{len(data)}')

print('2.5 划分训练验证集')
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
print(f'训练集序列长{len(train_data)},测试集序列长{len(val_data)}')

print('2.6 导出为存储numpy数组的bin文件')
#无符号16位整型
train_data = np.array(train_data, dtype=np.uint16)
val_data = np.array(val_data, dtype=np.uint16)
train_data.tofile(os.path.join(os.path.dirname(__file__),'train.bin'))
val_data.tofile(os.path.join(os.path.dirname(__file__),'val.bin'))

# 保存元信息，以便以后编码/解码
meta = {
    'data_size': data_size, #数据集大小
    'vocab_size': vocab_size,  # 词汇表大小
    'itos': itos,  # 整数到字符的映射
    'stoi': stoi,  # 字符到整数的映射
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:  # 打开文件用于写入
    pickle.dump(meta, f)  # 将元信息序列化并保存到文件





