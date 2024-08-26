#nanogpt
import math
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


class Head(nn.Module):
    def __init__(self, config, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size))) #创建的tril变量不是参数，因此需要用寄存器缓冲区将其分配给pytorch模块

        self.dropout = nn.Dropout(config.dropout) #丢弃一些，随机阻止某些节点间通信

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x) #[B,T,head_size]
        k = self.key(x) # [B,T,h_s]
        wei = q @ k.transpose(-2,-1) * C ** -0.5 #[B,T,h_s] @ [B,T,h_s] -> [B,T,T]
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #[B,T,T]
        wei = F.softmax(wei, dim=-1) #[B,T,T]
        wei = self.dropout(wei)
        v = self.value(x) #[B,T,hs]
        out = wei @ v #[B,T,T] @ [B,T,hs] -> [B,T,hs]
        return out

class MultHeadAttention(nn.Module):
    def __init__(self, config, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd) #回到残差路径的投影层,即做了一层线性变换
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) #[B,T,H,C]
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd), #论文中要求扩大隐层维度4倍
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd), #回到残差路径的投影层
            nn.Dropout(config.dropout), #在残差连接回到残差路径前可添加的内容
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_head
        # 每个头负责的隐藏层维数 =希望的总维数 // 注意力头数, 在最后各头cat连接后恢复总隐藏层数
        self.sa = MultHeadAttention(config, head_size)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x)) #[B,T,hd=n_embd]
        x = x + self.ffwd(self.ln2(x)) #[B,T,n_embd]
        return x


@dataclass
class NanogptConfig:
    #@dataclass告诉了解释器这个类有常见方法只需要存储数据就行
    # 1. data
    block_size: int = 32
    vocab_size: int = 65 #词表大小，不在train中做超参只在train中获取
    # 3.model
    n_embd: int = 64
    n_head: int = 2
    n_layer: int = 2
    dropout: float = 0.0
    bias: bool = True

class Nanogpt_LM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd) #词嵌入
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd) #位置嵌入
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd) #最后的Layernorm层
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    #前向传播函数
    def forward(self,idx,targets=None):
        '''前向传播，获得各位置对于所有词表的概率和与标签相比的损失'''
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) #[B,T,n_embd]
        pos_emb = self.position_embedding_table(torch.arange(T,)) #获取0到t-1整数表示位置 [T,C]
        x = tok_emb + pos_emb #[B,T,C]右对齐后整批次广播
        x = self.blocks(x) #[B,T,n_embd]
        x = self.ln_f(x)
        logits = self.lm_head(x) #[B,T,vocab_size]

        if targets == None: #若没有标签
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) #[B*T,C] [4*8,64]

            targets = targets.view(B*T) #[B*T,] 拉成一维的为了迎合交叉熵损失函数(B*T) [4*8,]
            #print(f'shape{logits.shape}and{targets.shape}')
            loss = F.cross_entropy(logits, targets)#[32,64]中从64类别分别选出32个词的最高分，32个标签词的结果（范围0-63）和相应位置ln值相乘
            logits = logits.view(B, T, C)

        return logits, loss

    def generate(self, idx, max_new_tokens): #idx [4,8]
        '''根据输入前向传播后的概率结果采样出新生成的字符，并更新延长输入'''
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:] #[B,<=T] 输入超过最大block_size需裁剪否则位置编码会失效
            logits, loss = self(idx_cond) #前向传播 logits [B,T,C]
            logits = logits[:,-1,:] #[B,T,C]->[B,C] 取最后一个字符判概率
            probs = F.softmax(logits,dim=-1) # [B,C] 得到最后一个字符在词表上的各概率
            idx_next = torch.multinomial(probs,num_samples=1) #[B,1] 按照随机种子进行概率采样，得到最后一字符的预测标号
            idx = torch.cat((idx,idx_next),dim=1) #[B,T+1]

        return idx

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
        config_args['bias'] = True  # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = NanogptConfig(**config_args)
        model = Nanogpt_LM(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def crop_block_size(self, block_size):
        #???????手动设定的block_size不是会将值赋给config中的吗，这两个值应该始终相同啊
        #手动规定的block_size如果更小，需要裁剪模型配置中的block_size和模型
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
