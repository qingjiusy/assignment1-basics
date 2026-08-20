import torch.nn as nn
import torch

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        # step 1: 建一个形状 (vocab_size, d_model) 的可学习查找表
        self.weight = nn.Parameter(torch.zeros(vocab_size, d_model))
        # step 2: 按 handout 初始化（可选）
        return

    def forward(self, token_ids):
        # step 3: 用 token_ids 作为“行下标”索引 self.weight
        #   —— 高级索引：weight[<某个整型张量>] 会一次取出所有对应行
        #   自问：返回的 shape 是不是 token_ids.shape + (d_model,)？
        #  整数张量索引！
        return self.weight[token_ids]
