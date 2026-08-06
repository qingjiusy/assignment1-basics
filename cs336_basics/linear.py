import torch.nn as nn
import torch

class Linear(nn.Module):
        def __init__(self, d_in: int, d_out: int):
            super().__init__()
            # step 1: 建一个形状为 (d_out, d_in) 的可学习参数
            # nn.Parameter(...)：告诉 PyTorch：“这是模型参数，需要被训练”
            # torch.randn 和 torch.empty 最大区别是：前者会写入随机值，后者只分配内存但不填值。
            self.weight = nn.Parameter(torch.empty(d_out, d_in))
            # step 2: 按 handout 的截断正态初始化它（可选，不影响单测）
            # 最后有_，表示此api是原地修改
            nn.init.trunc_normal_(self.weight)

        def forward(self, x):
            # step 3: 把 x 的最后一维 (d_in) 与 weight 的 d_in 维收缩，得到最后一维 d_out
            #   想清楚是 x @ weight 还是 x @ weight.T —— 用小例子核对形状
            #   （einsum 也可以：给每个维度起名，让收缩的维同名）
            # return ...
            return x @ self.weight.T
