from cs336_basics.linear import Linear
from cs336_basics.silu import silu
import torch
import torch.nn as nn

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # step 1: 建三条无 bias 的线性投影(可复用第01章的 Linear)
        #   w1: (d_ff, d_model)  上投影, 结果过 SiLU
        #   w3: (d_ff, d_model)  上投影, 当门
        #   w2: (d_model, d_ff)  下投影回 d_model
        #   注意: 用传入的 d_ff, 不要写死成 8/3*d_model
        #   self.w1 = ...   self.w2 = ...   self.w3 = ...
        self.w1 = Linear(d_model, d_ff)
        self.w3 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        
    def forward(self, x):
        # step 2: 一路上投影 W1 x, 过 SiLU -> "内容/激活"    (形状 (..., d_ff))
        # step 3: 另一路上投影 W3 x -> "门"                  (形状 (..., d_ff))
        #   想清楚: 哪一路过 SiLU? 哪一路当门? (对照公式, 别接反)
        # step 4: 两路逐元素相乘 (门控)                       (形状 (..., d_ff))
        # step 5: 用 W2 下投影回 d_model                     (形状 (..., d_model))
        # return ...
        return self.w2(silu(self.w1(x)) * self.w3(x))