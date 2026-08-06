import torch.nn as nn
import torch

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # step 1: 建一个形状为 (d_model,) 的可学习 gain 参数，初始化为全 1
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # step 0: 记住输入原始 dtype（最后要 cast 回去）
        in_dtype = x.dtype

        # step 1: upcast 到 float32 —— 关键！否则 atol=1e-4 处容易翻车
        x = x.to(torch.float32)

        # step 2: 沿最后一维算“均方” (mean of squares)，keepdim=True 以便广播
        mean_sq = x.pow(2).mean(dim=-1, keepdim=True)

        # step 3: 由 mean_sq 得到 rms。问：eps 放在 sqrt 里面还是外面？看 handout 公式
        rms = torch.sqrt(mean_sq + self.eps)

        # step 4: 用 rms 归一化 x，再乘 gain(self.weight, 形状 (d_model,) 会广播到最后一维)
        out = self.weight * (x / rms)
        
        # step 5: 把结果 cast 回 in_dtype 再返回
        return out.to(in_dtype)