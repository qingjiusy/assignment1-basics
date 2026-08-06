import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """沿指定 dim 做数值稳定的 softmax（引导骨架，非交付实现）。"""
    # step 1: 沿 dim 求最大值，keepdim=True 以便广播回 x 的形状
    #   理由 = 加性平移不变性：减去最大值不改变结果，却能防止 exp 上溢
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x = x - x_max
    
    # step 2: 平移后取指数（此时输入都 <= 0，exp 结果都在 (0, 1]）
    x_exp = torch.exp(x)
    
    # step 3: 沿同一个 dim 求和（keepdim=True）并归一化，返回与 x 同形状的分布
    ans = x_exp / x_exp.sum(dim=dim, keepdim=True)
    return ans