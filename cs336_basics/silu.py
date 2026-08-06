import torch

def silu(x):
    # step 1: 逐元素算门 sigmoid(x) —— 用 torch.sigmoid，落在 (0,1)
    # step 2: 把门逐元素乘回输入 x，得到与 x 同形状的结果
    # return ...
    return x * torch.sigmoid(x)