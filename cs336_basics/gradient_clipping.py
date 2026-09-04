import torch

def gradient_clipping(parameters, max_l2_norm, eps=1e-6):
    # 原地修改 .grad，无返回值：
    # step 1: 收集所有 grad 不为 None 的参数（冻结/未参与前向的参数 grad 是 None，跳过）
    grads = [ p.grad for p in parameters if p.grad is not None ]

    # step 2: 计算这些梯度拼在一起的“全局” L2 范数（不是逐参数分别算！）
    total_norm = torch.sqrt(sum(torch.sum(grad ** 2) for grad in grads))

    # step 3: 仅当 total_norm > max_l2_norm 时，算一个 < 1 的缩放因子 max_l2_norm / (total_norm + eps)
    # 否则因子为 1（什么都不做）
    scale = max_l2_norm / (total_norm + eps) if(total_norm > max_l2_norm) else 1

    # step 4: 用同一个 scale 原地缩放每个梯度（考虑 grad.mul_(scale) 之类的原地操作）
    # 注意这里的grads 里面保存的不是 p.grad 的拷贝，而是对这些 Tensor 对象的引用
    # 所以对 grads 中的 grad 的原地修改等价于对 p.grad 的原地修改
    for grad in grads:
        grad.mul_(scale)