import torch

def cross_entropy(inputs, targets):
    # 约定：inputs 是 (N, vocab)，targets 是 (N,)；返回标量(对 N 求平均)。

    # step 1: 沿 vocab 维做数值稳定的 log-sum-exp
    #   - 先取每行(沿 vocab)的最大值 m，注意 keepdim 以便广播（回顾第 00 章）
    #   - lse = m + log(sum(exp(inputs - m)))，结果形状 (N,)
    inputs_max = torch.max(inputs, dim=-1, keepdim=True).values # (N, vocab)
    log_sum_exp = inputs_max + torch.log(torch.sum(torch.exp(inputs - inputs_max), dim=-1, keepdim=False)) # (N,)

    # step 2: 按 targets 取出每个样本"正确类"对应的那个 logit -> (N,)
    # 用arange和targets同时索引第 0 维和第 1 维。而是按位置配对取
    target_logit = inputs[torch.arange(inputs.shape[0]), targets] # (N,)

    # step 3: 逐样本损失 = lse - target_logit   -> (N,)
    per_example = log_sum_exp - target_logit # (N,)

    # step 4: 对 N 个样本求平均，得到标量并 return
    return per_example.mean()