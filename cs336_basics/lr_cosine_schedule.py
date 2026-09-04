import math

def get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    # 三段式，返回一个标量学习率：

    # step 1: warmup 段 (it < warmup_iters)：从 0 线性升到 max
    #   —— 想清楚在 it == warmup_iters 时应恰好等于 max（对照期望：it=0 -> 0，it=warmup -> max）
    if(it < warmup_iters):
        lr = it * (max_learning_rate / warmup_iters)
    # step 2: 退火结束后 (it >= cosine_cycle_iters)：恒为 min
    elif(it >= cosine_cycle_iters):
        lr = min_learning_rate
    # step 3: 中间段 (warmup_iters <= it < cosine_cycle_iters)：在 max 与 min 之间做余弦插值
    #   先算归一化进度 progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters) ∈ [0, 1]
    #   再找一个“进度=0 时取 max、进度=1 时取 min”的平滑权重 weight 去混合两者
    #   所以目标是把 [0, 1] 的progress 映射到 [1, 0]的 weight
    #   （提示：cos 在 [0, π] 上从 1 单调降到 -1；怎样把它映射成一个 [0,1] 的混合系数？）
    else:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        weight = (1 + math.cos(math.pi * progress)) / 2
        lr = min_learning_rate + weight * (max_learning_rate - min_learning_rate)

    return lr