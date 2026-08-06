from cs336_basics.softmax import softmax
import einops

def scaled_dot_product_attention(Q, K, V, mask=None):
    # Q: (..., queries, d_k)   K: (..., keys, d_k)   V: (..., keys, d_v)
    # mask (可选): bool (..., queries, keys)，True=保留 / False=屏蔽
    # 要求：支持任意前导 batch 维（见 4D 测试），全程用广播、不要写死维数

    # step 1: 打分 scores = Q · Kᵀ。想清楚最后两维怎么乘：
    #   (queries, d_k) 与 (keys, d_k) 要收缩掉 d_k，得到 (queries, keys)。
    #   -> K 的最后两维需要转置；前导 batch 维保持广播。
    K_t = einops.rearrange(K, "... seq d -> ... d seq")
    scores = Q @ K_t # 因为 PyTorch 的 @ 对高维张量有固定规则：对最后两个维度做矩阵乘法，前面的维度当作 batch 维进行广播。

    # step 2: 缩放。除以 √d_k，其中 d_k = Q 最后一维的大小。（回忆第 2 节的方差实验：这一步让 softmax 输入不随 d_k 爆炸。）
    d_k = Q.shape[-1]
    scaled_scores = scores / (d_k ** 0.5)

    # step 3: 应用 mask（若提供）。在 softmax *之前*，把 mask 为 False 的位置加上一个极大的负数，使其 exp 后≈0。想清楚用什么值、怎么按 scores 广播。
    # 使用api：scores.masked_fill(condition, value)。含义是：在 condition == True 的位置，把 scores 替换成 value。
    if mask is not None:
        logits = scaled_scores.masked_fill(~mask, float("-inf")) # ~ 是取反操作

    # step 4: 沿 keys 维做 softmax，得到注意力权重。
    #   （复用你已实现的 softmax，并想清楚是最后哪一维归一化。）
    weights = softmax(logits, dim=-1)

    # step 5: 用权重对 V 加权求和，得到 (..., queries, d_v)。
    #   再想一次：这里哪两维在收缩？
    return weights @ V