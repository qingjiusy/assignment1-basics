import numpy as np
import torch

def get_batch(dataset, batch_size, context_length, device):
    # step 1: 采 batch_size 个随机起点
    # 想清楚合法区间：要能取到 input 和"右移一位"的 label，最大起点是多少？
    # 由于labels的存在，所以能取到的最后一位一定是在 dataset_len - context_length - 1
    # 而 np.random.randint(low, high, size)，其中 high 是开区间，所以就是：
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size) # (batch_size, )

    # step 2: 对每个起点，取长度 context_length 的输入窗口，堆成 (batch_size, context_length)
    # dataset 和 starts 目前都是 numpy中的对象
    # np.stack 和 torch.stack 非常像，np.stack([a, b], axis=0) 就类似于 torch.stack([a, b], dim=0)
    inputs = np.stack([dataset[s : s + context_length] for s in starts], axis=0) # (batch_size, context_length)

    # step 3: 取"右移一位"的标签窗口（label 是 next-token）
    labels = np.stack([dataset[s + 1 : s + context_length + 1] for s in starts], axis=0) # (batch_size, context_length)

    # step 4: 转成 torch.long，并搬到 device 上（两者都要在同一 device）
    x = torch.tensor(inputs, dtype = torch.int64, device = device)
    y = torch.tensor(labels, dtype = torch.int64, device = device)

    return x, y