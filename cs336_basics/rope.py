import einops
import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int):
        super().__init__()
        # step 1: 建维度对索引 i = 0,1,...,d_k/2 - 1，算频率表 inv_freq = theta^(-2i/d_k)
        i = torch.arange(d_k // 2, dtype=torch.float32) # (d_k/2, )
        inv_freq = theta ** (-2 * i / d_k) # (d_k/2, )

        # step 2: 位置 0..max_seq_len-1 与 inv_freq 做外积，得到角度表 (max_seq_len, d_k/2)
        # 需要构造一个pos 向量来储存位置，然后进行外积。这样得到的angles表就是我们的旋转角度对应表，对于每个pos的每一对两两组合的维度，我们有一个旋转角度
        # 这样就实现了加入位置信息
        pos = torch.arange(max_seq_len) # (max_seq_len, )
        angles = torch.outer(pos, inv_freq) # 两个一维向量，做外积操作，(max_seq_len, d_k/2)

        # step 3: 由 angles 预计算 cos / sin，并用 register_buffer 缓存（非可学习参数）
        # 用register的意义是：会把数据缓存到显卡上，减少了后续的数据搬运
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False) # (max_seq_len, d_k/2)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False) # (max_seq_len, d_k/2)
    

    def forward(self, x, token_positions):
        # x: (..., seq, d_k)   token_positions: (..., seq)
        # step 4: 用 token_positions 从缓存里取出对应的 cos / sin（注意任意前导维要能广播）
        # 使用整型张量索引，用token_positions中的每一个元素（数字）去做索引
        cos = self.cos_cached[token_positions] # (seq, d_k/2)
        sin = self.sin_cached[token_positions]

        # step 5: 把 x 的最后一维按你选定的方式配对，拆成两半 (x1, x2)
        # 先转变形状，然后沿着最后一维拆分
        # torch.unbind(x_pair, dim=-1)：沿最后一维拆成两个分量
        x1, x2 = torch.unbind(einops.rearrange(x, "... seq (pairs two) -> ... seq pairs two", two=2), dim=-1) # (..., seq, d_k // 2)

        # step 6: 对每一对施加二维旋转： x1' = x1*cos - x2*sin ; x2' = x1*sin + x2*cos
        x_1, x_2 = x1 * cos - x2 * sin, x1 * sin + x2 * cos

        # step 7: 把旋转后的两半重组回 (..., seq, d_k)，保证输出与输入同形状
        return einops.rearrange(torch.stack([x_1, x_2], dim=-1), "... seq pairs two -> ... seq (pairs two)", two=2)  # (..., seq, d_k)