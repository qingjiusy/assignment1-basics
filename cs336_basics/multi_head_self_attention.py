from cs336_basics.linear import Linear
from cs336_basics.rope import RotaryPositionalEmbedding
from cs336_basics.attention import scaled_dot_product_attention
from einops import rearrange
import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int,
                 use_rope: bool = False, max_seq_len: int | None = None,
                 theta: float | None = None):
        super().__init__()
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.d_head = d_model // num_heads
        self.use_rope = use_rope

        # 先想清楚: d_head = d_model // num_heads
        # step 1: 建 4 个投影 q_proj / k_proj / v_proj / o_proj —— 每个都是 (d_model, d_model)的线性层（复用你写的 Linear）。self.q_proj = ...  ...  self.o_proj = ...
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.o_proj = Linear(d_model, d_model)

        # step 2:（use_rope 时）准备一个 RoPE 模块，作用在每个头的 d_head 维上
        if(self.use_rope):
            self.RoPE = RotaryPositionalEmbedding(theta, self.d_head, max_seq_len)

    def forward(self, x, token_positions=None):
        # x: (..., seq, d_model)
        # step 1: 一次矩阵乘做投影，得到 q, k, v，每个 (..., seq, d_model)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # step 2: 拆头 —— 把最后一维拆成 (num_heads, d_head)，把 head 维挪到 seq 前
        # 目标形状 (..., num_heads, seq, d_head)
        q = rearrange(q, "... seq (num_heads d_head) -> ... num_heads seq d_head", num_heads = self.num_heads, d_head = self.d_head)   
        k = rearrange(k, "... seq (num_heads d_head) -> ... num_heads seq d_head", num_heads = self.num_heads, d_head = self.d_head)    
        v = rearrange(v, "... seq (num_heads d_head) -> ... num_heads seq d_head", num_heads = self.num_heads, d_head = self.d_head) 

        # step 3:（use_rope 时）对 q, k 施加 RoPE（复用你写的 RoPE；注意作用在 d_head 维）
        if(self.use_rope):
            q = self.RoPE(q, token_positions)
            k = self.RoPE(k, token_positions)

        # step 4: 构造因果 mask —— query i 只能看到 key j <= i，之前构造的attention中True=保留 / False=屏蔽
        # 构造一个(seq, seq)的mask，然后之后用的时候直接广播到(..., num_heads, seq, seq)
        mask = torch.tril(torch.ones(x.shape[-2], x.shape[-2], dtype=torch.bool)) # (seq, seq)

        # step 5: 调用你写的 SDPA，得到每个头的输出 
        attn = scaled_dot_product_attention(q, k, v, mask) # (..., num_heads, seq, d_head)

        # step 6: 拼头 —— 把 (num_heads, d_head) 拼回 d_model
        attn = rearrange(attn, "... num_heads seq d_head -> ... seq (num_heads d_head)") # (..., seq, d_model)

        # step 7: 过输出投影 o_proj 并返回
        # 过输出投影 o_proj是为了让不同 head 的信息发生融合。下一层就可以收到已经混合好的表示。
        return self.o_proj(attn)