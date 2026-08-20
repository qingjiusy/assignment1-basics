from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.multi_head_self_attention import MultiHeadSelfAttention
from cs336_basics.swiGLU import SwiGLU
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 max_seq_len: int | None = None, theta: float | None = None):
        super().__init__()
        # step 1: 两个 RMSNorm（ln1, ln2）、一个 MHA(attn, 带 RoPE)、一个 SwiGLU(ffn)
        self.ln1, self.ln2 = RMSNorm(d_model=d_model), RMSNorm(d_model=d_model)

        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            use_rope=True,
            max_seq_len=max_seq_len,
            theta=theta,
        )

        self.ffn = SwiGLU(
            d_model=d_model, 
            d_ff=d_ff,
        )
        

    def forward(self, x, token_positions=None):
        # 子层一（pre-norm 注意力）: 残差 + attn(ln1(x))，norm 在子层输入侧
        x = x + self.attn(self.ln1(x), token_positions)
        # 子层二（pre-norm 前馈）:   残差 + ffn(ln2(x))
        x = x + self.ffn(self.ln2(x))

        return x