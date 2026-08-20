from cs336_basics.embedding import Embedding
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.linear import Linear
import torch.nn as nn

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int,
                 num_layers: int, num_heads: int, d_ff: int, rope_theta: float):
        super().__init__()
        # step 1: token 嵌入表 (vocab_size, d_model)（复用你的 Embedding）
        self.token_embeddings = Embedding(vocab_size, d_model)

        # step 2: num_layers 个 TransformerBlock，用 nn.ModuleList 堆起来
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model, 
                num_heads=num_heads, 
                d_ff=d_ff,
                max_seq_len=context_length, 
                theta=rope_theta,
            )
            for _ in range(num_layers)
        ])

        # step 3: 最终 RMSNorm（ln_final）
        self.ln_final = RMSNorm(d_model=d_model)

        # step 4: lm_head：线性 d_model -> vocab_size
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, token_ids):
        # token_ids:(batch, seq)
        # step 1: 查嵌入 -> (batch, seq, d_model)
        x = self.token_embeddings(token_ids) # (batch, seq, d_model)

        # step 2: 依次过每个 block（token_positions 可用 arange 生成）
        for block in self.layers: 
            x = block(x) # (batch, seq, d_model)

        # step 3: 过最终 norm
        x = self.ln_final(x) # (batch, seq, d_model)

        # step 4: 过 lm_head 得到 logits (batch, seq, vocab_size)
        return self.lm_head(x)