from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import torch.nn as nn

from cs336_basics.train_bpe import train_bpe
from cs336_basics.linear import Linear
from cs336_basics.embedding import Embedding
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.silu import silu
from cs336_basics.swiGLU import SwiGLU
from cs336_basics.softmax import softmax
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.attention import scaled_dot_product_attention
from cs336_basics.rope import RotaryPositionalEmbedding
from cs336_basics.multi_head_self_attention import MultiHeadSelfAttention
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.adamw import AdamW
from cs336_basics.lr_cosine_schedule import get_lr_cosine_schedule
from cs336_basics.gradient_clipping import gradient_clipping

def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    给定一个 Linear 层的权重，计算批量输入的线性变换。

    参数：
        in_dim (int): 输入维度的大小
        out_dim (int): 输出维度的大小
        weights (Float[Tensor, "d_out d_in"]): 要使用的线性层权重
        in_features (Float[Tensor, "... d_in"]): 要应用该函数的输入张量

    返回：
        Float[Tensor, "... d_out"]: 你的线性模块变换后的输出。
    """
    linear = Linear(d_in, d_out)
    linear.weight.data = weights # 必须要用.data方法，因为linear.weight是一个nn.Parameter，而weights是一个普通张量。只有用.data才可以正常导入。
    return linear.forward(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    给定一个 Embedding 层的权重，获取一批 token id 对应的 embedding。

    参数：
        vocab_size (int)：词表中的 embedding 数量
        d_model (int)：embedding 维度的大小
        weights (Float[Tensor, "vocab_size d_model"])：用于查询的 embedding 向量表
        token_ids (Int[Tensor, "..."])：要从 Embedding 层中查询的一组 token id

    返回：
        Float[Tensor, "... d_model"]：由你的 Embedding 层返回的一批 embeddings。
    """
    embedding = Embedding(vocab_size, d_model)
    embedding.weight.data = weights
    return embedding.forward(token_ids)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """
    给定一个 SwiGLU 网络的权重，返回你的实现使用这些权重后得到的输出。

    参数：
        d_model (int)：前馈网络输入和输出的维度。
        d_ff (int)：SwiGLU 内部进行上投影时的维度。
        w1_weight (Float[Tensor, "d_ff d_model"])：W1 保存的权重。
        w2_weight (Float[Tensor, "d_model d_ff"])：W2 保存的权重。
        w3_weight (Float[Tensor, "d_ff d_model"])：W3 保存的权重。
        in_features (Float[Tensor, "... d_model"])：输入到前馈层的 embedding。

    返回：
        Float[Tensor, "... d_model"]：输出 embedding，形状与输入 embedding 相同。
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight

    swiglu = SwiGLU(d_model, d_ff)
    swiglu.w1.weight.data = w1_weight
    swiglu.w2.weight.data = w2_weight
    swiglu.w3.weight.data = w3_weight

    return swiglu(in_features)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    给定 Key（K）、Query（Q）和 Value（V）张量，
    返回你实现的 Scaled Dot-Product Attention（SDPA）的输出。

    参数：
        Q (Float[Tensor, "... queries d_k"]):
            Query 张量。

        K (Float[Tensor, "... keys d_k"]):
            Key 张量。

        V (Float[Tensor, "... keys d_v"]):
            Value 张量。

        mask (Bool[Tensor, "... queries keys"] | None):
            Mask 张量，用于指定每个 Query 可以看到哪些 Key。
            若为 None，则不使用 Mask。

    返回：
        Float[Tensor, "... queries d_v"]:
            Scaled Dot-Product Attention（SDPA）的输出。
    """
    return scaled_dot_product_attention(Q, K, V, mask)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    给定一个朴素（naive）、无 batch（unbatched）实现的多头注意力（Multi-Head Attention）中的
    Key、Query 和 Value 投影权重，返回一个经过优化的、支持 batch 的多头注意力实现的输出。

    该实现应当在一次矩阵乘法中同时完成所有注意力头（heads）的
    Key、Query 和 Value 投影，而不是分别对每个头单独计算。

    本函数**不需要使用 RoPE（Rotary Positional Embedding）**。

    可参考 Vaswani 等人在 2017 年论文《Attention Is All You Need》的第 3.2.2 节。

    参数：
        d_model (int):
            输入和输出特征的维度（模型隐藏层维度）。

        num_heads (int):
            多头注意力中使用的注意力头数量。

        max_seq_len (int):
            最大序列长度。如果你的实现会预先缓存（pre-cache）某些数据，则缓存到该长度即可。

        q_proj_weight (Float[Tensor, "d_model d_model"]):
            Query（Q）投影矩阵的权重。

        k_proj_weight (Float[Tensor, "d_model d_model"]):
            Key（K）投影矩阵的权重。

        v_proj_weight (Float[Tensor, "d_model d_model"]):
            Value（V）投影矩阵的权重。

        o_proj_weight (Float[Tensor, "d_model d_model"]):
            输出投影（Output Projection）矩阵的权重。

        in_features (Float[Tensor, "... sequence_length d_model"]):
            输入特征张量，即需要送入多头注意力计算的数据。

    返回：
        Float[Tensor, "... sequence_length d_model"]:
            使用给定的 Q、K、V 投影权重和输入特征，
            经过优化后的、支持 batch 的多头注意力实现计算得到的输出张量。
    """
    multi_head_self_attention = MultiHeadSelfAttention(
        d_model=d_model, 
        num_heads=num_heads,
    )
    multi_head_self_attention.q_proj.weight.data = q_proj_weight
    multi_head_self_attention.k_proj.weight.data = k_proj_weight
    multi_head_self_attention.v_proj.weight.data = v_proj_weight
    multi_head_self_attention.output_proj.weight.data = o_proj_weight

    return multi_head_self_attention(in_features) 


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    给定一个朴素（naive）、无 batch（unbatched）实现的多头注意力（Multi-Head Attention）中的
    Key、Query 和 Value 投影权重，返回一个经过优化的、支持 batch 的多头注意力实现的输出。

    该实现应当在一次矩阵乘法中同时完成所有注意力头（heads）的
    Key、Query 和 Value 投影，而不是分别对每个头单独计算。

    **本版本的多头注意力（MHA）需要包含 RoPE（Rotary Positional Embedding）。**

    在本实现中，**RoPE 的嵌入维度必须等于每个注意力头的维度**，即：

        d_head = d_model // num_heads

    可参考 Vaswani 等人在 2017 年论文《Attention Is All You Need》的第 3.2.2 节。

    参数：
        d_model (int):
            模型的隐藏层维度，即输入和输出特征的维度。

        num_heads (int):
            多头注意力中使用的注意力头数量。

        max_seq_len (int):
            最大序列长度。如果你的实现会预先缓存（pre-cache）某些数据，则缓存到该长度即可。

        theta (float):
            RoPE 的参数 θ，用于控制旋转位置编码的频率。

        q_proj_weight (Float[Tensor, "d_model d_model"]):
            Query（Q）投影矩阵的权重。

        k_proj_weight (Float[Tensor, "d_model d_model"]):
            Key（K）投影矩阵的权重。

        v_proj_weight (Float[Tensor, "d_model d_model"]):
            Value（V）投影矩阵的权重。

        o_proj_weight (Float[Tensor, "d_model d_model"]):
            输出投影（Output Projection）矩阵的权重。

        in_features (Float[Tensor, "... sequence_length d_model"]):
            输入特征张量，即需要送入多头注意力计算的数据。

        token_positions (Int[Tensor, "... sequence_length"] | None):
            （可选）表示每个 token 在序列中的位置的张量。
            如果为 None，则通常默认使用连续的位置索引（如 0, 1, 2, ...）。

    返回：
        Float[Tensor, "... sequence_length d_model"]:
            使用给定的 Q、K、V 投影权重、RoPE 位置编码和输入特征，
            经过优化后的、支持 batch 的多头注意力实现计算得到的输出张量。
    """
    multi_head_self_attention = MultiHeadSelfAttention(
            d_model=d_model, 
            num_heads=num_heads,
            use_rope=True,
            max_seq_len=max_seq_len,
            theta=theta,
        )

    multi_head_self_attention.q_proj.weight.data = q_proj_weight
    multi_head_self_attention.k_proj.weight.data = k_proj_weight
    multi_head_self_attention.v_proj.weight.data = v_proj_weight
    multi_head_self_attention.output_proj.weight.data = o_proj_weight

    return multi_head_self_attention(in_features, token_positions) 


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    对给定的输入张量应用 RoPE（旋转位置编码）。

    参数：
        d_k (int)：Query 或 Key 张量的嵌入维度大小。
        theta (float)：RoPE 的参数。
        max_seq_len (int)：如果你的实现会预先缓存（pre-cache）RoPE 所需的值，则表示预缓存的最大序列长度。
        in_query_or_key (Float[Tensor, "... sequence_length d_k"])：需要应用 RoPE 的输入张量（Query 或 Key）。
        token_positions (Int[Tensor, "... sequence_length"])：表示各个 token 位置的张量，形状为 (batch_size, sequence_length)。

    返回：
        Float[Tensor, "... sequence_length d_k"]：应用了 RoPE 后的输出张量。
    """

    rotary_positional_embedding = RotaryPositionalEmbedding(theta, d_k, max_seq_len)
    return rotary_positional_embedding(in_query_or_key, token_positions)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    给定一个 **Pre-Norm Transformer Block** 的权重和输入特征，返回该 Transformer Block 对输入特征进行前向计算后的输出。

    本函数应使用 **RoPE（Rotary Positional Embedding）**。

    根据你的实现方式，你可能只需要将相关参数传递给 `TransformerBlock` 的构造函数；或者你也可能需要自己初始化一个 RoPE 类，并将其传入 TransformerBlock。

    参数：
        d_model (int):
            Transformer Block 输入的特征维度。

        num_heads (int):
            多头注意力（Multi-Head Attention）的头数。
            `d_model` 必须能够被 `num_heads` 整除。

        d_ff (int):
            前馈网络（Feed-Forward Network，FFN）隐藏层的维度。

        max_seq_len (int):
            最大序列长度。
            如果你的实现会预先缓存 RoPE 等信息，则缓存长度应为该值。

        theta (float):
            RoPE 的参数 θ。

        weights (dict[str, Tensor]):
            官方参考实现（reference implementation）的 state_dict。

            字典中包含以下权重：

            - `attn.q_proj.weight`
                所有 `num_heads` 个注意力头的 Query 投影矩阵。

                形状：
                    (d_model, d_model)

                这些行按照每个 Head 的权重依次拼接排列，即：

                    attn.q_proj.weight ==
                    torch.cat([q_heads.0.weight,
                            ...,
                            q_heads.N.weight], dim=0)

            - `attn.k_proj.weight`
                所有注意力头的 Key 投影矩阵。

                形状：
                    (d_model, d_model)

                排列方式同样是：

                    attn.k_proj.weight ==
                    torch.cat([k_heads.0.weight,
                            ...,
                            k_heads.N.weight], dim=0)

            - `attn.v_proj.weight`
                所有注意力头的 Value 投影矩阵。

                形状：
                    (d_model, d_model)

                排列方式：

                    attn.v_proj.weight ==
                    torch.cat([v_heads.0.weight,
                            ...,
                            v_heads.N.weight], dim=0)

            - `attn.output_proj.weight`
                多头自注意力输出投影（Output Projection）的权重。

                形状：
                    (d_model, d_model)

            - `ln1.weight`
                Transformer Block 中第一个 RMSNorm 的缩放参数（affine weight）。

                形状：
                    (d_model,)

            - `ffn.w1.weight`
                FFN 第一层线性变换的权重。

                形状：
                    (d_ff, d_model)

            - `ffn.w2.weight`
                FFN 第二层线性变换的权重。

                形状：
                    (d_model, d_ff)

            - `ffn.w3.weight`
                FFN 第三层线性变换的权重。

                形状：
                    (d_ff, d_model)

            - `ln2.weight`
                Transformer Block 中第二个 RMSNorm 的缩放参数（affine weight）。

                形状：
                    (d_model,)

        in_features (Float[Tensor, "batch sequence_length d_model"]):
            需要输入到你的 Transformer Block 实现中的张量。

    返回：
        Float[Tensor, "batch sequence_length d_model"]

        返回 Transformer Block 前向传播后的输出张量。
    """
    transformer_block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads, 
        d_ff=d_ff,
        max_seq_len=max_seq_len, 
        theta=theta,
    )

    # transformer_block.attn.q_proj.weight.data = weights["attn.q_proj.weight"]
    # transformer_block.attn.k_proj.weight.data = weights["attn.k_proj.weight"]
    # transformer_block.attn.v_proj.weight.data = weights["attn.v_proj.weight"]
    # transformer_block.attn.o_proj.weight.data = weights["attn.output_proj.weight"]
    
    # transformer_block.ln1.weight.data = weights["ln1.weight"]
    # transformer_block.ln2.weight.data = weights["ln2.weight"]
    
    # transformer_block.ffn.w1.weight.data = weights["ffn.w1.weight"]
    # transformer_block.ffn.w2.weight.data = weights["ffn.w2.weight"]
    # transformer_block.ffn.w3.weight.data = weights["ffn.w3.weight"]
    transformer_block.load_state_dict(weights)

    return transformer_block(in_features)

def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """给定一个 Transformer 语言模型的权重以及输入 token 的索引，
    返回该模型对输入执行一次前向传播（forward pass）的输出。

    该函数应使用 RoPE（Rotary Positional Embedding，旋转位置编码）。

    参数：
        vocab_size (int)：
            输出词表（vocabulary）的大小，即需要预测的 token 种类数。

        context_length (int)：
            模型一次最多能够处理的 token 数（上下文长度）。

        d_model (int)：
            模型 embedding 和各子层输出的维度。

        num_layers (int)：
            Transformer Block 的层数。

        num_heads (int)：
            多头注意力（Multi-Head Attention）的头数。
            要求 d_model 能够被 num_heads 整除。

        d_ff (int)：
            前馈网络（Feed-Forward Network，论文 3.3 节）隐藏层的维度。

        rope_theta (float)：
            RoPE 的 θ 参数。

        weights (dict[str, Tensor])：
            参考实现的 state_dict。

            其中 {num_layers} 表示一个介于
            0 到 num_layers-1 之间的整数（即层编号）。

            字典中包含以下键：

            - token_embeddings.weight
                Token Embedding 矩阵。
                形状：(vocab_size, d_model)

            - layers.{i}.attn.q_proj.weight
                所有注意力头（num_heads）的 Query 投影矩阵。
                形状：
                    (num_heads × (d_model / num_heads), d_model)

                每个 head 的权重按行拼接，因此：

                    attn.q_proj.weight ==
                    torch.cat(
                        [q_heads.0.weight,
                        ...,
                        q_heads.N.weight],
                        dim=0
                    )

            - layers.{i}.attn.k_proj.weight
                所有注意力头的 Key 投影矩阵。
                形状：
                    (num_heads × (d_model / num_heads), d_model)

                排列方式与 q_proj 相同：

                    attn.k_proj.weight ==
                    torch.cat(
                        [k_heads.0.weight,
                        ...,
                        k_heads.N.weight],
                        dim=0
                    )

            - layers.{i}.attn.v_proj.weight
                所有注意力头的 Value 投影矩阵。
                形状：
                    (num_heads × (d_model / num_heads), d_model)

                排列方式与 q_proj 相同：

                    attn.v_proj.weight ==
                    torch.cat(
                        [v_heads.0.weight,
                        ...,
                        v_heads.N.weight],
                        dim=0
                    )

            - layers.{i}.attn.output_proj.weight
                多头自注意力输出投影层（Output Projection）的权重。

                形状：
                    ((d_model / num_heads) × num_heads, d_model)

            - layers.{i}.ln1.weight
                Transformer Block 中第一个 RMSNorm 的可学习缩放参数（affine transform）。

                形状：
                    (d_model,)

            - layers.{i}.ffn.w1.weight
                FFN 第一层线性变换权重。

                形状：
                    (d_ff, d_model)

            - layers.{i}.ffn.w2.weight
                FFN 第二层线性变换权重。

                形状：
                    (d_model, d_ff)

            - layers.{i}.ffn.w3.weight
                FFN 第三层线性变换权重。

                形状：
                    (d_ff, d_model)

            - layers.{i}.ln2.weight
                Transformer Block 中第二个 RMSNorm 的可学习缩放参数。

                形状：
                    (d_model,)

            - ln_final.weight
                最后一个 Transformer Block 输出之后所使用 RMSNorm 的可学习缩放参数。

                形状：
                    (d_model,)

            - lm_head.weight
                语言模型输出层（LM Head）的权重。

                形状：
                    (vocab_size, d_model)

        in_indices (Int[Tensor, "batch_size sequence_length"])：
            输入 token 的索引张量。

            形状：
                (batch_size, sequence_length)

            其中 sequence_length 不超过 context_length。

    返回：
        Float[Tensor, "batch_size sequence_length vocab_size"]

        返回每个位置预测的下一个 token 的未归一化分数（logits）。

        输出形状：
            (batch_size, sequence_length, vocab_size)
    """
    transformer_lM = TransformerLM(
        vocab_size=vocab_size, 
        context_length=context_length, 
        d_model=d_model,
        num_layers=num_layers, 
        num_heads=num_heads, 
        d_ff=d_ff, 
        rope_theta=rope_theta,
    )

    transformer_lM.load_state_dict(weights)
    return transformer_lM(in_indices)



def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """
    给定 RMSNorm 仿射变换（affine transform）的权重，
    返回输入特征经过 RMSNorm 后的输出。

    Args:
        d_model (int): RMSNorm 输入的特征维度。
        eps (float): 为了数值稳定性而加到分母中的一个很小的常数。
        weights (Float[Tensor, "d_model"]): RMSNorm 的权重（weight / gain）。
        in_features (Float[Tensor, "... d_model"]): 需要进行 RMSNorm 的输入特征，
            可以具有任意数量的前导维度。

    Returns:
        Float[Tensor, "... d_model"]: 与 `in_features` 形状相同的张量，
            表示对 `in_features` 执行 RMSNorm 后得到的输出。
    """
    rmsnorm = RMSNorm(d_model, eps)
    rmsnorm.weight.data = weights

    return rmsnorm(in_features)



def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """
    给定一个输入张量，对其中的每个元素应用 SiLU 激活函数，并返回结果。

    参数：
        in_features (Float[Tensor, "..."])：
            要应用 SiLU 的输入特征张量，张量的形状可以是任意的。

    返回：
        Float[Tensor, "..."]：
            一个与 `in_features` 形状相同的张量，其中每个元素都是对对应输入元素应用
            SiLU 激活函数后的结果。
    """
    return silu(in_features)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    给定一个输入张量，对指定维度 `dim` 应用 Softmax，并返回结果。

    参数：
        in_features (Float[Tensor, "..."])：
            要进行 Softmax 的输入特征张量，张量的形状可以是任意的。

        dim (int)：
            要应用 Softmax 的 `in_features` 的维度。

    返回：
        Float[Tensor, "..."]：
            一个与 `in_features` 形状相同的张量，其中指定维度 `dim`
            上的元素经过 Softmax 归一化后的结果。
    """
    return softmax(in_features, dim)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """
    给定输入张量和目标标签张量，计算所有样本的平均交叉熵损失。

    参数：
        inputs (Float[Tensor, "batch_size vocab_size"])：
            输入张量，其中 `inputs[i][j]` 表示第 `i` 个样本在第 `j` 个类别上的
            未归一化 logit（原始分数）。

        targets (Int[Tensor, "batch_size"])：
            形状为 `(batch_size,)` 的张量，表示每个样本正确类别的索引。
            每个值都必须位于 `0` 到 `num_classes - 1` 之间。

    返回：
        Float[Tensor, ""]：
            所有样本的平均交叉熵损失。
    """
    return cross_entropy(inputs, targets)


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """给定一组参数，对它们的**整体梯度（combined gradients）**进行裁剪，使其 L2 范数最大不超过 max_l2_norm。

    参数（Args）：

    * parameters (Iterable[torch.nn.Parameter])：一组可训练参数。
    * max_l2_norm (float)：一个正数，表示允许的最大 L2 范数。

    参数的梯度（parameter.grad）应该进行原地修改（in-place）。
    """
    gradient_clipping(parameters = parameters, max_l2_norm = max_l2_norm)

def get_adamw_cls() -> Any:
    """
    返回一个实现了 AdamW 算法的 torch.optim.Optimizer 优化器。
    """
    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    r"""
    给定**余弦学习率衰减调度（包含线性 warmup）**的参数以及一个迭代次数，返回在指定学习率调度下，该次迭代对应的学习率。

    参数（Args）：

    * it (int)：要获取学习率的迭代次数。
    * max_learning_rate (float)：\alpha_{\max}，余弦学习率调度（包含 warmup）中的最大学习率。
    * min_learning_rate (float)：\alpha_{\min}，余弦学习率调度（包含 warmup）中的最小/最终学习率。
    * warmup_iters (int)：T_w，学习率进行线性 warmup 的迭代次数。
    * cosine_cycle_iters (int)：T_c，余弦退火结束时的 iteration 编号。

    返回值（Returns）：

    返回在指定学习率调度下，给定迭代次数所对应的学习率。
    """
    return get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    raise NotImplementedError


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    raise NotImplementedError


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    给定输入语料库的路径，训练一个 BPE 分词器，并输出其词表和合并规则。

    参数：
    input_path（str | os.PathLike）：用于训练 BPE 分词器的数据路径。
    vocab_size（int）：分词器词表中的元素总数，包括特殊 token。
    special_tokens（list[str]）：需要添加到分词器词表中的特殊 token 字符串列表。
    这些字符串永远不会被拆分成多个 token，并且始终会作为一个完整 token 保留。
    如果这些特殊 token 出现在 input_path 指定的语料中，
    它们会像其他普通字符串一样被处理。

    返回值：
    tuple[dict[int, bytes], list[tuple[bytes, bytes]]]：
    vocab：
    训练完成后的分词器词表。
    它是一个从 int（词表中的 token ID）到 bytes（token 对应的字节序列）的映射。

    merges：
        BPE 合并规则。列表中的每个元素都是一个字节元组
        （<token1>, <token2>），表示将 <token1> 与 <token2> 合并。
        合并规则按照它们被创建的先后顺序排列。
    """

    return train_bpe(input_path, vocab_size, special_tokens)
