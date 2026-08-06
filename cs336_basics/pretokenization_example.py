import os
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    将文件拆分为可独立计数的多个部分。
    我们希望每个分块的大小大致相等，并且每个分块都以指定的特殊 token 结尾。
    如果文件中没有足够的特殊 token 来满足所需的分块数量，则返回的分块数量可能会少于预期。

    传入的三个参数：
    file：已经打开的二进制文件。
    desired_num_chunks：希望把文件切成多少块，比如 4 块。
    split_special_token：用于切分的特殊 token，必须是 bytes，例如：b"<|endoftext|>"

    返回值是一个一个整数列表，每个整数是文件中的字节位置。比如返回 [0, 1200, 2500, 4000]，
    就表示可以切成[0,1200)、[1200,2500)、[2500,4000) 这些区间。
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # 通过seek来移动文件指针到文件末尾，获取文件大小，然后把文件指针重新移动到开头
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # 对分块边界位置的初始估计，边界均匀分布
    # 每个分块从前一个索引位置开始，但不包含最后一个索引位置
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)] # 获得每个分块的初始边界位置
    chunk_boundaries[-1] = file_size # 最后一个分块的边界位置是文件末尾

    mini_chunk_size = 4096  # 每次向前预读 4 KB（4096 字节）的数据

    # 对每一个“初步估算出来的分块边界”，从该位置开始向后读取文件，直到找到指定的特殊分隔符；然后把边界移动到这个分隔符所在的位置。
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # 先把文件指针移动到初步估算的分块边界位置，然后向后寻找真正的分块边界
        while True:
            mini_chunk = file.read(mini_chunk_size)  # 读一个mini chunk（4KB）大小的数据，比如第一次 read：读取 [1000, 5096)

            # 判断是否到达文件末尾，如果到达文件末尾，就把分块边界设置为文件末尾
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # 在mini chunk中查找特殊分隔符的位置，如果找到了，就把分块边界设置为这个位置
            # .find() 方法返回子字符串在字符串中首次出现的位置，如果没有找到子字符串，则返回 -1。
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # 保证分块边界是唯一的，并且按升序排列，所以要对分块边界进行去重
    return sorted(set(chunk_boundaries))


## 使用
with open(..., "rb") as f:
    num_processes = 4
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # 下面是串行实现，但你可以将每一组起始结束位置分配给多个进程，从而实现并行处理。
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        # 读取当前分块的字节数据，并尝试将其解码为 UTF-8 字符串
        chunk = f.read(end - start).decode("utf-8", errors="ignore")# 表示如果读取到无法按照 UTF-8 解码的字节，就直接忽略这些错误字节，而不是抛出异常
        
        
        # 对当前分块执行预分词，并统计每个预分词单元出现的次数。