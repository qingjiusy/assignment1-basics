import regex as re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from cs336_basics.pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def process_chunk(input_path, start, end, special_tokens):
    """
    一个子进程处理 [start, end) 这一段文件。
    返回这个 chunk 内部的 pretoken 频率表。
    """
    local_pretoken = defaultdict(int)

    # 注意这里用 rb，因为 start/end 是 byte offset
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)

    text = chunk.decode("utf-8")

    # 先按 special token 切
    if(special_tokens):
        special_pattern = "|".join(re.escape(tok) for tok in special_tokens)
        parts = re.split(special_pattern, text)
    else:
        parts = [text]

    # 再做 GPT-2 pretokenization
    for part in parts:
        for token in re.findall(PAT, part):
            token_bytes = token.encode("utf-8")

            byte_tuple = tuple(bytes([byte]) for byte in token_bytes)
            local_pretoken[byte_tuple] += 1

    return local_pretoken

def train_bpe_multiprocess(input_path, vocab_size, special_tokens):
    vocab = {i: bytes([i]) for i in range(256)}

    for index, special_token in enumerate(special_tokens):
        vocab[index + 256] = special_token.encode("utf-8")

    # 通过 find_chunk_boundaries 找到了切分成4段的 boundaries
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, 4, b"<|endoftext|>")

    # 保存每个“异步提交出去的任务”
    jobs = []

    # 开一个进程池
    with ProcessPoolExecutor(max_workers=4) as executor:
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            # submit() 返回的并不是 process_chunk() 的真正结果，而是一个 Future 对象
            # 后面再用过 job.result() 拿到返回值
            jobs.append(
                # 表示函数 process_chunk 调用下面的四个参数
                executor.submit(
                    process_chunk,
                    input_path,
                    start,
                    end,
                    special_tokens,
                )
            )

    # 合并所有进程的统计结果
    pretoken = defaultdict(int)

    for job in jobs:
        local_counts = job.result()

        for token, count in local_counts.items():
            pretoken[token] += count

    # 之后就和前面一样，开始进行 merge
    # step 3: 循环，直到 len(vocab) == vocab_size:
    merges = []
    while(len(vocab) < vocab_size):

        # 3a. 统计所有相邻 token 对的频率(按 pretoken 数量加权)，记录在 pair_count 中
        pair_count = defaultdict(int)
        for pretoken_tuple, count in pretoken.items():
            for i in range(len(pretoken_tuple) - 1):
                pair_count[(pretoken_tuple[i], pretoken_tuple[i + 1])] += count

        if(not pair_count):
            break

        # 3b. 选频率最高的对；平局用 handout 的 tie-break 规则，也就是：频率相同时用更大的字典序的
        max_pair = max(pair_count, key=lambda pair: (pair_count[pair], pair))

        # 3c. 记录这条 merge，并把合并后的新 token 加进 vocab
        vocab[len(vocab)] = max_pair[0] + max_pair[1] #max_pair是(b"h",b"e")，vocab中加入的新词应该是b"he"
        merges.append(max_pair)

        # 更新pretoken
        # 遍历每一个pre_token，检查有没有可以合并的bytes
        # 3d. 在每个 new_pretoken 序列里把 max_pair 替换成新 token
        new_pretoken_counts = defaultdict(int)

        for pretoken_tuple, count in pretoken.items():
            new_pretoken = []
            i = 0

            while i < len(pretoken_tuple):
                # 当前 token 和下一个 token 正好组成 max_pair
                if (
                    i < len(pretoken_tuple) - 1
                    and pretoken_tuple[i] == max_pair[0]
                    and pretoken_tuple[i + 1] == max_pair[1]
                ):
                    # 合并两个 bytes token
                    new_pretoken.append(max_pair[0] + max_pair[1])
                    i += 2 # 已经处理了两个 token，所以向后移动两位
                else:
                    # 没有匹配到 max_pair，保留当前 token
                    new_pretoken.append(pretoken_tuple[i])
                    i += 1

            # list 转回 tuple
            new_pretoken_tuple = tuple(new_pretoken)

            # 保留原 pretoken 的出现次数
            new_pretoken_counts[new_pretoken_tuple] += count

        # 用新的统计结果替换旧的 pretoken
        pretoken = new_pretoken_counts
    
    # step 4: 返回 (vocab, merges)
    return vocab, merges