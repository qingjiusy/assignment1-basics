from collections import defaultdict
import regex as re
def train_bpe(input_path, vocab_size, special_tokens):
    """
    这个函数本质上要做一件事：从语料里训练出 byte-level BPE 的词表和 merge 规则。
    它要做：
    1. 读取训练语料
    2. 初始化vocabulary(初始化的256单字节token + 特殊token)
    3. 预切分语料
    4. 统计pretoken的频率,之后用于统计相邻token pair频率
    5. 循环选择最高频pair并记录merge,记录到vocab(训练后的合并部分)中
    6. return结果
    """

    # 1. 从路径读取输入文件
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read() # text是UTF-8编码的字符串，type是str

    # 2. 初始化词汇表
    # 在adapters接口中，要求返回的是dict[int, bytes]，因此我们构造的词汇表应该是一个哈希表
    # key: token_id(int) value: token(字节)
    vocab = {index : bytes([index]) for index in range(256)} #注意这里的bytes([index])写法，不应该写bytes(index)，因为我们是想让index变成单个byte token
    for index, special_token in enumerate(special_tokens):
        vocab[index + 256] = special_token.encode("utf-8")
    
    # 3. 预切分语料
    # 在切分语料的时候，为了实现更效率的切分，更好的利用我们的CPU资源，可以使用多进程（因为python的多线程并不能帮助我们使用多个CPU）
    # 多进程切分完成以后，先用special token切分，然后用regex实现正则表达式的切分，我们的text（str）就变成了一个个单词
    # 然后单词的str去调用encode方法，就变成了bytes数组
    # 准备一个记数表（byte-token tuple -> count）去记录每种pretoken的byte序列出现了多少次，用来实现去重
    # 计数表中的例子：(b"h", b"e", b"l", b"l", b"o") -> 5,意思是hello出现了五次
    pretoken = defaultdict(int)

    special_pattern = "|".join(re.escape(tok) for tok in special_tokens)
    if(special_tokens != []):
        parts = re.split(special_pattern, text)
    else:
        parts = [text]

    # 接下来把text的str列表进行regex分割，然后转成bytes tuple并计数
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" # GPT-2正则

    # 4. 统计pretoken的频率,之后用于统计相邻token pair频率
    for part in parts:
        parts_after_re = re.findall(PAT, part)
        for part_after_re in parts_after_re:
            pretoken[tuple(bytes([byte]) for byte in part_after_re.encode("utf-8"))] += 1

    # 5. 循环选择最高频pair并记录merge,记录到vocab(训练后的合并部分)中
    merges = []
    while(len(vocab) < vocab_size):
        # 重新统计目前这个pretoken中的pair对的情况
        pair_count = defaultdict(int)
        for pretoken_tuple, count in pretoken.items():
            for i in range(len(pretoken_tuple) - 1):
                pair_count[(pretoken_tuple[i], pretoken_tuple[i + 1])] += count

        if(not pair_count):
            break

        # 找到目前的最常见pair，频率相同时用更大的字典序的
        max_pair = max(pair_count, key=lambda pair: (pair_count[pair], pair))

        # 更新vocab和merges
        vocab[len(vocab)] = max_pair[0] + max_pair[1] #max_pair是(b"h",b"e")，vocab中加入的新词应该是b"he"
        merges.append(max_pair)

        # 更新pretoken
        # 遍历每一个pre_token，检查有没有可以合并的bytes
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

    return vocab, merges