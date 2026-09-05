import regex as re

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        # step 1: 存下 vocab / merges / special_tokens
        self.vocab = vocab # id -> bytes
        self.merges = merges # 记录需要合并的 token pair
        self.special_tokens = special_tokens

        # step 2: 预计算好方便查的结构
        # id -> bytes 的逆表，方便 encode 时快速进行
        # bytes -> id
        self.byte_to_id = {token_bytes: token_id for token_id, token_bytes in vocab.items()}

        # merges 的优先级表
        # 找到正确的优先级对应才能正确的进行merge
        self.ranks = {pair: rank for rank, pair in enumerate(merges)}

    def encode(self, text):
        # step 1: 先把 special_tokens 从 text 中切出来(它们直接映射到自己的 id)
        if(self.special_tokens):
            # special token 可能互相重叠，因此优先匹配更长的
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)

            # 通过加 ( 和 )，变成捕获组
            special_pattern = "(" + "|".join(re.escape(tok) for tok in sorted_special_tokens) + ")"
            parts = re.split(special_pattern, text)
        else:
            parts = [text]

        # step 2: 普通文本段落 -> 预分词(Part D 的正则) -> 每段变字节序列
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" # GPT-2正则
        ids = []

        for part in parts:
            # 是special_tokens，直接得到对应id
            if(self.special_tokens and part in self.special_tokens):
                ids.append(self.byte_to_id[part.encode("utf-8")])
            else:
                pre_tokens = re.findall(PAT, part)

                for pre_token in pre_tokens:
                    raw_bytes = pre_token.encode("utf-8")
                    byte_tokens = [bytes([b]) for b in raw_bytes]

                    # step 3: 对这个 byte_tokens，按 merges 的优先级反复合并相邻对，直到不能再合并
                    while True:
                        # 找出当前所有相邻 pair
                        pairs = [(byte_tokens[i], byte_tokens[i + 1]) for i in range(len(byte_tokens) - 1)]

                        # 只保留存在于 self.ranks 里的 pair
                        mergeable_pairs = [pair for pair in pairs if pair in self.ranks]

                        # 如果一个都不能合并，就结束
                        if not mergeable_pairs:
                            break

                        # 找优先级最高的 pair
                        best_pair = min(mergeable_pairs, key=lambda pair: self.ranks[pair])

                        # 把当前 byte_tokens 里所有这个 pair 合并
                        i = 0
                        new_byte_tokens = []

                        while(i <= len(byte_tokens) - 1):
                            if(i <= (len(byte_tokens) - 2) and (byte_tokens[i], byte_tokens[i + 1]) == best_pair):
                                new_byte_tokens.append(byte_tokens[i] + byte_tokens[i + 1]) # 两个bytes 可以用 + 来直接合并
                                i += 2
                            else:
                                new_byte_tokens.append(byte_tokens[i])
                                i += 1

                        byte_tokens = new_byte_tokens
        
                    # step 4: 把最终 token 的 bytes 逐个查逆表得到 id，拼成 id 列表返回
                    ids.extend(self.byte_to_id[byte] for byte in byte_tokens)

        return ids

    def encode_iterable(self, iterable):
        # step 1: 对 iterable(如文件的行) 逐段调用 encode 的逻辑，惰性 yield 出 id
        # iterable 是一个储存着一行行文本的数组
        # yield 用于流式处理大文件，避免一次性读入内存
        for text in iterable:
            ids = self.encode(text)

            for id in ids:
                yield id

    def decode(self, ids):
        # step 1: 每个 id -> self.vocab[id] 得到 bytes
        id_bytes = []
        for id in ids:
            id_bytes.append(self.vocab[id])

        # step 2: 把这些 bytes 按顺序拼接成一个大字节串
        byte_str = b"".join(id_bytes)

        # step 3: utf-8 解码(带容错) 得到最终字符串
        # 如果遇到非法 UTF-8 字节序列，不报错，而是用替代字符 � 代替。
        return byte_str.decode("utf-8", errors="replace")