"""tokenizer 训练流程的调试脚本。

    conda run -n cs336 python test.py

结构:
    ① 输入 -> segments -> counts    （已完成）
    ② 完整的多轮合并循环             （你在这一区从零写）
    ③ 参考: 你已经跑通的单轮版本      （保留，不要删）
"""

from collections import Counter

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# ══════════════════════════════════════════════════════════════
# ① 输入 -> segments -> counts
# ══════════════════════════════════════════════════════════════

text = open("mini_corpus.txt", encoding="utf-8").read()
special_tokens = ["<|endoftext|>"]

if special_tokens:
    segments = re.split("|".join(re.escape(t) for t in special_tokens), text)
else:
    segments = [text]

print("上一步的产物 segments =")
for i, s in enumerate(segments):
    print(f"  [{i}] {s!r}")

counts = Counter()
for seg in segments:
    for m in re.finditer(PAT, seg):
        b = m.group().encode("utf-8")
        key = tuple(b[i: i+1] for i in range(len(b)))
        counts[key] += 1

print("\n这一步的产物 counts =")
for k, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
    print(f"  {c:3d}  {k}")

# ══════════════════════════════════════════════════════════════
# ② 完整的多轮合并循环  —— 你在这一区从零写
# ══════════════════════════════════════════════════════════════
#
# 输入:  work_counts  dict[tuple[bytes, ...], int]   ① 的产物（副本）
#        vocab_size    300
#        num_initial   257                          = 256 字节 + 1 特殊 token
#
# 输出:  merges       list[tuple[bytes, bytes]]       按产生顺序
#
# 循环 vocab_size - num_initial = 43 轮，每轮:
#   ① 统计相邻对的加权频次          -> pair_counts
#   ② 选最优 pair（最高频；平局取字典序更大）
#   ③ 用 best 重写所有 key          -> new_counts
#   ④ merges.append(best); work_counts = new_counts
#
# 边界: 某一轮 pair_counts 为空 -> 提前 break
# 注意: 用 work_counts = Counter(counts) 开副本，别改坏下面 ③ 的参考代码

vocab_size = 300
num_initial = 257

work_counts = Counter(counts)
merges = []

# ← 你的代码写在这里


print("\n══════ 合并循环结果 ══════")
print("merges 条数 :", len(merges), "(上限 43；实际会在 20 条时提前结束)")
print("前 5 条     :", merges[:5])
print("最终格子数  :", len(work_counts), "(应等于 10，且全是单元素)")

EXPECTED_MERGES = [
    (b's', b't'),
    (b'e', b'st'),
    (b'o', b'w'),
    (b'l', b'ow'),
    (b'w', b'est'),
    (b'n', b'e'),
    (b'ne', b'west'),
    (b' ', b'low'),
    (b' ', b'newest'),
    (b'w', b'i'),
]
print("\n期望的前 10 条:")
for i, m in enumerate(EXPECTED_MERGES, 1):
    print(f"   {i:2d}  {m}")
print("你的前 10 条一致:", merges[:10] == EXPECTED_MERGES)

# ══════════════════════════════════════════════════════════════
# ③ 参考: 你已经跑通的单轮版本（保留，别删）
# ══════════════════════════════════════════════════════════════

#目标： 上一步产物变成，相邻对统计次数的表
pair_counts = Counter()
for key, freq in counts.items():
    for a,b in zip(key, key[1:]):
        pair_counts[(a, b)] += freq

# 目标找最多次相邻对
top = max(pair_counts.values())
candidates = [p for p,c in pair_counts.items() if c == top]
best = max(candidates)


# 用best改写格子中的数
new_token = best[0] + best[1]

print("\n══════ 单轮参考 ══════")
print("best      :", best)
print("new_token :", new_token)
new_counts = Counter()
for key,freq in counts.items():
    out = []
    i = 0
    while i < len(key):
        if i + 1 < len(key) and (key[i], key[i+1]) == best:
            out.append(new_token)
            i += 2
        else:
            out.append(key[i])
            i += 1
    new_counts[tuple(out)] += freq
print("new_counts:", dict(new_counts))

# ══════ 期望值 ═════════════════════════════════════════════════
# EXPECTED = {
#     (b"\n",): 5,
#     (b" ", b"l", b"o", b"w"): 4,
#     (b" ", b"n", b"e", b"w", b"e", b"s", b"t"): 4,
#     (b" ", b"w", b"i", b"d", b"e", b"s", b"t"): 3,
#     (b" ", b"l", b"o", b"w", b"e", b"r"): 1,
#     (b"a", b"b", b"c"): 1,
#     (b"d", b"e", b"f"): 1,
#     (b"l", b"o", b"w"): 1,
#     (b"l", b"o", b"w", b"e", b"r"): 1,
#     (b"n", b"e", b"w", b"e", b"s", b"t"): 1,
# }
#
# print("\n══════ 对照 ══════")
# print("pre-token 总个数 :", sum(counts.values()), "(期望 22)")
# print("pre-token 种类数 :", len(counts), "(期望 10)")
# print("和期望完全一致   :", counts == EXPECTED)
