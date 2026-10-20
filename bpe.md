# BPE Tokenizer
## 1 Unicode 标准
自然语言不能直接用来训练，会把文本转换成数字序列。
Unicode 规定了每个符号对应的数字编号，相当于一种规则
utf-8 是一种把编号用字节存储到计算机中的编码方式，是一种变长编码
## 2 子词分词
### 2.1 两个极端
- word-level 分词，按空格标点分成词，每个词对应一个 token
- byte-level 分词，把文本编码成字节，每个字节对应一个 token
方法1会有 oov 和词表爆炸的问题，方法2会序列过长的问题。为了解决这两者的问题，所以采用折中方案，也就是下面介绍的 BPE。

## 3 byte-level BPE
### 3.1 词表初始化
词表就是 token-整数ID的映射，utf-8的表示字节有0-255，初始词表数量是256加上特殊token数，特殊token不参与 BPE 合并
```py
vocab = {
    0: b'\x00',
    1: b'\x01',
    ...
    255: b'\xff',
    # 特殊 token 也可以加进来
    256: b'<|endoftext|>',
}
```

### 3.2 预分词
直接在字节序列上进行合并效率不高，也会出现语义相近但ID不同的问题（dog.和 dog!），所以思路是分成粗粒度 预token，然后统计 预token 频次，不跨预token 边界进行合并
```text
"low low low lower" 
→ 预分词: ["low", "low", "low", "lower"]
→ 频次表: {"low": 3, "lower": 1}
→ 后续统计 pair 时，"low" 内部算 3 次，而不是扫 3 遍语料
```
预分词有按空格分词和正则分词，modern tokenizer 一般使用正则分词
### 3.3 合并流程
四步循环
```text
统计所有相邻 pair 的频率（只在预 token 内部）
找最高频的 pair (A, B)
把所有 A B 替换成新 token AB，加入词表
重复 2-3 直到词表达到目标大小
```
最终词表大小是 vocab_size = 256 + 合并次数 + 特殊 token 数

### 3.4 Example
