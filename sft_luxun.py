"""
sft_luxun.py - 对预训练模型进行有监督微调 (SFT)
基于预训练的鲁迅 Transformer，使用多轮对话数据训练助手行为。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import json
from tokenizers import Tokenizer

# 复用预训练脚本中的模型定义（或直接导入）
# 这里直接复制 FullTransformer / RMSNorm / RoPE 函数，保持独立
# （因篇幅省略具体实现，实际使用时请从 train_Luxun.py 中复制）

# ---- 模型定义 (与 train_Luxun.py 完全相同) ----
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6): ...

class FullTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=4, n_heads=4, max_seq_len=512, rope_theta=10000.0):
        ...

# ---- 对话数据处理 ----
def format_conversation(conv):
    """
    将对话列表格式化为模型输入字符串。
    例如: <|user|>你好<|assistant|>你好！<|user|>再见<|assistant|>再见！
    使用特殊 token 分隔角色。
    """
    special_tokens = {
        "system": "<|system|>",
        "user": "<|user|>",
        "assistant": "<|assistant|>",
        "tool": "<|tool|>",
    }
    text = ""
    for turn in conv:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role in special_tokens:
            text += special_tokens[role] + content
    return text

def load_sft_data(filepath):
    """加载 SFT 数据 (jsonl)"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            conversations = sample["conversations"]
            data.append(conversations)
    return data

class SFTDataset(torch.utils.data.Dataset):
    """SFT 数据集，仅对 assistant 回复部分计算损失"""
    def __init__(self, data, tokenizer, max_len=256):
        self.samples = []
        bos_id = tokenizer.token_to_id("<|bos|>")
        eos_id = tokenizer.token_to_id("<|eos|>")
        pad_id = tokenizer.token_to_id("<|pad|>")
        user_id = tokenizer.token_to_id("<|user|>")
        assistant_id = tokenizer.token_to_id("<|assistant|>")

        for conv in data:
            # 格式化为文本
            text = format_conversation(conv)
            # 编码
            ids = tokenizer.encode(text).ids
            # 构造 labels：只对 assistant 生成的内容计算损失
            labels = [-100] * len(ids)  # -100 表示忽略
            # 找到所有 assistant 回复的片段
            # 简化：使用 token 序列中的 <|assistant|> 标记
            # 将 ids 转为 list 以便遍历
            i = 0
            while i < len(ids):
                if ids[i] == assistant_id:
                    # 找到 assistant 回复的起始（<|assistant|> 之后）
                    j = i + 1
                    while j < len(ids) and ids[j] not in [user_id, assistant_id, pad_id, bos_id]:
                        j += 1
                    # 标记为需要计算损失
                    for k in range(i, j):
                        labels[k] = ids[k]
                    i = j
                else:
                    i += 1
            # 截断或填充
            if len(ids) > max_len:
                ids = ids[:max_len]
                labels = labels[:max_len]
            else:
                padding = [pad_id] * (max_len - len(ids))
                ids = ids + padding
                labels = labels + [-100] * (max_len - len(labels))
            self.samples.append((ids, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids, labels = self.samples[idx]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

def train_sft():
    # 超参数
    batch_size = 4
    epochs = 5
    lr = 1e-5
    max_len = 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = "checkpoints_sft"
    os.makedirs(save_dir, exist_ok=True)

    # 加载分词器
    tokenizer = Tokenizer.from_file("tokenizer_luxun/tokenizer.json")
    vocab_size = len(tokenizer.get_vocab())

    # 加载预训练模型
    ckpt = torch.load("checkpoints_luxun2/best.pth", map_location=device)
    model = FullTransformer(vocab_size, d_model=64, n_layers=3, n_heads=4, max_seq_len=512).to(device)
    model.load_state_dict({k: v.to(device=device, dtype=torch.float) for k, v in ckpt.items()})
    # 冻结参考模型 (SFT 后可作为 reference)
    ref_model = FullTransformer(vocab_size, d_model=64, n_layers=3, n_heads=4, max_seq_len=512).to(device)
    ref_model.load_state_dict(model.state_dict())
    for param in ref_model.parameters():
        param.requires_grad = False

    # 加载 SFT 数据
    train_data = load_sft_data("sft_data.jsonl")  # 请替换为实际路径
    dataset = SFTDataset(train_data, tokenizer, max_len)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda', enabled=(device != 'cpu'))

    model.train()
    global_step = 0
    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device != 'cpu')):
                logits = model(x)
                # 只对非 -100 的 labels 计算交叉熵
                loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1), ignore_index=-100)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            total_loss += loss.item()
            global_step += 1
        print(f"Epoch {epoch+1}: loss = {total_loss/len(loader):.4f}")

    # 保存 SFT 模型
    torch.save(model.state_dict(), os.path.join(save_dir, "sft_model.pth"))
    # 保存 reference 模型
    torch.save(ref_model.state_dict(), os.path.join(save_dir, "reference_model.pth"))
    print("SFT 训练完成！")

if __name__ == "__main__":
    train_sft()