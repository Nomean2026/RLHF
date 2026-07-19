"""
dpo_luxun.py - DPO 训练
直接使用偏好对数据，不需要奖励模型和 Critic。
"""

import torch
import torch.nn.functional as F
import os
import json
from tokenizers import Tokenizer

# 模型定义（同前）
class RMSNorm(nn.Module): ...
class FullTransformer(nn.Module): ...

def load_dpo_data(filepath):
    """加载 DPO 偏好数据"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            chosen = sample["chosen"]
            rejected = sample["rejected"]
            # 转换为文本
            chosen_text = "".join([f"<|{t['role']}|>{t['content']}" for t in chosen])
            rejected_text = "".join([f"<|{t['role']}|>{t['content']}" for t in rejected])
            data.append((chosen_text, rejected_text))
    return data

def dpo_train():
    batch_size = 4
    epochs = 3
    lr = 1e-6
    beta = 0.1  # DPO 温度系数
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_len = 256

    tokenizer = Tokenizer.from_file("tokenizer_luxun/tokenizer.json")
    vocab_size = len(tokenizer.get_vocab())
    pad_id = tokenizer.token_to_id("<|pad|>")

    # 加载 SFT 模型作为 Policy，并复制 Reference
    policy = FullTransformer(vocab_size, d_model=64, n_layers=3, n_heads=4).to(device)
    policy.load_state_dict(torch.load("checkpoints_sft/sft_model.pth", map_location=device))
    ref_model = FullTransformer(vocab_size, d_model=64, n_layers=3, n_heads=4).to(device)
    ref_model.load_state_dict(torch.load("checkpoints_sft/reference_model.pth", map_location=device))
    for param in ref_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

    # 加载 DPO 数据
    train_data = load_dpo_data("dpo_data.jsonl")  # 替换为实际路径

    def get_log_prob(model, text):
        """计算给定文本的对数概率（总和）"""
        ids = tokenizer.encode(text).ids
        if len(ids) > max_len:
            ids = ids[:max_len]
        ids_tensor = torch.tensor([ids], device=device)
        logits = model(ids_tensor)  # [1, S, V]
        # 使用 cross_entropy 计算 token 级 log_prob 并求和
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = ids_tensor[:, 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, vocab_size), shift_labels.view(-1), reduction='none')
        # 转换为 log_prob (sum)
        log_prob = -loss.sum()
        return log_prob

    policy.train()
    for epoch in range(epochs):
        total_loss = 0
        for chosen_text, rejected_text in train_data:
            # 计算两个回答在当前策略和参考模型下的对数概率
            logp_chosen_policy = get_log_prob(policy, chosen_text)
            logp_rejected_policy = get_log_prob(policy, rejected_text)
            with torch.no_grad():
                logp_chosen_ref = get_log_prob(ref_model, chosen_text)
                logp_rejected_ref = get_log_prob(ref_model, rejected_text)

            # DPO 损失
            diff_chosen = logp_chosen_policy - logp_chosen_ref
            diff_rejected = logp_rejected_policy - logp_rejected_ref
            loss = -F.logsigmoid(beta * (diff_chosen - diff_rejected))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"DPO epoch {epoch+1}: loss = {total_loss/len(train_data):.4f}")

    torch.save(policy.state_dict(), "dpo_policy.pth")
    print("DPO 训练完成！")

if __name__ == "__main__":
    dpo_train()