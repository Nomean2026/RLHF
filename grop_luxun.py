"""
grpo_luxun.py - GRPO 训练
组内相对策略优化：对同一 prompt 采样多个回答，组内标准化奖励，无需 Critic。
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
from tokenizers import Tokenizer

# 模型定义（同前）
class RMSNorm(nn.Module): ...
class FullTransformer(nn.Module): ...

def grpo_train():
    batch_size = 1  # 每次处理一个 prompt
    G = 4           # 每组回答数量
    epochs = 3
    lr = 1e-6
    clip_eps = 0.2
    kl_coef = 0.05
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_len = 256

    tokenizer = Tokenizer.from_file("tokenizer_luxun/tokenizer.json")
    vocab_size = len(tokenizer.get_vocab())
    eos_id = tokenizer.token_to_id("<|eos|>")

    # 加载 SFT 模型作为 Policy，Reference 冻结
    policy = FullTransformer(vocab_size, d_model=64, n_layers=3, n_heads=4).to(device)
    policy.load_state_dict(torch.load("checkpoints_sft/sft_model.pth", map_location=device))
    ref_model = FullTransformer(vocab_size, d_model=64, n_layers=3, n_heads=4).to(device)
    ref_model.load_state_dict(torch.load("checkpoints_sft/reference_model.pth", map_location=device))
    for param in ref_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

    # 奖励模型占位（同 PPO 中的 RewardModel）
    reward_model = RewardModel(FullTransformer(vocab_size, d_model=64, n_layers=3, n_heads=4)).to(device)
    # reward_model.load_state_dict(torch.load("reward_model.pth"))

    prompts = ["<|user|>你好<|assistant|>", "<|user|>今天天气不错<|assistant|>"]

    policy.train()
    for epoch in range(epochs):
        for prompt in prompts:
            # 1. 采样 G 个回答
            responses = []
            all_log_probs = []  # 旧策略下的 log_probs
            for _ in range(G):
                input_ids = tokenizer.encode(prompt).ids
                input_tensor = torch.tensor([input_ids], device=device)
                generated = []
                log_probs = []
                for step in range(50):
                    logits = policy(input_tensor)[:, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    generated.append(next_token.item())
                    log_prob = torch.log(probs[0, next_token.item()])
                    log_probs.append(log_prob)
                    input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)
                    if next_token.item() == eos_id:
                        break
                responses.append(generated)
                all_log_probs.append(log_probs)

            # 2. 计算奖励
            rewards = []
            for gen in responses:
                full_ids = torch.tensor([tokenizer.encode(prompt).ids + gen], device=device)
                r = reward_model(full_ids).item()
                rewards.append(r)

            # 3. 组内标准化得优势
            rewards_t = torch.tensor(rewards, device=device)
            mean_r = rewards_t.mean()
            std_r = rewards_t.std() + 1e-8
            advantages = (rewards_t - mean_r) / std_r

            # 4. 计算 GRPO 损失
            total_loss = 0
            total_tokens = 0
            for i in range(G):
                advantage = advantages[i]
                if advantage == 0:
                    continue
                # 重新计算当前策略下的 log_prob
                gen_ids = responses[i]
                old_log_probs = all_log_probs[i]
                inp = torch.tensor([tokenizer.encode(prompt).ids], device=device)
                curr_log_probs = []
                for token_id in gen_ids:
                    logits = policy(inp)[:, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    curr_log_prob = torch.log(probs[0, token_id])
                    curr_log_probs.append(curr_log_prob)
                    inp = torch.cat([inp, torch.tensor([[token_id]], device=device)], dim=1)
                # 将列表转换为 tensor
                curr_log_probs = torch.stack(curr_log_probs)
                old_log_probs = torch.stack(old_log_probs).detach()
                ratio = torch.exp(curr_log_probs - old_log_probs)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantage
                loss_i = -torch.min(surr1, surr2).sum()
                total_loss += loss_i
                total_tokens += len(gen_ids)

                # 额外 KL 惩罚
                # ref_log_probs = ... 可在这里添加

            if total_tokens > 0:
                loss = total_loss / total_tokens
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f"GRPO epoch {epoch+1} 完成")

    torch.save(policy.state_dict(), "grpo_policy.pth")
    print("GRPO 训练完成！")

if __name__ == "__main__":
    grpo_train()