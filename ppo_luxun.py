"""
ppo_luxun.py - RLHF PPO 训练
在 SFT 模型基础上，使用奖励模型和 Critic 进行在线优化。
为简化演示，奖励模型使用一个简单的规则函数（实际需替换为训练好的 Reward Model）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from tokenizers import Tokenizer
from typing import List

# 模型定义（同前）
class RMSNorm(nn.Module): ...
class FullTransformer(nn.Module): ...

# 奖励模型占位：实际应替换为训练好的 RewardModel
class RewardModel(nn.Module):
    """简单的奖励模型：用一个线性层将最后一个 token 的隐状态映射为标量"""
    def __init__(self, base_model: FullTransformer):
        super().__init__()
        self.base = base_model
        self.score_head = nn.Linear(base_model.d_model, 1, bias=False)
    def forward(self, input_ids):
        hidden = self.base(input_ids)  # 只取隐状态，不需要 lm_head
        # 取最后一个 token 的隐状态
        last_hidden = hidden[:, -1, :]  # [B, d_model]
        return self.score_head(last_hidden).squeeze(-1)  # [B]

# Critic 模型：与 Actor 结构相同，但最后输出一个标量值
class CriticModel(nn.Module):
    def __init__(self, base_model: FullTransformer):
        super().__init__()
        self.base = base_model
        self.value_head = nn.Linear(base_model.d_model, 1, bias=False)
    def forward(self, input_ids):
        hidden = self.base(input_ids)  # [B, S, d_model]
        values = self.value_head(hidden).squeeze(-1)  # [B, S]
        return values

def compute_kl(log_probs, ref_log_probs):
    """近似 KL 散度"""
    return log_probs - ref_log_probs

def ppo_train():
    # 超参数
    batch_size = 4
    epochs = 3
    lr_actor = 1e-6
    lr_critic = 5e-6
    clip_eps = 0.2
    kl_coef = 0.05
    gamma = 0.95
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_len = 256

    # 加载分词器
    tokenizer = Tokenizer.from_file("tokenizer_luxun/tokenizer.json")
    vocab_size = len(tokenizer.get_vocab())

    # 加载 SFT 模型作为 Actor，并复制一份 Reference（冻结）
    actor = FullTransformer(vocab_size, d_model=64, n_layers=3, n_heads=4).to(device)
    actor.load_state_dict(torch.load("checkpoints_sft/sft_model.pth", map_location=device))
    ref_model = FullTransformer(vocab_size, d_model=64, n_layers=3, n_heads=4).to(device)
    ref_model.load_state_dict(torch.load("checkpoints_sft/reference_model.pth", map_location=device))
    for param in ref_model.parameters():
        param.requires_grad = False

    # 奖励模型（此处使用简单版本，实际需加载训练好的 RewardModel）
    reward_model = RewardModel(FullTransformer(vocab_size, d_model=64, n_layers=3, n_heads=4)).to(device)
    # 加载预训练的奖励模型权重（假设已训练好）
    # reward_model.load_state_dict(torch.load("reward_model.pth"))

    # Critic 模型（随机初始化）
    critic = CriticModel(FullTransformer(vocab_size, d_model=64, n_layers=3, n_heads=4)).to(device)

    # 优化器
    opt_actor = torch.optim.AdamW(actor.parameters(), lr=lr_actor)
    opt_critic = torch.optim.AdamW(critic.parameters(), lr=lr_critic)

    # 加载 RL 数据（格式：最后一个 assistant 回复留空，等待 Actor 生成）
    # 这里简化：随机选取一些 prompt
    prompts = ["<|user|>你好<|assistant|>", "<|user|>今天天气不错<|assistant|>"]

    actor.train()
    critic.train()
    for epoch in range(epochs):
        for prompt in prompts:
            # 1. Actor 生成回答
            input_ids = tokenizer.encode(prompt).ids
            input_tensor = torch.tensor([input_ids], device=device)
            # 自回归生成
            generated = []
            log_probs = []
            ref_log_probs = []
            values = []
            with torch.no_grad():
                for _ in range(50):  # 最大生成长度
                    logits = actor(input_tensor)[:, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    generated.append(next_token.item())
                    # 记录 log_prob
                    log_prob = torch.log(probs[0, next_token.item()])
                    log_probs.append(log_prob)
                    # 参考模型 log_prob
                    ref_logits = ref_model(input_tensor)[:, -1, :]
                    ref_probs = F.softmax(ref_logits, dim=-1)
                    ref_log_prob = torch.log(ref_probs[0, next_token.item()])
                    ref_log_probs.append(ref_log_prob)
                    # Critic 价值预测（当前状态）
                    val = critic(input_tensor)[:, -1]  # 最后一个 token 的价值
                    values.append(val)
                    input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)
                    if next_token.item() == tokenizer.token_to_id("<|eos|>"):
                        break

            # 2. 计算奖励
            full_ids = torch.tensor([input_ids + generated], device=device)
            reward = reward_model(full_ids).item()  # 标量
            # 计算每步奖励：KL 惩罚 + 最后一步加上奖励
            rewards = []
            for t in range(len(generated)):
                kl_pen = kl_coef * (log_probs[t] - ref_log_probs[t])
                step_reward = -kl_pen
                if t == len(generated) - 1:
                    step_reward += reward
                rewards.append(step_reward)

            # 3. 计算回报和优势
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns, device=device)
            values = torch.stack(values).squeeze()
            advantages = returns - values

            # 4. PPO 更新
            # 重新计算 Actor 的当前 log_prob
            curr_log_probs = []
            inp = torch.tensor([input_ids], device=device)
            for t, token_id in enumerate(generated):
                logits = actor(inp)[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                curr_log_prob = torch.log(probs[0, token_id])
                curr_log_probs.append(curr_log_prob)
                inp = torch.cat([inp, torch.tensor([[token_id]], device=device)], dim=1)
            curr_log_probs = torch.stack(curr_log_probs)
            old_log_probs = torch.stack(log_probs).detach()
            ratio = torch.exp(curr_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            opt_actor.zero_grad()
            actor_loss.backward()
            opt_actor.step()

            # Critic 更新
            critic_loss = F.mse_loss(values, returns.detach())
            opt_critic.zero_grad()
            critic_loss.backward()
            opt_critic.step()

        print(f"PPO epoch {epoch+1} 完成")

    torch.save(actor.state_dict(), "ppo_actor.pth")
    print("PPO 训练完成！")

if __name__ == "__main__":
    ppo_train()