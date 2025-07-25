#!/usr/bin/env python3
"""
示例：使用 use_turn_scores=True 进行独立奖励分配（不融合）

这个示例展示了如何：
1. 配置 use_turn_scores=True 
2. 过程奖励分配到各轮次最后token
3. LLM评分独立分配到序列最后token  
4. 两种奖励信号保持独立，不进行融合
"""

import torch
from ragen.llm_agent.ctx_manager import get_masks_and_scores
from transformers import AutoTokenizer

def create_mock_env_outputs_with_step_rewards():
    """创建包含每步奖励的模拟环境输出"""
    env_outputs = [
        {
            "env_id": 0,
            "group_id": 0,
            "tag": "WebBrowser",
            "history": [
                {
                    "state": "网页状态1",
                    "llm_response": "我需要点击登录按钮",
                    "reward": 2.5,  # 第1轮过程奖励
                    "actions_left": 2
                },
                {
                    "state": "网页状态2", 
                    "llm_response": "现在输入用户名",
                    "reward": 3.8,  # 第2轮过程奖励
                    "actions_left": 1
                },
                {
                    "state": "网页状态3",
                    "llm_response": "输入密码并登录",
                    "reward": 8.2,  # 第3轮过程奖励
                    "actions_left": 0
                }
            ],
            "metrics": {}
        },
        {
            "env_id": 1, 
            "group_id": 0,
            "tag": "WebBrowser",
            "history": [
                {
                    "state": "另一个网页状态1",
                    "llm_response": "分析页面结构",
                    "reward": 1.5,  # 第1轮过程奖励
                    "actions_left": 1
                },
                {
                    "state": "另一个网页状态2",
                    "llm_response": "找到目标元素并完成任务",
                    "reward": 7.3,  # 第2轮过程奖励
                    "actions_left": 0
                }
            ],
            "metrics": {}
        }
    ]
    
    # 模拟LLM终局评分（0-10分）
    llm_final_scores = [9.2, 6.8]  # 对应两个轨迹的最终评分
    
    return env_outputs, llm_final_scores

def demo_independent_reward_allocation():
    """演示独立奖励分配的工作原理"""
    print("="*60)
    print("🎯 演示独立奖励分配机制（无融合）")
    print("="*60)
    
    # 1. 模拟环境输出数据
    env_outputs, llm_scores = create_mock_env_outputs_with_step_rewards()
    
    print("📋 模拟数据:")
    for i, env_output in enumerate(env_outputs):
        print(f"  轨迹{i}: {len(env_output['history'])}轮交互")
        for j, turn in enumerate(env_output['history']):
            if 'reward' in turn:
                print(f"    轮次{j+1}: 过程奖励={turn['reward']}")
        print(f"    LLM终局评分: {llm_scores[i]}")
    
    # 2. 提取过程奖励序列
    step_scores = [[turn['reward'] for turn in env_output['history'] if 'reward' in turn] 
                  for env_output in env_outputs]
    
    print(f"\n📊 过程奖励矩阵:")
    for i, score_list in enumerate(step_scores):
        print(f"  轨迹{i}: {score_list}")
    
    print(f"\n📊 LLM终局评分: {llm_scores}")
    
    # 3. 模拟tokenization和奖励分配
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    
    # 创建模拟的input_ids
    mock_input_ids = torch.tensor([
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 151645, 11, 12, 13, 14, 151645, 15, 16, 17, 18, 151645],  # 轨迹0: 3轮对话
        [1, 2, 3, 4, 5, 6, 7, 151645, 8, 9, 10, 11, 151645, 0, 0, 0, 0, 0, 0, 0]  # 轨迹1: 2轮对话，末尾padding
    ])
    
    print(f"\n🔤 模拟input_ids形状: {mock_input_ids.shape}")
    
    # 4. 分配过程奖励（use_turn_scores=True）
    print(f"\n🎯 Step 1: 分配过程奖励到各轮次:")
    step_reward_tensor, loss_mask, response_mask = get_masks_and_scores(
        mock_input_ids, tokenizer, step_scores, use_turn_scores=True, enable_response_mask=False
    )
    
    print(f"  过程奖励tensor形状: {step_reward_tensor.shape}")
    for i in range(step_reward_tensor.shape[0]):
        step_nonzero = torch.nonzero(step_reward_tensor[i]).flatten()
        step_values = step_reward_tensor[i][step_nonzero]
        print(f"    轨迹{i}: 位置{step_nonzero.tolist()} = {step_values.tolist()}")
    
    # 5. 分配LLM评分（只在最后token）
    print(f"\n🎯 Step 2: 分配LLM评分到最后token:")
    llm_reward_tensor = torch.zeros_like(step_reward_tensor)
    llm_final_scores_tensor = torch.tensor(llm_scores, dtype=step_reward_tensor.dtype)
    llm_reward_tensor[:, -1] = llm_final_scores_tensor
    
    print(f"  LLM评分tensor形状: {llm_reward_tensor.shape}")
    for i in range(llm_reward_tensor.shape[0]):
        print(f"    轨迹{i}: 最后位置 = {llm_reward_tensor[i, -1].item()}")
    
    # 6. 展示独立分配结果
    print(f"\n✅ 独立分配结果总结:")
    print(f"  📈 过程奖励: 分配到各轮次最后token，保持细粒度反馈")
    print(f"  📋 LLM评分: 分配到序列最后token，提供整体质量评估")
    print(f"  🔗 两种信号独立: 不融合，分别提供学习信号")
    
    # 7. 对比融合模式的区别
    print(f"\n🔍 与融合模式的区别:")
    print(f"  独立模式: step_reward_scores + llm_reward_scores (两个独立tensor)")
    print(f"  融合模式: 单一tensor，最后token = 加权融合后的分数")
    print(f"  优势: 保留更丰富的学习信号，支持多目标优化")

def create_config_for_independent_rewards():
    """创建支持独立奖励分配的配置示例"""
    print("\n" + "="*60)
    print("⚙️ 配置独立奖励分配的示例")
    print("="*60)
    
    config_example = """
# 在 config/base-browser.yaml 中的配置示例:

agent_proxy:
  # 启用按轮次奖励分配
  use_turn_scores: True
  
  # 其他相关配置
  max_turn: 20
  enable_think: True
  
  # 奖励归一化配置
  reward_normalization:
    grouping: "state"
    method: "identity"  # 建议使用identity保持原始奖励结构

# 重要变化:
# 1. 移除了 step_reward_weight 和 llm_reward_weight 配置
# 2. 过程奖励和LLM评分不再融合
# 3. 分别保存为 step_reward_scores 和 llm_reward_scores

# 训练时的数据结构:
# rollouts.batch["step_reward_scores"]  # (B, L-1) 过程奖励
# rollouts.batch["llm_reward_scores"]   # (B, L-1) LLM评分（只在最后token有值）
"""
    
    print(config_example)

def demonstrate_reward_independence():
    """演示奖励独立性的优势"""
    print("\n" + "="*60)
    print("💡 独立奖励分配的优势")
    print("="*60)
    
    advantages = """
🎯 独立奖励分配的优势:

1. 信号纯净性:
   - 过程奖励: 专注于每步行为的即时反馈
   - LLM评分: 专注于最终结果的整体质量
   - 避免信号混淆和权重调优困难

2. 训练灵活性:
   - 可以分别调整两种奖励的学习率
   - 支持多目标优化策略
   - 便于消融实验和效果分析

3. 可解释性:
   - 清晰区分过程监督和结果监督
   - 便于调试和理解模型行为
   - 支持分别分析两种奖励的贡献

4. 扩展性:
   - 易于添加新的奖励信号
   - 支持更复杂的多层次奖励结构
   - 为未来的奖励机制留下空间

🔧 使用建议:
- 确保环境输出每轮都有reward字段
- 设置use_turn_scores=True
- 监控训练日志中的独立奖励分配信息
- 根据任务特点调整奖励权重（在trainer层面）
"""
    
    print(advantages)

if __name__ == "__main__":
    demo_independent_reward_allocation()
    create_config_for_independent_rewards()
    demonstrate_reward_independence()
    
    print("\n" + "="*60)
    print("✅ 独立奖励分配演示完成！")
    print("💡 要在实际训练中使用，请:")
    print("   1. 修改配置文件: use_turn_scores: True")
    print("   2. 确保环境输出每轮都有reward字段")  
    print("   3. 监控训练日志中的独立分配调试信息")
    print("   4. 在trainer层面处理两种独立的奖励信号")
    print("="*60) 