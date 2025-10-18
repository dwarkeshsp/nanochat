"""
Simple RL training script for teaching a model to add two numbers.
Demonstrates REINFORCE and GRPO algorithms in a minimal, clean implementation.
"""
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from nanochat.common import compute_init, compute_cleanup
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine


# Config
num_steps = 10
problems_per_batch = 4
samples_per_problem = 16  
max_new_tokens = 256  
temperature = 1.0  
top_k = 50 

ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)


def generate_prompt_and_answer(tokenizer):
    a = random.randint(1, 100)
    b = random.randint(1, 100)
    question_text = f"Compute the sum of {a} and {b}."
    
    SOT_USER = tokenizer.encode_special("<|user_start|>")
    EOT_USER = tokenizer.encode_special("<|user_end|>")
    SOT_MODEL = tokenizer.encode_special("<|assistant_start|>")
    prompt_tokens = [SOT_USER] + tokenizer.encode(question_text) + [EOT_USER, SOT_MODEL]
    
    return prompt_tokens, str(a + b)

@torch.no_grad()
def get_problem(engine, tokenizer):
    """Generator that yields problems with multiple samples and rewards."""
    EOT_MODEL = tokenizer.encode_special("<|assistant_end|>")
    
    while True:
        prompt_tokens, correct_answer = generate_prompt_and_answer(tokenizer)
        prefix_length = len(prompt_tokens)

        with autocast_ctx:
            generated_token_sequences, masks = engine.generate_batch(
                prompt_tokens, num_samples=samples_per_problem,
                max_tokens=max_new_tokens, temperature=temperature, top_k=top_k
            )

        rewards = []
        for sample_tokens in generated_token_sequences:
            generated_text = tokenizer.decode(sample_tokens[prefix_length:])
            reward = 1.0 if correct_answer in generated_text else 0.0
            rewards.append(reward)
        
        # Pad sequences
        max_length = max(len(seq) for seq in generated_token_sequences)
        padded_sequences = [seq + [EOT_MODEL] * (max_length - len(seq)) for seq in generated_token_sequences]
        padded_masks = [mask + [0] * (max_length - len(mask)) for mask in masks]
        
        ids = torch.tensor(padded_sequences, dtype=torch.long, device=device) # (N, T)
        mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device) # (N, T)
        inputs = ids[:, :-1] # (N, T-1)
        targets = ids[:, 1:].clone() # (N, T-1)
        targets[mask_ids[:, 1:] == 0] = -1
        
        rewards = torch.tensor(rewards, dtype=torch.float, device=device) # (N)
        advantages = rewards - rewards.mean() # (N)
            
        yield generated_token_sequences, inputs, targets, rewards, advantages

def REINFORCE_loss(log_probs, rewards, advantages):
    """REINFORCE: weight by absolute rewards."""
    return (log_probs * rewards.unsqueeze(-1)).sum()

def GRPO_loss(log_probs, rewards, advantages):
    """GRPO: weight by advantages (centered rewards)."""
    return (log_probs * advantages.unsqueeze(-1)).sum()

def train(loss_function, algorithm_name):
    model, tokenizer, _ = load_model('sft', device, phase="eval")
    engine = Engine(model, tokenizer)
    optimizers = model.setup_optimizers(
        unembedding_lr=0.0004, embedding_lr=0.02, matrix_lr=0.002, weight_decay=0.0
    )
    
    problem_iterator = get_problem(engine, tokenizer)
    reward_history = []
    
    for step in range(num_steps):
        rewards_list = []
        
        for problem in range(problems_per_batch):
            sequences, inputs, targets, rewards, advantages = next(problem_iterator)
            
            model.train()
            with autocast_ctx:
                logp = -model(inputs, targets, loss_reduction='none').view_as(inputs) # (N, T)
            
            pg_obj = loss_function(logp, rewards, advantages) 
            num_valid = (targets >= 0).sum().clamp(min=1)
            pg_obj = pg_obj / (num_valid * problems_per_batch)
            
            loss = -pg_obj
            loss.backward()
            
            rewards_list.append(rewards.mean().item())
        
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)
        
        mean_reward = sum(rewards_list) / len(rewards_list)
        reward_history.append(mean_reward)
       
        print(f"{algorithm_name} | Step {step}/{num_steps} | Avg Reward: {mean_reward:.2f}")
    
    return reward_history

def plot_rewards(reinforce_rewards, grpo_rewards):
    plt.figure(figsize=(10, 6))
    plt.plot(reinforce_rewards, label='REINFORCE', linewidth=2)
    plt.plot(grpo_rewards, label='GRPO', linewidth=2)
    plt.xlabel('Training Step'), plt.ylabel('Average Reward')
    plt.title('RL Algorithm Comparison: Addition Task')
    plt.legend(), plt.grid(alpha=0.3)
    plt.savefig('rl_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved comparison plot to rl_comparison.png")

if __name__ == "__main__":
    reinforce_rewards = train(REINFORCE_loss, "REINFORCE")
    grpo_rewards = train(GRPO_loss, "GRPO")

    plot_rewards(reinforce_rewards, grpo_rewards)
    
    compute_cleanup()



