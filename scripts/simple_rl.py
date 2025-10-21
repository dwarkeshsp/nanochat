"""
Simple RL training script for teaching a model to add.
Demonstrates REINFORCE and GRPO algorithms in a minimal implementation.

If you want to run this script, put it inside of nanochat/scripts/ and run it with:
python -m scripts.simple_rl

First add "matplotlib>=3.9.0" to pyproject.toml and run 'uv sync'

I wrote a separate script to download the weights for the model: 
https://gist.github.com/dwarkeshsp/7b456da6e219d2a0b0d45587d15c3421

Fundamentally, it's not that complicated: generate multiple trajectories per prompt ("Compute 12,323 + 43,765 ="). 
Mask out the prompt. Pad the trajectories to be of equal length. 
Find out what (log) probability your model put on every single token generated in the trajectories. 
Update your model to make it more likely that it produces the tokens in successful trajectories.
"""
import random
import time
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from nanochat.common import compute_init, compute_cleanup, get_base_dir
from nanochat.checkpoint_manager import load_model, save_checkpoint
from nanochat.engine import Engine


# Config
num_steps = 16
problems_per_step = 16
samples_per_problem = 16
max_new_tokens = 256  
temperature = 1.0  
top_k = 50

max_addend = int(1e5)
fraction_val = 0.1
len_val_addends = int(max_addend * fraction_val)
# Randomly split numbers into train/val sets
all_numbers = list(range(max_addend))
random.shuffle(all_numbers)
val_numbers = set(all_numbers[:len_val_addends])
train_numbers = set(all_numbers[len_val_addends:])

ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)


def generate_prompt_and_answer(tokenizer, number_set):    
    a = random.choice(list(number_set))
    b = random.choice(list(number_set))
    question_text = f"Compute the sum of {a} and {b}."
    
    SOT_USER = tokenizer.encode_special("<|user_start|>")
    EOT_USER = tokenizer.encode_special("<|user_end|>")
    SOT_MODEL = tokenizer.encode_special("<|assistant_start|>")
    prompt_tokens = [SOT_USER] + tokenizer.encode(question_text) + [EOT_USER, SOT_MODEL]
    
    return prompt_tokens, str(a + b)

@torch.no_grad()
def get_problem(engine, tokenizer, number_set):
    """Generator that yields problems with multiple samples and rewards."""
    EOT_MODEL = tokenizer.encode_special("<|assistant_end|>")
    
    while True:
        prompt_tokens, correct_answer = generate_prompt_and_answer(tokenizer, number_set)
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

@torch.no_grad()
def evaluate(engine, tokenizer, num_problems=problems_per_step):
    """Evaluate on validation set."""
    problem_iterator = get_problem(engine, tokenizer, number_set=val_numbers)
    total_reward = 0
    
    for _ in range(num_problems):
        _, _, _, rewards, _ = next(problem_iterator)
        total_reward += rewards.mean().item()
    
    return total_reward / num_problems

def train(loss_function):
    model, tokenizer, _ = load_model('sft', device, phase="eval")
    engine = Engine(model, tokenizer)
    optimizers = model.setup_optimizers(
        unembedding_lr=0.0004, embedding_lr=0.02, matrix_lr=0.002, weight_decay=0.0
    )
    python_start = tokenizer.encode_special("<|python_start|>")
    
    problem_iterator = get_problem(engine, tokenizer, number_set=train_numbers)
    train_rewards, val_rewards = [], []
    
    for step in range(num_steps):
        step_start = time.time()
        rewards_list = []
        seqs_with_python = 0

        for problem in range(problems_per_step):
            sequences, inputs, targets, rewards, advantages = next(problem_iterator)
            
            model.train()
            with autocast_ctx:
                logp = -model(inputs, targets, loss_reduction='none').view_as(inputs)
            
            pg_obj = loss_function(logp, rewards, advantages)
            num_valid = (targets >= 0).sum().clamp(min=1)
            pg_obj = pg_obj / (num_valid * problems_per_step)
            loss = -pg_obj
            
            
            rewards_list.append(rewards.mean().item())
            seqs_with_python += sum(python_start in seq for seq in sequences)

            loss.backward()

        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)

        # the rest of this is just logging
        train_reward = sum(rewards_list) / len(rewards_list)
        train_rewards.append(train_reward)

        val_reward = evaluate(engine, tokenizer) if step % 5 == 0 else None
        val_rewards.append(val_reward)
        
        step_time = time.time() - step_start
        frac_seqs_with_python = seqs_with_python / (problems_per_step * samples_per_problem)
        
        log_msg = f"Step {step + 1}/{num_steps} | Train: {train_reward:.2f} | Python Use: {frac_seqs_with_python:.2f} | Time: {step_time:.1f}s"
        if val_reward is not None:
            log_msg += f" | Val: {val_reward:.2f}"
        print(log_msg)
    
    return model, train_rewards, val_rewards

def save_model(model, name):
    base_dir = get_base_dir()
    depth = model.config.n_layer
    model_tag = f"d{depth}"
    checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", f"{model_tag}_{name}")
    save_checkpoint(
        checkpoint_dir,
        num_steps,
        model.state_dict(),
        None,
        {"model_config": model.config.__dict__}
    )
    print(f"✅ Saved {name} model to {checkpoint_dir}")

def plot_rewards(reinforce_train, reinforce_val, grpo_train, grpo_val):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    steps = range(len(reinforce_train))
    ax.plot(steps, reinforce_train, '-', color='blue')
    ax.plot(steps, grpo_train, '-', color='orange')
    
    val_steps = [i for i, v in enumerate(reinforce_val) if v is not None]
    r_val = [v for v in reinforce_val if v is not None]
    g_val = [v for v in grpo_val if v is not None]
    ax.plot(val_steps, r_val, 'o--', color='blue', alpha=0.6)
    ax.plot(val_steps, g_val, 'o--', color='orange', alpha=0.6)
    
    # Dual legends
    algo_handles = [Line2D([0], [0], color=c, lw=2, label=l) 
                    for c, l in [('blue', 'REINFORCE'), ('orange', 'GRPO')]]
    split_handles = [Line2D([0], [0], color='gray', lw=2, ls=ls, marker=m, label=l)
                     for ls, m, l in [('-', '', 'train'), ('--', 'o', 'val')]]
    
    first_legend = ax.legend(handles=algo_handles, loc='upper left')
    ax.add_artist(first_legend)
    ax.legend(handles=split_handles, loc='upper right')
    
    ax.set(xlabel='Training Step', ylabel='Avg Reward', 
           title='RLing Nanochat to add 5-digit numbers')
    ax.grid(alpha=0.3)
    fig.savefig('simple_rl_training_curves.png', dpi=150, bbox_inches='tight')
    print("\nSaved plot to simple_rl_training_curves.png")

if __name__ == "__main__":
    print("Training REINFORCE...")  
    model, reinforce_train, reinforce_val = train(REINFORCE_loss)
    save_model(model, "reinforce")
    print("Training GRPO...")
    model, grpo_train, grpo_val = train(GRPO_loss)
    save_model(model, "grpo")
    plot_rewards(reinforce_train, reinforce_val, grpo_train, grpo_val)
    compute_cleanup()



