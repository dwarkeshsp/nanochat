"""
Simple RL training script for teaching a model to add two numbers.
Demonstrates REINFORCE and GRPO algorithms in a minimal, clean implementation.
"""
import random
import torch
import torch.nn.functional as F

from nanochat.common import compute_init, compute_cleanup
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine


# Config
num_steps = 1 
problems_per_batch = 4  
samples_per_problem = 4  
max_new_tokens = 64  
temperature = 1.0  
top_k = 50 

# Init
source = "sft" 
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
model, tokenizer, meta = load_model(source, device, phase="eval")
engine = Engine(model, tokenizer)
dtype = torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)


# Special tokens
SOT_USER = tokenizer.encode_special("<|user_start|>")
EOT_USER = tokenizer.encode_special("<|user_end|>")
SOT_MODEL = tokenizer.encode_special("<|assistant_start|>")
EOT_MODEL = tokenizer.encode_special("<|assistant_end|>")


def generate_prompt_and_answer():
    a = random.randint(10, 100)
    b = random.randint(10, 100)
    
    # Build prompt tokens correctly (tokenizer.encode returns a list!)
    question_text = f"Compute the sum of {a} and {b}."
    prompt_tokens = [SOT_USER] + tokenizer.encode(question_text) + [EOT_USER, SOT_MODEL]
    
    # The correct answer as a string
    correct_answer = str(a + b)
    
    return prompt_tokens, correct_answer

@torch.no_grad()
def get_batch():
    """
    Generator that yields batches for RL training.
    For each problem, generates multiple samples and computes rewards.
    """
    for problem in range(problems_per_batch):
        prompt_tokens, correct_answer = generate_prompt_and_answer()
        prefix_length = len(prompt_tokens)

        with autocast_ctx:
            generated_token_sequences, masks = engine.generate_batch(
                prompt_tokens,
                num_samples=samples_per_problem,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )

        rewards = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            print(f"generated text: {generated_text}")
            # Simple reward: 1.0 if answer is in the text, 0.0 otherwise
            reward = 1.0 if correct_answer in generated_text else 0.0
            rewards.append(reward)
        
        max_length = max(len(seq) for seq in generated_token_sequences)
        padded_sequences = [seq + [EOT_MODEL] * (max_length - len(seq)) for seq in generated_token_sequences]
        padded_masks = [mask + [0] * (max_length - len(mask)) for mask in masks]
        
        ids = torch.tensor(padded_sequences, dtype=torch.long, device=device)  # (N, T)
        mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)  # (N, T)

        inputs = ids[:, :-1]  # (N, T-1)
        targets = ids[:, 1:].clone()  # (N, T-1)
        
        # Mask out prompt and padding tokens in targets
        targets[mask_ids[:, 1:] == 0] = -1  # -1 = ignore in loss
        
        # Compute advantages
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)  # (N,)
        advantages = rewards - rewards.mean() 
        print(f"generated_token_sequences shape: {len(generated_token_sequences)} x {[len(seq) for seq in generated_token_sequences]}")
        print(f"inputs shape: {inputs.shape}")
        print(f"targets shape: {targets.shape}")
        print(f"rewards shape: {rewards.shape}")
        print(f"advantages shape: {advantages.shape}")
            
        yield generated_token_sequences, inputs, targets, rewards, advantages

def REINFORCE_loss(log_probs, rewards, advantages):
    """REINFORCE: weight by absolute rewards."""
    return (log_probs * rewards.unsqueeze(-1)).sum()

def GRPO_loss(log_probs, rewards, advantages):
    """GRPO: weight by advantages (centered rewards)."""
    return (log_probs * advantages.unsqueeze(-1)).sum()

optimizers = model.setup_optimizers(
    unembedding_lr=0.004,
    embedding_lr=0.2,
    matrix_lr=0.02,
    weight_decay=0.0,
)

def train(loss_function):
    batch_iterator = get_batch()
    for step in range(num_steps):
        sequences, inputs, targets, rewards, advantages = next(batch_iterator)

        model.train()

        # Calculate log probabilities. Note that the model returns NLL = -logp, so we negate
        with autocast_ctx:
            logp = -model(inputs, targets, loss_reduction='none').view_as(inputs) # (N, T)
        
        # Calculate the policy gradient objective weighted by rewards or advantages
        pg_obj = loss_function(logp, rewards, advantages)
        
        # Normalize by the number of valid tokens
        num_valid = (targets >= 0).sum().clamp(min=1)
        pg_obj = pg_obj / num_valid
        
        # Formulate the loss to minimize (negate the objective we want to maximize)
        loss = -pg_obj
        loss.backward()

        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)

        if step % 10 == 0:
            print(f"Step {step}/{num_steps} | Loss: {loss.item():.4f} | Avg Reward: {rewards.mean().item():.2f}")


if __name__ == "__main__":
    train(GRPO_loss)
    compute_cleanup()



