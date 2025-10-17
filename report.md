# nanochat training report

Generated: 2025-10-16 17:09:09

## Environment

### Git Information
- Branch: master
- Commit: 4346536 (dirty)
- Message: also allow regenerating assistant message by clicking it, and make sure to feed 

### Hardware
- Platform: Linux
- CPUs: 104 cores (208 logical)
- Memory: 1771.7 GB
- GPUs: 8x NVIDIA H100 80GB HBM3
- GPU Memory: 633.5 GB total
- CUDA Version: 12.8
- Hourly Rate: $24.00/hour

### Software
- Python: 3.10.12
- PyTorch: 2.8.0+cu128


### Bloat
- Characters: 350,412
- Lines: 8,542
- Files: 43
- Tokens (approx): 87,603
- Dependencies (uv.lock lines): 2,004

Run started: 2025-10-16 17:09:12

---

## Tokenizer training
timestamp: 2025-10-16 17:10:34

- max_chars: 2,000,000,000
- doc_cap: 10,000
- vocab_size: 65,536
- train_time: 56.8138
- num_special_tokens: 9
- token_bytes_min: 1
- token_bytes_max: 32
- token_bytes_mean: 6.9197
- token_bytes_std: 2.8748


## Tokenizer evaluation
timestamp: 2025-10-16 17:10:41

### Comparison with GPT-2

| Text Type | Bytes | GPT-2 Tokens | GPT-2 Ratio | Ours Tokens | Ours Ratio | Relative Diff % |
|-----------|-------|--------------|--------------|-------------|------------|-----------------|
| news | 1819 | 404 | 4.50 | 375 | 4.85 | +7.2% |
| korean | 893 | 745 | 1.20 | 712 | 1.25 | +4.4% |
| code | 1259 | 576 | 2.19 | 492 | 2.56 | +14.6% |
| math | 1834 | 936 | 1.96 | 966 | 1.90 | -3.2% |
| science | 1112 | 260 | 4.28 | 228 | 4.88 | +12.3% |
| fwe-train | 4208518 | 900364 | 4.67 | 856883 | 4.91 | +4.8% |
| fwe-val | 4908443 | 1059062 | 4.63 | 1010352 | 4.86 | +4.6% |

### Comparison with GPT-4

| Text Type | Bytes | GPT-4 Tokens | GPT-4 Ratio | Ours Tokens | Ours Ratio | Relative Diff % |
|-----------|-------|--------------|--------------|-------------|------------|-----------------|
| news | 1819 | 387 | 4.70 | 375 | 4.85 | +3.1% |
| korean | 893 | 364 | 2.45 | 712 | 1.25 | -95.6% |
| code | 1259 | 309 | 4.07 | 492 | 2.56 | -59.2% |
| math | 1834 | 832 | 2.20 | 966 | 1.90 | -16.1% |
| science | 1112 | 249 | 4.47 | 228 | 4.88 | +8.4% |
| fwe-train | 4208518 | 874799 | 4.81 | 856883 | 4.91 | +2.0% |
| fwe-val | 4908443 | 1029691 | 4.77 | 1010352 | 4.86 | +1.9% |


## Base model training
timestamp: 2025-10-16 20:16:58

- run: dummy
- depth: 20
- max_seq_len: 2048
- num_iterations: -1
- target_flops: -1.0000
- target_param_data_ratio: 20
- device_batch_size: 32
- total_batch_size: 524,288
- embedding_lr: 0.2000
- unembedding_lr: 0.0040
- weight_decay: 0.0000
- matrix_lr: 0.0200
- grad_clip: 1.0000
- eval_every: 250
- eval_tokens: 10,485,760
- core_metric_every: 2000
- core_metric_max_per_task: 500
- sample_every: 2000
- model_tag: 
- Number of parameters: 560,988,160
- Number of FLOPs per token: 3.491758e+09
- Calculated number of iterations: 21,400
- Number of training tokens: 11,219,763,200
- Tokens : Params ratio: 20.0000
- DDP world size: 8
- warmup_ratio: 0.0000
- warmdown_ratio: 0.2000
- final_lr_frac: 0.0000
- Minimum validation bpb: 0.8118
- Final validation bpb: 0.8118
- CORE metric estimate: 0.2086
- MFU %: 48.65%
- Total training flops: 3.917670e+19
- Total training time: 170.39m
- Peak memory usage: 75422.02MiB


## Base model loss
timestamp: 2025-10-16 20:18:04

- train bpb: 0.8147
- val bpb: 0.8119
- sample 0: <|bos|>The capital of France is Paris. It is the largest city in France and the second largest in Europe.
- sample 1: <|bos|>The chemical symbol of gold is Au. It is a soft, malleable, ductile, and lustrous metal.
- sample 2: <|bos|>If yesterday was Friday, then tomorrow will be Saturday. If you’re like most people, you probably don’t think about it
- sample 3: <|bos|>The opposite of hot is cold. The opposite of cold is hot. The opposite of hot is cold.
- sample 4: <|bos|>The planets of the solar system are: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune,
- sample 5: <|bos|>My favorite color is red. I love red. I love red. I love red. I love
- sample 6: <|bos|>If 5*x + 3 = 13, then x is the solution to the equation.
x = 13
x = 3



## Base model evaluation
timestamp: 2025-10-16 20:22:26

- Model: base_model (step 21400)
- CORE metric: 0.1957
- hellaswag_zeroshot: 0.2622
- jeopardy: 0.1238
- bigbench_qa_wikidata: 0.5231
- arc_easy: 0.5258
- arc_challenge: 0.1320
- copa: 0.3200
- commonsense_qa: 0.1595
- piqa: 0.3896
- openbook_qa: 0.1280
- lambada_openai: 0.3697
- hellaswag: 0.2639
- winograd: 0.2527
- winogrande: 0.0639
- bigbench_dyck_languages: 0.1080
- agi_eval_lsat_ar: 0.0543
- bigbench_cs_algorithms: 0.3742
- bigbench_operators: 0.1333
- bigbench_repeat_copy_logic: 0.0000
- squad: 0.2355
- coqa: 0.1912
- boolq: -0.4816
- bigbench_language_identification: 0.1771


## Midtraining
timestamp: 2025-10-16 20:31:20

- run: dummy
- dtype: bfloat16
- max_seq_len: 2048
- device_batch_size: 32
- unembedding_lr: 0.0040
- embedding_lr: 0.2000
- matrix_lr: 0.0200
- init_lr_frac: 1.0000
- weight_decay: 0.0000
- eval_every: 150
- eval_tokens: 10,485,760
- total_batch_size: 524,288
- dry_run: 0
- Number of iterations: 765
- DDP world size: 8
- Minimum validation bpb: 0.3954


## Chat evaluation mid
timestamp: 2025-10-16 20:38:25

- source: mid
- task_name: None
- dtype: bfloat16
- temperature: 0.0000
- max_new_tokens: 512
- num_samples: 1
- top_k: 50
- batch_size: 8
- model_tag: None
- step: None
- max_problems: None
- ARC-Easy: 0.3838
- ARC-Challenge: 0.2952
- MMLU: 0.3165
- GSM8K: 0.0417
- HumanEval: 0.0854
- ChatCORE metric: 0.0909


## Chat SFT
timestamp: 2025-10-16 20:41:11

- run: dummy
- source: mid
- dtype: bfloat16
- device_batch_size: 4
- num_epochs: 1
- max_iterations: -1
- target_examples_per_step: 32
- unembedding_lr: 0.0040
- embedding_lr: 0.2000
- matrix_lr: 0.0200
- weight_decay: 0.0000
- init_lr_frac: 0.0200
- eval_every: 100
- eval_steps: 100
- eval_metrics_every: 200
- Training rows: 20,843
- Number of iterations: 651
- Training loss: 1.1076
- Validation loss: 1.0118


## Chat evaluation sft
timestamp: 2025-10-16 20:47:57

- source: sft
- task_name: None
- dtype: bfloat16
- temperature: 0.0000
- max_new_tokens: 512
- num_samples: 1
- top_k: 50
- batch_size: 8
- model_tag: None
- step: None
- max_problems: None
- ARC-Easy: 0.3969
- ARC-Challenge: 0.3029
- MMLU: 0.3159
- GSM8K: 0.0531
- HumanEval: 0.0854
- ChatCORE metric: 0.0985


## Summary

- Characters: 350,412
- Lines: 8,542
- Files: 43
- Tokens (approx): 87,603
- Dependencies (uv.lock lines): 2,004

| Metric          | BASE     | MID      | SFT      | RL       |
|-----------------|----------|----------|----------|----------|
| CORE            | 0.1957   | -        | -        | -        |
| ARC-Challenge   | -        | 0.2952   | 0.3029   | -        |
| ARC-Easy        | -        | 0.3838   | 0.3969   | -        |
| GSM8K           | -        | 0.0417   | 0.0531   | -        |
| HumanEval       | -        | 0.0854   | 0.0854   | -        |
| MMLU            | -        | 0.3165   | 0.3159   | -        |
| ChatCORE        | -        | 0.0909   | 0.0985   | -        |

Total wall clock time: 3h38m
