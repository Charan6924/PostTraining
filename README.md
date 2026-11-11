# LLM Post-Training: DPO, GRPO, and PPO

Learn and compare three post-training algorithms on math reasoning using **Qwen3-1.7B** and a **single NVIDIA RTX 4090 (24 GB VRAM)**.

## Experiment scope

- Run **DPO, GRPO, and PPO separately**.
- Initialize every run from the original [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B) checkpoint.
- **No supervised fine-tuning (SFT) stage.**
- Do not initialize one algorithm from another algorithm's output.
- Evaluate the original model as the baseline.
- Full-parameter training versus adapters remains an implementation decision; LoRA is not a requirement of these algorithms. Separate runs avoid holding all experiments in memory together, but each run still has its own memory requirements.

## Task and datasets

The task is to solve arithmetic word problems and produce an automatically checkable final answer.

| Purpose | Dataset and split | Examples | Usage |
| --- | --- | ---: | --- |
| Training | `openai/gsm8k`, configuration `main`, split `train` | 6,973 | Training prompts and answer keys |
| Validation | Held out from the same GSM8K training split | 500 | Hyperparameter tuning and checkpoint selection |
| Primary benchmark | `openai/gsm8k`, configuration `main`, split `test` | 1,319 | Final in-domain evaluation |
| Transfer benchmark | `HuggingFaceH4/MATH-500`, split `test` | 500 | Final evaluation on harder math problems |

GSM8K has 7,473 original training examples. Make the 6,973/500 split once using a fixed seed, save the example identifiers, and reuse it across all runs. Never use benchmark questions or answers for training, preference generation, or checkpoint selection.

Sources: [GSM8K](https://huggingface.co/datasets/openai/gsm8k), [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500).

## Algorithms

### DPO — Direct Preference Optimization

Train on preference pairs with a prompt, a chosen response, and a rejected response.

1. Sample multiple candidate responses from the original Qwen3-1.7B for each training question.
2. Extract and check each candidate's final answer against the answer key.
3. Choose a correct candidate as `chosen` and an incorrect candidate as `rejected`.
4. Keep only questions for which both kinds of candidate exist.
5. Train DPO on the resulting fixed preference dataset.

No SFT stage or separate learned reward model is needed. The usable prompt count will be smaller than or equal to 6,973; record it alongside candidate sampling settings and generation cost. Correctness labels verify final answers, not every step in the response.

Reference: [TRL DPO documentation](https://huggingface.co/docs/trl/dpo_trainer).

### GRPO — Group Relative Policy Optimization

Generate groups of responses during training and compare their rewards within each group.

- Input: training questions and their answer keys.
- Initial group size to try: four responses per prompt, subject to memory profiling.
- Reward: **1 for a correct final answer; 0 otherwise**.
- Missing or unparseable final answers receive 0.
- No separate learned reward model or value model is required.

Groups with identical rewards provide no within-group correctness advantage; track how often this happens.

Reference: [TRL GRPO documentation](https://huggingface.co/docs/trl/grpo_trainer).

### PPO — Proximal Policy Optimization

Generate responses during training and optimize against the same correctness reward used for GRPO.

- Input: training questions and their answer keys.
- Reward: **1 for a correct final answer; 0 otherwise**.
- Missing or unparseable final answers receive 0.
- Train a value estimator to support advantage estimation.
- Use the answer checker directly; a separate learned reward model is unnecessary for this experiment.

Account for the policy, value estimator, reference-policy configuration, optimizer states, activations, and generation caches when profiling memory. Record any KL regularization separately from the correctness reward.

Reference: [TRL PPO documentation](https://huggingface.co/docs/trl/ppo_trainer). PPO APIs are version-sensitive; select a supported implementation and pin its dependencies.

## Evaluation

Evaluate these four checkpoints:

1. Original Qwen3-1.7B baseline.
2. Qwen3-1.7B after DPO.
3. Qwen3-1.7B after GRPO.
4. Qwen3-1.7B after PPO.

**Primary metric:** final-answer accuracy on GSM8K test, using one response per question.

**Transfer metric:** final-answer accuracy on MATH-500, also using one response per question. Use mathematical equivalence checking for MATH-500 rather than raw string matching.

For every checkpoint, keep the evaluation prompt template, thinking mode, decoding settings, token budget, answer extraction, and scoring implementation identical. Select and record these settings before comparing results; report truncation rates so a short token budget does not silently distort the comparison.

Also report:

- Total GPU time, including DPO preference generation.
- Peak GPU memory.
- Number of training prompts and responses processed.
- Total generated tokens and average response length.
- Invalid-answer and truncated-response rates.
- Training seed and, when feasible, variation across multiple seeds.

The runs share the same source question pool, but DPO filters prompts and the algorithms consume different amounts of generation compute. Report these differences; equal epoch counts do not imply equal compute.

Public benchmarks may overlap with the model's pretraining or earlier post-training data. Interpret results as improvement over the original checkpoint, not proof of entirely unseen reasoning ability.

## Hardware and implementation

- GPU: one RTX 4090 with 24 GB VRAM; run one experiment at a time.
- Suggested stack: PyTorch, Transformers, Datasets, and a compatible TRL version; PEFT if adapters are selected.
- Pin exact model and dataset revisions, package versions, and run configurations.
- Profile each algorithm before choosing full-parameter training, optimizer/offloading options, sequence lengths, and batch sizes. Full-parameter training is not assumed to fit with a conventional optimizer.
- Use a consistent Qwen3 thinking-mode setting. Non-thinking mode is a practical starting choice for shorter generations; decide and record the setting before running experiments.

## Results

No training or evaluation has been run yet.

| Checkpoint | GSM8K accuracy | MATH-500 accuracy | GPU hours including data generation | Peak VRAM |
| --- | --- | --- | --- | --- |
| Original Qwen3-1.7B | TBD | TBD | TBD | TBD |
| DPO | TBD | TBD | TBD | TBD |
| GRPO | TBD | TBD | TBD | TBD |
| PPO | TBD | TBD | TBD | TBD |
