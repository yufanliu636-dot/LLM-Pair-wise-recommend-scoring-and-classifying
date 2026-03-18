# LLM-Pair-wise-recommend-scoring-and-classifying
## Reward Modeling

To mitigate length-based reward hacking in reinforcement learning, we explored common techniques such as reward clipping and length penalty. However, instead of relying solely on these heuristics, we address the problem at its root by removing spurious length signals from the reward.

We propose a **two-head reward model**:

- **Preference head**: learns to score response quality based on human preferences  
- **Length head**: captures length-related signals

During reinforcement learning, we **discard the length head** and only use the preference head. This ensures that the model is optimized purely for content quality, rather than response length.

## Get started
Package installing:
```bash
pip install requirements.txt
```

## Run LoRA
```bash
python train_lora.py --config config.yaml
```

## After LoRA Fine-tuning ,we merge weights:
```bash
python merge_lora.py \
    --base_path gemma-2-2b-it \
    --lora_path  my \
    --save_path meged_gemma-2-2b-it \
    --device cuda:0
```



## Loss Function

We adopt a pairwise preference learning objective with an explicit correction for length bias.

Given an input prompt \( x \), a chosen response \( y_c \), and a rejected response \( y_r \),  
the reward model produces scalar scores:

$$
r_c = R(x, y_c), \quad r_r = R(x, y_r)
$$

### Length-adjusted Preference

To remove spurious length signals, we define an adjusted reward difference:

$$
\tilde{r} = (r_c - r_r) - \alpha (|y_c| - |y_r|)
$$

where:
- \( |y| \) denotes the response length (number of tokens)
- \( \alpha \) controls the strength of length bias correction

### Training Objective

We optimize the standard Bradley-Terry objective:

$$
\mathcal{L}_{BT} = -\log \sigma(\tilde{r})
$$

To stabilize training, we further apply L2 regularization on reward magnitudes:

$$
\mathcal{L}_{reg} = \lambda (r_c^2 + r_r^2)
$$

The final loss is:

$$
\mathcal{L} = \mathcal{L}_{BT} + \mathcal{L}_{reg}
$$

### Intuition

Standard reward models often exploit response length as a shortcut signal,  
leading to a bias toward longer outputs.

Our formulation explicitly subtracts a length-dependent term from the reward difference,  
making the preference objective invariant to response length.

This encourages the model to focus on **content quality rather than verbosity**.


## Training Setup

We fine-tune the Gemma-2-2B model using LoRA on a high-quality dataset of 40K+ pairwise samples,  
each consisting of a prompt, two candidate responses (answer A / B), and a preference label.

- Model: Gemma-2-2B
- Method: LoRA fine-tuning
- Data: 40K+ preference pairs (prompt, answer A, answer B, label)
- Warmup ratio: 0.1
- Learning rate: 1e-5 with decay

## Model Evaluation

To ensure **objective and unbiased scoring**, we adopt a position-agnostic evaluation strategy:

- For each prompt, we score the responses in **both orders**: (Prompt + Answer A + Answer B) and (Prompt + Answer B + Answer A).  
- For each response, we **average the two scores** to produce the final evaluation.  

This approach eliminates positional bias and ensures fair comparison between answers.

## Results

Our approach significantly improves the model's ability to judge pairwise response quality.

- Arena55k accuracy: **53.78% → 88.77%**

This demonstrates a substantial gain in preference modeling performance.
The results highlight the effectiveness of our training pipeline in improving reward modeling accuracy with minimal parameter updates.


## NEXT：Handling "Tie" Cases with Reinforcement Learning

We extend our evaluation to pairwise datasets that include **"tie" labels**,  
where two responses are considered equally good.

To handle these ambiguous cases, we integrate **reinforcement learning** into the training pipeline:

- The reward model is trained to give precise scores even when responses are tied.  
- During RLHF, the language model is optimized to **maximize the calibrated reward**, not just win/lose outcomes.

This approach improves the reward model’s robustness and ensures fairer scoring,  
helping it act as a **competent judge** in pairwise comparisons, including subtle tie scenarios.
