# LLMReasonBench

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Benchmark and enhance reasoning capabilities in large language models (LLMs)

## 📖 Overview

LLMReasonBench is an evaluation and training framework that measures and improves how well LLMs separate **memory recall** from **logical reasoning**. Based on the methodology from *[Disentangling Memory and Reasoning Ability in Large Language Models](https://github.com/MingyuJ666/Disentangling-Memory-and-Reasoning)*, this tool uses customizable special tokens to explicitly isolate these cognitive processes during inference.

In addition to standard static reasoning tasks, LLMReasonBench supports **interactive benchmarks** such as ALFWorld and ARC‑AGI‑3, where an agent must perform multi‑turn actions in a simulated environment.

## 🚀 Quick Start

### Installation

Clone the repository and run the installation script:

```bash
git clone https://github.com/metalearningnet/LLMReasonBench.git
cd LLMReasonBench
./install.sh
```

### Generate Chain‑of‑Thought (CoT) Data (Static Datasets)

Before training, you must generate structured CoT steps for your dataset. The generator uses an LLM (vLLM for local inference or OpenAI API) to create steps annotated with the special tokens defined in `COT_TOKENS` (by default, `<memory>` for factual extraction and `<reason>` for logical operations).

> **💡 Teacher‑LLM generation control**  
>  
> Whether the teacher LLM is invoked to produce CoT steps depends on the `parameter_efficient_mode` setting in `conf/settings.yaml`.  
>  
> - **With CoT reasoning** (teacher generates structured steps): set the mode to `lora-cog-frozen` or `lora-cog-tuned`.  
> - **Without CoT reasoning** (fast dataset preparation): set the mode to any other value, e.g., `none` or `lora`. The generator will simply format the questions and answers, saving a plain dataset suitable for standard fine‑tuning or evaluation.  
>  
> The generation command itself runs identically in both cases.

```bash
# Generate CoT data for TruthfulQA training set (using vLLM backend by default)
./run.sh --generate --dataset truthfulqa --mode train
```

If the teacher LLM should produce reasoning steps, make sure `parameter_efficient_mode` is `lora-cog-frozen` or `lora-cog-tuned` *before* running the command.  
If the mode is `none` or `lora`, the generator skips the teacher entirely and produces a clean question‑answer dataset with empty CoT fields – perfect for baselines or initial data inspection.

> **Note:** Generation requires a valid LLM backend. The default is `vllm` with teacher model `Qwen/Qwen3.5-27B`. Configure paths or API keys in `conf/settings.yaml`.

### Generate Expert Trajectories (Interactive Datasets)

For interactive benchmarks like **ALFWorld** and **ARC‑AGI‑3**, the generation process uses the teacher model (or an optional scripted fallback) to interact with the environment and record full action‑observation trajectories. The generator employs enhanced prompting strategies, including goal‑locking, systematic surface search, and task‑specific guidance, to ensure high‑quality trajectory generation.

```bash
# Generate expert trajectories for ALFWorld (TextWorld)
./run.sh --generate --dataset alfworld

# Generate expert trajectories for ARC‑AGI‑3
./run.sh --generate --dataset arc3
```

Interactive generation parameters are defined in `conf/datasets.yaml` (e.g., `task_source`, `max_steps`, `output_format`, `filter_success`, `game_ids`). The ALFWorld generator automatically parses the goal to extract target objects and receptacles, adapting to various task templates. The ARC‑3 generator interfaces with the `arc-agi` Python package to run puzzle environments in terminal‑rendered mode.

### Train a Model

After generating CoT data (or expert trajectories), you can fine‑tune a base model (configured via `common.model` in `conf/settings.yaml`):

```bash
# Standard supervised fine‑tuning for single‑turn datasets
./run.sh --train --dataset truthfulqa

# Multi‑turn conversational fine‑tuning (e.g., ALFWorld, ARC‑3)
./run.sh --train --dataset alfworld --mode multiturn
./run.sh --train --dataset arc3 --mode multiturn

# Reinforcement learning (DPO example)
./run.sh --train --rl dpo --dataset truthfulqa
```

**Multi‑turn training** uses the `--mode multiturn` flag to format conversation histories with the tokenizer's chat template and mask loss only on assistant turns. This mode works with datasets generated with `output_format: "messages"` (like ALFWorld and ARC‑3).

Supported RL methods: `dpo`, `cpo`, `kto`, `orpo`.  
RL configuration files (e.g., `conf/dpo.yaml`) control hyperparameters.

### Evaluate a Model

Evaluate any fine‑tuned model, adapter, or base model:

```bash
# Single‑turn dataset evaluation
./run.sh --eval --model /path/to/checkpoint --dataset truthfulqa

# Multi‑turn dataset – offline evaluation (compare final action)
./run.sh --eval --model /path/to/checkpoint --dataset alfworld
./run.sh --eval --model /path/to/checkpoint --dataset arc3

# Multi‑turn dataset – interactive evaluation (run in environment)
./run.sh --eval --model /path/to/checkpoint --dataset alfworld --interactive
./run.sh --eval --model /path/to/checkpoint --dataset arc3 --interactive
```

For multi‑turn datasets, the evaluator automatically detects the format.
- Without `--interactive`, it performs **offline evaluation** by comparing the generated final action with the recorded expert action.
- With `--interactive`, it launches the actual environment (ALFWorld or ARC‑3) and measures **task success rate**.

## 📊 Model Evaluation

The evaluation script supports:
- Standard Hugging Face models
- PEFT adapters
- vLLM accelerated inference
- OpenAI API models (with `--backend api`)

```bash
# Evaluate with vLLM
./run.sh --eval --model meta-llama/Llama-3.2-3B --dataset gsm8k --backend vllm

# Evaluate using OpenAI GPT-4o
./run.sh --eval --model gpt-4o --dataset mmlupro --backend api
```

## 🗂️ Dataset Management

### Supported Datasets

The following datasets are pre‑configured and ready for immediate use:

| Dataset | Config Key | Answer Type | Type | Hugging Face / Source |
|---------|------------|-------------|------|------------------------|
| **GSM8K** | `gsm8k` | Numeric | Static | `openai/gsm8k` |
| **AIME24** | `aime24` | Numeric | Static | `HuggingFaceH4/aime_2024` |
| **AIME25** | `aime25` | Numeric | Static | `math-ai/aime25` |
| **AQUA-RAT** | `aqua` | Multiple Choice | Static | `deepmind/aqua_rat` |
| **MMLU-Pro** | `mmlupro` | Multiple Choice | Static | `TIGER-Lab/MMLU-Pro` |
| **TruthfulQA** | `truthfulqa` | Multiple Choice | Static | `truthfulqa/truthful_qa` |
| **StrategyQA** | `strategyqa` | Boolean | Static | `ChilleD/StrategyQA` |
| **MetaMathQA** | `metamathqa` | Numeric | Static | `meta-math/MetaMathQA` |
| **CommonsenseQA** | `commonsenseqa` | Multiple Choice | Static | `tau/commonsense_qa` |
| **ALFWorld** | `alfworld` | Action Sequence | Interactive | ALFWorld TextWorld |
| **ARC-AGI-3** | `arc3` | Action Sequence | Interactive | ARC Prize / `arc-agi` package |

### Direct Dataset Download

You can quickly download and prepare **any** Hugging Face dataset without running the full generation pipeline. This is useful for datasets that already contain CoT‑style reasoning or for test sets.

```bash
./install.sh --dataset <hf_dataset_id> --name <output_name>
```

**Example:**
```bash
./install.sh --dataset metalearningnet/qwen3.5-metamathqa-cot --name metamathqa
```

By default, this downloads the **train** split and saves it as `data/<output_name>_train.json`. To download a different split, use `--split`. The dataset will be saved to `data/<output_name>_<split>.json`.  
> **Important:** You still need to register the dataset in `conf/datasets.yaml` and create a Python dataset class (see below).

### Adding Custom Datasets

#### Static Datasets (Single‑Turn)

To add a new static dataset, follow these three steps:

1. **Update `conf/datasets.yaml`**

   ```yaml
   my_dataset:
     source: "my-org/my-dataset"
     split_mapping:
       train: "train"
       test: "test"
     answer_type: "multiple_choice"   # one of: multiple_choice, numeric, boolean, action_sequence
     field_mapping:
       question: "question_field"
       answer: "answer_field"
       options: "choices_field"       # optional, for multiple choice
     clean_latex: false               # optional, strip LaTeX formatting
   ```

2. **Create a Dataset Class** (`src/dataset/my_dataset.py`):

   ```python
   from preprocess import JsonDataset
   from generator import DatasetGenerator

   class MyDatasetGenerator(DatasetGenerator):
       """Optional custom generator logic."""
       pass

   class MyDataset(JsonDataset):
       INSTRUCTION = "Solve the following problem step by step."
   ```

3. **Register in `src/dataset/__init__.py`**:

   ```python
   from .my_dataset import MyDatasetGenerator, MyDataset

   GENERATOR_MAP['my_dataset'] = MyDatasetGenerator
   DATASET_MAP['my_dataset'] = MyDataset
   ```

#### Interactive Datasets (Multi‑Turn)

For interactive benchmarks (e.g., ALFWorld, ARC‑3), use the `InteractiveGenerator` base class and create a corresponding `TrajectoryDataset` subclass.

1. **Configure `conf/datasets.yaml`** with `interactive: true`:

   ```yaml
   my_interactive:
     name: "My Interactive Dataset"
     answer_type: "action_sequence"
     interactive: true
     output_format: "messages"              # "messages" (chat-style) or "trajectory"
     filter_success: true                   # Keep only successful episodes
     max_steps: 50
     # Environment‑specific parameters:
     # game_ids: ["ls20", "ft09"]           # For ARC-3
     # task_source: "pick_and_place_simple" # For ALFWorld
     # use_butler_fallback: true            # ALFWorld fallback policy
   ```

2. **Create Generator and Dataset Classes** (`src/dataset/my_interactive_dataset.py`):

   ```python
   from typing import Any, Dict, List
   from preprocess import TrajectoryDataset
   from generator import InteractiveGenerator

   class MyInteractiveGenerator(InteractiveGenerator):
       def setup_environment(self, task: Dict[str, Any], split: str = "train") -> Any:
           # Initialize environment (e.g., arc-agi)
           pass

       def load_builtin_tasks(self) -> List[Dict[str, Any]]:
           # Return list of task dicts (with 'id', 'goal'/'instruction', etc.)
           pass

   class MyInteractiveDataset(TrajectoryDataset):
       INSTRUCTION = "Your task instruction here."
   ```

3. **Register in `src/dataset/__init__.py`**:

   ```python
   from .my_interactive_dataset import MyInteractiveGenerator, MyInteractiveDataset

   GENERATOR_MAP['my_interactive'] = MyInteractiveGenerator
   DATASET_MAP['my_interactive'] = MyInteractiveDataset
   ```

Interactive generation automatically leverages the LLM client defined in `conf/settings.yaml` (`generator` section). Reference implementations are available in `src/dataset/alfworld.py` and `src/dataset/arc3.py`.

## ⚙️ Configuration

All configuration files reside in the `conf/` directory.

### Global Settings

`conf/settings.yaml` controls:
- Base model (`common.model`)
- LoRA configuration (`common.lora_config`)
- Training hyperparameters (`train` section)
- Generation backend and teacher model (`generator` section)

> **Important for CoT generation:** When you want the teacher LLM to produce CoT steps, set `common.parameter_efficient_mode` to `lora-cog-frozen` or `lora-cog-tuned`. If you choose any other mode (e.g., `none` or `lora`), the generator will produce a plain dataset without reasoning chains – an efficient way to pre‑process data for standard supervised fine‑tuning.

Example snippet (showing key defaults):

```yaml
common:
  model: "Qwen/Qwen3.5-4B"
  # Choose a mode:
  #   lora-cog-frozen / lora-cog-tuned  → teacher LLM generates CoT steps
  #   none / lora                       → plain dataset, no teacher LLM
  parameter_efficient_mode: "lora-cog-tuned"
  lora_config:
    r: 16
    alpha: 16
    use_rslora: true

train:
  learning_rate: 0.00001
  num_train_epochs: 1
  max_prompt_length: 512
  max_completion_length: 2048

generator:
  backend: "vllm"                     # 'vllm' or 'api'
  temperature: 0.3
  vllm:
    model: "Qwen/Qwen3.5-27B"         # Teacher model for CoT generation
    tensor_parallel_size: 1
    gpu_memory_utilization: 0.8
  api:                                # Used when backend='api'
    api_key: ""
    api_base: "https://api.openai.com/v1"
    model: "gpt-4o"
```

### Dataset Configuration

`conf/datasets.yaml` defines each dataset’s properties. For static datasets, the main fields are:

| Field | Description |
|-------|-------------|
| `source` | Hugging Face dataset identifier or local path |
| `config_name` | (Optional) Dataset configuration name |
| `split_mapping` | Maps `train`/`test` to actual split names |
| `answer_type` | `boolean`, `numeric`, or `multiple_choice` |
| `valid_answers` | (Multiple choice) Allowed letters, e.g., `["A","B","C"]` |
| `field_mapping` | Maps dataset fields to `question`, `answer`, `options` |
| `clean_latex` | Remove LaTeX formatting (e.g., `\frac{1}{2}` → `1/2`) |

For interactive datasets, you must set `interactive: true` and provide environment‑specific parameters. Common fields include:

| Field | Description |
|-------|-------------|
| `task_source` | (ALFWorld) Built‑in task type or path to JSON file |
| `game_ids` | (ARC-3) List of game IDs to use (e.g., `["ls20", "ft09"]`) |
| `max_steps` | Maximum steps per episode |
| `output_format` | `"messages"` (chat-style) or `"trajectory"` (step-by-step) |
| `filter_success` | Keep only successful episodes |
| `use_butler_fallback` | (ALFWorld) Enable scripted fallback policy |

### Special Token Configuration

`conf/tokens.py` defines the CoT tokens and their expected behavior. The following configuration is used throughout LLMReasonBench:

```python
COT_TOKENS = {
    'memory': {
        'description': 'Extract and state ONLY the given facts, numbers, or formulas from the problem statement.',
        'prerequisite': True # Generate these steps first and use as context for the rest
    },
    'reason': {
        'description': 'Perform calculations and logical operations using the facts from the prerequisite memory steps to derive conclusions or solve the problem.'
    }
}

END_MARK = True # True: <token> content </token> | False: <token>: content
```

- The **`prerequisite`** flag indicates that steps of that token type are generated **before** the main reasoning and then used as context (like memory) for the remaining steps.
- All token types without `prerequisite: True` are generated in the second stage, after the prerequisite steps are available.

For interactive datasets like ALFWorld and ARC‑3, the injection of these CoT tokens into the prompts is controlled by the `parameter_efficient_mode` setting. When the mode is set to `lora-cog-frozen` or `lora-cog-tuned`, the generator will explicitly instruct the model to structure its reasoning using the configured tags.

### Chain‑of‑Thought (CoT) Format

LLMReasonBench supports two output formats controlled by the `END_MARK` flag in `conf/tokens.py`:

- **Colon format** (`END_MARK = False`):
   ```
   <memory>: The problem states that x = 5.
   <reason>: Using x = 5, calculate 2*x = 10.
   ```

- **Closing‑tag format** (`END_MARK = True`):
   ```
   <memory> The problem states that x = 5. </memory>
   <reason> Using x = 5, calculate 2*x = 10. </reason>
   ```

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](https://github.com/metalearningnet/LLMReasonBench/blob/main/LICENSE) file for details.
