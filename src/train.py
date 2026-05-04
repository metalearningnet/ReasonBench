import sys
import yaml
import torch
import shutil
import transformers
from pathlib import Path
from training import LMTrainer
from dataset import DATASET_MAP
from preprocess import DataConfig
from huggingface_hub import login
from model_loader import CausalLM
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from peft_model import PeftModelForCausalLMWrapper
from multiturn_dataset import make_multiturn_data_module
from supervised_dataset import make_supervised_data_module
from fixed_length_dataset import make_fixed_length_data_module
from transformers.utils import logging as transformers_logging
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from config import (
    COT_TOKEN_NAMES, END_MARK, DEFAULT_TRAIN_OUTPUT_DIR, DEFAULT_CHECKPOINT_DIR,
    load_config, load_datasets_config, update_dataclass_from_config,
    setup_directories, logger, dataset_names
)

if END_MARK:
    COT_TOKEN_LIST = [f"<{i}>" for i in COT_TOKEN_NAMES] + [f"</{i}>" for i in COT_TOKEN_NAMES]
else:
    COT_TOKEN_LIST = [f"<{i}>" for i in COT_TOKEN_NAMES]

transformers_logging.set_verbosity_info()
transformers_logging.enable_explicit_format()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f"Using device: {device}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"CUDA device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        logger.info(f"  Device {i}: {torch.cuda.get_device_name(i)}")

@dataclass
class ModelArguments:
    random_initialize: Optional[bool] = field(default=False)
    model: Optional[str] = field(
        default=None,
        metadata={"help": "Pre-trained model name on Huggingface or checkpoint path."}
    )
    hf_hub_token: Optional[str] = field(default="none")
    parameter_efficient_mode: Optional[str] = field(
        default="none",
        metadata={"choices": ["none", "lora", "lora-cog-frozen", "lora-cog-tuned"]}
    )
    use_calculator: Optional[bool] = field(default=False)
    lora_module: Optional[str] = field(default="mlp")
    lora_r: Optional[int] = field(default=16)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_bias: Optional[str] = field(
        default="none",
        metadata={"choices": ["none", "all", "lora_only"]}
    )
    lora_inference_mode: Optional[bool] = field(default=False)
    lora_target_modules: Optional[List[str]] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    lora_init_lora_weights: Optional[bool] = field(default=True)
    lora_fan_in_fan_out: Optional[bool] = field(default=False)

@dataclass
class DataArguments:
    dataset: str = field(
        default=None,
        metadata={"choices": dataset_names}
    )
    mode: str = field(
        default="supervised",
        metadata={"choices": ["supervised", "fixed_length", "multiturn"]}
    )
    num_train: Optional[int] = field(default=None)
    num_test: Optional[int] = field(default=None)
    embedding_model_name: Optional[str] = field(default=None)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    int8_training: Optional[bool] = field(default=False)
    load_in_8bit: Optional[bool] = field(default=False)
    load_in_4bit: Optional[bool] = field(default=False)
    num_train_epochs: Optional[int] = field(default=1)
    learning_rate: Optional[float] = field(default=1e-5)
    warmup_steps: Optional[int] = field(default=100)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    logging_steps: Optional[int] = field(default=10)
    eval_steps: Optional[int] = field(default=50)
    save_steps: Optional[int] = field(default=100)
    save_total_limit: Optional[int] = field(default=1)
    fp16: Optional[bool] = field(default=False)
    bf16: Optional[bool] = field(default=False)
    output_dir: str = field(default=str(DEFAULT_TRAIN_OUTPUT_DIR))
    checkpoint_dir: str = field(default=str(DEFAULT_CHECKPOINT_DIR))
    max_prompt_length: Optional[int] = field(default=512)
    max_completion_length: Optional[int] = field(default=1024)
    per_device_train_batch_size: Optional[int] = field(default=8)
    per_device_eval_batch_size: Optional[int] = field(default=8)

def log_trainable_parameters(model: torch.nn.Module):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_param = sum(p.numel() for p in model.parameters())
    trainable_percent = 100 * trainable_params / all_param if all_param > 0 else 0
    
    logger.info(
        f"Trainable parameters: {trainable_params:,} || "
        f"All parameters: {all_param:,} || "
        f"Trainable: {trainable_percent:.2f}%"
    )
    
    return trainable_params, all_param, trainable_percent

def enable_cog_tuning(model: torch.nn.Module):
    input_emb = model.get_input_embeddings()
    output_emb = model.get_output_embeddings()
    
    if hasattr(input_emb, 'new_embedding') and input_emb.new_embedding is not None:
        input_emb.new_embedding.weight.requires_grad = True
        logger.info("Enabled prompt tuning on input embeddings")
    else:
        logger.warning("Input embeddings lack 'new_embedding' – prompt tuning will not affect new tokens (none exist).")
    
    if hasattr(output_emb, 'new_linear') and output_emb.new_linear is not None:
        output_emb.new_linear.weight.requires_grad = True
        logger.info("Enabled prompt tuning on output embeddings")
    else:
        logger.warning("Output embeddings lack 'new_linear' – prompt tuning will not affect new tokens (none exist).")

def create_tokenizer(model_name: str, cache_dir: Optional[str]) -> transformers.PreTrainedTokenizer:
    logger.debug(f"Creating tokenizer for model: {model_name}")
    
    if 'llama' in model_name.lower() or 'alpaca' in model_name.lower():
        tokenizer_class = transformers.LlamaTokenizer
        logger.debug("Using LlamaTokenizer")
    else:
        tokenizer_class = transformers.AutoTokenizer
        logger.debug(f"Using AutoTokenizer for {model_name}")
    
    try:
        tokenizer = tokenizer_class.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        logger.info(f"Tokenizer loaded successfully: {tokenizer_class.__name__}")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise
    
    special_tokens = {}
    if tokenizer.eos_token is None:
        special_tokens["eos_token"] = '</s>'
        logger.debug("Added eos_token")
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token
        logger.debug("Set bos_token to eos_token")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.debug("Set pad_token to eos_token")
    
    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
    
    tokenizer.padding_side = "left"
    logger.debug(f"Set padding_side to: {tokenizer.padding_side}")
    
    return tokenizer

def create_cot_tokens(
    config: dict,
    tokenizer: transformers.PreTrainedTokenizer
) -> tuple[Dict[str, str], List[int], List[int]]:
    if 'tag' not in config['common']['parameter_efficient_mode']:
        return []
    
    logger.info(f"Defining CoT tokens: {COT_TOKEN_LIST}")
    
    before_size = len(tokenizer)
    logger.info(f"Tokenizer vocabulary before adding CoT tokens: {before_size}")

    num_new = tokenizer.add_tokens(COT_TOKEN_LIST)
    cot_token_ids = tokenizer.convert_tokens_to_ids(COT_TOKEN_LIST)

    after_size = len(tokenizer)
    logger.info(f"Tokenizer vocabulary after adding CoT tokens: {after_size}")
    logger.debug(f"Added {num_new} new token(s) to tokenizer")
    
    for token in COT_TOKEN_LIST:
        token_id = tokenizer.convert_tokens_to_ids(token)
        logger.debug(f"Token '{token}' -> ID {token_id}")
    
    return cot_token_ids

def get_data_class(dataset_name: str):
    if dataset_name not in DATASET_MAP:
        error_msg = f"Dataset {dataset_name} not implemented. Available: {list(DATASET_MAP.keys())}"
        logger.error(error_msg)
        raise NotImplementedError(error_msg)
    
    logger.debug(f"Selected dataset class: {DATASET_MAP[dataset_name].__name__}")
    return DATASET_MAP[dataset_name]

def create_peft_config(model_args: ModelArguments, model_name: str) -> Optional[LoraConfig]:
    if 'lora' not in model_args.parameter_efficient_mode:
        logger.debug("No LoRA configuration needed")
        return None
    
    logger.info(f"Creating LoRA configuration for model: {model_name}")
    
    target_modules = model_args.lora_target_modules
    
    if not target_modules:
        if any(x in model_name.lower() for x in ['llama', 'alpaca', 'qwen']):
            target_modules = []
            if model_args.lora_module == 'mlp':
                target_modules.extend(["gate_proj", "up_proj", "down_proj"])
                logger.debug("LoRA targeting MLP layers")
            elif model_args.lora_module == 'atten':
                target_modules.extend(["q_proj", "k_proj", "v_proj", "o_proj"])
                logger.debug("LoRA targeting attention layers")
            else:
                target_modules.extend(["q_proj", "v_proj"])
                logger.debug("LoRA targeting q_proj and v_proj (default)")
        elif 'gpt2' in model_name.lower():
            target_modules = ["c_attn", "c_proj"]
            logger.debug("LoRA targeting GPT2 layers")
        else:
            logger.warning(f"Unknown model architecture for {model_name}, using default target modules")
            target_modules = ["q_proj", "v_proj"]
    
    logger.info(f"LoRA target modules: {target_modules}")
    logger.info(f"LoRA configuration: r={model_args.lora_r}, alpha={model_args.lora_alpha}, dropout={model_args.lora_dropout}")
    
    modules_to_save = ["embed_tokens", "lm_head", "wte", "wpe"]

    return LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=model_args.lora_dropout,
        bias=model_args.lora_bias,
        task_type=TaskType.CAUSAL_LM,
        inference_mode=model_args.lora_inference_mode,
        init_lora_weights=model_args.lora_init_lora_weights,
        fan_in_fan_out=model_args.lora_fan_in_fan_out,
        modules_to_save=modules_to_save
    )

def load_model(
    model_args: ModelArguments,
    training_args: TrainingArguments,
    num_new_tokens: int,
    tokenizer: transformers.PreTrainedTokenizer
) -> torch.nn.Module:
    model_path = model_args.model
    
    logger.info(f"Loading model from: {model_path}")
    
    load_kwargs = {
        "n_tokens": num_new_tokens,
        "parameter_efficient_mode": model_args.parameter_efficient_mode,
        "pretrained_model_name_or_path": model_path,
        "cache_dir": training_args.cache_dir
    }
    
    logger.debug(f"Loading model with kwargs: {load_kwargs}")
    
    if training_args.load_in_8bit:
        load_kwargs.update({
            "device_map": "auto",
            "load_in_8bit": True,
            "offload_folder": "offload",
            "offload_state_dict": True
        })
        logger.info("Loading model in 8-bit precision")
    elif training_args.load_in_4bit:
        load_kwargs.update({
            "device_map": "auto",
            "load_in_4bit": True
        })
        logger.info("Loading model in 4-bit precision")
    elif training_args.bf16:
        load_kwargs["dtype"] = torch.bfloat16
        logger.info(f"Loading model with dtype: torch.bfloat16")
    elif training_args.fp16:
        load_kwargs["dtype"] = torch.float32
        logger.info("Loading model in fp32 for mixed-precision training")
    else:
        logger.info("Loading model in default precision (fp32)")
    
    try:
        model = CausalLM.from_pretrained(**load_kwargs)
        input_emb_size = model.get_input_embeddings().weight.size(0)
        tokenizer_vocab_size = len(tokenizer)
        model_vocab_size = model.config.vocab_size
        
        logger.info(f"Model config vocab_size: {model_vocab_size}")
        logger.info(f"Model input embeddings size: {input_emb_size}")
        logger.info(f"Tokenizer vocabulary size: {tokenizer_vocab_size}")
        
        if input_emb_size != tokenizer_vocab_size:
            logger.warning(f"MISMATCH: Model embeddings ({input_emb_size}) != Tokenizer vocab ({tokenizer_vocab_size})")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    if training_args.load_in_8bit or training_args.load_in_4bit:
        logger.info("Preparing model for k-bit training")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing
        )
        logger.debug("Model prepared for k-bit training")
    
    return model

def train_rl_mode(rl_mode: str, data_args: DataArguments, model_args: ModelArguments, training_args: TrainingArguments):
    logger.info(f"Starting RL training mode: {rl_mode}")
    
    base_config = load_config()
    config = base_config.copy()
    config['rl']['training_mode'] = rl_mode
    config['common']['dataset'] = data_args.dataset
    
    if data_args.num_train:
        config['num_train'] = data_args.num_train

    if data_args.num_test:
        config['num_test'] = data_args.num_test
    
    if model_args.model:
        config['common']['model'] = model_args.model
    
    if model_args.parameter_efficient_mode != "none":
        config['common']['parameter_efficient_mode'] = model_args.parameter_efficient_mode
    
    if model_args.lora_r:
        if 'lora_config' not in config['common']:
            config['common']['lora_config'] = {}
        config['common']['lora_config']['r'] = model_args.lora_r
    
    if model_args.lora_alpha:
        if 'lora_config' not in config['common']:
            config['common']['lora_config'] = {}
        config['common']['lora_config']['alpha'] = model_args.lora_alpha
    
    if model_args.lora_target_modules:
        if 'lora_config' not in config['common']:
            config['common']['lora_config'] = {}
        config['common']['lora_config']['target_modules'] = model_args.lora_target_modules
    
    if training_args.output_dir:
        config['train']['output_dir'] = str(training_args.output_dir)
    
    if training_args.learning_rate:
        config['train']['learning_rate'] = training_args.learning_rate
    
    if training_args.num_train_epochs:
        config['train']['num_train_epochs'] = training_args.num_train_epochs
    
    if training_args.fp16:
        config['train']['fp16'] = training_args.fp16
    
    if training_args.bf16:
        config['train']['bf16'] = training_args.bf16
    
    setup_directories(config)
    
    if model_args.hf_hub_token:
        try:
            login(token=model_args.hf_hub_token)
            logger.info("Successfully logged in to Hugging Face Hub")
        except Exception as e:
            logger.warning(f"Failed to login to Hugging Face Hub: {e}")
    
    wandb_run = None
    if config.get('logging', {}).get('wandb_project'):
        try:
            import wandb
            wandb_run = wandb.init(
                project=config['logging']['wandb_project'],
                entity=config['logging'].get('wandb_entity'),
                name=config['experiment'].get('name', f'rl-{rl_mode}-run'),
                tags=config['experiment'].get('tags', []) + ['rl', rl_mode],
                notes=config['experiment'].get('notes', ''),
                config={
                    **config['common'],
                    **config.get('train', {}),
                    **config.get('rl', {}),
                    'training_mode': rl_mode,
                    'parameter_efficient_mode': model_args.parameter_efficient_mode,
                    'model': model_args.model
                }
            )
            logger.info("WandB initialized successfully for RL training")
        except ImportError:
            logger.warning("WandB not installed, skipping initialization")
        except Exception as e:
            logger.warning(f"Failed to initialize WandB: {e}")
    
    try:
        from training import create_rl_trainer
        
        trainer = create_rl_trainer(config)
        
        logger.info(f"Starting {rl_mode.upper()} training...")
        results = trainer.train()
        
        logger.info(f"\nRL training completed successfully!")
        
        if wandb_run:
            wandb.log(results.get("train_metrics", {}))
            logger.info("Results logged to WandB")
        
        return results
    except Exception as e:
        logger.error(f"Error during RL training: {e}", exc_info=True)
        raise
    finally:
        if wandb_run:
            wandb_run.finish()
            logger.info("WandB run finished")

def train():
    user_provided_mode = False
    for arg in sys.argv[1:]:
        if arg == '--mode' or arg.startswith('--mode='):
            user_provided_mode = True
            break
    
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args, remaining_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    config = load_config()
    rl_mode = None

    if "--rl" in remaining_args:
        rl_index = remaining_args.index("--rl")
        if rl_index + 1 < len(remaining_args) and not remaining_args[rl_index + 1].startswith("--"):
            rl_mode = remaining_args[rl_index + 1]
        else:
            rl_mode = config['rl']['training_mode']

    checkpoint_dir = Path(training_args.checkpoint_dir)
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if rl_mode:
        return train_rl_mode(
            rl_mode=rl_mode,
            data_args=data_args,
            model_args=model_args,
            training_args=training_args
        )

    if not data_args.dataset:
        data_args.dataset = config['common']['dataset']

    model_args = update_dataclass_from_config(model_args, config, ['common', 'train'])
    data_args = update_dataclass_from_config(data_args, config, ['common', 'train'])
    training_args = update_dataclass_from_config(training_args, config, ['common', 'train'])

    if 'batch_size' in config['common']:
        training_args.per_device_train_batch_size = config['common']['batch_size']
        training_args.per_device_eval_batch_size = config['common']['batch_size']

    logger.info("Starting training process")
    logger.info(f"Model arguments: {model_args}")
    logger.info(f"Data arguments: {data_args}")
    logger.info(f"Training arguments: {training_args}")

    setup_directories(config)

    if 'lora' in model_args.parameter_efficient_mode:
        logger.info(f"LoRA Configuration loaded:")
        logger.info(f"  - r: {model_args.lora_r}")
        logger.info(f"  - alpha: {model_args.lora_alpha}")
        logger.info(f"  - dropout: {model_args.lora_dropout}")
        logger.info(f"  - bias: {model_args.lora_bias}")
        logger.info(f"  - inference_mode: {model_args.lora_inference_mode}")
        logger.info(f"  - module: {model_args.lora_module}")
        logger.info(f"  - target_modules: {model_args.lora_target_modules}")
        logger.info(f"  - init_lora_weights: {model_args.lora_init_lora_weights}")
        logger.info(f"  - fan_in_fan_out: {model_args.lora_fan_in_fan_out}")

    if model_args.hf_hub_token:
        try:
            login(token=model_args.hf_hub_token)
            logger.info("Successfully logged in to Hugging Face Hub")
        except Exception as e:
            logger.warning(f"Failed to login to Hugging Face Hub: {e}")

    model_max_length = training_args.max_prompt_length + training_args.max_completion_length
    tokenizer = create_tokenizer(model_args.model, training_args.cache_dir)
    tokenizer.model_max_length = model_max_length
    logger.info(f"Tokenizer configured with max length: {model_max_length}")

    cot_tokens = create_cot_tokens(config, tokenizer)
    if cot_tokens:
        logger.info(f"Added {len(cot_tokens)} CoT tokens")

    from transformers import AutoConfig
    orig_config = AutoConfig.from_pretrained(model_args.model)
    if hasattr(orig_config, "vocab_size"):
        orig_vocab_size = orig_config.vocab_size
    else:
        logger.warning(f"'vocab_size' missing from config. Fetching from base tokenizer.")
        temp_tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model, trust_remote_code=True)
        orig_vocab_size = len(temp_tokenizer)
        del temp_tokenizer
    n_tokens_needed = max(len(cot_tokens), len(tokenizer) - orig_vocab_size)
    logger.info(f"Model base vocab size: {orig_vocab_size}, tokenizer final size: {len(tokenizer)} -> need {n_tokens_needed} new tokens")

    try:
        data_class = get_data_class(data_args.dataset)
        logger.debug(f"Using dataset class: {data_class.__name__}")
    except NotImplementedError as e:
        logger.error(f"Dataset error: {e}")
        return

    data_config = DataConfig()
    datasets_config = load_datasets_config()
    data_config.dataset = datasets_config[data_args.dataset]

    if not user_provided_mode:
        dataset_cfg = datasets_config.get(data_args.dataset, {})
        if dataset_cfg.get('interactive', False) and dataset_cfg.get('output_format') == 'messages':
            logger.info(f"Auto‑inferring mode='multiturn' from dataset configuration (interactive with messages)")
            data_args.mode = "multiturn"
        elif dataset_cfg.get('fixed_length', False):
            logger.info(f"Auto‑inferring mode='fixed_length' from dataset configuration")
            data_args.mode = "fixed_length"

    try:
        train_dataset = data_class(data_args.dataset, "train", config=data_config)
        logger.info(f"Successfully loaded training dataset {data_args.dataset} with {len(train_dataset)} examples")
        if data_args.num_train is not None:
            limit = min(data_args.num_train, len(train_dataset))
            if hasattr(train_dataset, 'x') and hasattr(train_dataset, 'y'):
                train_dataset.x = train_dataset.x[:limit]
                train_dataset.y = train_dataset.y[:limit]
            if hasattr(train_dataset, 'data'):
                train_dataset.data = train_dataset.data[:limit]
            logger.info(f"Limited training dataset to {limit} examples (num_train={data_args.num_train})")
    except Exception as e:
        logger.error(f"Failed to load training dataset: {e}", exc_info=True)
        return

    try:
        eval_dataset = data_class(data_args.dataset, "test", config=data_config)
        logger.info(f"Evaluation dataset loaded successfully: {len(eval_dataset)} examples")
        if data_args.num_test is not None:
            limit = min(data_args.num_test, len(eval_dataset))
            if hasattr(eval_dataset, 'x') and hasattr(eval_dataset, 'y'):
                eval_dataset.x = eval_dataset.x[:limit]
                eval_dataset.y = eval_dataset.y[:limit]
            if hasattr(eval_dataset, 'data'):
                eval_dataset.data = eval_dataset.data[:limit]
            logger.info(f"Limited evaluation dataset to {limit} examples (num_test={data_args.num_test})")
    except Exception as e:
        logger.warning(f"No evaluation dataset: {e}")
        eval_dataset = None

    try:
        if data_args.mode == "supervised":
            data_module = make_supervised_data_module(
                tokenizer,
                train_dataset,
                eval_dataset,
                seed=config['common']['seed']
            )
            logger.info("Created supervised data module")
        elif data_args.mode == "fixed_length":
            data_module = make_fixed_length_data_module(
                tokenizer,
                train_dataset,
                eval_dataset,
                model_max_length,
                seed=config['common']['seed']
            )
            logger.info("Created fixed length data module")
        elif data_args.mode == "multiturn":
            data_module = make_multiturn_data_module(
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                max_length=model_max_length,
                seed=config['common']['seed']
            )
            logger.info("Created multi‑turn data module")
        else:
            logger.error(f"Unknown mode: {data_args.mode}")
            return
    except Exception as e:
        logger.error(f"Failed to create data module: {e}", exc_info=True)
        return

    try:
        model = load_model(model_args, training_args, n_tokens_needed, tokenizer)
        logger.info(f"Model loaded successfully: {model_args.model}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        return

    try:
        peft_mode = model_args.parameter_efficient_mode
        if 'lora' in model_args.parameter_efficient_mode:
            peft_config = create_peft_config(model_args, model_args.model)
            add_tokens = 'cog' in peft_mode
            model = PeftModelForCausalLMWrapper(model, peft_config, add_tokens=add_tokens)
            logger.info(f"Applied LoRA PEFT wrapper (add_tokens={add_tokens})")
        else:
            for param in model.parameters():
                param.requires_grad = False
            logger.info("LoRA is not enabled; froze all base model parameters")

        if 'cog-tuned' in peft_mode:
            enable_cog_tuning(model)
            logger.info("Applied cognitive token tuning")
        elif 'cog-frozen' in peft_mode:
            logger.info("Cognitive token tuning disabled")
        elif 'cog' in peft_mode:
            logger.error(f"Invalid peft mode: '{peft_mode}'. Must specify 'cog-tuned' or 'cog-frozen'.")
            return
    except Exception as e:
        logger.error(f"Failed to apply PEFT: {e}", exc_info=True)
        return

    log_trainable_parameters(model)

    wandb_run = None
    if config.get('logging', {}).get('wandb_project'):
        try:
            import wandb
            wandb_run = wandb.init(
                project=config['logging']['wandb_project'],
                entity=config['logging'].get('wandb_entity'),
                name=config['experiment'].get('name', 'training-run'),
                tags=config['experiment'].get('tags', []),
                notes=config['experiment'].get('notes', ''),
                config={
                    **config['common'],
                    **config['train'],
                    'parameter_efficient_mode': model_args.parameter_efficient_mode,
                    'dataset': data_args.dataset,
                    'model': model_args.model
                }
            )
            logger.info("WandB initialized successfully")
        except ImportError:
            logger.warning("WandB not installed, skipping initialization")
        except Exception as e:
            logger.warning(f"Failed to initialize WandB: {e}")

    try:
        trainer = LMTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            **data_module
        )
        logger.info("Trainer created successfully")
    except Exception as e:
        logger.error(f"Failed to create trainer: {e}", exc_info=True)
        return

    try:
        logger.info("Starting new training session")
        trainer.train()
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        if wandb_run:
            wandb_run.finish()
        return

    try:
        trainer.save_state()
        trainer.save_model(output_dir=str(training_args.checkpoint_dir))
        logger.info(f"Model saved successfully to: {training_args.checkpoint_dir}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}", exc_info=True)

    try:
        config_output_path = Path(training_args.checkpoint_dir) / 'training_config.yaml'
        with open(config_output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Training config saved to: {config_output_path}")
    except Exception as e:
        logger.error(f"Failed to save training config: {e}")

    if wandb_run:
        wandb_run.finish()
        logger.info("WandB run finished")

    logger.info(f"Model: {model_args.model}")
    logger.info(f"Model saved at: {str(training_args.checkpoint_dir)}")
    logger.info(f"Output directory: {str(training_args.output_dir)}")

if __name__ == "__main__":
    train()
