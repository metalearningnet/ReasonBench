import torch
import logging
from config import logger
from datasets import load_dataset
from training.rl import RLConfigManager
from peft import TaskType, LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.generation.configuration_utils").setLevel(logging.ERROR)

def create_trainer(model, tokenizer, training_args, trainer_class, preprocess_fn):
    dataset_name = training_args.dataset_name
    logger.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=training_args.dataset_split)
    train_dataset = preprocess_fn(dataset)

    eval_dataset = None
    if training_args.eval_split:
        try:
            eval_dataset = load_dataset(dataset_name, split=training_args.eval_split)
            eval_dataset = preprocess_fn(eval_dataset, limit=training_args.num_test)
            logger.info(f"Loaded evaluation dataset with {len(eval_dataset)} examples")
        except Exception as e:
            logger.warning(f"Could not load evaluation dataset: {e}")

    return trainer_class(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

def create_rl_trainer(config):
    training_mode = config["rl"]["training_mode"]
    logger.info(f"Creating {training_mode.upper()} trainer from configuration")
    training_args = RLConfigManager.create_training_args(config, training_mode)
    common_config = config.get("common", {})
    model_path = common_config["model"]
    logger.info(f"Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")
    
    load_kwargs = {
        "pretrained_model_name_or_path": model_path,
        "torch_dtype": torch.float16 if training_args.fp16 else torch.float32,
        "device_map": "auto" if torch.cuda.is_available() else None,
        "trust_remote_code": True
    }
    
    if training_args.load_in_8bit:
        load_kwargs["load_in_8bit"] = True
        logger.info("Loading model in 8-bit precision")
    elif training_args.load_in_4bit:
        load_kwargs["load_in_4bit"] = True
        logger.info("Loading model in 4-bit precision")
    
    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    peft_mode = common_config.get("parameter_efficient_mode", "none")
    if peft_mode != "none":
        lora_config_args = common_config.get("lora_config", {})
        peft_kwargs = {
            "r": lora_config_args.get("r", 16),
            "lora_alpha": lora_config_args.get("alpha", 32),
            "target_modules": lora_config_args.get("target_modules", ["q_proj", "v_proj"]),
            "lora_dropout": lora_config_args.get("dropout", 0.1),
            "bias": lora_config_args.get("bias", "none"),
            "task_type": TaskType.CAUSAL_LM
        }

        if "cog-tuned" in peft_mode:
            peft_kwargs["modules_to_save"] = ["embed_tokens", "lm_head"]
            logger.info("Configured LoRA to tune cognitive tokens (modules_to_save applied)")
        elif "cog-frozen" in peft_mode:
            logger.info("Cognitive token tuning disabled")

        peft_config = LoraConfig(**peft_kwargs)
        model = get_peft_model(model, peft_config)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Applied LoRA: {trainable_params:,} trainable params ({100*trainable_params/total_params:.2f}%)")
    
    from training import TRAINER_MAP, PREPROCESSOR_MAP
    TrainerClass = TRAINER_MAP[training_mode]
    preprocessor = PREPROCESSOR_MAP[training_mode](training_args)
    trainer = create_trainer(model, tokenizer, training_args, TrainerClass, preprocessor.process)
    logger.info(f"{training_mode.upper()} trainer created successfully")
    return trainer
