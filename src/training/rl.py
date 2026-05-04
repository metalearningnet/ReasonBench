import json
import torch
import logging
import transformers
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datasets import Dataset as HFDataset
from transformers import PreTrainedTokenizer
from peft import PeftModel, get_peft_model_state_dict
from config import DEFAULT_OUTPUT_DIR, DEFAULT_CHECKPOINT_DIR, logger, load_rl_config

logging.getLogger("trl.trainer.utils").setLevel(logging.ERROR)

@dataclass
class RLTrainingArguments(transformers.TrainingArguments):
    beta: float = field(default=0.1)
    loss_type: str = field(default="sigmoid")
    desirable_weight: float = field(default=1.0)
    undesirable_weight: float = field(default=1.0)
    top_p: Optional[float] = field(default=None)
    temperature: Optional[float] = field(default=None)
    max_prompt_length: int = field(default=512)
    max_completion_length: int = field(default=1024)
    dataset_name: str = field(default=None)
    dataset_split: str = field(default="train")
    eval_split: str = field(default=None)
    field_mappings: Optional[Dict[str, str]] = field(default=None)
    load_in_8bit: bool = field(default=False)
    load_in_4bit: bool = field(default=False)
    training_mode: str = field(default="dpo")
    model_init_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)
    ref_model_init_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)
    num_train: Optional[int] = field(default=None)
    num_test: Optional[int] = field(default=None)
    checkpoint_dir: str = field(default=None)

class RLConfigManager:
    @staticmethod
    def create_training_args(
        config: Dict[str, Any],
        training_mode: str
    ) -> RLTrainingArguments:
        train_config = config.get("train", {})
        common_config = config.get("common", {})
        dataset_name = common_config.get("dataset")
        generator_config = config.get("generator", {})
        use_bf16 = train_config.get("bf16", False)
        use_fp16 = False if use_bf16 else train_config.get("fp16", False)
        args = {
            "bf16": use_bf16,
            "fp16": use_fp16,
            "eval_steps": train_config.get("eval_steps", 50),
            "save_steps": train_config.get("save_steps", 100),
            "warmup_steps": train_config.get("warmup_steps", 100),
            "logging_steps": train_config.get("logging_steps", 1000),
            "learning_rate": train_config.get("learning_rate", 1e-5),
            "num_train_epochs": train_config.get("num_train_epochs", 1),
            "save_total_limit": train_config.get("save_total_limit", 1),
            "max_prompt_length": train_config.get("max_prompt_length", 512),
            "output_dir": str(train_config.get("output_dir", DEFAULT_OUTPUT_DIR)),
            "checkpoint_dir": str(train_config.get("checkpoint_dir", DEFAULT_CHECKPOINT_DIR)),
            "gradient_accumulation_steps": train_config.get("gradient_accumulation_steps", 4),
            "max_completion_length": train_config.get("max_completion_length", 1024),
            "load_in_8bit": common_config.get("load_in_8bit", False),
            "load_in_4bit": common_config.get("load_in_4bit", False),
            "temperature": generator_config.get("temperature", 0.3),
            "top_p": generator_config.get("top_p", 1.0),
            "num_train": config.get("num_train", None),
            "num_test": config.get("num_test", None),
            "seed": common_config.get("seed", 42),
            "training_mode": training_mode,
            "ref_model_init_kwargs": {},
            "model_init_kwargs": {}
        }
        from training import CONFIG_MAP
        cfg = CONFIG_MAP[training_mode](config, training_mode, dataset_name)
        args = cfg.update(args)
        return RLTrainingArguments(**args)

class RLPreprocessor:
    def __init__(self, training_args):
        self.training_args = training_args
    
    def process(self, dataset, limit=None):
        raise NotImplementedError("Subclasses must implement the process() method.")

class RLConfig:
    def __init__(self, config, training_mode, dataset_name):
        self.config = config
        self.rl_config = load_rl_config(training_mode)
        common_config = self.config.get("common", {})
        self.batch_size = common_config.get("batch_size", 1)
        self.field_mappings = self.rl_config.get("field_mappings", {})
        self.dataset_name = dataset_name if dataset_name is not None else self.rl_config["dataset"]
    
    def update(self, args, dataset_name=None):
        raise NotImplementedError("Subclasses must implement the update() method.")

class RLTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizer,
        args: RLTrainingArguments,
        train_dataset: Optional[HFDataset] = None,
        eval_dataset: Optional[HFDataset] = None
    ):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.train_dataset = train_dataset
        logger.info(f"Initialized {self.__class__.__name__} with {len(train_dataset) if train_dataset else 0} training examples")
    
    def train(self):
        raise NotImplementedError
    
    def evaluate(self):
        return {}
    
    def save_model(self, output_dir: str):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(output_dir)
        
        is_peft_model = False
        peft_config = None
        
        try:
            if isinstance(self.model, PeftModel):
                is_peft_model = True
                peft_config = self.model.peft_config
                logger.info("Detected PeftModel instance")
        except ImportError:
            logger.warning("PEFT library not available")
        
        if not is_peft_model and hasattr(self.model, 'peft_config'):
            peft_config = self.model.peft_config
            if peft_config is not None:
                is_peft_model = True
                logger.info("Detected model with peft_config attribute")
        
        if is_peft_model and peft_config:
            logger.info("Saving LoRA adapters separately")
            
            try:
                if isinstance(self.model, PeftModel):
                    self.model.save_pretrained(output_dir)
                    logger.info(f"PEFT adapters saved to {output_dir}")
                else:
                    logger.warning("Model has peft_config but isn't PeftModel instance")
                    logger.warning("Attempting manual adapter saving...")
                    
                    adapter_state_dict = get_peft_model_state_dict(self.model)
                    torch.save(adapter_state_dict, Path(output_dir) / "adapter_model.bin")
                    
                    config_dict = peft_config.to_dict()
                    with open(Path(output_dir) / "adapter_config.json", "w") as f:
                        json.dump(config_dict, f, indent=4)
                    
                    logger.info(f"Manual adapter saving completed")
            except Exception as e:
                logger.error(f"Error saving PEFT adapters: {e}")
                logger.warning("Falling back to standard model saving")
                self.model.save_pretrained(output_dir)
        else:
            logger.info("Saving standard model (full weights)")
            self.model.save_pretrained(output_dir)
        
        args_dict = self.args.to_dict()
        args_dict.pop('checkpoint_dir', None)
        with open(Path(output_dir) / "training_args.json", "w") as f:
            json.dump(args_dict, f, indent=4, default=str)
        
        logger.info(f"Model saved to {output_dir}")
