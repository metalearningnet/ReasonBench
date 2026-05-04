import os
import re
import sys
import yaml
import json
import torch
import random
import transformers
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from dataset import DATASET_MAP
from model_loader import CausalLM
from huggingface_hub import login
from preprocess import DataConfig
from train import create_cot_tokens
from peft import PeftModel, PeftConfig
from torch.utils.data import DataLoader
from generator import format_trajectory
from dataclasses import dataclass, field, asdict
from peft_model import PeftModelForCausalLMWrapper
from typing import Optional, Dict, List, Tuple, Any
from config import (
    COT_TOKEN_NAMES, DEFAULT_EVAL_OUTPUT_DIR, DEFAULT_CHECKPOINT_DIR, MD_PATH, MD_SRC, RESERVED_MODELS, SHOW_EVAL, SHOW_PROMPT, SHOW_TRAJECTORY_ON_FAIL,
    load_config, load_datasets_config, update_dataclass_from_config, setup_directories, dataset_names, logger
)

@dataclass
class ModelArguments:
    model: Optional[str] = field(
        default=None,
        metadata={"help": "Path to trained model checkpoint. Required for evaluation."}
    )
    cache_dir: Optional[str] = field(default=None)
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the output directory."}
    )
    max_length: Optional[int] = field(default=1000)
    save_result: Optional[bool] = field(default=True)
    load_in_8bit: Optional[bool] = field(default=False)
    load_in_4bit: Optional[bool] = field(default=False)
    use_calculator: Optional[bool] = field(default=False)
    decoding_scheme: Optional[str] = field(default="default")
    generation_config: Optional[str] = field(
        default=None,
        metadata={"help": "Path to generation config file or JSON string"}
    )
    temperature: Optional[float] = field(default=None)
    top_p: Optional[float] = field(default=None)
    top_k: Optional[int] = field(default=None)
    num_beams: Optional[int] = field(default=None)
    do_sample: Optional[bool] = field(default=None)
    parameter_efficient_mode: Optional[str] = field(
        default='none',
        metadata={"choices": ["none", "lora", "lora-cog-frozen", "lora-cog-tuned"]}
    )
    hf_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Hugging Face token required for llama family models."}
    )
    interactive: Optional[bool] = field(
        default=False,
        metadata={"help": "Use interactive environment evaluation for multi‑turn datasets (e.g., ALFWorld)."}
    )
    enable_cpu_offload: Optional[bool] = field(default=False)
    config: Optional[str] = field(
        default=None,
        metadata={"help": "Path to config file"}
    )

@dataclass
class DataArguments:
    dataset: str = field(
        default=None,
        metadata={
            "help": "Dataset name.",
            "choices": dataset_names
        }
    )
    seed: Optional[int] = field(default=42)
    batch_size: Optional[int] = field(default=None)
    num_test: Optional[int] = field(default=None)

class TokenizerFactory:
    @staticmethod
    def create(model_name: str, cache_dir: Optional[str]) -> transformers.PreTrainedTokenizer:
        model_name_lower = model_name.lower()
        if 'llama' in model_name_lower or 'alpaca' in model_name_lower:
            tokenizer_class = transformers.LlamaTokenizer
        else:
            tokenizer_class = transformers.AutoTokenizer
        
        if os.path.isdir(model_name):
            try:
                logger.info(f"Attempting to load tokenizer from local directory: {model_name}")
                tokenizer = tokenizer_class.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    trust_remote_code=True,
                    local_files_only=True
                )
                logger.info("Successfully loaded tokenizer from local directory")
                TokenizerFactory._post_process_tokenizer(tokenizer)
                return tokenizer
            except Exception as e:
                logger.warning(f"Failed to load tokenizer from local directory: {e}")
                logger.info("Attempting to load original model name from training config...")
                
                config_path = os.path.join(model_name, "training_config.yaml")
                if os.path.isfile(config_path):
                    with open(config_path, "r") as f:
                        training_config = yaml.safe_load(f)
                    original_model = training_config.get("common", {}).get("model")
                    if original_model:
                        logger.info(f"Found original model name: {original_model}")
                        tokenizer = tokenizer_class.from_pretrained(
                            original_model,
                            cache_dir=cache_dir,
                            trust_remote_code=True
                        )
                        TokenizerFactory._post_process_tokenizer(tokenizer)
                        return tokenizer
                    else:
                        logger.error("No model name found in training_config.yaml")
                else:
                    logger.error(f"training_config.yaml not found in {model_name}")
                
                raise ValueError(f"Could not load tokenizer from '{model_name}'")
        else:
            logger.info(f"Loading tokenizer from Hugging Face hub: {model_name}")
            tokenizer = tokenizer_class.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            TokenizerFactory._post_process_tokenizer(tokenizer)
            return tokenizer

    @staticmethod
    def _post_process_tokenizer(tokenizer: transformers.PreTrainedTokenizer):
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.debug(f"Set pad_token to eos_token: {tokenizer.pad_token}")
        tokenizer.padding_side = "left"
        logger.debug(f"Set padding_side to: {tokenizer.padding_side}")

class ModelPathResolver:
    @staticmethod
    def resolve(model_args: ModelArguments) -> Tuple[str, Optional[Path], Optional[Path], Optional[Path]]:
        model_path = Path(model_args.model)
        
        if model_path.exists() and model_path.is_dir():
            logger.info(f"Model path is a directory: {model_path}")
            
            base_model_name = None
            checkpoint_dir = model_path
            
            base_info_path = model_path / "base_model_info.json"
            if base_info_path.exists():
                try:
                    with open(base_info_path, "r") as f:
                        base_info = json.load(f)
                    base_model_name = base_info.get("base_model_name_or_path")
                    if base_model_name:
                        logger.info(f"Found base model in checkpoint: {base_model_name}")
                        return base_model_name, None, None, checkpoint_dir
                except Exception as e:
                    logger.warning(f"Failed to read base_model_info.json: {e}")
            
            adapter_config_path = model_path / "adapter_config.json"
            if adapter_config_path.exists():
                try:
                    config = PeftConfig.from_pretrained(str(model_path))
                    base_model_name = config.base_model_name_or_path
                    logger.info(f"Base model from adapter config: {base_model_name}")
                    return base_model_name, None, None, checkpoint_dir
                except Exception as e:
                    logger.warning(f"Could not get base model from adapter config: {e}")
            
            training_args_path = model_path / "training_args.json"
            if training_args_path.exists():
                try:
                    with open(training_args_path, "r") as f:
                        training_args = json.load(f)
                    
                    if 'model_name_or_path' in training_args:
                        base_model_name = training_args['model_name_or_path']
                    elif 'model' in training_args:
                        base_model_name = training_args['model']
                    
                    if base_model_name:
                        logger.info(f"Base model from training args: {base_model_name}")
                        return base_model_name, None, None, checkpoint_dir
                except Exception as e:
                    logger.warning(f"Failed to read training_args.json: {e}")
            
            return str(model_path), None, None, None
        
        logger.info(f"Model path appears to be a model identifier: {model_path}")
        
        checkpoint_dir = Path(DEFAULT_CHECKPOINT_DIR)
        if checkpoint_dir.exists():
            input_embedding_file = checkpoint_dir / "input_embeddings.pt"
            output_embedding_file = checkpoint_dir / "output_embeddings.pt"
            
            if input_embedding_file.exists() and output_embedding_file.exists():
                logger.info(f"Found custom embedding files in {checkpoint_dir}")
                return str(model_path), input_embedding_file, output_embedding_file, checkpoint_dir
        
        return str(model_path), None, None, None

class ModelLoader:
    @staticmethod
    def load(
        model_args: ModelArguments,
        model_name: str,
        input_embedding_file: Optional[Path] = None,
        output_embedding_file: Optional[Path] = None,
        num_new_tokens: Optional[int] = None,
        checkpoint_dir: Optional[Path] = None
    ) -> torch.nn.Module:
        logger.info(f"Loading model from: {model_name}")
        
        if model_name in RESERVED_MODELS:
            logger.info(f"Loading reserved model: {model_name}")
            if model_name == MD_PATH:
                if not MD_SRC.exists():
                    logger.error(f"Failed to find {MD_SRC}")
                sys.path.append(str(MD_SRC))
                from md import MD
                from utils import md_cfg
                model = MD.from_pretrained(ckpt_path=md_cfg.ckpt_path)
                return model
            else:
                raise ValueError(f"Unknown reserved model: {model_name}")
        
        model_type = ModelLoader._detect_model_type(
            model_args, model_name, checkpoint_dir,
            input_embedding_file, output_embedding_file
        )
        
        logger.info(f"Detected model type: {model_type}")
        
        if model_type == "rl_peft":
            return ModelLoader._load_rl_peft_model(
                model_args=model_args,
                base_model_name=model_name,
                checkpoint_dir=checkpoint_dir,
                merge_lora=getattr(model_args, 'merge_lora', False)
            )
        elif model_type == "custom_peft":
            return ModelLoader._load_custom_peft_model(
                model_args=model_args,
                base_model_name=model_name,
                checkpoint_dir=checkpoint_dir,
                input_embedding_file=input_embedding_file,
                output_embedding_file=output_embedding_file,
                num_new_tokens=num_new_tokens
            )
        elif model_type == "base_model_with_lora":
            return ModelLoader._load_base_model_with_lora(
                model_args=model_args,
                model_name=model_name,
                checkpoint_dir=checkpoint_dir,
                num_new_tokens=num_new_tokens
            )
        elif model_type == "standalone_model":
            return ModelLoader._load_standalone_model(
                model_args=model_args,
                model_name=model_name
            )
        else:
            logger.error(f"Unknown model type: {model_type}")
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def _detect_model_type(
        model_args: ModelArguments,
        model_name: str,
        checkpoint_dir: Optional[Path],
        input_embedding_file: Optional[Path],
        output_embedding_file: Optional[Path]
    ) -> str:
        if checkpoint_dir and checkpoint_dir.exists():
            has_training_args = (checkpoint_dir / "training_args.json").exists()
            has_adapter_config = (checkpoint_dir / "adapter_config.json").exists()
            has_adapter_model = (
                (checkpoint_dir / "adapter_model.safetensors").exists() or
                (checkpoint_dir / "adapter_model.bin").exists()
            )
            
            has_custom_embeddings = (
                (checkpoint_dir / "input_embeddings.pt").exists() and
                (checkpoint_dir / "output_embeddings.pt").exists()
            )
            
            if has_training_args and has_adapter_config and has_adapter_model:
                return "rl_peft"
            elif has_adapter_config and has_custom_embeddings:
                return "custom_peft"
            elif has_adapter_config and has_adapter_model:
                logger.info("Detected standard PEFT model (not RL-trained)")
                return "base_model_with_lora"
        
        if input_embedding_file and output_embedding_file:
            if input_embedding_file.exists() and output_embedding_file.exists():
                return "custom_peft"
        
        if os.path.isdir(model_name) and model_args.parameter_efficient_mode and 'lora' in model_args.parameter_efficient_mode:
            return "base_model_with_lora"
        
        return "standalone_model"
    
    @staticmethod
    def _load_rl_peft_model(
        model_args: ModelArguments,
        base_model_name: str,
        checkpoint_dir: Path,
        merge_lora: bool = False
    ) -> torch.nn.Module:
        logger.info("Loading RL-trained PEFT model")
        logger.info(f"Base model: {base_model_name}")
        logger.info(f"Checkpoint: {checkpoint_dir}")
        logger.info(f"Merge LoRA: {merge_lora}")
        
        try:
            base_model = ModelLoader._load_base_model_with_kwargs(
                model_name=base_model_name,
                model_args=model_args
            )
            
            model = PeftModel.from_pretrained(base_model, str(checkpoint_dir))
            logger.info("Loaded PEFT adapters")
            
            if merge_lora:
                logger.info("Merging LoRA adapters into base model...")
                model = model.merge_and_unload()
                logger.info("LoRA adapters merged successfully")
            
            return model
        except Exception as e:
            logger.error(f"Failed to load RL-trained PEFT model: {e}")
            raise
    
    @staticmethod
    def _load_custom_peft_model(
        model_args: ModelArguments,
        base_model_name: str,
        checkpoint_dir: Optional[Path] = None,
        input_embedding_file: Optional[Path] = None,
        output_embedding_file: Optional[Path] = None,
        num_new_tokens: Optional[int] = None
    ) -> torch.nn.Module:
        logger.info("Loading custom PEFT model")
        
        try:
            load_kwargs = {
                "pretrained_model_name_or_path": base_model_name,
                "parameter_efficient_mode": model_args.parameter_efficient_mode,
                "cache_dir": model_args.cache_dir,
                "n_tokens": num_new_tokens or 0,
                "input_embedding_file": str(input_embedding_file) if input_embedding_file else None,
                "output_embedding_file": str(output_embedding_file) if output_embedding_file else None
            }
            
            device_map = "auto" if torch.cuda.is_available() else None
            load_kwargs["device_map"] = device_map
            
            if model_args.load_in_8bit:
                load_kwargs["load_in_8bit"] = True
                load_kwargs["torch_dtype"] = torch.float16
                logger.info("Loading base model in 8-bit precision")
            elif model_args.load_in_4bit:
                load_kwargs["load_in_4bit"] = True
                load_kwargs["torch_dtype"] = torch.float16
                logger.info("Loading base model in 4-bit precision")
            else:
                load_kwargs["torch_dtype"] = torch.float32
                logger.info("Loading base model with float32 precision")
            
            model = CausalLM.from_pretrained(**load_kwargs)
            logger.info("Base model loaded successfully")
            
            if checkpoint_dir and checkpoint_dir.exists():
                try:
                    model = PeftModelForCausalLMWrapper.from_pretrained(
                        model,
                        checkpoint_dir,
                        load_embeddings=False,
                        n_tokens=num_new_tokens or 0
                    )
                    logger.info("Loaded custom PEFT adapters")
                except Exception as e:
                    logger.warning(f"Failed to load custom PEFT adapters: {e}")
                    logger.info("Trying to load as standard PEFT model...")
                    
                    try:
                        model = PeftModel.from_pretrained(model, str(checkpoint_dir))
                        logger.info("Loaded as standard PEFT model (fallback)")
                    except Exception as e2:
                        logger.warning(f"Failed to load as standard PEFT: {e2}")
                        logger.info("Returning model without adapters")
            
            return model
        except Exception as e:
            logger.error(f"Failed to load custom PEFT model: {e}")
            raise
    
    @staticmethod
    def _load_base_model_with_lora(
        model_args: ModelArguments,
        model_name: str,
        checkpoint_dir: Path,
        num_new_tokens: Optional[int] = None
    ) -> torch.nn.Module:
        logger.info("Loading base model with LoRA adapters")
        
        try:
            base_model = ModelLoader._load_base_model_with_kwargs(
                model_name=model_name,
                model_args=model_args
            )
            
            model = PeftModel.from_pretrained(base_model, str(checkpoint_dir))
            logger.info("Loaded LoRA adapters as standard PEFT")
            
            if getattr(model_args, 'merge_lora', False):
                logger.info("Merging LoRA adapters...")
                model = model.merge_and_unload()
            
            return model
        except Exception as e:
            logger.error(f"Failed to load base model with LoRA: {e}")
            
            try:
                logger.info("Trying custom PEFT wrapper as fallback...")
                
                base_model = ModelLoader._load_base_model_with_kwargs(
                    model_name=model_name,
                    model_args=model_args
                )
                
                model = PeftModelForCausalLMWrapper.from_pretrained(
                    base_model,
                    checkpoint_dir,
                    load_embeddings=True,
                    n_tokens=num_new_tokens or 0
                )
                logger.info("Loaded with custom PEFT wrapper (fallback)")
                
                return model
            except Exception as e2:
                logger.error(f"Failed to load with custom wrapper: {e2}")
                raise
    
    @staticmethod
    def _load_standalone_model(
        model_args: ModelArguments,
        model_name: str
    ) -> torch.nn.Module:
        logger.info(f"Loading standalone model: {model_name}")
        
        return ModelLoader._load_base_model_with_kwargs(
            model_name=model_name,
            model_args=model_args
        )
    
    @staticmethod
    def _load_base_model_with_kwargs(
        model_name: str,
        model_args: ModelArguments
    ) -> torch.nn.Module:
        load_kwargs = {
            "pretrained_model_name_or_path": model_name,
            "device_map": "auto" if torch.cuda.is_available() else None,
            "trust_remote_code": True
        }
        
        if model_args.load_in_8bit:
            load_kwargs["load_in_8bit"] = True
            load_kwargs["torch_dtype"] = torch.float16
            logger.info("Loading model in 8-bit precision")
        elif model_args.load_in_4bit:
            load_kwargs["load_in_4bit"] = True
            load_kwargs["torch_dtype"] = torch.float16
            logger.info("Loading model in 4-bit precision")
        else:
            load_kwargs["torch_dtype"] = torch.float32
            logger.info("Loading model with float32 precision")
        
        if model_args.cache_dir:
            load_kwargs["cache_dir"] = model_args.cache_dir
        
        if getattr(model_args, 'enable_cpu_offload', False):
            load_kwargs["offload_folder"] = "offload"
            load_kwargs["offload_state_dict"] = True
            logger.info("CPU offloading enabled")
        
        try:
            try:
                model = CausalLM.from_pretrained(**load_kwargs)
            except:
                logger.info("CausalLM not available, using AutoModelForCausalLM")
                model = transformers.AutoModelForCausalLM.from_pretrained(**load_kwargs)
            
            logger.info(f"Base model loaded successfully: {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load base model {model_name}: {e}")
            raise
    
    @staticmethod
    def _prepare_load_kwargs(
        model_args: ModelArguments,
        model_name: str,
        input_embedding_file: Optional[Path],
        output_embedding_file: Optional[Path],
        num_new_tokens: int
    ) -> Dict[str, Any]:
        logger.warning("Using legacy _prepare_load_kwargs method")
        
        load_kwargs = {
            "n_tokens": num_new_tokens,
            "input_embedding_file": str(input_embedding_file) if input_embedding_file else None,
            "output_embedding_file": str(output_embedding_file) if output_embedding_file else None,
            "pretrained_model_name_or_path": model_name,
            "parameter_efficient_mode": model_args.parameter_efficient_mode,
            "cache_dir": model_args.cache_dir,
            "offload_folder": "offload",
            "offload_state_dict": True
        }
        
        device_map = "auto"

        if model_args.load_in_8bit:
            load_kwargs.update({
                "dtype": torch.float16,
                "device_map": device_map,
                "load_in_8bit": True
            })
            logger.info("Loading model in 8-bit precision")
        elif model_args.load_in_4bit:
            load_kwargs.update({
                "dtype": torch.float16,
                "device_map": device_map,
                "load_in_4bit": True
            })
            logger.info("Loading model in 4-bit precision")
        else:
            load_kwargs.update({
                "dtype": torch.float32,
                "device_map": device_map
            })
            logger.info("Loading model with float32 precision")
        
        return load_kwargs
    
    @staticmethod
    def _load_base_model(load_kwargs: Dict[str, Any]) -> torch.nn.Module:
        logger.warning("Using legacy _load_base_model method")
        
        try:
            model = CausalLM.from_pretrained(**load_kwargs)
            logger.info(f"Base model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load base model: {e}")
            raise
    
    @staticmethod
    def _place_model_on_device(model: torch.nn.Module, model_args: ModelArguments) -> torch.nn.Module:
        try:
            if hasattr(model, 'device'):
                logger.info(f"Model device attribute: {model.device}")
            
            if model_args.load_in_8bit or model_args.load_in_4bit:
                logger.info("Using quantized model with auto device mapping")
                return model
            
            if torch.cuda.is_available():
                first_param = next(model.parameters())
                if first_param.device.type != 'cuda':
                    logger.info("Moving model to GPU...")
                    model = model.to('cuda')
                
                all_on_gpu = True
                for name, param in model.named_parameters():
                    if param.device.type != 'cuda':
                        logger.warning(f"Parameter {name} is NOT on GPU: {param.device}")
                        all_on_gpu = False
                
                if all_on_gpu:
                    logger.info("All model parameters are on GPU")
                else:
                    logger.warning("Some parameters are NOT on GPU!")
            else:
                logger.info("Model on CPU")
        except Exception as e:
            logger.warning(f"Could not check model device: {e}")
        
        return model

class DatasetManager:
    @classmethod
    def get_data_class(cls, dataset_name: str) -> Any:
        if dataset_name not in DATASET_MAP:
            available = list(DATASET_MAP.keys())
            raise ValueError(f"Dataset '{dataset_name}' not implemented. Available: {available}")
        
        return DATASET_MAP[dataset_name]
    
    @staticmethod
    def create_dataset(
        data_class: Any,
        name: str,
        split: str,
        data_config: DataConfig
    ) -> torch.utils.data.Dataset:
        try:
            dataset = data_class(name, split, config=data_config)
            logger.info(f"Dataset created: {split} split with {len(dataset)} examples")
            return dataset
        except Exception as e:
            logger.error(f"Failed to create dataset: {e}")
            raise
    
    @staticmethod
    def sample_dataset(
        dataset: torch.utils.data.Dataset,
        num_samples: int,
        seed: int = 42
    ) -> torch.utils.data.Dataset:
        if len(dataset) <= num_samples:
            logger.debug(f"Dataset size {len(dataset)} <= requested samples {num_samples}, skipping sampling")
            return dataset
        
        random.seed(seed)
        indices = random.sample(range(len(dataset)), num_samples)
        
        class SampledDataset(torch.utils.data.Dataset):
            def __init__(self, original_dataset, indices):
                self.original_dataset = original_dataset
                self.indices = indices
                
            def __len__(self):
                return len(self.indices)
                
            def __getitem__(self, idx):
                return self.original_dataset[self.indices[idx]]
                
            def __getattr__(self, name):
                return getattr(self.original_dataset, name)
        
        return SampledDataset(dataset, indices)

class BatchEvaluator:
    @staticmethod
    def evaluate(
        model: torch.nn.Module,
        model_name: str,
        tokenizer: transformers.PreTrainedTokenizer,
        x_text: List[str],
        y_text: List[str],
        dataset: Any,
        max_length: int,
        decoding_scheme: str,
        generation_kwargs: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, int, List[Dict], Dict[str, int], Dict[str, int]]:
        use_quantization = hasattr(model, 'is_quantized') or (
            hasattr(model, 'config') and
            hasattr(model.config, 'quantization_config')
        )

        if use_quantization:
            logger.debug("Using quantization-aware device handling")
            encoding = BatchEvaluator._encode_inputs_no_device(
                tokenizer,
                x_text,
                max_length
            )
        else:
            try:
                model_device = next(model.parameters()).device
                logger.debug(f"Model device: {model_device}")
                encoding = BatchEvaluator._encode_inputs(
                    tokenizer,
                    x_text,
                    max_length,
                    model_device
                )
            except StopIteration:
                logger.warning("Could not get model device, using default")
                encoding = BatchEvaluator._encode_inputs_no_device(
                    tokenizer,
                    x_text,
                    max_length
                )
        
        generated_ids = BatchEvaluator._generate_text(
            model,
            model_name,
            encoding,
            max_length,
            decoding_scheme,
            generation_kwargs
        )
        
        generated_texts = BatchEvaluator._decode_generated_text(
            tokenizer,
            generated_ids,
            encoding
        )
        
        cleaned_texts = BatchEvaluator._clean_generated_texts(
            generated_texts,
            dataset
        )
        
        return BatchEvaluator._process_results(
            cleaned_texts,
            x_text,
            y_text,
            dataset
        )
    
    @staticmethod
    def _clean_generated_texts(
        generated_texts: List[str],
        dataset: Any
    ) -> List[str]:
        cleaned_texts = []
        dataset_type = dataset.__class__.__name__ if hasattr(dataset, '__class__') else 'unknown'
        
        for text in generated_texts:
            if not text or not text.strip():
                cleaned_texts.append("")
                continue
            
            text = text.strip()
            
            text = BatchEvaluator._apply_general_cleaning(text)
            
            text = BatchEvaluator._apply_dataset_specific_cleaning(text, dataset_type)
            
            text = BatchEvaluator._extract_likely_answer(text, dataset_type)
            
            cleaned_texts.append(text)
        
        return cleaned_texts
    
    @staticmethod
    def _apply_general_cleaning(text: str) -> str:
        if not text:
            return text
        
        conversational_patterns = [
            (r'Okay,\s*let\'s see\s*', ''),
            (r'Let me think\s*', ''),
            (r'Hmm,\s*', ''),
            (r'Wait,\s*', ''),
            (r'Well,\s*', ''),
            (r'So,\s*', ''),
            (r'Now,\s*', ''),
            (r'First,\s*', ''),
            (r'Next,\s*', ''),
            (r'Then,\s*', ''),
            (r'Finally,\s*', ''),
            (r'In summary,\s*', ''),
            (r'To summarize,\s*', '')
        ]
        
        for pattern, replacement in conversational_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        
        if COT_TOKEN_NAMES:
            for token in COT_TOKEN_NAMES:
                patterns_to_fix = [
                    (rf'<{re.escape(token)}>\s*:\s*', f'<{token}>: '),
                    (rf'<{re.escape(token)}>:\s+', f'<{token}>: '),
                    (rf'<{re.escape(token)}>:(?!\s)', f'<{token}>: ')
                ]
                
                for pattern, replacement in patterns_to_fix:
                    text = re.sub(pattern, replacement, text)
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            is_cot_line = False
            if COT_TOKEN_NAMES:
                for token in COT_TOKEN_NAMES:
                    if line.startswith(f'<{token}>:'):
                        is_cot_line = True
                        if ': ' in line:
                            parts = line.split(': ', 1)
                            if len(parts) == 2:
                                tag, content = parts
                                content = re.sub(r'\s+', ' ', content).strip()
                                line = f'{tag}: {content}'
                        break
            
            if not is_cot_line:
                line = re.sub(r'\s+', ' ', line).strip()
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    @staticmethod
    def _apply_dataset_specific_cleaning(text: str, dataset_type: str) -> str:
        if not text:
            return text
        
        has_math_indicators = any(indicator in text.lower() for indicator in
                                 ['$', '\\boxed', 'sqrt', 'frac', '^', 'pi', 'equation'])
        
        has_multiple_choice = any(pattern in text for pattern in
                                 ['A)', 'B)', 'C)', 'D)', 'E)', 'F)', 'G)', 'H)', 'I)', 'J)', 'K)'
                                  'A.', 'B.', 'C.', 'D.', 'E.', 'F.', 'G.', 'H.', 'I.', 'J.', 'K.'])
        
        has_code = any(pattern in text for pattern in
                      ['def ', 'class ', 'import ', 'print(', 'return ', 'function'])
        
        if has_math_indicators:
            text = BatchEvaluator._clean_math_content(text)
        
        if has_multiple_choice:
            text = BatchEvaluator._clean_multiple_choice_content(text)
        
        if has_code:
            text = BatchEvaluator._clean_code_content(text)
        
        return text
    
    @staticmethod
    def _clean_math_content(text: str) -> str:
        if not text:
            return text
        
        boxed_pattern = r'\\boxed\{([^}]+)\}'
        boxed_match = re.search(boxed_pattern, text)
        if boxed_match:
            return boxed_match.group(1).strip()
        
        math_patterns = [
            r'\$\$([^$]+)\$\$',
            r'\$([^$]+)\$',
            r'\\\[([^\]]+)\\\]',
            r'\\begin\{equation\}(.+?)\\end\{equation\}'
        ]
        
        for pattern in math_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                last_math = matches[-1].strip()
                last_math = re.sub(r'\s+', ' ', last_math)
                return last_math
        
        math_answer_patterns = [
            r'=\s*([\d\.\+\-\*/\(\)\s]+)(?=[\s\.\n]|$)',
            r'is\s*([\d\.\+\-\*/\(\)\s]+)(?=[\s\.\n]|$)'
        ]
        
        for pattern in math_answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                answer = re.sub(r'\s+', '', answer)
                return answer
        
        return text
    
    @staticmethod
    def _clean_multiple_choice_content(text: str) -> str:
        if not text:
            return text
        
        choice_patterns = [
            r'[\(\[]?\s*([A-G])\s*[\)\]]',
            r'Answer:\s*([A-G])',
            r'Option\s*([A-G])',
            r'Choice\s*([A-G])'
        ]
        
        for pattern in choice_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[-1].upper()
        
        lines = text.split('\n')
        for line in reversed(lines):
            line = line.strip()
            if re.match(r'^[A-G][\.\)]\s*', line):
                return line[0].upper()
        
        return text
    
    @staticmethod
    def _clean_code_content(text: str) -> str:
        if not text:
            return text
        
        code_block_pattern = r'```(?:\w+)?\s*\n(.*?)\n```'
        code_blocks = re.findall(code_block_pattern, text, re.DOTALL)
        
        if code_blocks:
            return code_blocks[-1].strip()
        
        lines = text.split('\n')
        code_lines = []
        
        for line in lines:
            line = line.strip()
            if any(pattern in line for pattern in
                  ['def ', 'return ', 'print(', 'import ', 'from ', 'class ']):
                code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines[-3:])
        
        return text
    
    @staticmethod
    def _extract_likely_answer(text: str, dataset_type: str) -> str:
        if not text:
            return text
        
        extraction_strategies = [
            BatchEvaluator._extract_via_structured_patterns,
            BatchEvaluator._extract_via_keywords,
            BatchEvaluator._extract_via_numeric_patterns,
            BatchEvaluator._extract_via_final_line
        ]
        
        for strategy in extraction_strategies:
            extracted = strategy(text)
            if extracted and extracted.strip():
                return extracted.strip()
        
        if len(text) > 200:
            return text[-100:].strip()
        
        return text.strip()
    
    @staticmethod
    def _extract_via_structured_patterns(text: str) -> str:
        patterns = [
            (r'Answer:\s*([^\n\.]+)', 1),
            (r'The answer is:\s*([^\n\.]+)', 1),
            (r'Final answer:\s*([^\n\.]+)', 1),
            (r'Solution:\s*([^\n\.]+)', 1),
            (r'Result:\s*([^\n\.]+)', 1),
            (r'Therefore,\s*([^\n\.]+)', 1),
            (r'Thus,\s*([^\n\.]+)', 1),
            (r'Hence,\s*([^\n\.]+)', 1),
            (r'So,\s*([^\n\.]+)', 1)
        ]
        
        for pattern, group in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                answer = match.group(group).strip()
                answer = re.sub(r'[\.\,\;\:\!\?]*$', '', answer)
                return answer
        
        return ""
    
    @staticmethod
    def _extract_via_keywords(text: str) -> str:
        lines = text.split('\n')
        answer_lines = []
        
        answer_keywords = ['answer', 'result', 'solution', 'final', 'therefore',
                          'thus', 'hence', 'equals', 'is', 'are', 'gives']
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in answer_keywords):
                for keyword in answer_keywords:
                    if keyword in line_lower:
                        parts = re.split(keyword, line_lower, maxsplit=1, flags=re.IGNORECASE)
                        if len(parts) > 1:
                            answer = parts[1].strip()
                            answer = re.sub(r'^[:\s]*', '', answer)
                            if answer:
                                answer_lines.append(answer)
                        break
        
        if answer_lines:
            return answer_lines[-1]
        
        return ""
    
    @staticmethod
    def _extract_via_numeric_patterns(text: str) -> str:
        numeric_patterns = [
            r'=\s*([\d\.]+)',
            r'is\s*([\d\.]+)',
            r'are\s*([\d\.]+)',
            r'equals\s*([\d\.]+)',
            r'gives\s*([\d\.]+)'
        ]
        
        for pattern in numeric_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[-1]
        
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            last_num = numbers[-1]
            num_pos = text.rfind(last_num)
            if num_pos >= 0:
                context_start = max(0, num_pos - 30)
                context_end = min(len(text), num_pos + len(last_num) + 30)
                context = text[context_start:context_end].lower()
                
                if any(indicator in context for indicator in
                      ['answer', 'result', 'solution', '=', 'is', 'are', 'equals']):
                    return last_num
        
        return ""
    
    @staticmethod
    def _extract_via_final_line(text: str) -> str:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if not lines:
            return ""
        
        last_line = lines[-1]
        
        if len(last_line.split()) <= 10:
            last_line = re.sub(r'^(Answer|Result|Solution|Final):?\s*', '',
                              last_line, flags=re.IGNORECASE)
            return last_line
        
        return ""
    
    @staticmethod
    def _encode_inputs_no_device(
        tokenizer: transformers.PreTrainedTokenizer,
        x_text: List[str],
        max_length: int,
    ) -> Dict[str, torch.Tensor]:
        return tokenizer(
            x_text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
    
    @staticmethod
    def _encode_inputs(
        tokenizer: transformers.PreTrainedTokenizer,
        x_text: List[str],
        max_length: int,
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        encoding = tokenizer(
            x_text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        return {k: v.to(device) for k, v in encoding.items()}
    
    @staticmethod
    def _generate_text(
        model: torch.nn.Module,
        model_name: str,
        encoding: Dict[str, Any],
        max_length: int,
        decoding_scheme: str,
        generation_kwargs: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        max_new_tokens = max_length - encoding['input_ids'].shape[1]
        
        default_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": model.config.pad_token_id if hasattr(model.config, 'pad_token_id') else None,
            "eos_token_id": model.config.eos_token_id if hasattr(model.config, 'eos_token_id') else None
        }
        
        if hasattr(model, 'generation_config') and model.generation_config is not None:
            model_config_dict = model.generation_config.to_dict()
            model_config_dict.pop('max_length', None)
            model_config_dict.pop('max_new_tokens', None)
            default_kwargs.update(model_config_dict)
        
        if generation_kwargs:
            default_kwargs.update(generation_kwargs)
        
        if decoding_scheme != "default":
            if decoding_scheme == "greedy":
                default_kwargs.update({
                    "do_sample": False,
                    "temperature": 1.0,
                    "num_beams": 1
                })
            elif decoding_scheme == "beam":
                default_kwargs.update({
                    "num_beams": 4,
                    "early_stopping": True,
                    "do_sample": False
                })
            elif decoding_scheme == "sampling":
                default_kwargs.update({
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.9
                })
        
        generation_kwargs = {k: v for k, v in default_kwargs.items() if v is not None}
        
        if model_name not in RESERVED_MODELS:
            with torch.no_grad():
                return model.generate(
                    **encoding,
                    **generation_kwargs
                )
        else:
            with torch.no_grad():
                return model.generate(input_ids=encoding['input_ids'])
    
    @staticmethod
    def _decode_generated_text(
        tokenizer: transformers.PreTrainedTokenizer,
        generated_ids: torch.Tensor,
        encoding: Dict[str, torch.Tensor]
    ) -> List[str]:
        try:
            return tokenizer.batch_decode(
                generated_ids[:, encoding['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
        except Exception as e:
            logger.error(f"Error decoding generated IDs: {e}")
            return [""] * len(generated_ids)
    
    @staticmethod
    def _process_results(
        generated_texts: List[str],
        x_text: List[str],
        y_text: List[str],
        dataset: Any
    ) -> Tuple[int, int, List[Dict], Dict[str, int], Dict[str, int]]:
        batch_correct = 0
        batch_outputs = []
        
        for text, x, y in zip(generated_texts, x_text, y_text):
            text, x, y = str(text), str(x), str(y)
            
            try:
                if dataset.is_correct(text, y):
                    batch_correct += 1
                    result = 'correct'
                else:
                    result = 'wrong'
            except Exception as e:
                logger.warning(f"Error checking correctness: {e}")
                result = 'error'
            
            batch_outputs.append({
                'input': x,
                'target': y,
                'generated_text': text,
                'result': result
            })
        
        return batch_correct, len(x_text), batch_outputs

class ResultSaver:
    @staticmethod
    def save(
        model_args: ModelArguments,
        data_args: DataArguments,
        all_outputs: List[Dict],
        total_correct: int,
        total_examples: int,
        config: Dict[str, Any]
    ) -> Tuple[Path, Path]:
        accuracy = total_correct / total_examples if total_examples > 0 else 0
        
        results_dir = ResultSaver._create_results_dir(model_args, data_args)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_identifier = ResultSaver._get_model_identifier(model_args)
        
        json_path = ResultSaver._save_json_results(
            results_dir, timestamp, model_identifier,
            model_args, data_args, all_outputs,
            total_correct, total_examples, accuracy, config
        )
        
        csv_path = ResultSaver._save_csv_summary(
            results_dir, timestamp, model_identifier, all_outputs
        )
        
        ResultSaver._print_summary(
            model_args, data_args, accuracy,
            total_correct, total_examples, json_path, csv_path
        )
        
        return json_path, csv_path
    
    @staticmethod
    def _create_results_dir(
        model_args: ModelArguments,
        data_args: DataArguments
    ) -> Path:
        results_dir = Path(model_args.output_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_dir = results_dir / data_args.dataset
        dataset_dir.mkdir(exist_ok=True)
        
        if model_args.model:
            model_name = Path(model_args.model).name
            model_dir = dataset_dir / model_name
            model_dir.mkdir(exist_ok=True)
            return model_dir
        
        return dataset_dir
    
    @staticmethod
    def _get_model_identifier(model_args: ModelArguments) -> str:
        if not model_args.model:
            return "unknown_model"
        
        model_path = Path(model_args.model)
        
        if model_path.exists() and model_path.is_dir():
            if (model_path / "training_args.json").exists():
                try:
                    with open(model_path / "training_args.json", "r") as f:
                        training_args = json.load(f)
                    
                    if "run_name" in training_args:
                        return training_args["run_name"]
                    
                    if "dataset_name" in training_args:
                        dataset_name = training_args["dataset_name"]
                        return f"{model_path.name}_{dataset_name}"
                except Exception as e:
                    logger.debug(f"Could not extract identifier from training args: {e}")
            
            return model_path.name
        else:
            return str(model_args.model).replace('/', '_')
    
    @staticmethod
    def _save_json_results(
        results_dir: Path,
        timestamp: str,
        model_identifier: str,
        model_args: ModelArguments,
        data_args: DataArguments,
        all_outputs: List[Dict],
        total_correct: int,
        total_examples: int,
        accuracy: float,
        config: Dict[str, Any]
    ) -> Path:
        filename = f"{model_identifier}_{timestamp}_results.json"
        json_path = results_dir / filename
        model_info = config.pop('model_info', {}) if isinstance(config, dict) else {}
        
        results = {
            "model": model_args.model,
            "dataset": data_args.dataset,
            "accuracy": accuracy,
            "total_examples": total_examples,
            "correct_predictions": total_correct,
            "timestamp": timestamp,
            "model_info": {
                **model_info,
                "model_args": ResultSaver._clean_dataclass_for_json(asdict(model_args)),
                "data_args": ResultSaver._clean_dataclass_for_json(asdict(data_args)),
            },
            "config": config,
            "predictions": all_outputs
        }
        
        results = ResultSaver._convert_paths_to_strings(results)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4, default=str)
        
        logger.debug(f"Saved detailed results to: {json_path}")
        return json_path
    
    @staticmethod
    def _clean_dataclass_for_json(obj: Any) -> Any:
        if hasattr(obj, '__dict__'):
            obj = vars(obj)
        elif hasattr(obj, '_asdict'):
            obj = obj._asdict()
        
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                if not key.startswith('_'):
                    result[key] = ResultSaver._clean_dataclass_for_json(value)
            return result
        elif isinstance(obj, list):
            return [ResultSaver._clean_dataclass_for_json(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    @staticmethod
    def _convert_paths_to_strings(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: ResultSaver._convert_paths_to_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ResultSaver._convert_paths_to_strings(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj
    
    @staticmethod
    def _save_csv_summary(
        results_dir: Path,
        timestamp: str,
        model_identifier: str,
        all_outputs: List[Dict]
    ) -> Optional[Path]:
        try:
            if not all_outputs:
                logger.warning("No outputs to save to CSV")
                return None
            
            csv_path = results_dir / f"{model_identifier}_{timestamp}_summary.csv"
            
            csv_data = []
            for i, item in enumerate(all_outputs):
                csv_data.append({
                    'index': i,
                    'input': str(item.get('input', ''))[:500],
                    'target': str(item.get('target', '')),
                    'generated_text': str(item.get('generated_text', '')),
                    'result': item.get('result', 'unknown'),
                    'is_correct': 1 if item.get('result') == 'correct' else 0
                })
            
            summary_df = pd.DataFrame(csv_data)
            summary_df.to_csv(csv_path, index=False)
            
            logger.debug(f"Saved summary to: {csv_path}")
            return csv_path
            
        except Exception as e:
            logger.warning(f"Could not save CSV summary: {e}")
            return None
    
    @staticmethod
    def _print_summary(
        model_args: ModelArguments,
        data_args: DataArguments,
        accuracy: float,
        total_correct: int,
        total_examples: int,
        json_path: Path,
        csv_path: Optional[Path]
    ):
        logger.info("=" * 60)
        logger.info("EVALUATION RESULTS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Model: {model_args.model}")
        logger.info(f"Dataset: {data_args.dataset}")
        logger.info(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        logger.info(f"Total examples: {total_examples}")
        logger.info(f"Correct predictions: {total_correct}")
        logger.info(f"Incorrect predictions: {total_examples - total_correct}")
        logger.info(f"Results saved to: {json_path}")
        
        if csv_path:
            logger.info(f"Summary saved to: {csv_path}")
        
        if hasattr(model_args, 'parameter_efficient_mode'):
            logger.info(f"Parameter efficient mode: {model_args.parameter_efficient_mode}")
        
        logger.info("=" * 60)
    
    @staticmethod
    def save_rl_specific_info(
        checkpoint_dir: Path,
        output_dir: Path,
        results: Dict[str, Any]
    ) -> Path:
        rl_info_path = output_dir / "rl_training_info.json"
        
        try:
            training_args_path = checkpoint_dir / "training_args.json"
            if training_args_path.exists():
                with open(training_args_path, "r") as f:
                    training_args = json.load(f)
                
                rl_info = {
                    "checkpoint_dir": str(checkpoint_dir),
                    "training_args": training_args,
                    "evaluation_results": {
                        "timestamp": datetime.now().isoformat(),
                        **results
                    }
                }
                
                with open(rl_info_path, "w") as f:
                    json.dump(rl_info, f, indent=4, default=str)
                
                logger.info(f"Saved RL-specific info to: {rl_info_path}")
            
            return rl_info_path
            
        except Exception as e:
            logger.warning(f"Could not save RL-specific info: {e}")
            return None

def is_multiturn_dataset(dataset) -> bool:
    if hasattr(dataset, 'output_format') and dataset.output_format == 'messages':
        return True

    if len(dataset) > 0:
        sample = dataset[0]
        if isinstance(sample, dict) and 'messages' in sample:
            return True
    return False

def evaluate_multiturn(
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    dataset: torch.utils.data.Dataset,
    model_args: ModelArguments,
    generation_kwargs: Dict[str, Any]
) -> Tuple[int, int, List[Dict]]:
    total_correct = 0
    total_examples = len(dataset)
    all_outputs = []

    generation_kwargs.setdefault('max_new_tokens', 128)

    for idx in tqdm(range(total_examples), desc="Evaluating multi‑turn"):
        sample = dataset[idx]
        messages = sample["messages"]

        context_messages = messages[:-1]
        ground_truth = messages[-1]["content"]

        try:
            input_text = tokenizer.apply_chat_template(
                context_messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            logger.warning(f"Failed to apply chat template for sample {idx}: {e}")
            input_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in context_messages])
            input_text += "\nassistant: "

        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=model_args.max_length
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)

        generated = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        cleaned = BatchEvaluator._clean_generated_texts([generated], dataset)[0]

        if SHOW_EVAL:
            logger.info(f"Sample {idx}:\n  Target: {ground_truth[:200]}\n  Generated: {cleaned[:200]}")

        if hasattr(dataset, 'is_correct'):
            is_correct = dataset.is_correct(cleaned, ground_truth)
        else:
            is_correct = cleaned.strip() == ground_truth.strip()

        if is_correct:
            total_correct += 1
        else:
            if SHOW_EVAL:
                logger.info(f"Mismatch at sample {idx}: target and generated differ.")

        all_outputs.append({
            'input': input_text,
            'target': ground_truth,
            'generated_text': cleaned,
            'result': 'correct' if is_correct else 'wrong'
        })

    return total_correct, total_examples, all_outputs

def evaluate_interactive(
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    dataset: torch.utils.data.Dataset,
    model_args: ModelArguments,
    generation_kwargs: Dict[str, Any]
) -> Tuple[int, int, List[Dict]]:
    total_success = 0
    total_episodes = len(dataset)
    all_outputs = []

    max_steps = getattr(dataset, 'max_steps', 50)
    generation_kwargs.setdefault('max_new_tokens', 128)

    completion_prefixes = dataset.get_completion_action_prefixes()
    logger.info(f"Using completion action prefixes: {completion_prefixes}")

    for idx in tqdm(range(total_episodes), desc="Interactive evaluation"):
        sample = dataset[idx]
        task_id = sample.get("task_id", f"task_{idx}")
        goal = sample.get("goal", "")
        split = getattr(dataset, 'split', 'test')

        try:
            env = dataset.create_interactive_env(sample, split)
            obs_list, info_dict = env.reset()
            obs = obs_list[0]
            info = {k: v[0] for k, v in info_dict.items()}
        except Exception as e:
            logger.error(f"Failed to initialize environment for task {task_id}: {e}")
            all_outputs.append({
                'task_id': task_id,
                'success': False,
                'error': str(e),
                'steps': 0,
                'history': [],
                'input': f"Task {task_id}: {goal}",
                'target': 'SUCCESS',
                'generated_text': f"Environment initialization failed: {str(e)[:100]}",
                'result': 'failure'
            })
            continue

        done = False
        step_count = 0
        history = []
        reward = 0

        target_obj, target_rec = dataset.get_goal_components(goal)

        while not done and step_count < max_steps:
            raw_admissible = info.get("admissible_commands", [])
            flat_commands = dataset.flatten_commands(raw_admissible)

            if not flat_commands:
                logger.warning(f"No admissible commands at step {step_count}. Breaking episode.")
                break

            if target_obj and target_rec and completion_prefixes:
                placement_commands = [
                    cmd for cmd in flat_commands
                    if any(cmd.startswith(prefix) for prefix in completion_prefixes)
                ]
                for cmd in placement_commands:
                    if target_obj.lower() in cmd.lower() and target_rec.lower() in cmd.lower():
                        action = cmd
                        logger.info(f"Auto-selected placement action: {action}")
                        try:
                            obs_list, reward_list, done_list, info_dict = env.step([action])
                            obs = obs_list[0]
                            reward = reward_list[0]
                            done = done_list[0]
                            info = {k: v[0] for k, v in info_dict.items()} if info_dict else {}
                        except Exception as e:
                            logger.error(f"Placement shortcut failed for task {task_id}: {e}")
                            done = True
                            break
                        step_count += 1
                        history.append({
                            'observation': obs,
                            'action': action,
                            'admissible_commands': flat_commands.copy()
                        })
                        break
                if done:
                    break

            prompt = dataset.build_prompt(obs, goal, flat_commands, history)

            if SHOW_PROMPT:
                logger.info(f"Prompt:\n{prompt}")

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=model_args.max_length
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(**inputs, **generation_kwargs)

            generated_text = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )

            action = dataset.parse_action(generated_text, flat_commands)
            if action is None:
                logger.debug(f"Failed to parse action from: {generated_text[:100]}")
                if flat_commands:
                    action = random.choice(flat_commands)
                    logger.warning(f"Using random fallback action: {action}")
                else:
                    break

            try:
                obs_list, reward_list, done_list, info_dict = env.step([action])
                obs = obs_list[0]
                reward = reward_list[0]
                done = done_list[0]
                info = {k: v[0] for k, v in info_dict.items()}
            except Exception as e:
                logger.error(f"Environment step failed for task {task_id}: {e}")
                break

            history.append({
                'observation': obs,
                'action': action,
                'admissible_commands': flat_commands.copy()
            })
            step_count += 1

        success = (reward == 1)

        if not success and hasattr(dataset, 'infer_success_from_obs'):
            if dataset.infer_success_from_obs(obs):
                success = True
                logger.info(f"Task {task_id} inferred as successful from observation.")

        if success:
            total_success += 1
        elif SHOW_TRAJECTORY_ON_FAIL:
            logger.info(f"Failed trajectory for task {task_id}:\n{format_trajectory(history, goal)}")

        all_outputs.append({
            'task_id': task_id,
            'goal': goal,
            'success': success,
            'steps': step_count,
            'history': history,
            'max_steps_reached': step_count >= max_steps and not done,
            'input': goal,
            'target': 'SUCCESS' if success else 'FAILURE',
            'generated_text': f"Episode completed in {step_count} steps",
            'result': 'success' if success else 'failure'
        })

        if hasattr(env, 'close'):
            env.close()

    return total_success, total_episodes, all_outputs

def evaluate() -> float:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    if model_args.model is None:
        logger.error("--model is required for evaluation")
        raise ValueError("model is required")

    if data_args.dataset is None:
        logger.error("--dataset is required for evaluation")
        raise ValueError("dataset is required")

    try:
        config = load_config(model_args.config if hasattr(model_args, 'config') else None)
        logger.info("Configuration loaded successfully")
    except FileNotFoundError as e:
        logger.warning(f"Configuration file not found: {e}")
        config = {'eval': {}}
        logger.info("Using default configuration")

    model_args = update_dataclass_from_config(model_args, config, ['common', 'eval'])
    data_args = update_dataclass_from_config(data_args, config, ['common', 'eval'])

    if not data_args.batch_size:
        data_args.batch_size = config['common']['batch_size']

    setup_directories(config)

    if model_args.output_dir is None:
        model_args.output_dir = str(DEFAULT_EVAL_OUTPUT_DIR)
        logger.info(f"Using default output directory: {model_args.output_dir}")

    if model_args.hf_hub_token:
        try:
            login(token=model_args.hf_hub_token)
            logger.info("Logged in to Hugging Face Hub")
        except Exception as e:
            logger.warning(f"Failed to login to Hugging Face Hub: {e}")

    model_name, input_embedding_file, output_embedding_file, checkpoint_dir = ModelPathResolver.resolve(model_args)

    logger.info(f"Resolved model name: {model_name}")

    is_trl_trained_model = False

    if model_name in RESERVED_MODELS:
        logger.info(f"Loading reserved model: {model_name}")
        model = ModelLoader.load(model_args, model_name)
        tokenizer = model.tokenizer
    else:
        if checkpoint_dir and checkpoint_dir.exists():
            has_training_args = (checkpoint_dir / "training_args.json").exists()
            has_adapter_config = (checkpoint_dir / "adapter_config.json").exists()
            has_adapter_model = (
                (checkpoint_dir / "adapter_model.safetensors").exists() or
                (checkpoint_dir / "adapter_model.bin").exists()
            )
            if has_training_args and has_adapter_config and has_adapter_model:
                is_trl_trained_model = True
                logger.info("Detected RL-trained model (standard PEFT format)")

        tokenizer_path = str(checkpoint_dir) if checkpoint_dir and checkpoint_dir.exists() else model_name
        logger.info(f"Loading tokenizer from: {tokenizer_path}")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            padding_side="left"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")

        cot_tokens = []
        num_new_tokens = 0
        if not is_trl_trained_model:
            try:
                cot_tokens = create_cot_tokens(config, tokenizer)
                if cot_tokens:
                    logger.info(f"Created {len(cot_tokens)} CoT delimiters")
                from transformers import AutoConfig
                orig_config = AutoConfig.from_pretrained(model_name)
                orig_vocab_size = orig_config.vocab_size
                num_new_tokens = max(len(cot_tokens), len(tokenizer) - orig_vocab_size)
                logger.info(f"Model base vocab size: {orig_vocab_size}, tokenizer final size: {len(tokenizer)} -> need {num_new_tokens} new tokens")
            except Exception as e:
                logger.warning(f"Failed to create CoT delimiters: {e}")
                cot_tokens = []
        else:
            logger.info("CoT delimiters disabled")

        logger.info("Loading model...")
        model = ModelLoader.load(
            model_args=model_args,
            model_name=model_name,
            input_embedding_file=input_embedding_file,
            output_embedding_file=output_embedding_file,
            num_new_tokens=num_new_tokens,
            checkpoint_dir=checkpoint_dir
        )

        logger.info("=" * 60)
        logger.info("MODEL CONFIGURATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"PEFT Model Class: {type(model).__name__}")
        logger.info(f"TRL Fine-tuning Status: {is_trl_trained_model}")
        logger.info(f"Chain-of-Thought Delimiters: {num_new_tokens}")
        logger.info(f"Active Device: {next(model.parameters()).device}")
        logger.info("=" * 60)

    model.eval()
    logger.info("Model set to evaluation mode")

    try:
        data_class = DatasetManager.get_data_class(data_args.dataset)
        logger.info(f"Using dataset class: {data_class.__name__}")
    except ValueError as e:
        logger.error(f"Dataset error: {e}")
        raise

    data_config = DataConfig()
    try:
        datasets_config = load_datasets_config()
        data_config.dataset = datasets_config.get(data_args.dataset, {})
    except Exception as e:
        logger.warning(f"Failed to load dataset config: {e}")
        data_config.dataset = {}

    logger.info(f"Loading dataset: {data_args.dataset}")
    try:
        dataset = DatasetManager.create_dataset(
            data_class,
            data_args.dataset,
            "test",
            data_config
        )
        logger.info(f"Dataset loaded with {len(dataset)} examples")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    if data_args.num_test and len(dataset) > data_args.num_test:
        logger.info(f"Sampling {data_args.num_test} examples from dataset")
        dataset = DatasetManager.sample_dataset(
            dataset, data_args.num_test, data_args.seed
        )
        logger.info(f"Sampled dataset has {len(dataset)} examples")

    generation_kwargs = {}
    if model_args.temperature is not None:
        generation_kwargs["temperature"] = model_args.temperature
    if model_args.top_p is not None:
        generation_kwargs["top_p"] = model_args.top_p
    if model_args.top_k is not None:
        generation_kwargs["top_k"] = model_args.top_k
    if model_args.num_beams is not None:
        generation_kwargs["num_beams"] = model_args.num_beams
    if model_args.do_sample is not None:
        generation_kwargs["do_sample"] = model_args.do_sample

    if model_args.generation_config:
        try:
            if Path(model_args.generation_config).exists():
                with open(model_args.generation_config, 'r') as f:
                    file_config = json.load(f)
                    generation_kwargs.update(file_config)
                logger.info(f"Loaded generation config from file: {model_args.generation_config}")
            else:
                file_config = json.loads(model_args.generation_config)
                generation_kwargs.update(file_config)
                logger.info("Loaded generation config from JSON string")
        except Exception as e:
            logger.warning(f"Failed to load generation config: {e}")

    if generation_kwargs:
        logger.info("Generation configuration:")
        for key, value in generation_kwargs.items():
            logger.info(f"  {key}: {value}")

    if is_multiturn_dataset(dataset):
        if model_args.interactive:
            logger.info("Using interactive environment evaluation (multi‑turn)")
            total_correct, total_examples, all_outputs = evaluate_interactive(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                model_args=model_args,
                generation_kwargs=generation_kwargs,
            )
        else:
            logger.info("Using offline multi‑turn evaluation")
            total_correct, total_examples, all_outputs = evaluate_multiturn(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                model_args=model_args,
                generation_kwargs=generation_kwargs,
            )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=data_args.batch_size,
            shuffle=False,
            num_workers=0
        )
        logger.info(f"Created dataloader with batch size {data_args.batch_size}")

        total_correct = 0
        total_examples = 0
        all_outputs = []

        with tqdm(dataloader, desc="Evaluation", unit="batch") as progress_bar:
            for batch_idx, batch in enumerate(progress_bar):
                x_text, y_text = batch['x'], batch['y']
                try:
                    batch_correct, batch_size, batch_outputs = BatchEvaluator.evaluate(
                        model=model,
                        model_name=model_name,
                        tokenizer=tokenizer,
                        x_text=x_text,
                        y_text=y_text,
                        dataset=dataset,
                        max_length=model_args.max_length,
                        decoding_scheme=model_args.decoding_scheme,
                        generation_kwargs=generation_kwargs
                    )
                except Exception as e:
                    logger.error(f"Error evaluating batch {batch_idx}: {e}")
                    continue

                total_correct += batch_correct
                total_examples += batch_size
                all_outputs.extend(batch_outputs)

                current_accuracy = total_correct / total_examples if total_examples > 0 else 0
                progress_bar.set_postfix({
                    'accuracy': f'{current_accuracy:.4f}',
                    'correct': f'{total_correct}/{total_examples}'
                })

    final_accuracy = total_correct / total_examples if total_examples > 0 else 0

    logger.info(f"Model type: {'TRL-trained PEFT' if is_trl_trained_model else 'Standard model'}")
    logger.info(f"Total examples evaluated: {total_examples}")
    logger.info(f"Correct predictions / Successes: {total_correct}")
    logger.info(f"Incorrect predictions / Failures: {total_examples - total_correct}")
    logger.info(f"Final accuracy / Success rate: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")

    if model_args.save_result:
        try:
            config['model_info'] = {
                'is_trl_trained': is_trl_trained_model,
                'model_type': type(model).__name__,
                'checkpoint_dir': str(checkpoint_dir) if checkpoint_dir else None
            }
            ResultSaver.save(
                model_args, data_args, all_outputs,
                total_correct, total_examples,
                config
            )
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    else:
        logger.info("Results saving disabled (save_result=False)")

    return final_accuracy

def evaluate_rl_checkpoint(checkpoint_path: str, dataset_name: str, **kwargs) -> float:
    logger.info(f"Evaluating RL checkpoint: {checkpoint_path}")
    
    original_argv = sys.argv
    
    try:
        args = ['--model', checkpoint_path, '--dataset', dataset_name]
        
        if 'batch_size' in kwargs:
            args.extend(['--batch_size', str(kwargs['batch_size'])])
        if 'num_test' in kwargs:
            args.extend(['--num_test', str(kwargs['num_test'])])
        if 'temperature' in kwargs:
            args.extend(['--temperature', str(kwargs['temperature'])])
        if 'output_dir' in kwargs:
            args.extend(['--output_dir', kwargs['output_dir']])
        
        sys.argv = ['evaluate.py'] + args
        
        return evaluate()
    finally:
        sys.argv = original_argv

if __name__ == "__main__":
    try:
        accuracy = evaluate()
        sys.exit(0 if accuracy > 0 else 1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
