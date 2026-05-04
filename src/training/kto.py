import os
import copy
from peft import PeftModel
from trl import KTOTrainer, KTOConfig
from config import logger, load_rl_config
from training.rl import RLConfig, RLTrainer, RLPreprocessor

class RLKTOConfig(RLConfig):
    def update(self, args):
        args.update({
            "dataset_name": self.dataset_name,
            "beta": self.rl_config.get("beta", 0.1),
            "desirable_weight": self.rl_config.get("desirable_weight", 1.0),
            "undesirable_weight": self.rl_config.get("undesirable_weight", 1.0),
            "per_device_train_batch_size": max(1, self.batch_size),
            "per_device_eval_batch_size": max(1, self.batch_size),
            "field_mappings": self.field_mappings,
            "eval_split": self.rl_config.get("eval_split", None)
        })
        return args

class RLKTOPreprocessor(RLPreprocessor):
    def process(self, dataset, limit=None):
        field_mappings = self.training_args.field_mappings
        if field_mappings is None:
            raise ValueError("KTO field mappings must be provided in training_args.")
        
        logger.info(f"KTO field mappings: {field_mappings}")
        required_columns = list(field_mappings.values())
        missing_columns = [col for col in required_columns if col not in dataset.column_names]
        
        if missing_columns:
            raise ValueError(
                f"Dataset missing required columns: {missing_columns}. "
                f"Available columns: {dataset.column_names}"
            )
        
        rename_dict = {
            source: target for target, source in field_mappings.items()
            if source != target
        }
        
        if rename_dict:
            dataset = dataset.rename_columns(rename_dict)
            logger.info(f"Renamed dataset columns: {rename_dict}")
        
        num_proc = min(8, os.cpu_count() or 1)
        
        def unpair_batch(batch):
            prompts = []
            completions = []
            labels = []
            
            for p, c, r in zip(batch["prompt"], batch["chosen"], batch["rejected"]):
                p, c, r = str(p).strip(), str(c).strip(), str(r).strip()
                if not p or not c or not r or c == r:
                    continue
                
                prompts.append(p)
                completions.append(c)
                labels.append(True)
                
                prompts.append(p)
                completions.append(r)
                labels.append(False)
            
            return {
                "prompt": prompts,
                "completion": completions,
                "label": labels
            }
        
        initial_size = len(dataset)
        
        dataset = dataset.map(
            unpair_batch,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=num_proc
        )
        
        unpaired_size = len(dataset)
        logger.info(f"Unpaired {initial_size} DPO pairs into {unpaired_size} KTO individual examples.")
        
        effective_limit = limit if limit is not None else getattr(self.training_args, 'num_train', None)
        if effective_limit is not None and effective_limit > 0:
            effective_limit = min(effective_limit, len(dataset))
            dataset = dataset.select(range(effective_limit))
            logger.info(f"Limited dataset to {effective_limit} examples.")
        
        if len(dataset) > 0:
            logger.info(f"Sample prompt: {dataset[0]['prompt'][:100]}...")
            logger.info(f"Sample completion: {dataset[0]['completion'][:100]}...")
            logger.info(f"Sample label: {dataset[0]['label']}")
        
        return dataset

class RLKTOTrainer(RLTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        is_peft = isinstance(self.model, PeftModel) or hasattr(self.model, "peft_config")
        
        if is_peft:
            logger.info("PEFT model detected. Setting ref_model=None to save VRAM.")
            ref_model = None
        else:
            logger.info("Full fine-tuning detected. Deepcopying model for reference.")
            ref_model = copy.deepcopy(self.model)
            for param in ref_model.parameters():
                param.requires_grad = False
        
        rl_config = load_rl_config("kto")
        desirable_weight = rl_config.get("desirable_weight", 1.0)
        undesirable_weight = rl_config.get("undesirable_weight", 1.0)
        
        kto_config = KTOConfig(
            output_dir=self.args.output_dir,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            per_device_eval_batch_size=self.args.per_device_eval_batch_size,
            fp16=self.args.fp16,
            bf16=self.args.bf16,
            learning_rate=self.args.learning_rate,
            num_train_epochs=self.args.num_train_epochs,
            logging_steps=self.args.logging_steps,
            save_steps=self.args.save_steps,
            eval_steps=self.args.eval_steps,
            save_total_limit=self.args.save_total_limit,
            warmup_steps=self.args.warmup_steps,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            seed=self.args.seed,
            beta=self.args.beta,
            desirable_weight=desirable_weight,
            undesirable_weight=undesirable_weight,
            max_length=self.args.max_prompt_length + self.args.max_completion_length
        )
        
        self.trltrainer = KTOTrainer(
            model=self.model,
            ref_model=ref_model,
            args=kto_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer
        )
        
        logger.info(f"RLKTOTrainer initialized with beta={self.args.beta}, desirable={desirable_weight}, undesirable={undesirable_weight}")
    
    def train(self):
        try:
            train_result = self.trltrainer.train()
            logger.info("KTO training completed successfully")
            
            if self.args.checkpoint_dir:
                self.save_model(self.args.checkpoint_dir)
            
            return {
                "train_metrics": train_result.metrics,
                "output_dir": self.args.checkpoint_dir
            }
        except Exception as e:
            logger.error(f"Error during KTO training: {e}")
            raise
