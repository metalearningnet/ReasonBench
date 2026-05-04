import os
from config import logger, load_rl_config
from trl.experimental.cpo import CPOTrainer, CPOConfig
from training.rl import RLConfig, RLTrainer, RLPreprocessor

class RLCPOConfig(RLConfig):
    def update(self, args):
        args.update({
            "dataset_name": self.dataset_name,
            "beta": self.rl_config.get("beta", 0.1),
            "loss_type": self.rl_config.get("loss_type", "sigmoid"),
            "per_device_train_batch_size": max(1, self.batch_size),
            "per_device_eval_batch_size": max(1, self.batch_size),
            "field_mappings": self.field_mappings,
            "eval_split": self.rl_config.get("eval_split", None)
        })
        return args

class RLCPOPreprocessor(RLPreprocessor):
    def process(self, dataset, limit=None):
        field_mappings = self.training_args.field_mappings
        if field_mappings is None:
            raise ValueError("CPO field mappings must be provided in training_args.")
        
        logger.info(f"CPO field mappings: {field_mappings}")
        required_columns = list(field_mappings.values())
        missing_columns = [col for col in required_columns if col not in dataset.column_names]
        
        if missing_columns:
            raise ValueError(
                f"CPO dataset missing required columns: {missing_columns}. "
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
        
        def clean_batch(batch):
            for key in ["prompt", "chosen", "rejected"]:
                batch[key] = [
                    str(val).strip() if val is not None else ""
                    for val in batch.get(key, [])
                ]
            return batch
        
        dataset = dataset.map(
            clean_batch,
            batched=True,
            num_proc=num_proc,
            desc="Cleaning CPO dataset"
        )
        
        initial_size = len(dataset)
        
        def filter_batch(batch):
            return [
                bool(p and c and r and c != r)
                for p, c, r in zip(batch["prompt"], batch["chosen"], batch["rejected"])
            ]
        
        dataset = dataset.filter(
            filter_batch,
            batched=True,
            num_proc=num_proc,
            desc="Filtering invalid CPO pairs"
        )
        filtered_size = len(dataset)
        
        if filtered_size < initial_size:
            logger.warning(f"Filtered out {initial_size - filtered_size} examples with empty or identical responses.")
        
        effective_limit = limit if limit is not None else getattr(self.training_args, 'num_train', None)
        if effective_limit is not None and effective_limit > 0:
            effective_limit = min(effective_limit, len(dataset))
            dataset = dataset.select(range(effective_limit))
            logger.info(f"Limited dataset to {effective_limit} examples.")
        
        logger.info(f"Processed CPO dataset with {len(dataset)} examples.")
        if len(dataset) > 0:
            logger.info(f"Sample prompt: {dataset[0]['prompt'][:100]}...")
            logger.info(f"Sample chosen: {dataset[0]['chosen'][:100]}...")
            logger.info(f"Sample rejected: {dataset[0]['rejected'][:100]}...")
        
        return dataset

class RLCPOTrainer(RLTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        rl_config = load_rl_config("cpo")
        cpo_alpha = rl_config.get("cpo_alpha", 1.0)
        
        cpo_config = CPOConfig(
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
            cpo_alpha=cpo_alpha,
            loss_type=self.args.loss_type,
            max_length=self.args.max_prompt_length + self.args.max_completion_length
        )
        
        self.trltrainer = CPOTrainer(
            model=self.model,
            args=cpo_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer
        )
        
        logger.info(f"RLCPOTrainer initialized with beta={self.args.beta}, cpo_alpha={cpo_alpha}, loss_type={self.args.loss_type}")
    
    def train(self):
        try:
            train_result = self.trltrainer.train()
            logger.info("CPO training completed successfully")
            
            if self.args.checkpoint_dir:
                self.save_model(self.args.checkpoint_dir)
            
            return {
                "train_metrics": train_result.metrics,
                "output_dir": self.args.checkpoint_dir
            }
        except Exception as e:
            logger.error(f"Error during CPO training: {e}")
            raise
