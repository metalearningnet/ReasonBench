import torch
import dataset
from config import logger
from transformers import Trainer
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional, Dict, Any, Tuple
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.trainer_utils import EvalLoopOutput, has_length

class LMTrainer(Trainer):
    def __init__(self, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
    
    def get_eval_dataloader(
        self,
        eval_dataset: Optional[Dataset] = None
    ) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if isinstance(eval_dataset, dataset.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last

        return self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))
    
    def _setup_model_for_evaluation(
        self,
        dataloader: DataLoader
    ) -> torch.nn.Module:
        model = self._wrap_model(
            self.model,
            training=False,
            dataloader=dataloader
        )

        if len(self.accelerator._models) == 0 and model is self.model:
            if self.is_deepspeed_enabled:
                model = self.accelerator.prepare(model)
            else:
                model = self.accelerator.prepare_model(model, evaluation_mode=True)

            if self.is_fsdp_enabled:
                self.model = model

            if model is not self.model:
                self.model_wrapped = model

        if not self.is_in_train:
            if self.args.fp16_full_eval:
                model = model.to(
                    dtype=torch.float16,
                    device=self.args.device
                )
            elif self.args.bf16_full_eval:
                model = model.to(
                    dtype=torch.bfloat16,
                    device=self.args.device
                )

        return model

    def _process_batch(
        self,
        model: torch.nn.Module,
        batch: Dict[str, Any]
    ) -> Tuple[List[str], List[str], List[str]]:
        if "x" in batch:
            inputs = batch["x"]
        elif "input_ids" in batch:
            input_ids = batch["input_ids"]
            
            inputs = self.tokenizer.batch_decode(
                input_ids,
                skip_special_tokens=True
            )
        else:
            raise ValueError(f"Batch must contain either 'x' or 'input_ids'. Keys: {batch.keys()}")
        
        encoding = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors='pt'
        )
        
        encoding = self._prepare_inputs(encoding)
        input_length = encoding['input_ids'].size(1)
        
        generation_config = {
            'max_new_tokens': 512,
            'num_beams': 1,
            'do_sample': False,
            'temperature': 1.0,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id
        }
        
        with torch.no_grad():
            generated_ids = model.generate(
                **encoding,
                **generation_config
            )
        
        generated_texts = self.tokenizer.batch_decode(
            generated_ids[:, input_length:],
            skip_special_tokens=True
        )
        
        if "y" in batch:
            labels = batch["y"]
        elif "labels" in batch:
            labels = self.tokenizer.batch_decode(
                batch["labels"],
                skip_special_tokens=True
            )
        else:
            labels = [""] * len(inputs)
        
        return generated_texts, labels, inputs

    def _calculate_accuracy(
        self,
        predictions: List[str],
        inputs: List[str],
        labels: List[str],
        dataset: Dataset
    ) -> Tuple[float, List[Tuple[str, str, str]]]:
        if len(predictions) != len(labels) or len(predictions) != len(inputs):
            raise ValueError(
                f"Mismatched lengths: predictions={len(predictions)}, "
                f"labels={len(labels)}, inputs={len(inputs)}"
            )
        
        num_correct = 0
        outputs = []
        
        for pred, x, y in zip(predictions, inputs, labels):
            pred_str, x_str, y_str = str(pred), str(x), str(y)
            
            if dataset.is_correct(pred_str, y_str):
                num_correct += 1
            
            outputs.append((pred_str, x_str, y_str))
        
        accuracy = num_correct / len(predictions) if predictions else 0.0
        
        return accuracy, outputs

    def _get_num_samples(
        self,
        dataloader: DataLoader,
        eval_dataset: Dataset
    ) -> int:
        if has_length(eval_dataset):
            return len(eval_dataset)
        
        if isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            return eval_dataset.num_examples
        
        if has_length(dataloader):
            return self.num_examples(dataloader)
        
        return 0

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval"
    ) -> EvalLoopOutput:
        args = self.args
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only
        
        model = self._setup_model_for_evaluation(dataloader)
        
        logger.info(f"Running {description}")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {args.eval_batch_size}")
        
        model.eval()
        
        self.callback_handler.eval_dataloader = dataloader
        
        eval_dataset = getattr(dataloader, "dataset", None)
        
        if args.past_index >= 0:
            self._past = None
        
        all_predictions: List[str] = []
        all_labels: List[str] = []
        all_inputs: List[str] = []
        
        observed_num_examples = 0
        
        for step, batch in enumerate(dataloader):
            observed_batch_size = len(batch) if batch else 0
            observed_num_examples += observed_batch_size
            
            predictions, labels, inputs = self._process_batch(model, batch)
            
            all_predictions.extend(predictions)
            all_labels.extend(labels)
            all_inputs.extend(inputs)
            
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)
        
        if args.past_index and hasattr(self, "_past"):
            delattr(self, "_past")
        
        num_samples = self._get_num_samples(dataloader, eval_dataset)
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples
        
        accuracy, outputs = self._calculate_accuracy(
            all_predictions, all_inputs, all_labels, dataloader.dataset
        )
        
        metrics = {f"{metric_key_prefix}_acc": accuracy}
        
        logger.info(f"Evaluation accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        logger.info(f"Total examples evaluated: {len(all_predictions)}")
        
        return EvalLoopOutput(
            predictions=all_predictions,
            label_ids=all_labels,
            metrics=metrics,
            num_samples=num_samples
        )
