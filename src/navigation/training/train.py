import logging
import os
import pickle

import hydra
import torch
from dataloader import MultiChoiceDataset, get_data_split
from hydra.core.hydra_config import HydraConfig
from metric import ActionEvaluatorGeneration, ActionEvaluatorMultiChoice
from omegaconf import DictConfig
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main_process = local_rank in [-1, 0]

    if not is_main_process:
        logging.disable(logging.INFO)

    logger.info(f"Use model {cfg.model.model_name_or_path}")
    logger.info(f"Task mode: {cfg.task_mode}")
    logger.info(f"Running on {world_size} GPU(s)")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)

    candidate_results = None
    if cfg.data.score_file and os.path.exists(cfg.data.score_file):
        with open(cfg.data.score_file, "rb") as f:
            candidate_results = pickle.load(f)
            logger.info(f"Loaded candidate scores from {cfg.data.score_file}")

    output_dir = HydraConfig.get().runtime.output_dir

    train_data = get_data_split(
        cfg.data.base_path,
        cfg.data.train_split,
        is_train=True,
        candidate_results=candidate_results,
    )

    train_dataset = MultiChoiceDataset(
        train_data,
        tokenizer,
        neg_ratio=cfg.train.neg_ratio,
        num_candidates=cfg.train.num_candidates,
        max_context_len=cfg.train.max_context_len,
        mode=cfg.model_mode,
        task_mode=cfg.task_mode,
    )

    model_kwargs = {}

    if cfg.train.get('qlora', False):
        logger.info("Using QLora (4bit quantization)")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = quantization_config

        # Standard pattern for QLora device mapping
        device_map = "auto"
        ddp = world_size != 1
        if ddp:
            device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        model_kwargs["device_map"] = device_map

    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model.model_name_or_path, **model_kwargs)

    if is_main_process:
        if hasattr(model, 'hf_quantizer') and model.hf_quantizer is not None:
            logger.info(f"Model is quantized with: {model.hf_quantizer}")
            
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total parameters: {total_params:,}")
        
        param_dtypes = {}
        for _, param in model.named_parameters():
            dtype_str = str(param.dtype)
            param_dtypes[dtype_str] = param_dtypes.get(dtype_str, 0) + param.numel()
        logger.info("Parameter dtypes:")
        for dtype, count in param_dtypes.items():
            logger.info(f"  {dtype}: {count:,} parameters ({count/total_params*100:.1f}%)")

    if cfg.train.lora:
        if hasattr(cfg.model, 'lora'):
            lora_config = LoraConfig(
                r=cfg.model.lora.r,
                lora_alpha=cfg.model.lora.alpha,
                target_modules=["q", "v"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM,
            )
        else:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q", "v"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM,
            )
        model = get_peft_model(model, lora_config)
        if is_main_process:
            model.print_trainable_parameters()

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    num_gpus = world_size if world_size > 0 else 1
    steps_per_epoch = len(train_dataset) // (
        cfg.train.per_device_train_batch_size * num_gpus * cfg.train.gradient_accumulation_steps
    )
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Total training examples: {len(train_dataset)}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.train.per_device_eval_batch_size,
        dataloader_num_workers=2,
        predict_with_generate=True,
        fp16=False,
        bf16=cfg.train.bf16,
        tf32=cfg.train.tf32,
        learning_rate=cfg.train.learning_rate,
        num_train_epochs=cfg.train.epoch,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=int(steps_per_epoch * 0.2),
        save_strategy="epoch",
        save_total_limit=2,
        optim=cfg.train.optim,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        report_to="tensorboard",
        ddp_find_unused_parameters=False
    )

    if cfg.model_mode == "multichoice":
        evaluator = ActionEvaluatorMultiChoice(tokenizer)
    else:
        evaluator = ActionEvaluatorGeneration(tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        compute_metrics=evaluator,
    )

    trainer.train()

    if is_main_process:
        trainer.save_state()
        trainer.save_model(output_dir=output_dir)
        logger.info(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()