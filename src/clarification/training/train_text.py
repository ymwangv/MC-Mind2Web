import logging
import os

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from torch.nn.utils.rnn import pad_sequence
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

from dataloader import ClarificationDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextDataCollator:
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        text_feats = []
        for f in features:
            text_item = {}
            for key in ("input_ids", "attention_mask"):
                if key in f:
                    v = f[key]
                    if isinstance(v, torch.Tensor):
                        v = v.tolist()
                    text_item[key] = v
            text_feats.append(text_item)

        batch = self.tokenizer.pad(
            text_feats,
            padding=True,
            # pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )

        label_tensors = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
        labels_padded = pad_sequence(label_tensors, batch_first=True, padding_value=-100)
        batch["labels"] = labels_padded

        return batch


def setup_model_and_tokenizer(model_name: str, use_4bit: bool = True, device_map=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    return model, tokenizer


def setup_lora(model, r=16, alpha=32, dropout=0.05):
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    # target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    return model


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    is_main_process = local_rank in [-1, 0]
    is_ddp = local_rank != -1

    if not is_main_process:
        logging.disable(logging.INFO)

    logger.info(f"Training text-only clarification model")
    logger.info(f"Model: {cfg.model.model_name_or_path}")

    mode = cfg.mode
    env = cfg.env

    logger.info(f"Mode: {mode}, Env: {env}")
    logger.info(f"Working directory: {HydraConfig.get().runtime.output_dir}")

    max_length_key = mode if mode == 'conv_only' else f"{mode}_{env}"
    max_length = cfg.model.max_length.get(max_length_key, 2048)
    logger.info(f"Max length: {max_length}")

    if is_ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    else:
        device_map = "auto"

    model, tokenizer = setup_model_and_tokenizer(cfg.model.model_name_or_path, cfg.use_4bit, device_map=device_map)

    if cfg.use_lora:
        lora_config = cfg[cfg.lora_config]
        model = setup_lora(model, lora_config.lora_r, lora_config.lora_alpha, lora_config.lora_dropout)
        logger.info(f"LoRA enabled ({cfg.lora_config}): r={lora_config.lora_r}, alpha={lora_config.lora_alpha}")

    train_dataset = ClarificationDataset(
        data_path=cfg.data_path,
        mode=mode,
        env=env,
        split=cfg.split,
        tokenizer=tokenizer,
        processor=None,
        max_length=max_length
    )

    logger.info(f"Train samples: {len(train_dataset)}")

    # Original data collator - trains on everything (WRONG)
    # data_collator = DataCollatorForLanguageModeling(
    #     tokenizer=tokenizer,
    #     mlm=False,
    #     pad_to_multiple_of=8
    # )

    # Use custom data collator that preserves masked labels
    data_collator = TextDataCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=8
    )

    output_dir = HydraConfig.get().runtime.output_dir

    effective_batch_size = world_size * cfg.train.per_device_train_batch_size * cfg.train.gradient_accumulation_steps

    logger.info(f"Batch size calculation:")
    logger.info(f"  Number of GPUs: {world_size}")
    logger.info(f"  Per device batch size: {cfg.train.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation steps: {cfg.train.gradient_accumulation_steps}")
    logger.info(f"  Effective global batch size: {effective_batch_size}")

    steps_per_epoch = len(train_dataset) // (cfg.train.per_device_train_batch_size * world_size * cfg.train.gradient_accumulation_steps)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,

        optim=cfg.train.optim,
        learning_rate=cfg.train.learning_rate,
        num_train_epochs=cfg.train.num_epochs,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        warmup_ratio=cfg.train.warmup_ratio,

        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=max(1, int(steps_per_epoch * 0.1)),

        save_strategy="epoch",
        save_total_limit=2,

        fp16=False,
        bf16=True,
        tf32=True,

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        ddp_find_unused_parameters=True,

        dataloader_num_workers=4,

        report_to="tensorboard"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    trainer.train()

    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    logger.info(f"Training completed. Model saved to {output_dir}")


if __name__ == "__main__":
    main()