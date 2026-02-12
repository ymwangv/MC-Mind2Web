import json
import logging
import os
import re
from tqdm import tqdm

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel

from dataloader import ClarificationDataset
from metric import calculate_metrics, calculate_split_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(cfg: DictConfig):
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_name_or_path,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = None
    if cfg.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    quantization_kwargs = {"quantization_config": bnb_config} if bnb_config else {}

    if cfg.use_tuned and cfg.model_path:
        if cfg.use_lora:
            logger.info(f"Loading base model: {cfg.model.model_name_or_path}")
            base_model = AutoModelForCausalLM.from_pretrained(
                cfg.model.model_name_or_path,
                device_map="auto",
                torch_dtype=dtype,
                trust_remote_code=True,
                **quantization_kwargs
            )
            logger.info(f"Loading tuned LoRA adapter from {cfg.model_path}")
            model = PeftModel.from_pretrained(base_model, cfg.model_path)
            model = model.merge_and_unload()
        else:
            logger.info(f"Loading tuned full model from {cfg.model_path}")
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_path,
                device_map="auto",
                torch_dtype=dtype,
                trust_remote_code=True,
                **quantization_kwargs
            )
    else:
        logger.info(f"Loading original base model: {cfg.model.model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.model_name_or_path,
            device_map="auto",
            torch_dtype=dtype,
            trust_remote_code=True,
            **quantization_kwargs
        )

    model.eval()
    return model, tokenizer


def extract_yes_no(text):
    if text.startswith("assistant"):
        text = text[len("assistant"):].strip()
    
    text = text.strip().lower()

    match = re.search(r'\b(yes|no)\b', text, re.IGNORECASE)
    if match:
        return match.group(1).lower() == "yes"

    return False


def extract_question(text):
    if text.startswith("assistant"):
        text = text[len("assistant"):].strip()
    
    return text.strip()
    # return text.split("\n")[0].strip() 


def generate_response(model, tokenizer, system_prompt, user_prompt, max_length, cfg, prompt_type="ans"):
    assistant_prompt = "Answer:" if prompt_type == "ans" else "Answer: Yes. Question:"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False
    ).to(model.device)

    with torch.no_grad():
        if cfg.eval.do_sample:
            outputs = model.generate(
                **inputs,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=cfg.eval.max_new_tokens,
                temperature=cfg.eval.temperature,
                top_p=cfg.eval.top_p,
                do_sample=True
            )
        else:
            outputs = model.generate(
                **inputs,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=cfg.eval.max_new_tokens,
                do_sample=False
            )

    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return output_text


@hydra.main(version_base=None, config_path="conf", config_name="eval_config")
def main(cfg: DictConfig):
    logger.info(f"Model: {cfg.model.model_name_or_path}")
    logger.info(f"Mode: {cfg.mode}, Env: {cfg.env}")

    output_dir = HydraConfig.get().runtime.output_dir
    logger.info(f"Output directory: {output_dir}")

    model, tokenizer = load_model_and_tokenizer(cfg)

    max_length_key = cfg.mode if cfg.mode == 'conv_only' else f"{cfg.mode}_{cfg.env}"
    max_length = cfg.model.max_length.get(max_length_key, 2048)
    logger.info(f"Max input length: {max_length} (key: {max_length_key})")

    all_results = []

    for split in cfg.splits:
        logger.info(f"\nEvaluating on {split} split...")

        test_dataset = ClarificationDataset(
            data_path=cfg.data_path,
            mode=cfg.mode,
            env=cfg.env if cfg.mode != 'conv_only' else None,
            split=split,
            tokenizer=tokenizer,
            processor=None,
            max_length=max_length
        )

        logger.info(f"Test samples: {len(test_dataset)}")

        split_results = []

        # for idx in tqdm(range(20), desc=f"Processing {split}"):
        for idx in tqdm(range(len(test_dataset)), desc=f"Processing {split}"):
            example = test_dataset.examples[idx]

            # First inference: Get Yes/No answer
            first_output = generate_response(
                model, tokenizer,
                example['system_prompt'],
                example['user_prompt'],
                max_length, cfg,
                prompt_type="ans"
            )

            # Extract Yes/No
            predicted_needs = extract_yes_no(first_output)

            predicted_question = ""
            second_output = ""

            # Second inference if needed: Get clarification question
            if predicted_needs:
                second_output = generate_response(
                    model, tokenizer,
                    example['system_prompt'],
                    example['user_prompt'],
                    max_length, cfg,
                    prompt_type="ques"
                )
                predicted_question = extract_question(second_output)

            # Process ground truth
            ground_truth = example['response']
            actual_needs = "Answer: Yes" in ground_truth
            actual_question = ""
            if actual_needs and "Question:" in ground_truth:
                actual_question = ground_truth.split("Question:", 1)[1].strip()

            result = {
                'sample_id': example['sample_id'],
                'annotation_id': example['annotation_id'],
                'action_id': example['action_id'],
                'predicted_needs_clarification': predicted_needs,
                'predicted_question': predicted_question,
                'actual_needs_clarification': actual_needs,
                'actual_question': actual_question,
                'first_output': first_output,
                'second_output': second_output,
                'split': split
            }

            split_results.append(result)

        all_results.extend(split_results)

        if True:
            split_output_file = os.path.join(output_dir, f"results_{split}.json")
            with open(split_output_file, 'w') as f:
                json.dump(split_results, f, indent=2)
            logger.info(f"Saved {split} results to {split_output_file}")

            calculate_split_metrics(split, output_dir)
            logger.info(f"Calculated metrics for {split} split")

    output_file = os.path.join(output_dir, "results_all.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved all results to {output_file}")

    metrics = calculate_metrics(all_results)

    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_file}")

    logger.info("\n=== Overall Metrics ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"{key}: {value:.4f}")
        else:
            logger.info(f"{key}: {value}")


if __name__ == "__main__":
    main()