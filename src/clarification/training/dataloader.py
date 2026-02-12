import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Optional
import warnings

import lxml.etree
from PIL import Image
from torch.utils.data import Dataset
from datasets import load_from_disk

sys.path.append(str(Path(__file__).parent.parent.parent))
from clarification.training.prompt import get_finetuning_prompt
from utils.dom_utils import prune_tree, get_tree_repr

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
MAX_PIXELS = 89_478_485


def generate_html_content(item, candidate_ranks, top_k=50):
    sample_id = f"{item['annotation_id']}_{item['action_uid']}"

    if sample_id not in candidate_ranks:
        return None

    candidate_rank_dict = candidate_ranks[sample_id]
    sorted_candidates = sorted(candidate_rank_dict.items(), key=lambda x: x[1])
    top_candidate_ids = [str(candidate_id) for candidate_id, _ in sorted_candidates[:top_k]]

    if not top_candidate_ids:
        return None

    dom_tree = lxml.etree.fromstring(item["cleaned_html"])
    pruned_tree = prune_tree(dom_tree, top_candidate_ids)

    tree_repr, _ = get_tree_repr(
        pruned_tree,
        id_mapping={},
        keep_html_brackets=True
    )

    return tree_repr


class ClarificationDataset(Dataset):

    def __init__(
        self,
        data_path: str,
        mode: str,
        env: Optional[str] = None,
        split: str = "train",
        top_k: int = 50,
        processor=None,
        tokenizer=None,
        max_length: int = 2048
    ):
        self.mode = mode
        self.env = env
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        dataset_path = os.path.join(data_path, split if split == "train" else f"test_{split}")
        dataset = load_from_disk(os.path.join(dataset_path, "dataset"))
        logger.info(f"Loading {len(dataset)} samples from {dataset_path}")

        candidate_ranks = {}
        if env in ["html", "both"]:
            candidate_results_file = os.path.join("eval", "candidate", "conv_clar", "scores_all.pkl")
            if os.path.exists(candidate_results_file):
                with open(candidate_results_file, 'rb') as f:
                    candidate_results = pickle.load(f)
                candidate_ranks = candidate_results.get("ranks", {})
                logger.info(f"Loaded candidate ranks for HTML processing")

        screenshot_dir = os.path.join(dataset_path, "screenshots")
        processed_count = 0
        skipped_count = 0

        for item in dataset:
            annotation_id = item.get('annotation_id')
            action_id = item.get('action_uid')
            sample_id = f"{annotation_id}_{action_id}"

            screenshot_file = os.path.join(screenshot_dir, f"{sample_id}.jpg")
            if not os.path.exists(screenshot_file):
                # logger.info("No screenshot")
                skipped_count += 1
                continue
            
            try:
                with Image.open(screenshot_file) as img:
                    if img.width * img.height > MAX_PIXELS:
                        logger.warning(
                            f"[Skip] Oversized image detected: {screenshot_file} "
                            f"({img.width}x{img.height}={img.width*img.height} pixels)"
                        )
                        skipped_count += 1
                        continue
            except Exception as e:
                logger.warning(f"[Skip] Cannot open image {screenshot_file}: {e}")
                skipped_count += 1
                continue
            
            crop_bbox = item.get('crop_bbox')
            if not crop_bbox:
                # logger.info("No crop bbox")
                skipped_count += 1
                continue

            task = item.get('vague_task')
            target_action_index = item.get('target_action_index', 0)
            previous_actions = item.get('action_reprs', [])[:int(target_action_index)]
            previous_conversations = item.get('previous_conversations', [])
            current_conversation = item.get('current_conversation')
            if current_conversation is None:
                current_conversation = {}
            clarification_point = item.get('clarification_point', False)

            html_content = None
            if env in ["html", "both"]:
                html_content = generate_html_content(item, candidate_ranks, top_k)
                if not html_content:
                    skipped_count += 1
                    continue

            clarification_need = clarification_point
            clarification_question = ""
            if clarification_need and current_conversation:
                clarification_question = current_conversation.get('question', '')

            system_prompt, user_prompt = get_finetuning_prompt(
                mode=mode,
                env=env,
                task=task,
                previous_conversations=previous_conversations,
                previous_actions=previous_actions,
                html_content=html_content
            )

            if clarification_need:
                if clarification_question:
                    response = f"Answer: Yes. Question: {clarification_question}"
                else:
                    response = "Answer: Yes."
            else:
                response = "Answer: No."

            example = {
                'sample_id': sample_id,
                'annotation_id': annotation_id,
                'action_id': action_id,
                'system_prompt': system_prompt,
                'user_prompt': user_prompt,
                'response': response,
                'needs_image': mode != "conv_only" and env in ["screenshot", "both"],
                'screenshot_path': screenshot_file,
                'crop_bbox': crop_bbox
            }

            self.examples.append(example)
            processed_count += 1

        logger.info(f"Created {processed_count} training examples, skipped {skipped_count}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        if self.processor:
            image = Image.open(example['screenshot_path']).convert('RGB')
            crop_bbox = json.loads(example['crop_bbox'])
            top = crop_bbox.get('top')
            bottom = crop_bbox.get('bottom')
            image = image.crop((0, top, image.width, bottom))

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": example['system_prompt']}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": example['user_prompt']}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": example['response']}]
                }
            ]

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

            inputs = self.processor(
                text=text,
                images=image,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=False
            )

            input_ids = inputs["input_ids"].squeeze(0)
            labels = input_ids.clone()

            image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")
            image_token_positions = (input_ids == image_token_id)
            labels[image_token_positions] = -100

            return {
                "input_ids": input_ids.tolist(),
                "attention_mask": inputs["attention_mask"].squeeze(0).tolist(),
                "labels": labels.tolist(),
                'pixel_values': inputs['pixel_values'].squeeze(0),
                "image_sizes": tuple(inputs["image_sizes"].squeeze(0).tolist())
            }

        elif self.tokenizer:
            messages = [
                {"role": "system", "content": example['system_prompt']},
                {"role": "user", "content": example['user_prompt']},
                {"role": "assistant", "content": example['response']}
            ]

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=False
            )
            
            input_ids = inputs['input_ids'].squeeze(0)
            labels = input_ids.clone()

            return {
                'input_ids': input_ids.tolist(),
                'attention_mask': inputs['attention_mask'].squeeze(0).tolist(),
                'labels': labels.tolist()
            }

        else:
            raise ValueError("Either processor (for LLaVA) or tokenizer (for text models) must be provided")
