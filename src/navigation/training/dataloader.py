import json
import os
import pathlib
import random
import sys

import lxml
from datasets import load_from_disk
from torch.utils.data import Dataset

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
from utils.dom_utils import get_tree_repr, prune_tree


def format_conversations(sample):
    conversations = sample.get("previous_conversations", []).copy()
    if sample.get("current_conversation"):
        conversations.append(sample["current_conversation"])

    conv_texts = []
    for turn in conversations:
        conv_texts.append(f"assistant: {turn['question']}\nuser: {turn['response']}")

    return "\n".join(conv_texts) if conv_texts else ""


def format_task_input(sample, task_mode="full", previous_k=5):
    if task_mode == "full":
        task_text = sample["confirmed_task"]
    else:
        task_text = sample["vague_task"]

    seq_input = (
        "Based on the HTML webpage above, try to complete the following task:\n"
        f"Task: {task_text}\n"
    )

    if task_mode == "conv_pred":
        seq_input += "Previous conversations:\n"
        conv_text = format_conversations(sample)
        if conv_text:
            seq_input += conv_text + "\n"
        else:
            seq_input += "None\n"

    seq_input += "Previous actions:\n"
    target_idx = int(sample["target_action_index"])
    prev_actions = sample["action_reprs"][:target_idx][-previous_k:]

    if prev_actions:
        for action in prev_actions:
            seq_input += f"{action}\n"
    else:
        seq_input += "None\n"

    return seq_input


def format_input_generation(
    sample, 
    candidate_ids, 
    gt=-1, 
    previous_k=5, 
    keep_html_brackets=False, 
    task_mode="full"
):
    dom_tree = lxml.etree.fromstring(sample["cleaned_html"])
    dom_tree = prune_tree(dom_tree, candidate_ids)

    tree_repr, id_mapping = get_tree_repr(
        dom_tree, id_mapping={}, keep_html_brackets=keep_html_brackets
    )

    candidate_nodes = dom_tree.xpath("//*[@backend_node_id]")

    choices = []
    for idx, node in enumerate(candidate_nodes):
        choices.append(
            [
                node.attrib["backend_node_id"],
                " ".join(
                    get_tree_repr(
                        node,
                        id_mapping=id_mapping,
                        keep_html_brackets=keep_html_brackets,
                    )[0].split()[:10]
                ),
            ]
        )

    gt = id_mapping.get(gt, -1)

    seq_input = format_task_input(sample, task_mode, previous_k)
    seq_input += (
        "What should be the next action? "
        "Please select the element to interact with, and the action to perform along with the value to type in or select. "
        "If the task cannot be completed, output None."
    )

    if gt == -1:
        seq_target = "None"
    else:
        operation = json.loads(sample["operation"]) if isinstance(sample["operation"], str) else sample["operation"]
        current_action_op = operation["op"]
        current_action_value = operation["value"]
        seq_target = f"Element: {choices[gt][1]}\n"
        seq_target += f"Action: {current_action_op}\n"
        if current_action_op != "CLICK":
            seq_target += f"Value: {current_action_value}"
    return tree_repr, seq_input, seq_target, choices


def format_input_multichoice(
    sample, 
    candidate_ids, 
    gt=-1, 
    previous_k=5, 
    keep_html_brackets=False, 
    task_mode="full"
):
    dom_tree = lxml.etree.fromstring(sample["cleaned_html"])
    dom_tree = prune_tree(dom_tree, candidate_ids)
    
    tree_repr, id_mapping = get_tree_repr(
        dom_tree, id_mapping={}, keep_html_brackets=keep_html_brackets
    )
    
    candidate_nodes = dom_tree.xpath("//*[@backend_node_id]")
    
    choices = []
    for idx, node in enumerate(candidate_nodes):
        choices.append(
            [
                node.attrib["backend_node_id"],
                " ".join(
                    get_tree_repr(
                        node,
                        id_mapping=id_mapping,
                        keep_html_brackets=keep_html_brackets,
                    )[0].split()[:10]
                ),
            ]
        )
    
    gt = id_mapping.get(gt, -1)

    seq_input = format_task_input(sample, task_mode, previous_k)
    seq_input += (
        "What should be the next action? Please select from the following choices "
        "(If the correct action is not in the page above, please select A. 'None of the above'):\n\n"
        "A. None of the above\n"
    )
    
    for idx, choice in enumerate(choices):
        # convert to ascii A, B, C, D, ...
        seq_input += f"{chr(66 + idx)}. {choice[1]}\n"

    if gt == -1:
        seq_target = "A."
    else:
        gt += 1
        operation = json.loads(sample["operation"]) if isinstance(sample["operation"], str) else sample["operation"]
        current_action_op = operation["op"]
        current_action_value = operation["value"]
        seq_target = f"{chr(65 + gt)}.\n" f"Action: {current_action_op}\n"
        if current_action_op != "CLICK":
            seq_target += f"Value: {current_action_value}"
    return tree_repr, seq_input, seq_target, choices


class MultiChoiceDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        neg_ratio=0.2,
        num_candidates=5,
        max_context_len=512,
        mode="multichoice",
        task_mode="full",
        top_k=-1
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.neg_ratio = neg_ratio
        self.num_candidates = num_candidates
        self.max_context_len = max_context_len
        self.mode = mode
        self.task_mode = task_mode
        self.top_k = top_k

    def __len__(self):
        return len(self.data) * 10

    def __getitem__(self, idx):
        sample = self.data[idx // 10]

        # Prioritize hard negatives (with high rank) using 8:2 sampling
        if self.top_k > 0:
            top_negatives = [c for c in sample["neg_candidates"] if c["rank"] < self.top_k]
            other_negatives = [c for c in sample["neg_candidates"] if c["rank"] >= self.top_k]
        else:
            top_negatives = []
            other_negatives = sample["neg_candidates"]

        if random.random() < 0.8 and len(top_negatives) > 0:
            neg_candidates = top_negatives
        else:
            neg_candidates = other_negatives

        # Select positive sample (80 percent) or pure negative sample (20 percent)
        if len(sample["pos_candidates"]) != 0 and (
            random.random() > self.neg_ratio or len(neg_candidates) == 0
        ):
            pos_candidate = random.choice(sample["pos_candidates"])
            neg_candidate = random.sample(
                neg_candidates,
                min(len(neg_candidates), self.num_candidates - 1),
            )
            
            gt = pos_candidate["backend_node_id"]
            candidate_ids = [gt] + [c["backend_node_id"] for c in neg_candidate]
            
            if self.mode == "multichoice":
                seq_context, seq_in, seq_out, _ = format_input_multichoice(
                    sample, candidate_ids, gt, task_mode=self.task_mode
                )
            else:
                seq_context, seq_in, seq_out, _ = format_input_generation(
                    sample, candidate_ids, gt, task_mode=self.task_mode
                )
        else:
            neg_candidate = random.sample(
                neg_candidates,
                min(len(neg_candidates), self.num_candidates),
            )
            
            gt = -1
            candidate_ids = [c["backend_node_id"] for c in neg_candidate]
            
            if self.mode == "multichoice":
                seq_context, seq_in, seq_out, _ = format_input_multichoice(
                    sample, candidate_ids, gt, task_mode=self.task_mode
                )
            else:
                seq_context, seq_in, seq_out, _ = format_input_generation(
                    sample, candidate_ids, gt, task_mode=self.task_mode
                )

        seq_context = self.tokenizer(
            seq_context,
            truncation=True,
            max_length=self.max_context_len,
            add_special_tokens=False,
        )
        seq_in = self.tokenizer(
            seq_in,
            truncation=True,
            max_length=self.max_context_len,
            add_special_tokens=True,
        )
        model_input = {
            "input_ids": seq_context["input_ids"] + seq_in["input_ids"],
            "attention_mask": seq_context["attention_mask"] + seq_in["attention_mask"],
        }
        seq_out = self.tokenizer(seq_out)
        model_input["labels"] = seq_out["input_ids"]
        return model_input


def get_data_split(
    base_path,
    split_name,
    is_train=False,
    candidate_results=None
):
    dataset_path = os.path.join(base_path, split_name, "dataset")
    dataset = load_from_disk(dataset_path)

    if candidate_results is not None:
        candidate_scores = candidate_results["scores"]
        candidate_ranks = candidate_results["ranks"]

        def get_score(sample):
            sample_id = f"{sample['annotation_id']}_{sample['action_uid']}"

            # Process positive candidates
            pos_candidates_updated = []
            for candidate_str in sample["pos_candidates"]:
                candidate = json.loads(candidate_str) if isinstance(candidate_str, str) else candidate_str
                candidate_id = candidate["backend_node_id"]
                if sample_id in candidate_scores and candidate_id in candidate_scores[sample_id]:
                    candidate["score"] = candidate_scores[sample_id][candidate_id]
                    candidate["rank"] = candidate_ranks[sample_id][candidate_id]
                pos_candidates_updated.append(candidate)

            # Process negative candidates
            neg_candidates_updated = []
            for candidate_str in sample["neg_candidates"]:
                candidate = json.loads(candidate_str) if isinstance(candidate_str, str) else candidate_str
                candidate_id = candidate["backend_node_id"]
                if sample_id in candidate_scores and candidate_id in candidate_scores[sample_id]:
                    candidate["score"] = candidate_scores[sample_id][candidate_id]
                    candidate["rank"] = candidate_ranks[sample_id][candidate_id]
                neg_candidates_updated.append(candidate)

            sample["pos_candidates"] = pos_candidates_updated
            sample["neg_candidates"] = neg_candidates_updated
            return sample

        dataset = dataset.map(get_score)

    if is_train:
        original_len = len(dataset)
        dataset = dataset.filter(lambda x: len(x["pos_candidates"]) > 0)
        print(f"{len(dataset)}/{original_len} samples filtered with positive candidates")

    return dataset
