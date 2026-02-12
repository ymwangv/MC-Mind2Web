import logging
import os
import random

from datasets import load_from_disk
from sentence_transformers import InputExample
from torch.utils.data import Dataset


logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def format_query(sample, mode="full"):
    if mode == "full":
        task_text = sample["confirmed_task"]
    elif mode in ["vague", "conv_pred", "conv_clar"]:
        task_text = sample["vague_task"]

    query = f"Task is: {task_text}\n"

    if mode in ["conv_pred", "conv_clar"]:
        query += "Previous conversations:\n"

        conversations = sample["previous_conversations"].copy()
        if mode == "conv_pred" and sample.get("current_conversation"):
            conversations.append(sample["current_conversation"])

        conv_texts = []
        for turn in conversations:
            conv_texts.append(f"assistant: {turn['question']}\nuser: {turn['response']}")
        query += "\n".join(conv_texts)
        query += "\n"

    query += "Previous actions:\n"
    target_idx = int(sample["target_action_index"])
    prev_actions = sample["action_reprs"][:target_idx][-3:]
    query += "\n".join(prev_actions)

    return query


class CandidateRankDataset(Dataset):

    def __init__(self, data=None, neg_ratio=5, mode="full"):
        self.data = data
        self.neg_ratio = neg_ratio
        self.mode = mode

    def __len__(self):
        return len(self.data) * (1 + self.neg_ratio)

    def __getitem__(self, idx):
        sample = self.data[idx // (1 + self.neg_ratio)]

        if idx % (1 + self.neg_ratio) == 0 or len(sample["neg_candidates_formatted"]) == 0:
            candidate = random.choice(sample["pos_candidates_formatted"])
            label = 1
        else:
            candidate = random.choice(sample["neg_candidates_formatted"])
            label = 0

        query = format_query(sample, self.mode)

        return InputExample(
            texts=[candidate[1], query],
            label=label,
        )


def get_data_split(
    base_path,
    split_name,
    is_train=False
):
    dataset_path = os.path.join(base_path, split_name, "dataset")
    dataset = load_from_disk(dataset_path)

    if is_train:
        original_len = len(dataset)
        dataset = dataset.filter(lambda x: len(x["pos_candidates_formatted"]) > 0)
        logger.info(f"{len(dataset)}/{original_len} samples filtered with positive candidates")

    return dataset