import gc
import json
import logging
import pathlib
import re
import shutil
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


import lxml.etree
from datasets import Dataset, load_dataset
from tqdm import tqdm

from dom_utils import get_tree_repr, prune_tree

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class DataFormatter:
    def __init__(self, data_dir=None):
        if data_dir is None:
            project_root = pathlib.Path(__file__).parent.parent.parent
            self.data_dir = project_root / "data"
        else:
            self.data_dir = pathlib.Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def format_candidate(self, dom_tree, candidate, keep_html_brackets=False):
        node_tree = prune_tree(dom_tree, [candidate["backend_node_id"]])
        c_node = node_tree.xpath("//*[@backend_node_id]")[0]

        if c_node.getparent() is not None:
            c_node.getparent().remove(c_node)
            ancestor_repr, _ = get_tree_repr(
                node_tree, id_mapping={}, keep_html_brackets=keep_html_brackets
            )
        else:
            ancestor_repr = ""

        subtree_repr, _ = get_tree_repr(
            c_node, id_mapping={}, keep_html_brackets=keep_html_brackets
        )

        if subtree_repr.strip():
            subtree_repr = " ".join(subtree_repr.split()[:100])
        else:
            subtree_repr = ""

        if ancestor_repr.strip():
            ancestor_repr = re.sub(r"\s*\(\s*", "/", ancestor_repr)
            ancestor_repr = re.sub(r"\s*\)\s*", "", ancestor_repr)
            ancestor_repr = " ".join(ancestor_repr.split()[-50:])
        else:
            ancestor_repr = ""

        return f"ancestors: {ancestor_repr}\ntarget: {subtree_repr}"

    def save_screenshot(self, sample, output_dir):
        annotation_id = sample["annotation_id"]
        action_uid = sample["action_uid"]

        if not sample.get("screenshot"):
            logger.info(f"No screenshot for task-action: {annotation_id} - {action_uid}")
            return None

        screenshot_dir = output_dir / "screenshots"
        screenshot_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{annotation_id}_{action_uid}.jpg"
        filepath = screenshot_dir / filename

        if filepath.exists():
            return filename

        image = sample["screenshot"]

        try:
            image.save(filepath)
            return filename
        except Exception as e:
            logger.info(f"Error saving screenshot: {e}")
            return None

    def process_dataset(self, dataset_name, split, num_proc=8):
        logger.info(f"Loading dataset {dataset_name}, split {split}")
        dataset = load_dataset(dataset_name, split=split)

        output_dir = self.data_dir / split
        output_dir.mkdir(parents=True, exist_ok=True)

        def format_candidates(sample):
            dom_tree = lxml.etree.fromstring(sample["cleaned_html"])

            pos_candidates_formatted = []
            for c in sample["pos_candidates"]:
                candidate = json.loads(c)
                pos_candidates_formatted.append(
                    (
                        candidate["backend_node_id"],
                        self.format_candidate(dom_tree, candidate, keep_html_brackets=False)
                    )
                )

            neg_candidates_formatted = []
            for c in sample["neg_candidates"]:
                candidate = json.loads(c)
                neg_candidates_formatted.append(
                    (
                        candidate["backend_node_id"],
                        self.format_candidate(dom_tree, candidate, keep_html_brackets=False)
                    )
                )

            sample["pos_candidates_formatted"] = pos_candidates_formatted
            sample["neg_candidates_formatted"] = neg_candidates_formatted
            return sample

        logger.info(f"Saving {len(dataset)} screenshots (num_proc={num_proc})")
        with ThreadPoolExecutor(max_workers=num_proc) as executor:
            futures = [
                executor.submit(self.save_screenshot, sample, output_dir)
                for sample in dataset
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Saving screenshots"):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error saving screenshot: {e}")

        logger.info(f"Processing {len(dataset)} candidates (num_proc={num_proc})")
        processed_dataset = dataset.map(
            format_candidates,
            num_proc=num_proc,
            desc="Processing candidates",
            load_from_cache_file=False
        )
        
        logger.info("Removing screenshots form processed dataset")
        processed_dataset = processed_dataset.remove_columns(["screenshot"])
        
        logger.info("Saving processed dataset temporally for saving memory")
        temp_dataset_path = output_dir / "temp"
        processed_dataset.save_to_disk(str(temp_dataset_path))

        del processed_dataset
        gc.collect()
        
        processed_dataset = Dataset.load_from_disk(str(temp_dataset_path))

        logger.info("Grouping samples by task for conversation processing")

        task_groups = defaultdict(list)
        for idx in range(len(processed_dataset)):
            sample = processed_dataset[idx]
            task_groups[sample["annotation_id"]].append(
                (idx, int(sample["target_action_index"]))
            )

        for annotation_id in task_groups:
            task_groups[annotation_id].sort(key=lambda x: x[1])

        logger.info("Building conversation histories and saving final dataset")

        def format_conversations():
            for task_indices in tqdm(task_groups.values(), desc="Processing task groups"):
                previous_conversations = []

                for idx, action_index in task_indices:
                    sample = processed_dataset[idx]

                    current_conversation = None
                    if sample["clarification_details"]:
                        clarification = json.loads(sample["clarification_details"])
                        if clarification["clarifying_question"] and clarification["user_response"]:
                            current_conversation = {
                                "action_idx": action_index,
                                "question": clarification["clarifying_question"],
                                "response": clarification["user_response"]
                            }

                    final_sample = dict(sample)
                    final_sample["current_conversation"] = current_conversation
                    final_sample["previous_conversations"] = previous_conversations.copy()

                    yield final_sample

                    if current_conversation:
                        previous_conversations.append(current_conversation)

        dataset_path = output_dir / "dataset"
        final_dataset = Dataset.from_generator(format_conversations)

        logger.info(f"Saving dataset to {dataset_path}")
        final_dataset.save_to_disk(str(dataset_path))

        logger.info(f"Successfully saved dataset to {dataset_path}")
        
        del processed_dataset
        gc.collect()
        time.sleep(1)

        if temp_dataset_path.exists():
            try:
                shutil.rmtree(temp_dataset_path)
                logger.info("Cleaned up temporary files")
            except Exception as e:
                logger.warning(f"Could not remove temp files: {e}")

        processed_data = list(final_dataset)

        with_pos = sum(1 for s in processed_data if s["pos_candidates_formatted"])
        with_neg = sum(1 for s in processed_data if s["neg_candidates_formatted"])
        with_curr_conv = sum(1 for s in processed_data if s["current_conversation"])
        with_prev_conv = sum(1 for s in processed_data if s["previous_conversations"])

        logger.info(f"Statistics for {split}:")
        logger.info(f"  - Samples with positive candidates: {with_pos}/{len(processed_data)}")
        logger.info(f"  - Samples with negative candidates: {with_neg}/{len(processed_data)}")
        logger.info(f"  - Samples with current conversation: {with_curr_conv}/{len(processed_data)}")
        logger.info(f"  - Samples with previous conversations: {with_prev_conv}/{len(processed_data)}")

    def load_data(self, split):
        dataset_path = self.data_dir / split / "dataset"

        if not dataset_path.exists():
            logger.error(f"Dataset not found: {dataset_path}")
            return None

        dataset = Dataset.load_from_disk(str(dataset_path))

        logger.info(f"Loaded {len(dataset)} samples from {dataset_path}")
        return dataset


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Format and cache dataset")
    parser.add_argument("--dataset", type=str, default="ymwangv/MC-Mind2Web", help="Dataset name")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--num-proc", type=int, default=8, help="Number of processes for parallel processing")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    formatter = DataFormatter()
    formatter.process_dataset(dataset_name=args.dataset, split=args.split, num_proc=args.num_proc)


if __name__ == "__main__":
    main()