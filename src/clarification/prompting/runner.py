import argparse
import json
import logging
import os
from typing import Tuple

from agent import ClarificationAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClarificationRunner:

    def __init__(self, agent: ClarificationAgent, model_name: str):
        self.agent = agent
        self.model_name = model_name

    def process_file(self, query_file_path: str, output_dir: str) -> bool:
        with open(query_file_path, 'r', encoding='utf-8') as f:
            query_data = json.load(f)

        logger.info(f"Processing {query_file_path}")
        result = self.agent.process_query(query_data)

        os.makedirs(output_dir, exist_ok=True)

        result_file = os.path.join(output_dir, f"result_{self.model_name}.json")

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=4)

        logger.info(f"Saved result to {result_file}")
        return True

    def process_directory(self, data_dir: str, test_mode: bool = False) -> Tuple[int, int]:

        processed_count = 0
        failed_count = 0

        if not os.path.exists(data_dir):
            logger.error(f"Data directory does not exist: {data_dir}")
            return 0, 0

        sample_dirs = []
        for sample_id in os.listdir(data_dir):
            sample_dir = os.path.join(data_dir, sample_id)
            if os.path.isdir(sample_dir):
                query_file = os.path.join(sample_dir, "query.json")
                if os.path.exists(query_file):
                    sample_dirs.append((sample_id, sample_dir, query_file))

        sample_dirs.sort(key=lambda x: x[0])
        total_samples = len(sample_dirs)
        logger.info(f"Found {total_samples} samples in {data_dir}")

        if test_mode:
            sample_dirs = sample_dirs[:3]
            logger.info(f"Test mode: Processing only first 3 samples")

        for idx, (sample_id, sample_dir, query_file) in enumerate(sample_dirs, 1):
            logger.info(f"Processing [{idx}/{len(sample_dirs)}] sample: {sample_id}")
            if self.process_file(query_file, sample_dir):
                processed_count += 1
            else:
                failed_count += 1

        return processed_count, failed_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="task", choices=["task", "website", "domain"])
    parser.add_argument("--mode", default="proact", choices=["proact", "conv_only", "env_only"])
    parser.add_argument("--env-type", default="screenshot", choices=["screenshot", "html", "both"])
    parser.add_argument("--provider", default="gemini", choices=["gpt", "gemini"])
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--key-number", type=int, default=1)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    logger.info(f"Data split: {args.split}")
    logger.info(f"Experiment Mode: {args.mode}")
    if args.mode != "conv_only":
        logger.info(f"Environment Type: {args.env_type}")
    logger.info(f"Provider: {args.provider}")
    logger.info(f"Model: {args.model}")
    logger.info(f"API Key Number: {args.key_number}")

    if args.mode == "conv_only":
        args.env_type = None

    if args.env_type:
        data_dir = os.path.join("results", "prompting", "clarification", args.split, f"{args.mode}_{args.env_type}")
    else:
        data_dir = os.path.join("results", "prompting", "clarification", args.split, args.mode)

    logger.info(f"Data directory: {data_dir}")

    if args.test:
        logger.info(f"TEST MODE: Will process only first 3 samples")

    agent = ClarificationAgent(
        provider=args.provider,
        model=args.model,
        key_number=args.key_number,
        mode=args.mode,
        env_type=args.env_type
    )

    runner = ClarificationRunner(agent, model_name=args.model)

    processed_count, failed_count = runner.process_directory(data_dir,test_mode=args.test)

    logger.info(f"Processing complete: {processed_count} succeeded, {failed_count} failed")


if __name__ == "__main__":
    main()