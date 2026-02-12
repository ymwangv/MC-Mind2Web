import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from clarification.prompting.metric import ClarificationMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_all(args):

    env_type = None if args.mode == "conv_only" else args.env_type

    all_results = []
    splits = ["task", "website", "domain"]

    logger.info(f"Loading results from all splits...")

    for split in splits:
        if env_type:
            data_dir = os.path.join("results", "prompting", "clarification", split, f"{args.mode}_{env_type}")
        else:
            data_dir = os.path.join("results", "prompting", "clarification", split, args.mode)

        split_metrics = ClarificationMetrics()
        num_samples = split_metrics.load_results(data_dir, model_name=args.model)

        if num_samples > 0:
            logger.info(f"  {split}: loaded {num_samples} samples")
            all_results.extend(split_metrics.results)
        else:
            logger.warning(f"  {split}: no results found")

    if len(all_results) == 0:
        logger.error("No results found across any split!")
        return

    logger.info(f"\nTotal samples loaded: {len(all_results)}")

    metrics = ClarificationMetrics()
    metrics.results = all_results

    if env_type:
        metrics_dir = os.path.join("eval", "prompting", "clarification", "all_splits", f"{args.mode}_{env_type}")
    else:
        metrics_dir = os.path.join("eval", "prompting", "clarification", "all_splits", args.mode)

    os.makedirs(metrics_dir, exist_ok=True)

    metrics_file = os.path.join(metrics_dir, f"metrics_{args.model}.json")
    metrics.save_metrics(metrics_file)

    logger.info(f"Metrics saved to: {metrics_file}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", default="proact", choices=["proact", "conv_only", "env_only"])
    parser.add_argument("--env-type", default="screenshot", choices=["screenshot", "html", "both"])
    parser.add_argument("--model", default="gemini-2.0-flash")

    args = parser.parse_args()

    if args.mode == "conv_only":
        args.env_type = None

    logger.info("Evaluation Configuration:")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Mode: {args.mode}")
    if args.env_type:
        logger.info(f"  Environment: {args.env_type}")
    logger.info(f"  Combining all splits: task + website + domain")

    evaluate_all(args)

    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()