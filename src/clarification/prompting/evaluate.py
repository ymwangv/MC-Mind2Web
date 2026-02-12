import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from clarification.prompting.metric import ClarificationMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(args):
    if args.env_type:
        data_dir = os.path.join("results", "prompting", "clarification", args.split, f"{args.mode}_{args.env_type}")
        metrics_dir = os.path.join("eval", "prompting", "clarification", args.split, f"{args.mode}_{args.env_type}")
    else:
        data_dir = os.path.join("results", "prompting", "clarification", args.split, args.mode)
        metrics_dir = os.path.join("eval", "prompting", "clarification", args.split, args.mode)
    os.makedirs(metrics_dir, exist_ok=True)

    if args.mode == "conv_only":
        args.env_type = None

    logger.info(f"Evaluating split: {args.split}, mode: {args.mode}")
    if args.env_type:
        logger.info(f"Environment type: {args.env_type}")

    metrics = ClarificationMetrics()
    num_results = metrics.load_results(data_dir, model_name=args.model)

    metrics_file = os.path.join(metrics_dir, f"metrics_{args.model}.json")
    metrics.save_metrics(metrics_file)

    if num_results == 0:
        logger.warning("No results found to calculate metrics")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--split", default="task", choices=["task", "website", "domain"])
    parser.add_argument("--mode", default="proact", choices=["proact", "conv_only", "env_only"])
    parser.add_argument("--env-type", default="screenshot", choices=["screenshot", "html", "both"])
    parser.add_argument("--model", default="gemini-2.5-pro")
    args = parser.parse_args()
    
    if args.mode == "conv_only":
        args.env_type = None

    logger.info(f"Data split: {args.split}")
    logger.info(f"Experiment Mode: {args.mode}")
    if args.mode != "conv_only":
        logger.info(f"Environment Type: {args.env_type}")
    logger.info(f"Model: {args.model}")

    evaluate(args)

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()