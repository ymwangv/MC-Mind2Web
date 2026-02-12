import argparse
import os
import json
import logging
import re

from metric import calculate_metrics

logger = logging.getLogger(__name__)

def process_file(file_path):
    logger.info(f"Processing file: {file_path}")

    with open(file_path, 'r') as f:
        data = json.load(f)
    
    for sample in data:
        first_output = sample.get("first_output", "").strip()

        if first_output:
            sample["predicted_needs_clarification"] = True
            
            match = re.search(r"(?:question:\s*)?([^\n]*)", first_output, flags=re.IGNORECASE)
            if match:
                question = match.group(1).strip()
                sample["predicted_question"] = question
        
        file_dir = os.path.dirname(file_path)
        processed_file_name = os.path.basename(file_path).replace(".json", "_processed.json")
        processed_file_path = os.path.join(file_dir, processed_file_name)
        with open(processed_file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    return data

def post_evaluate(eval_dir):

    # files = ["all", "task", "website", "domain"]
    files = ["task", "website"]
    for file in files:
        file_path = os.path.join(eval_dir, f"results_{file}.json")
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            continue
        
        processed_results = process_file(file_path)
        metrics = calculate_metrics(processed_results)
    
        metrics_file = os.path.join(eval_dir, f"metrics_{file}_post.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Saved metrics to {metrics_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="proact", choices=["proact", "conv_only"])
    parser.add_argument("--env", default="screenshot", choices=["screenshot", "both"])
    parser.add_argument("--model", default="llava-mistral", choices=["llava-mistral", "llava-vicuna"])
    args = parser.parse_args()
    
    mode = args.mode
    env = args.env
    model = args.model
    use_tuned = True
    
    model_path = ""
    if model == 'llava-mistral':
        model_path = "llava-1.6-mistral-7b"
    else:
        model_path = "llava-1.6-vicuna-7b"
    
    eval_dir = ""
    if use_tuned:
        eval_dir = os.path.join('eval', 'clarification', f"{mode}_{env}", model_path, 'True')
    else:
        eval_dir = os.path.join('eval', 'clarification', f"{mode}_{env}", model_path, 'False')
    
    post_evaluate(eval_dir)
