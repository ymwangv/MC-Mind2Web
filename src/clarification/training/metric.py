import json
import logging
import os
from typing import Dict, Any, List
from collections import defaultdict

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def calculate_metrics(results: List[Dict]) -> Dict[str, Any]:

    if not results:
        return {}

    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    rouge_metric = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    # accuracy and f1 - micro
    y_true = []
    y_pred = []

    for result in results:
        y_true.append(result['actual_needs_clarification'])
        y_pred.append(result['predicted_needs_clarification'])

    total_samples = len(results)
    pred_clarification_count = sum(y_pred)
    actual_clarification_count = sum(y_true)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[False, True])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # accuracy and f1 - macro
    task_results = defaultdict(list)
    for result in results:
        annotation_id = result.get('annotation_id')
        if annotation_id:
            task_results[annotation_id].append(result)

    task_metrics = []
    for annotation_id, samples in task_results.items():
        task_y_true = [s['actual_needs_clarification'] for s in samples]
        task_y_pred = [s['predicted_needs_clarification'] for s in samples]

        if task_y_true:
            task_precision, task_recall, task_f1, _ = precision_recall_fscore_support(
                task_y_true, task_y_pred, average='binary', zero_division=0
            )
            task_accuracy = accuracy_score(task_y_true, task_y_pred)

            task_metrics.append({
                'precision': task_precision,
                'recall': task_recall,
                'f1_score': task_f1,
                'accuracy': task_accuracy
            })

    if task_metrics:
        macro_precision = sum(m['precision'] for m in task_metrics) / len(task_metrics)
        macro_recall = sum(m['recall'] for m in task_metrics) / len(task_metrics)
        macro_f1 = sum(m['f1_score'] for m in task_metrics) / len(task_metrics)
        macro_accuracy = sum(m['accuracy'] for m in task_metrics) / len(task_metrics)
    else:
        macro_precision = macro_recall = macro_f1 = macro_accuracy = 0.0

    # quality - micro
    quality_samples = []
    for result in results:
        if result.get('predicted_needs_clarification') and result.get('actual_needs_clarification'):
            pred_q = result.get('predicted_question', '').strip()
            actual_q = result.get('actual_question', '').strip()
            if pred_q and actual_q:
                quality_samples.append({
                    'generated': pred_q,
                    'reference': actual_q
                })

    micro_quality_metrics = {}
    if quality_samples:
        # BLEU-1
        bleu_scores = []
        smoothing = SmoothingFunction().method1
        for sample in quality_samples:
            candidate_tokens = sample['generated'].lower().split()
            reference_tokens = sample['reference'].lower().split()
            if candidate_tokens and reference_tokens:
                bleu_1 = sentence_bleu(
                    [reference_tokens],
                    candidate_tokens,
                    weights=(1.0, 0, 0, 0),
                    smoothing_function=smoothing
                )
                bleu_scores.append(bleu_1)

        # ROUGE-L
        rouge_scores = []
        for sample in quality_samples:
            scores = rouge_metric.score(sample['reference'], sample['generated'])
            rouge_scores.append(scores['rougeL'].fmeasure)

        # BERTScore
        candidates = [s['generated'] for s in quality_samples]
        references = [s['reference'] for s in quality_samples]
        _, _, bert_f1 = bert_score(candidates, references, lang='en', verbose=False)
        bert_scores = bert_f1.tolist()

        # SBERT similarity
        sbert_scores = []
        ref_embeddings = sbert_model.encode(references)
        cand_embeddings = sbert_model.encode(candidates)
        for i in range(len(candidates)):
            similarity = cosine_similarity(
                [ref_embeddings[i]],
                [cand_embeddings[i]]
            )[0][0]
            sbert_scores.append(float(similarity))

        micro_quality_metrics = {
            'micro_quality_samples': len(quality_samples),
            'micro_bleu_1': round(sum(bleu_scores) / len(bleu_scores), 2) if bleu_scores else 0,
            'micro_rouge_l': round(sum(rouge_scores) / len(rouge_scores), 2) if rouge_scores else 0,
            'micro_bert_score': round(sum(bert_scores) / len(bert_scores), 2) if bert_scores else 0,
            'micro_sbert_score': round(sum(sbert_scores) / len(sbert_scores), 2) if sbert_scores else 0
        }
    else:
        micro_quality_metrics = {
            'micro_quality_samples': 0,
            'micro_bleu_1': 0,
            'micro_rouge_l': 0,
            'micro_bert_score': 0,
            'micro_sbert_score': 0
        }

    # quality - macro
    task_quality_metrics = []
    for annotation_id, samples in task_results.items():
        task_quality_samples = []
        for result in samples:
            if result.get('predicted_needs_clarification') and result.get('actual_needs_clarification'):
                pred_q = result.get('predicted_question', '').strip()
                actual_q = result.get('actual_question', '').strip()
                if pred_q and actual_q:
                    task_quality_samples.append({
                        'generated': pred_q,
                        'reference': actual_q
                    })

        if task_quality_samples:
            # BLEU-1
            task_bleu_scores = []
            for sample in task_quality_samples:
                candidate_tokens = sample['generated'].lower().split()
                reference_tokens = sample['reference'].lower().split()
                if candidate_tokens and reference_tokens:
                    bleu_1 = sentence_bleu(
                        [reference_tokens],
                        candidate_tokens,
                        weights=(1.0, 0, 0, 0),
                        smoothing_function=smoothing
                    )
                    task_bleu_scores.append(bleu_1)

            # ROUGE-L
            task_rouge_scores = []
            for sample in task_quality_samples:
                scores = rouge_metric.score(sample['reference'], sample['generated'])
                task_rouge_scores.append(scores['rougeL'].fmeasure)

            # BERTScore
            candidates = [s['generated'] for s in task_quality_samples]
            references = [s['reference'] for s in task_quality_samples]
            _, _, bert_f1 = bert_score(candidates, references, lang='en', verbose=False)
            task_bert_scores = bert_f1.tolist()

            # SBERT similarity
            task_sbert_scores = []
            ref_embeddings = sbert_model.encode(references)
            cand_embeddings = sbert_model.encode(candidates)
            for i in range(len(candidates)):
                similarity = cosine_similarity(
                    [ref_embeddings[i]],
                    [cand_embeddings[i]]
                )[0][0]
                task_sbert_scores.append(float(similarity))

            task_quality_metrics.append({
                'bleu_1': sum(task_bleu_scores) / len(task_bleu_scores) if task_bleu_scores else 0,
                'rouge_l': sum(task_rouge_scores) / len(task_rouge_scores) if task_rouge_scores else 0,
                'bert_score': sum(task_bert_scores) / len(task_bert_scores) if task_bert_scores else 0,
                'sbert_score': sum(task_sbert_scores) / len(task_sbert_scores) if task_sbert_scores else 0
            })

    macro_quality_metrics = {}
    if task_quality_metrics:
        macro_quality_metrics = {
            'macro_quality_tasks': len(task_quality_metrics),
            'macro_bleu_1': round(sum(m['bleu_1'] for m in task_quality_metrics) / len(task_quality_metrics), 2),
            'macro_rouge_l': round(sum(m['rouge_l'] for m in task_quality_metrics) / len(task_quality_metrics), 2),
            'macro_bert_score': round(sum(m['bert_score'] for m in task_quality_metrics) / len(task_quality_metrics), 2),
            'macro_sbert_score': round(sum(m['sbert_score'] for m in task_quality_metrics) / len(task_quality_metrics), 2)
        }
    else:
        macro_quality_metrics = {
            'macro_quality_tasks': 0,
            'macro_bleu_1': 0,
            'macro_rouge_l': 0,
            'macro_bert_score': 0,
            'macro_sbert_score': 0
        }

    # Compile all metrics
    metrics = {
        # Basic stats
        'total_samples': total_samples,
        'predicted_clarifications': pred_clarification_count,
        'actual_clarifications': actual_clarification_count,
        'prediction_rate': round(pred_clarification_count / total_samples, 2) if total_samples > 0 else 0,
        'actual_rate': round(actual_clarification_count / total_samples, 2) if total_samples > 0 else 0,

        # Confusion matrix
        'confusion_matrix': {
            'true_positive': int(tp),
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn)
        },

        # Micro metrics (sample-level)
        'micro_precision': round(precision, 2),
        'micro_recall': round(recall, 2),
        'micro_f1': round(f1, 2),
        'micro_accuracy': round(accuracy, 2),

        # Macro metrics (task-level)
        'num_tasks': len(task_metrics),
        'macro_precision': round(macro_precision, 2),
        'macro_recall': round(macro_recall, 2),
        'macro_f1': round(macro_f1, 2),
        'macro_accuracy': round(macro_accuracy, 2),

        # Micro quality metrics
        **micro_quality_metrics,

        # Macro quality metrics
        **macro_quality_metrics
    }

    return metrics


def calculate_split_metrics(split: str, output_dir: str) -> Dict[str, Any]:
    split_results_file = os.path.join(output_dir, f"results_{split}.json")

    if not os.path.exists(split_results_file):
        logger.warning(f"Results file not found: {split_results_file}")
        return {}

    with open(split_results_file, 'r') as f:
        split_results = json.load(f)

    logger.info(f"Loaded {len(split_results)} results for {split} split from {split_results_file}")

    metrics = calculate_metrics(split_results)

    metrics_file = os.path.join(output_dir, f"metrics_{split}.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved {split} metrics to {metrics_file}")

    return metrics