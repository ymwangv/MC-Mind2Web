import json
import logging
import os
from typing import Dict, Any
from collections import defaultdict

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClarificationMetrics:

    def __init__(self):
        self.results = []
        self.metrics = {}
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def load_results(self, data_dir: str, model_name: str = None) -> int:
        self.results = []

        for sample_id in os.listdir(data_dir):
            sample_dir = os.path.join(data_dir, sample_id)
            if os.path.isdir(sample_dir):
                result_file = os.path.join(sample_dir, f"result_{model_name}.json")
                if os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        result_data = json.load(f)

                    query_file = os.path.join(sample_dir, 'query.json')
                    if os.path.exists(query_file):
                        with open(query_file, 'r') as qf:
                            query_data = json.load(qf)

                        extracted_data = {
                            'sample_id': sample_id,
                            'annotation_id': query_data.get('annotation_id'),
                            'action_id': query_data.get('action_id'),
                            'pred_clarification_need': result_data.get('result', {}).get('clarification_need', False),
                            'pred_clarification_question': result_data.get('result', {}).get('clarification_question', ''),
                            'gt_clarification_need': query_data.get('clarification_point', False),
                            'gt_clarification_question': query_data.get('current_conversation').get('question', '') if query_data.get('current_conversation') else '',
                            'api_meta': result_data.get('api_meta', {})
                        }

                        self.results.append(extracted_data)

        logger.info(f"Loaded {len(self.results)} results from {data_dir} with model {model_name}")
        return len(self.results)

    def calculate_basic_metrics(self) -> Dict[str, Any]:
        if not self.results:
            return {}

        total = len(self.results)

        pred_clarification_need = sum(1 for r in self.results if r['pred_clarification_need'])
        pred_no_clarification_need = total - pred_clarification_need

        gt_clarification_need = sum(1 for r in self.results if r['gt_clarification_need'])
        gt_no_clarification_need = total - gt_clarification_need

        metrics = {
            'total_samples': total,
            'pred_clarification_need': pred_clarification_need,
            'pred_no_clarification_need': pred_no_clarification_need,
            'pred_clarification_rate': pred_clarification_need / total if total > 0 else 0,
            'gt_clarification_need': gt_clarification_need,
            'gt_no_clarification_need': gt_no_clarification_need,
            'gt_clarification_rate': gt_clarification_need / total if total > 0 else 0
        }

        return metrics

    def calculate_api_metrics(self) -> Dict[str, Any]:
        total_time = 0
        total_attempts = 0
        total_tokens = defaultdict(int)
        total_costs = defaultdict(float)

        for result in self.results:
            api_meta = result.get('api_meta', {})

            total_time += api_meta.get('time_seconds', 0)
            total_attempts += api_meta.get('attempts', 0)

            tokens = api_meta.get('tokens', {})
            for key, value in tokens.items():
                total_tokens[key] += value

            costs = api_meta.get('costs', {})
            for key, value in costs.items():
                if isinstance(value, (int, float)):
                    total_costs[key] += value

        num_results = len(self.results)

        metrics = {
            'total_time_seconds': total_time,
            'avg_time_seconds': total_time / num_results if num_results > 0 else 0,
            'total_attempts': total_attempts,
            'avg_attempts': total_attempts / num_results if num_results > 0 else 0,
            'total_tokens': dict(total_tokens),
            'total_costs': dict(total_costs)
        }

        return metrics

    def calculate_classification_metrics(self) -> Dict[str, Any]:
        y_true = []
        y_pred = []

        for result in self.results:
            y_true.append(result['gt_clarification_need'])
            y_pred.append(result['pred_clarification_need'])

        if not y_true:
            return {}

        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=[False, True])

        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        return {
            'micro_precision': round(precision, 2),
            'micro_recall': round(recall, 2),
            'micro_f1_score': round(f1, 2),
            'micro_accuracy': round(accuracy, 2),
            'micro_confusion_matrix': {
                'true_positive': int(tp),
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn)
            }
        }

    def calculate_task_level_classification_metrics(self) -> Dict[str, Any]:
        if not self.results:
            return {}

        task_results = defaultdict(list)
        for result in self.results:
            annotation_id = result['annotation_id']
            if annotation_id:
                task_results[annotation_id].append(result)

        if not task_results:
            return {}

        task_metrics = []

        for annotation_id, samples in task_results.items():
            y_true = [s['gt_clarification_need'] for s in samples]
            y_pred = [s['pred_clarification_need'] for s in samples]

            if y_true:
                precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
                accuracy = accuracy_score(y_true, y_pred)

                task_metrics.append({
                    'annotation_id': annotation_id,
                    'num_samples': len(samples),
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'accuracy': accuracy
                })

        if not task_metrics:
            return {}

        avg_precision = sum(m['precision'] for m in task_metrics) / len(task_metrics)
        avg_recall = sum(m['recall'] for m in task_metrics) / len(task_metrics)
        avg_f1 = sum(m['f1_score'] for m in task_metrics) / len(task_metrics)
        avg_accuracy = sum(m['accuracy'] for m in task_metrics) / len(task_metrics)

        return {
            'num_tasks': len(task_metrics),
            'macro_precision': round(avg_precision, 2),
            'macro_recall': round(avg_recall, 2),
            'macro_f1_score': round(avg_f1, 2),
            'macro_accuracy': round(avg_accuracy, 2)
        }

    def calculate_question_quality_metrics(self) -> Dict[str, Any]:
        correct_predictions = []

        for result in self.results:
            pred = result['pred_clarification_need']
            truth = result['gt_clarification_need']

            if pred and truth:
                generated = result['pred_clarification_question']
                reference = result['gt_clarification_question']

                if generated and reference:
                    correct_predictions.append({
                        'generated': generated,
                        'reference': reference
                    })

        if not correct_predictions:
            return {}

        gen_lengths = [len(p['generated'].split()) for p in correct_predictions]
        ref_lengths = [len(p['reference'].split()) for p in correct_predictions]
        avg_gen_length = sum(gen_lengths) / len(gen_lengths) if gen_lengths else 0
        avg_ref_length = sum(ref_lengths) / len(ref_lengths) if ref_lengths else 0


        bleu_1_scores = []
        rouge_l_scores = []
        bert_scores = []
        sbert_scores = []

        candidates = [pred['generated'] for pred in correct_predictions]
        references = [pred['reference'] for pred in correct_predictions]

        if candidates and references:
            smoothing = SmoothingFunction().method1
            for pred in correct_predictions:
                candidate_tokens = pred['generated'].lower().split()
                reference_tokens = pred['reference'].lower().split()
                if candidate_tokens and reference_tokens:
                    bleu_1 = sentence_bleu([reference_tokens], candidate_tokens, weights=(1.0, 0, 0, 0), smoothing_function=smoothing)
                    bleu_1_scores.append(bleu_1)

            for pred in correct_predictions:
                rouge_scores = self.rouge_scorer.score(pred['reference'], pred['generated'])
                rouge_l_scores.append(rouge_scores['rougeL'].fmeasure)

            _, _, F1 = bert_score(candidates, references, lang='en', verbose=False)
            bert_scores = F1.tolist()

            ref_embeddings = self.sbert_model.encode(references)
            cand_embeddings = self.sbert_model.encode(candidates)

            for i in range(len(candidates)):
                similarity = cosine_similarity([ref_embeddings[i]], [cand_embeddings[i]])[0][0]
                sbert_scores.append(float(similarity))

        avg_bleu_1 = round(sum(bleu_1_scores) / len(bleu_1_scores), 2) if bleu_1_scores else 0.0
        avg_rouge_l = round(sum(rouge_l_scores) / len(rouge_l_scores), 2) if rouge_l_scores else 0.0
        avg_bert_score = round(sum(bert_scores) / len(bert_scores), 2) if bert_scores else 0.0
        avg_sbert_score = round(sum(sbert_scores) / len(sbert_scores), 2) if sbert_scores else 0.0

        return {
            'correct_predictions': len(correct_predictions),
            'micro_avg_generated_length': avg_gen_length,
            'micro_avg_reference_length': avg_ref_length,
            'micro_bleu_1': avg_bleu_1,
            'micro_rouge_l': avg_rouge_l,
            'micro_bert_score': avg_bert_score,
            'micro_sbert_score': avg_sbert_score
        }

    def calculate_task_level_question_quality_metrics(self) -> Dict[str, Any]:
        task_results = defaultdict(list)
        for result in self.results:
            annotation_id = result['annotation_id']
            if annotation_id:
                task_results[annotation_id].append(result)

        if not task_results:
            return {}

        task_quality_metrics = []

        for annotation_id, samples in task_results.items():
            correct_predictions = []

            for result in samples:
                pred = result['pred_clarification_need']
                truth = result['gt_clarification_need']

                if pred and truth:
                    generated = result['pred_clarification_question']
                    reference = result['gt_clarification_question']

                    if generated and reference:
                        correct_predictions.append({
                            'generated': generated,
                            'reference': reference
                        })

            if not correct_predictions:
                continue

            gen_lengths = [len(p['generated'].split()) for p in correct_predictions]
            ref_lengths = [len(p['reference'].split()) for p in correct_predictions]
            task_avg_gen_length = sum(gen_lengths) / len(gen_lengths) if gen_lengths else 0
            task_avg_ref_length = sum(ref_lengths) / len(ref_lengths) if ref_lengths else 0

            bleu_1_scores = []
            rouge_l_scores = []
            bert_scores = []
            sbert_scores = []

            candidates = [pred['generated'] for pred in correct_predictions]
            references = [pred['reference'] for pred in correct_predictions]

            if candidates and references:
                smoothing = SmoothingFunction().method1
                for pred in correct_predictions:
                    candidate_tokens = pred['generated'].lower().split()
                    reference_tokens = pred['reference'].lower().split()
                    if candidate_tokens and reference_tokens:
                        bleu_1 = sentence_bleu([reference_tokens], candidate_tokens, weights=(1.0, 0, 0, 0), smoothing_function=smoothing)
                        bleu_1_scores.append(bleu_1)

                for pred in correct_predictions:
                    rouge_scores = self.rouge_scorer.score(pred['reference'], pred['generated'])
                    rouge_l_scores.append(rouge_scores['rougeL'].fmeasure)

                _, _, F1 = bert_score(candidates, references, lang='en', verbose=False)
                bert_scores = F1.tolist()

                ref_embeddings = self.sbert_model.encode(references)
                cand_embeddings = self.sbert_model.encode(candidates)

                for i in range(len(candidates)):
                    similarity = cosine_similarity([ref_embeddings[i]], [cand_embeddings[i]])[0][0]
                    sbert_scores.append(float(similarity))

            task_bleu = sum(bleu_1_scores) / len(bleu_1_scores) if bleu_1_scores else 0.0
            task_rouge = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0.0
            task_bert = sum(bert_scores) / len(bert_scores) if bert_scores else 0.0
            task_sbert = sum(sbert_scores) / len(sbert_scores) if sbert_scores else 0.0

            task_quality_metrics.append({
                'annotation_id': annotation_id,
                'num_correct_predictions': len(correct_predictions),
                'avg_generated_length': task_avg_gen_length,
                'avg_reference_length': task_avg_ref_length,
                'bleu_1': task_bleu,
                'rouge_l': task_rouge,
                'bert_score': task_bert,
                'sbert_score': task_sbert
            })

        if not task_quality_metrics:
            return {}

        avg_gen_length = sum(m['avg_generated_length'] for m in task_quality_metrics) / len(task_quality_metrics)
        avg_ref_length = sum(m['avg_reference_length'] for m in task_quality_metrics) / len(task_quality_metrics)
        avg_bleu = sum(m['bleu_1'] for m in task_quality_metrics) / len(task_quality_metrics)
        avg_rouge = sum(m['rouge_l'] for m in task_quality_metrics) / len(task_quality_metrics)
        avg_bert = sum(m['bert_score'] for m in task_quality_metrics) / len(task_quality_metrics)
        avg_sbert = sum(m['sbert_score'] for m in task_quality_metrics) / len(task_quality_metrics)

        return {
            'num_tasks_with_correct_predictions': len(task_quality_metrics),
            'macro_avg_generated_length': avg_gen_length,
            'macro_avg_reference_length': avg_ref_length,
            'macro_bleu_1': round(avg_bleu, 2),
            'macro_rouge_l': round(avg_rouge, 2),
            'macro_bert_score': round(avg_bert, 2),
            'macro_sbert_score': round(avg_sbert, 2)
        }

    def calculate_all_metrics(self) -> Dict[str, Any]:
        metrics = {
            'basic': self.calculate_basic_metrics(),
            'classification_micro': self.calculate_classification_metrics(),
            'classification_macro': self.calculate_task_level_classification_metrics(),
            'question_quality_micro': self.calculate_question_quality_metrics(),
            'question_quality_macro': self.calculate_task_level_question_quality_metrics(),
            'api': self.calculate_api_metrics()
        }

        return metrics

    def save_metrics(self, output_file: str):
        metrics = self.calculate_all_metrics()

        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=4)

        logger.info(f"Saved metrics to {output_file}")