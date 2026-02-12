import os
import pickle
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


def write_pickle(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


def generate_pickle(eval_dir, save_path):
    split_files = ['train', 'test_task', 'test_website', 'test_domain']

    all_dict = {"scores": {}, "ranks": {}}

    total_candidates = 0
    total_samples = 0

    for split_name in split_files:
        file_path = os.path.join(eval_dir, f'scores_{split_name}.pkl')

        logger.info(f"Loading {file_path}")
        data = load_pickle(file_path)

        split_samples = len(data['scores'])
        split_candidates = 0
        for id, res in data['scores'].items():
            split_candidates += len(res)

        total_samples += split_samples
        total_candidates += split_candidates
        logger.info(f"  - {split_name}: {split_samples} samples, {split_candidates} candidates")

        for key in ["scores", "ranks"]:
            for id, res in data[key].items():
                all_dict[key].setdefault(id, {}).update(res)

    final_samples = len(all_dict['scores'])
    final_candidates = sum(len(res) for res in all_dict['scores'].values())

    logger.info(f"Sum of splits: {total_samples} samples, {total_candidates} candidates")
    logger.info(f"Final merged: {final_samples} samples, {final_candidates} candidates")

    write_pickle(all_dict, save_path)

    return all_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_mode', type=str, default='full')
    args = parser.parse_args()
    
    eval_dir = f'eval/candidate/{args.eval_mode}'
    save_path = f'{eval_dir}/scores_all.pkl'

    all_dict = generate_pickle(eval_dir, save_path)
    
    logger.info(f"Generated {len(all_dict['scores'])} samples")
    logger.info(f"Total score entries: {len(all_dict['scores'])}")
    logger.info(f"Total rank entries: {len(all_dict['ranks'])}")


if __name__ == "__main__":
    main()