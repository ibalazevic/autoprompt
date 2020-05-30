import os
import json
import argparse
from tqdm import tqdm
from transformers import RobertaTokenizer
import main.utils as utils


def load_blacklist(filepath):
    blacklist = []
    with open(filepath, 'r') as f_in:
        lines = f_in.readlines()
        for line in lines:
            sample = json.loads(line)
            sub_uri = sample['sub_uri']
            pred_id = sample['predicate_id']
            obj_uri = sample['obj_uri']
            blacklist.append((sub_uri, pred_id, obj_uri))
    return blacklist


def trim_dataset(args):
    # tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Make out dir if it doesn't exist
    os.makedirs(args.out_dir, exist_ok=True)

    before_total = after_total = 0
    
    for f in os.listdir(args.data_dir):
        filename = os.fsdecode(f)
        if filename.endswith('.jsonl'):
            rel_id = os.path.basename(filename).replace('.jsonl', '')

            # Skip P527 and P1376 because RE doesn't consider them
            if rel_id == 'P527' or rel_id == 'P1376':
                print('Skipping {}'.format(rel_id))
                continue

            print('Trimming {}'.format(rel_id))
            trimmed_samples = []
            filepath_bl = os.path.join(args.blacklist_dir, filename)
            blacklist = load_blacklist(filepath_bl)
            filepath_in = os.path.join(args.data_dir, filename)
            with open(filepath_in, 'r') as f_in:
                lines = f_in.readlines()
                print('Before:', len(lines))
                before_total += len(lines)
                for line in tqdm(lines):
                    sample = json.loads(line)
                    sub_uri = sample['sub_uri']
                    pred_id = sample['predicate_id']
                    obj_uri = sample['obj_uri']
                    # obj_label = sample['obj_label']
                    # if len(tokenizer.tokenize(obj_label)) == 1:
                    #     trimmed_samples.append(sample)
                    # else:
                    #     print(tokenizer.encode(obj_label, add_special_tokens=False))
                    if (sub_uri, pred_id, obj_uri) not in blacklist:
                        trimmed_samples.append(sample)

            print('After:', len(trimmed_samples))
            after_total += len(trimmed_samples)
            filepath_out = os.path.join(args.out_dir, rel_id + '.jsonl')
            with open(filepath_out, 'w+') as f_out:
                for sample in trimmed_samples:
                    f_out.write(json.dumps(sample) + '\n')
            print()

    print('Num samples before trimming:', before_total)
    print('Num samples after trimming:', after_total)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get the intersection of common vocab and RoBERTa vocab')
    parser.add_argument('data_dir', type=str, help='Directory containing TREx test set')
    parser.add_argument('blacklist_dir', type=str, help='Blacklist that contains facts that should be filtered out')
    parser.add_argument('out_dir', type=str, help='Directory to store trimmed dataset')
    args = parser.parse_args()
    trim_dataset(args)
