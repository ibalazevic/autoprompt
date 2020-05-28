import os
import json
import argparse
from tqdm import tqdm
from transformers import RobertaTokenizer
import main.utils as utils


def trim_dataset(args):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Make out dir if it doesn't exist
    os.makedirs(args.out_dir, exist_ok=True)

    before_total = after_total = 0
    
    for f in os.listdir(args.data_dir):
        filename = os.fsdecode(f)
        if filename.endswith('.jsonl'):
            rel_id = os.path.basename(filename).replace('.jsonl', '')
            print('Trimming {}'.format(rel_id))
            trimmed_samples = []
            filepath_in = os.path.join(args.data_dir, filename)
            with open(filepath_in, 'r') as f_in:
                lines = f_in.readlines()
                print('Before:', len(lines))
                before_total += len(lines)
                for line in tqdm(lines):
                    sample = json.loads(line)
                    obj_label = sample['obj_label']
                    if len(tokenizer.tokenize(obj_label)) == 1:
                        trimmed_samples.append(sample)
                    else:
                        print(tokenizer.encode(obj_label, add_special_tokens=False))

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
    parser.add_argument('out_dir', type=str, help='Directory to store trimmed dataset')
    args = parser.parse_args()
    trim_dataset(args)
