import os
import json
import argparse
import constants


def main(args):
    rel_to_prompt = {}
    with open(args.in_file, encoding='utf-8', mode='r') as f_in:
        lines = f_in.readlines()
        rel_id = ''
        start_idx = len('Best prompt: ')
        for line in lines:
            # Make sure order is P1001, P101, ...
            if line.startswith('P'):
                rel_id = line.replace('.txt', '').strip()
            elif line.startswith('Best prompt'):
                rel_to_prompt[rel_id] = line[start_idx:].strip()

    filepath = os.path.join(args.out_dir, 'relations.jsonl')
    filepath_L = os.path.join(args.out_dir, 'relations_latex.jsonl') # Latex version
    with open(filepath, encoding='utf-8', mode='w+') as f_out, open(filepath_L, encoding='utf-8', mode='w+') as f_out_L:
        for rel_id in constants.TREX_RELATIONS_ORDERED:
            # prompt = rel_to_prompt[rel_id].replace(' ##', '').replace('##', '')
            # prompt_L = rel_to_prompt[rel_id].replace('#', '\#')
            f_out.write(json.dumps({
                'relation': rel_id,
                'template': rel_to_prompt[rel_id].replace(' ##', '').replace('##', '')
            }, ensure_ascii=False) + '\n')
            f_out_L.write(json.dumps({
                'relation': rel_id,
                'template': rel_to_prompt[rel_id] # .replace('##', '\#\#')
            }, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert aggregated experiment results into JSONL relation templates')
    parser.add_argument('in_file', type=str, help='File containing aggregated experiment results (i.e. each relations best trigger, dev loss, and elapsed time')
    parser.add_argument('out_dir', type=str, help='Directory to store JSONL relation templates (latex version and non-latex version')
    args = parser.parse_args()

    main(args)
