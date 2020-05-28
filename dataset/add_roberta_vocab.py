import os
import argparse
from tqdm import tqdm
from transformers import RobertaTokenizer
import main.utils as utils


def main(args):
    # Load common vocab
    common_vocab = utils.load_vocab(args.common_vocab)

    # Go through each subword in common vocab and tokenize it with Roberta tokenizer.
    # If the subword consists of multiple tokens, don't include it in the updated common vocab
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    new_common_vocab = []
    for subword in tqdm(common_vocab):
        subword_rob = ' ' + str(subword).strip()
        tokens = tokenizer.tokenize(subword_rob)
        if len(tokens) == 1:
            # print(subword, tokens[1:-1])
            new_common_vocab.append(subword)

    print('New common vocab size:', len(new_common_vocab))
    filepath = os.path.join(args.out_dir, 'common_vocab_cased_roberta.txt')
    with open(filepath, 'w+') as f_out:
        for subword in new_common_vocab:
            f_out.write(subword + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get the intersection of common vocab and RoBERTa vocab')
    parser.add_argument('common_vocab', type=str, help='File containing common vocab subset')
    parser.add_argument('out_dir', type=str, help='Directory to save new common vocab to')
    args = parser.parse_args()
    main(args)
