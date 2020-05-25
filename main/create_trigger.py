import os
import sys
import time
import random
import logging
import argparse
import numpy as np
import heapq
import spacy
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from copy import deepcopy
from operator import itemgetter
from transformers import AutoConfig, AutoModelWithLMHead, AutoTokenizer
import constants
import utils

import torch.nn.functional as F

ID_TO_POS = {}

nlp = spacy.load("en_core_web_sm")

# logger = logging.getLogger(__name__)

def hotflip_attack(averaged_grad, embedding_matrix, trigger_token_ids,
                   increase_loss=False, num_candidates=1):
    averaged_grad = averaged_grad.cpu()
    embedding_matrix = embedding_matrix.cpu()
    trigger_token_embeds = torch.nn.functional.embedding(torch.LongTensor(trigger_token_ids),
                                                         embedding_matrix).detach().unsqueeze(0)
    averaged_grad = averaged_grad.unsqueeze(0)
    gradient_dot_embedding_matrix = torch.einsum("bij,kj->bik",
                                                 (averaged_grad, embedding_matrix))
    if not increase_loss:
        gradient_dot_embedding_matrix *= -1    # lower versus increase the class probability.
    if num_candidates > 1: # get top k options
        _, best_k_ids = torch.topk(gradient_dot_embedding_matrix, num_candidates, dim=2)
        return best_k_ids.detach().cpu().numpy()[0]
    _, best_at_each_step = gradient_dot_embedding_matrix.max(2)
    return best_at_each_step[0].detach().cpu().numpy()


# Returns the wordpiece embedding weight matrix
def get_embedding_weight(config, model):
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            if module.weight.shape[0] == config.vocab_size: # only add a hook to wordpiece embeddings, not position embeddings
                return module.weight.detach()


# Add hooks for embeddings
extracted_grads = []
def extract_grad_hook(module, grad_in, grad_out):
    extracted_grads.append(grad_out[0])


def add_hooks(config, model):
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            if module.weight.shape[0] == config.vocab_size: # only add a hook to wordpiece embeddings, not position
                module.weight.requires_grad = True
                module.register_backward_hook(extract_grad_hook)


def get_best_candidates(model, tokenizer, source_tokens, target_tokens, trigger_tokens, trigger_mask, segment_ids, attention_mask, candidates, beam_size, token_to_flip, unique_objects, special_token_ids, device):
    best_cand_loss = sys.maxsize
    best_cand_trigger_tokens = None

    # Filter candidates
    filtered_candidates = []
    for cand in candidates:
        print('CAND:', cand, tokenizer.decode([cand]))
        # Make sure to exclude special tokens like [CLS] from candidates
        if cand in special_token_ids:
            print("Skipping candidate {} because it's a special symbol: {}".format(cand, tokenizer.decode([cand])))
            continue

        # Make sure object/answer token is not included in the trigger -> prevents biased/overfitted triggers for each relation
        if tokenizer.decode([cand]).strip().lower() in unique_objects:
            print("Skipping candidate {} because it's the same as a gold object: {}".format(cand, tokenizer.decode([cand])))
            continue

        # Ignore capitalized word pieces and hopefully this heuristic will deal with proper nouns like Antarctica and ABC
        if any(c.isupper() for c in tokenizer.decode([cand])):
            print("Skipping candidate {} because it's probably a proper noun: {}".format(cand, tokenizer.decode([cand])))
            continue
        
        filtered_candidates.append(cand)

    for cand in filtered_candidates:
        # Replace current token with new candidate
        cand_trigger_tokens = deepcopy(trigger_tokens)
        cand_trigger_tokens[token_to_flip] = cand

        # Get loss of candidate and update current best if it has lower loss
        with torch.no_grad():
            cand_loss = get_loss(model, source_tokens, target_tokens, cand_trigger_tokens, trigger_mask, segment_ids, attention_mask, device).detach().cpu().numpy()
            if cand_loss < best_cand_loss:
                best_cand_loss = cand_loss
                best_cand_trigger_tokens = deepcopy(cand_trigger_tokens)
    
    return best_cand_trigger_tokens, best_cand_loss


def get_loss(model, source_tokens, target_tokens, trigger_tokens, trigger_mask, segment_ids, attention_mask, device):
    batch_size = source_tokens.size()[0]
    trigger_tokens = torch.tensor(trigger_tokens, device=device).repeat(batch_size, 1)
    # Make sure to not modify the original source tokens
    src = source_tokens.clone()
    src = src.masked_scatter_(trigger_mask.to(torch.bool), trigger_tokens).to(device)
    dst = target_tokens.to(device)
    outputs = model(src, masked_lm_labels=dst, token_type_ids=segment_ids, attention_mask=attention_mask)
    loss, pred_scores = outputs[:2]
    return loss


def get_prediction(model, source_tokens, target_tokens, trigger_tokens, trigger_mask, segment_ids, attention_mask, device):
    batch_size = source_tokens.size()[0]
    trigger_tokens = torch.tensor(trigger_tokens, device=device).repeat(batch_size, 1)
    # Make sure to not modify the original source tokens
    src = source_tokens.clone()
    src = src.masked_scatter_(trigger_mask.to(torch.bool), trigger_tokens).to(device)
    dst = target_tokens.to(device)

    src = src[0].unsqueeze(0)
    dst = dst[0].unsqueeze(0)
    segment_ids = segment_ids[0].unsqueeze(0)
    attention_mask = attention_mask[0].unsqueeze(0)

    outputs = model(src, masked_lm_labels=dst, token_type_ids=segment_ids, attention_mask=attention_mask)
    loss, pred_scores = outputs[:2]
    print('PRED SCORES:', pred_scores.size())
    log_probs = F.log_softmax(pred_scores, dim=-1).cpu()

    masked_index = torch.where(dst.squeeze(0) != constants.MASKED_VALUE)[0].detach().cpu().numpy()
    print('MASKED INDEX:', masked_index)

    logits = log_probs[:, masked_index, :]
    print('LOGITS:', logits, logits.size())
    # print('LOG PROBS:', log_probs, log_probs.size())


def build_prompt(tokenizer, pair, trigger_tokens, use_ctx, prompt_format, masking=False):
    prompt_list = []

    if use_ctx:
        sub, obj, context = pair
        # print('SUBJECT: {}, OBJECT: {}, CONTEXT: {}'.format(sub, obj, context))
    else:
        sub, obj = pair

    if masking:
        obj = tokenizer.mask_token

    # Convert triggers from ids to tokens
    print('TRIGGER TOKENS:', trigger_tokens)
    triggers = tokenizer.decode(trigger_tokens)
    print('TRIGGERS:', triggers)

    if use_ctx:
        prompt_list.append(context)

    trigger_idx = 0
    for part in prompt_format:
        if part == 'X':
            prompt_list.append(sub)
        elif part == 'Y':
            prompt_list.append(obj)
        else:
            num_trigger_tokens = int(part)
            # TODO: FIX THIS FOR BERT!!!!!!!!!!!!!!!!!!!!!!!!!!
            # prompt_list.extend(triggers[trigger_idx:trigger_idx+num_trigger_tokens])
            # Update trigger idx
            # trigger_idx += num_trigger_tokens
            prompt_list.append(triggers)

    # Add period
    prompt_list.append('.')
    # Detokenize output and remove hashtags in subwords
    prompt = ' '.join(prompt_list)
    # Combine subwords with the previous word
    # TODO: I DONT THINK ROBERTA HAS HASHTAGS IN SUBWORDSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
    prompt = prompt.replace(' ##', '')
    # Remove hashtags from subword if its in the beginning of the prompt
    prompt = prompt.replace('##', '')
    return prompt


def build_prompt_final(tokenizer, trigger_tokens, prompt_format):
    prompt_list = []

    # Convert triggers from ids to tokens
    triggers = tokenizer.convert_ids_to_tokens(trigger_tokens)

    trigger_idx = 0
    for part in prompt_format:
        if part == 'X':
            prompt_list.append('[X]')
        elif part == 'Y':
            prompt_list.append('[Y]')
        else:
            num_trigger_tokens = int(part)
            prompt_list.extend(triggers[trigger_idx:trigger_idx+num_trigger_tokens])
            # Update trigger idx
            trigger_idx += num_trigger_tokens

    # Add period
    prompt_list.append('.')
    # Detokenize output
    prompt = ' '.join(prompt_list)
    return prompt


def load_pretrained(model_name):
    config = AutoConfig.from_pretrained(args.model_name)
    model = AutoModelWithLMHead.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    return config, model, tokenizer


def set_seed(seed: int):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def run_model(args):
    # TODO: REMOVE
    roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
    roberta.eval()

    set_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config, model, tokenizer = load_pretrained(args.model_name)
    # Don't want model in train mode because triggers will be impacted by dropout and such
    model.eval()
    model.to(device)

    add_hooks(config, model) # add gradient hooks to embeddings
    embedding_weight = get_embedding_weight(config, model) # save the word embedding matrix

    # Number of iterations to wait before early stop if there is no progress on dev set
    dev_patience = args.patience

    # Measure elapsed time of trigger search algorithm
    start = time.time()

    # Load TREx train and dev data
    train_file = os.path.join(args.data_dir, 'train.jsonl')
    train_data = utils.load_TREx_data(args, train_file, tokenizer)
    print('Num samples in train data: {}'.format(len(train_data)))
    dev_file = os.path.join(args.data_dir, 'dev.jsonl')
    dev_data = utils.load_TREx_data(args, dev_file, tokenizer)
    print('Num samples in dev data: {}'.format(len(dev_data)))

    # Get all unique objects from train data and check if hotflip candidates == object later on
    unique_objects = utils.get_unique_objects(train_data, args.use_ctx)
    print('Unique objects:', unique_objects)

    # NOTE: add_prefix_space has no effect on special tokens
    # TODO: add unused BERT tokens to special tokens
    special_token_ids = [
        tokenizer.encode(tokenizer.cls_token, add_special_tokens=False),
        tokenizer.encode(tokenizer.unk_token, add_special_tokens=False),
        tokenizer.encode(tokenizer.sep_token, add_special_tokens=False),
        tokenizer.encode(tokenizer.mask_token, add_special_tokens=False),
        tokenizer.encode(tokenizer.pad_token, add_special_tokens=False)
    ]

    """
    Trigger Initialization Options
    1. Manual -> provide prompt format + manual prompt
    2. Random (manual length OR different format/length) -> provide only prompt format
    """
    # Parse prompt format
    prompt_format = args.format.split('-')
    trigger_token_length = sum([int(x) for x in prompt_format if x.isdigit()])
    if args.manual:
        print('Trigger initialization: MANUAL')
        init_trigger = args.manual
        init_tokens = tokenizer.tokenize(init_trigger)
        trigger_token_length = len(init_tokens)
        trigger_tokens = tokenizer.convert_tokens_to_ids(init_tokens)
    else:
        print('Trigger initialization: RANDOM')
        init_token = tokenizer.convert_tokens_to_ids([constants.INIT_WORD])[0]
        trigger_tokens = np.array([init_token] * trigger_token_length)
        # trigger_tokens = tokenizer.encode([constants.INIT_WORD] * trigger_token_length, add_special_tokens=False, add_prefix_space=True)
    print('Initial trigger tokens: {}, Length: {}'.format(trigger_tokens, trigger_token_length))

    best_dev_loss = 999999
    best_trigger_tokens = None
    best_iter = 0
    train_losses = []
    dev_losses = []
    dev_num_no_improve = 0
    count = 0

    for i in range(args.iters):
        print('Iteration: {}'.format(i))
        end_iter = False
        best_loss_iter = 999999
        counter = 0

        losses_batch_train = []
        # Full pass over training data set in batches of subject-object-context triplets
        for batch in utils.iterate_batches(train_data, args.bsz, True):
            # Tokenize and pad batch
            source_tokens, target_tokens, trigger_mask, segment_ids, attention_mask = utils.make_batch(tokenizer, batch, trigger_tokens, prompt_format, args.use_ctx, device)
            print('SOURCE TOKENS:', source_tokens, source_tokens.size())#, tokenizer.decode(source_tokens.squeeze(0)))
            print('TARGET TOKENS:', target_tokens, target_tokens.size())
            print('TRIGGER MASK:', trigger_mask, trigger_mask.size())
            print('SEGMENT IDS:', segment_ids, segment_ids.size())
            print('ATTENTION MASK:', attention_mask, attention_mask.size())

            # Iteratively update tokens in the trigger
            prev_best_curr_loss = 0
            for token_to_flip in range(trigger_token_length):
                # Early stopping if no improvements to trigger
                if end_iter:
                    continue

                model.zero_grad() # clear previous gradients
                loss = get_loss(model, source_tokens, target_tokens, trigger_tokens, trigger_mask, segment_ids, attention_mask, device)
                loss.backward() # compute derivative of loss w.r.t. params using backprop

                global extracted_grads
                grad = extracted_grads[0]
                bsz, _, emb_dim = grad.size() # middle dimension is number of trigger tokens
                extracted_grads = [] # clear gradients from past iterations
                trigger_mask_matrix = trigger_mask.unsqueeze(-1).repeat(1, 1, emb_dim).to(torch.bool).to(device)
                grad = torch.masked_select(grad, trigger_mask_matrix).view(bsz, -1, emb_dim)

                # Get averaged gradient of current trigger token
                averaged_grad = grad.sum(dim=0)[token_to_flip].unsqueeze(0)
                # Use hotflip (linear approximation) attack to get the top num_candidates
                candidates = hotflip_attack(averaged_grad,
                                            embedding_weight,
                                            [trigger_tokens[token_to_flip]],
                                            increase_loss=False,
                                            num_candidates=args.num_cand)[0]

                best_curr_trigger_tokens, best_curr_loss = get_best_candidates(model,
                                                                    tokenizer,
                                                                    source_tokens,
                                                                    target_tokens,
                                                                    trigger_tokens,
                                                                    trigger_mask,
                                                                    segment_ids,
                                                                    attention_mask,
                                                                    candidates,
                                                                    args.beam_size,
                                                                    token_to_flip,
                                                                    unique_objects,
                                                                    special_token_ids,
                                                                    device)

                # If ALL candidates were invalid, replace best_curr_loss with previous best_curr_loss 
                if best_curr_loss == sys.maxsize:
                    best_curr_loss = prev_best_curr_loss

                prev_best_curr_loss = best_curr_loss
                losses_batch_train.append(best_curr_loss)

                if best_curr_loss < best_loss_iter:
                    counter = 0
                    best_loss_iter = best_curr_loss
                    trigger_tokens = deepcopy(best_curr_trigger_tokens)
                elif counter == len(trigger_tokens) * 2:
                    print('Early stopping: better trigger not found for a while, skipping to next iteration')
                    end_iter = True
                else:
                    counter += 1

                # DEBUGO MODE
                if args.debug:
                    input()

        # Compute average train loss across all batches
        train_loss = best_loss_iter

        # Evaluate on dev set
        losses_batch_dev = []
        for batch in utils.iterate_batches(dev_data, args.bsz, True):
            source_tokens, target_tokens, trigger_mask, segment_ids, attention_mask = utils.make_batch(tokenizer, batch, trigger_tokens, prompt_format, args.use_ctx, device)
            # Don't compute gradient to save memory
            with torch.no_grad():
                loss = get_loss(model, source_tokens, target_tokens, trigger_tokens, trigger_mask, segment_ids, attention_mask, device)
                losses_batch_dev.append(loss.item())
        dev_loss = np.mean(losses_batch_dev)

        # TODO: Print model's prediction on dev sample with current trigger
        # get_prediction(model, source_tokens, target_tokens, trigger_tokens, trigger_mask, segment_ids, attention_mask, device)

        # Early stopping on dev set
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_trigger_tokens = deepcopy(trigger_tokens)
            dev_num_no_improve = 0
            best_iter = i
        else:
            dev_num_no_improve += 1
        if dev_num_no_improve == dev_patience:
            print('Early Stopping: dev loss not decreasing')
            break

        # Only print out train loss, dev loss, and sample prediction before early stopping
        print('Trigger tokens: {}'.format(tokenizer.decode(trigger_tokens)))
        print('Train loss: {}'.format(train_loss))
        print('Dev loss: {}'.format(dev_loss))
        # Store train loss of last batch, which should be the best because we update the same trigger
        train_losses.append(train_loss)
        dev_losses.append(dev_loss)
        # Print model prediction on dev data point with current trigger
        rand_idx = random.randint(0, len(dev_data) - 1) # Follow progress of random dev data pair
        prompt = build_prompt(tokenizer, dev_data[rand_idx], trigger_tokens, args.use_ctx, prompt_format, masking=True)
        print('Prompt:', prompt)
        print('RoBERTa predictions:', roberta.fill_mask(prompt, topk=3))
        # Sanity check
        # original_log_probs_list, [token_ids], [masked_indices], _, _ = model.get_batch_generation([[prompt]], try_cuda=False)
        # print_sentence_predictions(original_log_probs_list[0], token_ids, model.vocab, masked_indices=masked_indices)
        # get_prediction(model, dev_data[rand_idx], trigger_tokens, trigger_mask, segment_ids, device)

    print('Best dev loss: {} (iter {})'.format(round(best_dev_loss, 3), best_iter))
    # print('Best trigger: ', ' '.join(tokenizer.convert_ids_to_tokens(best_trigger_tokens)))
    print('Best prompt: {}'.format(build_prompt_final(tokenizer, best_trigger_tokens, prompt_format)))

    # Measure elapsed time
    end = time.time()
    print('Elapsed time: {} sec'.format(end - start))

    # Plot loss/learning curve
    num_iters = len(train_losses)
    plt.plot(range(num_iters), train_losses)
    plt.plot(range(num_iters), dev_losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Dev'])
    plt.savefig(os.path.join(args.out_dir, 'loss.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    parser.add_argument('out_dir', type=str)
    parser.add_argument('--model_name', type=str, default='bert-base-cased', help='Model name passed to HuggingFace AutoX classes')
    parser.add_argument('--use_ctx', action='store_true', help='Use context sentences for open-book probing')
    parser.add_argument('--iters', type=int, default='100', help='Number of iterations to run trigger search algorithm')
    parser.add_argument('--bsz', type=int, default=32, help='Batch size')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--num_cand', type=int, default=10)
    parser.add_argument('--beam_size', type=int, default=1)
    parser.add_argument('--format', type=str, default='X-5-Y', help='Prompt format')
    parser.add_argument('--manual', type=str, help='Manual prompt')
    parser.add_argument('--debug', default=False, action='store_true')
    args = parser.parse_args()

    # if args.debug:
    #     level = logging.DEBUG
    # else:
    #     level = logging.INFO
    # logging.basicConfig(stream=sys.stdout, level=level)

    run_model(args)
