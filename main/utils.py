import os
import re
import json
import random
import asyncio
import numpy as np
import torch
from copy import deepcopy
from pytorch_transformers import BertTokenizer
import constants

# Set random seed so randomly picking context sentences is consistent across runs
random.seed(0)
np.random.seed(0)

async def map_async(fn, iterator, max_tasks=10, sleep_time=0.01):
    tasks = set()
    
    for x in iterator:
        if len(tasks) >= max_tasks:
            _, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        
        new_task = asyncio.ensure_future(fn(x))
        tasks.add(new_task)
        await asyncio.sleep(random.random() * sleep_time)

    await asyncio.wait(tasks)


def load_TREx_data(args, filename, tokenizer):
    facts = []
    with open(filename, newline='') as f:
        lines = f.readlines()
        num_invalid_facts = 0
        for line in lines:
            sample = json.loads(line)
            sub_label = sample['sub_label']
            obj_label = sample['obj_label']

            # Skip facts with objects that consist of multiple tokens
            if len(tokenizer.tokenize(obj_label)) != 1:
                # print(tokenizer.tokenize(obj_label))
                num_invalid_facts += 1
                continue

            if args.use_ctx:
                # For conditional probing, skip facts that don't have context sentence
                if 'evidences' not in sample:
                    num_invalid_facts += 1
                    continue
                evidences = sample['evidences']
                # Randomly pick a context sentence
                obj_surface, masked_sent = random.choice([(evidence['obj_surface'], evidence['masked_sentence']) for evidence in evidences])
                words = masked_sent.split()
                if len(words) > constants.MAX_CONTEXT_LEN:
                    # If the masked sentence is too long, use the first X tokens (it's ok if obj isn't included)
                    masked_sent = ' '.join(words[:constants.MAX_CONTEXT_LEN])
                
                # If truncated context sentence still has MASK, we need to replace it with object surface but if it left out MASK, it's fine
                context = masked_sent.replace(constants.MASK, obj_surface)
                facts.append((sub_label, obj_label, context))
            else:
                # Facts only consist of sub and obj for unconditional probing
                facts.append((sub_label, obj_label))

        print('Total facts before:', len(lines))
        print('Invalid facts:', num_invalid_facts)
        print('Total facts after:', len(facts))

    return facts


def iterate_batches(inputs, batch_size, shuffle=False):
    """
    Split data into batches and return them as a generator
    """
    size = len(inputs)
    inputs = np.array(inputs)
    if shuffle:
        indices = np.arange(size)
        np.random.shuffle(indices)
    for start_idx in range(0, size, batch_size):
        end_idx = min(start_idx + batch_size, size)
        if shuffle:
            excerpt = indices[start_idx:end_idx]
        else:
            excerpt = slice(start_idx, end_idx)
        yield inputs[excerpt]


def make_batch(tokenizer, batch, trigger_tokens, prompt_format, use_ctx, device):
    """
    For BERT, [CLS] token marks the beginning of a sentence and [SEP] marks separation/end of sentences
    """
    source_tokens_batch = []
    target_tokens_batch = []
    trigger_mask_batch = []
    segment_ids_batch = []

    cls_token = tokenizer.encode(tokenizer.cls_token, add_special_tokens=False)
    sep_token = tokenizer.encode(tokenizer.sep_token, add_special_tokens=False)
    mask_token = tokenizer.encode(tokenizer.mask_token, add_special_tokens=False)
    pad_token = tokenizer.encode(tokenizer.pad_token, add_special_tokens=False)
    period_token = tokenizer.encode('.', add_special_tokens=False)
    
    for sample in batch:
        source_tokens = []
        target_tokens = []
        trigger_mask = []
        segment_ids = [] # used to distinguish different sentences

        if use_ctx:
            sub, obj, ctx = sample
            sub_tokens = tokenizer.encode(sub, add_special_tokens=False, add_prefix_space=False)
            obj_tokens = tokenizer.encode(obj, add_special_tokens=False, add_prefix_space=False)
            ctx_tokens = tokenizer.encode(ctx, add_special_tokens=False, add_prefix_space=False)
        else:
            sub, obj = sample
            sub_tokens = tokenizer.encode(sub, add_special_tokens=False, add_prefix_space=False)
            obj_tokens = tokenizer.encode(obj, add_special_tokens=False, add_prefix_space=False)

        trigger_idx = 0
        # Add CLS token at the beginning
        source_tokens.extend(cls_token)
        target_tokens.append(constants.MASKED_VALUE)
        trigger_mask.append(0)
        # Add context if probe setting is open-book (use context)
        if use_ctx:
            # From CLS token right before
            segment_ids.append(0)
            # Add context tokens
            source_tokens.extend(ctx_tokens)
            target_tokens.extend([constants.MASKED_VALUE] * len(ctx_tokens))
            trigger_mask.extend([0] * len(ctx_tokens))
            segment_ids.extend([0] * len(ctx_tokens))
            # Add SEP token to distinguish sentences
            source_tokens.extend(sep_token)
            target_tokens.append(constants.MASKED_VALUE)
            trigger_mask.append(0)
            segment_ids.append(0)
        
        # Keep track of length of first sentence only for conditional probing which uses two sentences
        first_sent_len = len(segment_ids)

        for part in prompt_format:
            if part == 'X':
                # Add subject
                source_tokens.extend(sub_tokens)
                target_tokens.extend([constants.MASKED_VALUE] * len(sub_tokens))
                trigger_mask.extend([0] * len(sub_tokens))
            elif part == 'Y':
                # Add MASKED object
                source_tokens.extend(mask_token)
                target_tokens.extend(obj_tokens)
                trigger_mask.extend([0] * len(obj_tokens))
            else:
                # Add triggers
                num_trigger_tokens = int(part)
                source_tokens.extend(trigger_tokens[trigger_idx:trigger_idx+num_trigger_tokens])
                target_tokens.extend([constants.MASKED_VALUE] * (num_trigger_tokens))
                trigger_mask.extend([1] * (num_trigger_tokens))
                # Update trigger idx
                trigger_idx += num_trigger_tokens

        # Add period at end of prompt
        source_tokens.extend(period_token)
        target_tokens.append(constants.MASKED_VALUE)
        trigger_mask.append(0)
        # Add SEP token at the end
        source_tokens.extend(sep_token)
        target_tokens.append(constants.MASKED_VALUE)
        trigger_mask.append(0)

        if use_ctx:
            segment_ids.extend([1] * (len(source_tokens) - first_sent_len))
        else:
            segment_ids.extend([0] * len(source_tokens))

        # Add encoded prompt to batch
        source_tokens_batch.append(torch.tensor(source_tokens))
        target_tokens_batch.append(torch.tensor(target_tokens))
        trigger_mask_batch.append(torch.tensor(trigger_mask))
        segment_ids_batch.append(torch.tensor(segment_ids))

    # Get max length sequence for padding
    seq_len = [s.size(0) for s in source_tokens_batch]
    max_len = np.max(seq_len)

    # Pad the batch
    source_tokens_batch = torch.nn.utils.rnn.pad_sequence(source_tokens_batch, batch_first=True, padding_value=pad_token[0])
    target_tokens_batch = torch.nn.utils.rnn.pad_sequence(target_tokens_batch, batch_first=True, padding_value=constants.MASKED_VALUE)
    trigger_mask_batch = torch.nn.utils.rnn.pad_sequence(trigger_mask_batch, batch_first=True)
    segment_ids_batch = torch.nn.utils.rnn.pad_sequence(segment_ids_batch, batch_first=True, padding_value=pad_token[0])

    # Create attention mask that makes sure that padding is not attended to by the model
    attention_mask_batch = source_tokens_batch.clone()
    attention_mask_batch[attention_mask_batch != pad_token[0]] = 1

    # Move to GPU
    source_tokens_batch = source_tokens_batch.to(device)
    target_tokens_batch = target_tokens_batch.to(device)
    trigger_mask_batch = trigger_mask_batch.to(device)
    segment_ids_batch = segment_ids_batch.to(device)
    attention_mask_batch = attention_mask_batch.to(device)

    return source_tokens_batch, target_tokens_batch, trigger_mask_batch, segment_ids_batch, attention_mask_batch


def get_unique_objects(data, use_ctx=False):
    objs = set()
    for sample in data:
        if use_ctx:
            sub, obj, ctx = sample
        else:
            sub, obj = sample
        # print('sub: {}, obj: {}, ctx: {}'.format(sub, obj, ctx))
        objs.add(obj.lower())
    return list(objs)


def load_vocab(vocab_filename):
    with open(vocab_filename, "r") as f:
        lines = f.readlines()
    vocab = [x.strip() for x in lines]
    return vocab


def get_id_from_url(url):
    """
    Extract Wikidata entity id from URL
    """
    return url.split('/')[-1]
