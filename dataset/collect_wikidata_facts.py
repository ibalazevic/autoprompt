import os
import re
import json
import time
import random
import asyncio
import argparse
import numpy as np
from collections import defaultdict
from nltk import tokenize
from tqdm import tqdm
from qwikidata.linked_data_interface import get_entity_dict_from_api
from qwikidata.entity import WikidataItem, WikidataProperty
from newspaper import Article
from pytorch_transformers import BertTokenizer
import utils
import constants

EID_TO_URL = {}
EID_TO_TEXT = {}
COUNT = 0

async def increment_count():
    global COUNT
    COUNT += 1


async def fetch_url(eid):
    """
    Get entitiy's site link
    """
    global EID_TO_URL
    if eid not in EID_TO_URL:
        try:
            q_dict = get_entity_dict_from_api(eid)
            q = WikidataItem(q_dict)
            url = q.get_sitelinks()["enwiki"]["url"]
            EID_TO_URL[eid] = url
            return url
        except:
            return None
    else:
        return EID_TO_URL[eid]


async def extract_context(url, obj_id, obj_label, sub_label, keyword):
    """
    Extract the context sentence from the Wikipedia page, if there is a sentence that contains sub, obj, and keyword
    """
    global EID_TO_TEXT
    if obj_id not in EID_TO_TEXT:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text
        EID_TO_TEXT[obj_id] = text
    else:
        text = EID_TO_TEXT[obj_id]

    sents = tokenize.sent_tokenize(text)
    # TODO: handle multiple context sentences
    for sent in sents:
        # Make sure object only appears once in sentence because newspaper sometimes combines image captions with main text
        # Ex: "History Saint-Etienne cathedral in Metz , capital of LorraineLorraine's borders have changed often in its long history."
        if sent.count(obj_label) == 1 and keyword in sent:
            sent_no_obj = sent.replace(obj_label, '')
            # Checking for object and the subject handles case when sub_label == obj_label (ex: Luxembourg is capital of Luxembourg)
            if sub_label in sent_no_obj:
                # Clean up sentence
                sent = sent.strip()
                sent = sent.replace('\n', '')
                sent = re.sub("[\[].*?[\]]", "", sent) # remove brackets and their contents since wikipedia has citations like [42]
                return sent

    return None


async def process_fact(args, fact_tuple, pred_id, tokenizer, common_vocab, trex_set, out_file):
    """
    Takes in a tuple of (sub_id, obj_id, sub_label, obj_label) and writes it to out file if it's a valid fact
    """
    # line = query.strip().split('\t')
    # sub_url, sub, obj_url, obj = line
    # sub_id = utils.get_id_from_url(sub_url)
    # obj_id = utils.get_id_from_url(obj_url)

    sub_id, sub_label, obj_id, obj_label = fact_tuple

    # Make sure object, which is already in canonical form at this point, is a single token
    if len(tokenizer.tokenize(obj_label)) != 1:
        return

    # First, make sure fact is not in TREx train/test set
    if (sub_id, obj_id) in trex_set:
        return

    # Make sure object is in common vocab subset
    if obj_label not in common_vocab:
        return

    # Make sure subject is prominent (has a Wikipedia page)
    # try:
    #     q_dict = get_entity_dict_from_api(sub_id)
    #     q = WikidataItem(q_dict)
    #     if not q.get_sitelinks():
    #         return
    # except ValueError:
    #     return

    # Some entities don't have labels so the subject label is the URI
    # if sub_id == sub_label:
    #     return

    print('SUB: {}, PRED: {}, OBJ: {}'.format(sub_label, pred_id, obj_label))

    fact_json = {
        'obj_uri': obj_id,
        'obj_label': obj_label,
        'sub_uri': sub_id,
        'sub_label': sub_label,
        'predicate_id': pred_id,
        'evidences': []
    }

    # Get Wikipedia page of the object
    url = await fetch_url(obj_id)
    if not url:
        # At this point, the subject and object are valid so save the fact without context sentence
        out_file.write(json.dumps(fact_json) + '\n')
        return

    # Extract the context sentence from the Wikipedia page
    ctx = await extract_context(url, obj_id, obj_label, sub_label, args.keyword)
    if not ctx:
        # Save fact that doesn't have context sentence but is still valid
        out_file.write(json.dumps(fact_json) + '\n')
        return

    # Mask object in context sentence
    masked_sent = ctx.replace(obj_label, constants.MASK, 1)
    # If MASK is followed by -ian, -ns, -ese (basically a letter) then skip the context sentence
    # MASK followed by single quote, space, or period is fine
    idx_after_mask = masked_sent.index(constants.MASK) + len(constants.MASK)
    if masked_sent[idx_after_mask].isalpha():
        out_file.write(json.dumps(fact_json) + '\n')
        return

    print('CTX:', ctx)
    print('MASKED:', masked_sent)

    # Finally write fact with context sentence to file
    fact_json['evidences'].append({
        'sub_surface': sub_label,
        'obj_surface': obj_label,
        'masked_sentence': masked_sent
    })
    out_file.write(json.dumps(fact_json) + '\n')

    # Increment global count
    # await increment_count()


async def main(args):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

    assert os.path.basename(os.path.dirname(args.in_file)) == os.path.basename(os.path.dirname(args.out_file))
    pred_id = os.path.basename(os.path.dirname(args.in_file))

    # Load common vocab subset
    common_vocab = utils.load_vocab(args.common_vocab)

    # Go though TREx test set and save every sub-obj pair/fact in a dictionary
    trex_set = set()
    with open(args.trex_test, 'r') as f_in:
        lines = f_in.readlines()
        for line in tqdm(lines):
            sample = json.loads(line)
            trex_set.add((sample['sub_uri'], sample['obj_uri']))

    # Do the same for TREx train set
    with open(args.trex_train, 'r') as f_in:
        lines = f_in.readlines()
        for line in tqdm(lines):
            sample = json.loads(line)
            trex_set.add((sample['sub_uri'], sample['obj_uri']))

    # Load TSV file
    with open(args.in_file, 'r') as f_in, open(args.out_file, 'a+') as f_out:
        lines = f_in.readlines()
        queries = []
        for line in tqdm(lines):
            sample = line.strip().split('\t')
            sub_url, sub_label, obj_url, obj_label = sample
            sub_id = utils.get_id_from_url(sub_url)
            obj_id = utils.get_id_from_url(obj_url)
            queries.append((sub_id, sub_label, obj_id, obj_label))
        
        # Async is good stuff
        random.shuffle(queries)
        await utils.map_async(lambda q: process_fact(args, q, pred_id, tokenizer, common_vocab, trex_set, f_out), queries, args.max_tasks, args.sleep_time)

    # # Get facts with SPARQL query
    # start = time.perf_counter()
    # offset = 0
    # keep_looping = True
    # while keep_looping:
    #     # Exit early if scraping is taking more than half an hour
    #     if time.perf_counter() - start > 1800:
    #         print('Scraping taking more than half an hour. Exiting early.')
    #         break

    #     try:
    #         sparql_query = """
    #         SELECT ?item ?itemLabel ?value ?valueLabel
    #         WHERE { ?item wdt:""" + args.rel + """ ?value SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". } }
    #         LIMIT """ + str(args.query_limit) + """ OFFSET """ + str(offset)
    #         # If there are no more facts to query, this function will return json.decoder.JSONDecodeError which inherits from ValueError
    #         res = return_sparql_query_results(sparql_query)

    #         if 'results' in res and 'bindings' in res['results'] and res['results']['bindings']:
    #             queries = []
    #             for b in res['results']['bindings']:
    #                 sub_id = utils.get_id_from_url(b['item']['value'])
    #                 obj_id = utils.get_id_from_url(b['value']['value'])
    #                 sub_label = b['itemLabel']['value']
    #                 obj_label = b['valueLabel']['value']
    #                 queries.append((sub_id, obj_id, sub_label, obj_label))

    #             with open(args.out_file, 'a+') as f_out:
    #                 await utils.map_async(lambda q: process_fact(q, args, tokenizer, trex_set, common_vocab, f_out), queries, args.max_tasks, args.sleep_time)

    #                 if args.num_samples > 0:
    #                     global COUNT
    #                     if COUNT >= args.num_samples:
    #                         return

    #             offset += args.query_limit
    #             print('OFFSET:', offset)
    #         else:
    #             # Results (bindings) is empty which means WQS ran out of facts
    #             print('Query result is empty.')
    #             keep_looping = False
    #     except json.decoder.JSONDecodeError:
    #         # Case where Wikidata Query Service has no more facts to return
    #         print('Wikidata Query Service ran out of facts.')
    #         keep_looping = False

    # # Measure elapsed time
    # end = time.perf_counter() - start
    # print('Elapsed time: {} sec'.format(round(end, 2)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gather more facts for TREx dataset directly from Wikidata')
    parser.add_argument('in_file', type=str, help='TSV file containing Wikidata subject-relation-object triplets')
    # parser.add_argument('rel', type=str, help='Wikidata relation ID')
    parser.add_argument('out_file', type=str, help='JSONL file with new facts')
    parser.add_argument('--trex_train', type=str, help='Path to TREx TRAIN set collected from the rest of TREx knowledge source')
    parser.add_argument('--trex_test', type=str, help='Path to TREx TEST set for a relation')
    parser.add_argument('--common_vocab', type=str, help='File containing common vocab subset')
    # parser.add_argument('--keywords', nargs='+', type=str, help='Keywords like capital that will help extract relevant context sentences')
    parser.add_argument('--keyword', type=str, default='', help='Keyword like capital that will help extract relevant context sentences')
    parser.add_argument('--num_samples', type=int, default=-1, help='Number of new samples to gather from Wikidata for a particular relation')
    # parser.add_argument('--query_limit', type=int, default=1000, help='SPARQL limit')
    parser.add_argument('--sleep_time', type=float, default=1e-4)
    parser.add_argument('--max_tasks', type=int, default=50)
    args = parser.parse_args()

    # print('Collecting more data for relation {}...'.format(args.rel))
    start = time.perf_counter()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(args))
    # Zero-sleep to allow underlying connections to close
    loop.run_until_complete(asyncio.sleep(0))
    loop.close()
    # Measure elapsed time
    end = time.perf_counter() - start
    print('Elapsed time: {} sec'.format(round(end, 2)))
