# -*- coding: utf-8 -*-
import json
import random

from transformers import BertTokenizer
from transformers import AutoTokenizer

from utils import *

tokenizer = None

import os

def get_shuffled_answer(alternatives):
    answers_index = [0, 1, 2]
    random.shuffle(answers_index)
    alternatives = [alternatives[x] for x in answers_index]
    label = list(answers_index).index(0)
    return alternatives, label


def get_one_sample_features(one,model_type=None):
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_type)
    alternatives, label = get_shuffled_answer(one['alternatives'].split('|'))
    query = one['query']
    paragraph = clean(one['passage'])
    alt_ids = [y for x in alternatives for y in [tokenizer.cls_token_id] + tokenizer.encode(x,add_special_tokens=False)]
    seq_ids = alt_ids \
                    + [tokenizer.sep_token_id] + tokenizer.encode(query,add_special_tokens=False) + [tokenizer.sep_token_id]
    token_type_ids = [0] * len(seq_ids)   
    global_attention_mask = [1] * len(seq_ids)
    
    seq_ids += tokenizer.encode(paragraph, max_length=tokenizer.max_len - len(seq_ids)-1,truncation=True,add_special_tokens=False)
    seq_ids += [tokenizer.sep_token_id]
    token_type_ids += [1] * (len(seq_ids) - len(token_type_ids))
    global_attention_mask += [0] * (len(seq_ids) - len(token_type_ids))

    attention_mask = [1] * len(seq_ids)
    return [seq_ids, label, attention_mask, token_type_ids, alternatives, global_attention_mask]

def convert_to_features(filename):
    with open(filename, encoding='utf-8') as f:
        raw = json.load(f)
        data = multi_process(get_one_sample_features, raw)
    print('get {} with {} samples'.format(filename, len(data)))
    return data


def prepare_bert_data(model_type='bert-base-chinese', data_path="/home/bzw/data/ReCO"):
    assert data_path.split('/')[-1] == 'ReCO'
    global tokenizer
    if "albert_chinese" in  model_type:
        tokenizer = BertTokenizer.from_pretrained(model_type)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_type)
    if not os.path.exists(os.path.join(data_path,'test.{}.obj'.format(model_type.replace('/', '.')))):
        test_data = convert_to_features(os.path.join(data_path,'ReCO.testa.json'))
        dump_file(test_data, os.path.join(data_path,'test.{}.obj'.format(model_type.replace('/', '.'))))
    if not os.path.exists(os.path.join(data_path,'valid.{}.obj'.format(model_type.replace('/', '.')))):
        valid_data = convert_to_features(os.path.join(data_path,'ReCO.validationset.json'))
        dump_file(valid_data, os.path.join(data_path,'valid.{}.obj'.format(model_type.replace('/', '.'))))
    if not os.path.exists(os.path.join(data_path,'train.{}.obj'.format(model_type.replace('/', '.')))):
        train_data = convert_to_features(os.path.join(data_path,'ReCO.trainingset.json'))
        dump_file(train_data, os.path.join(data_path,'train.{}.obj'.format(model_type.replace('/', '.'))))
