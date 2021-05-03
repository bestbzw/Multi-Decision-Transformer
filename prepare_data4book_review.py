# -*- coding: utf-8 -*-
import json
import random

from transformers import BertTokenizer
from transformers import AutoTokenizer

from utils import *

tokenizer = None

import os

def get_one_sample_features(one,model_type=None):
    #global tokenizer
    #if tokenizer is None:
    #    tokenizer = AutoTokenizer.from_pretrained(model_type)
        
    seq_ids = tokenizer.encode(one[1], max_length=tokenizer.max_len - 2, truncation=True)

    token_type_ids = [0] * len(seq_ids)
    attention_mask = [1] * len(seq_ids)
    
    global_attention_mask = [1] + [0]*(len(seq_ids) - 1)

    return [seq_ids, int(float(one[0])), attention_mask, token_type_ids,global_attention_mask]


def convert_to_features(filename):
    with open(filename, 'r') as f:
        lines = [line.strip().split('\t') for line in f.readlines()[1:] if len(line.split('\t')) == 2]

    data = multi_process(get_one_sample_features, lines)
    print('get {} with {} samples'.format(filename, len(data)))
    return data


def prepare_bert_data(model_type='bert-base-chinese', data_path="/home/bzw/data/douban"):
    assert data_path.split('/')[-1] == 'douban'
    global tokenizer
    if "albert_chinese" in  model_type:
        tokenizer = BertTokenizer.from_pretrained(model_type)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_type)
    if not os.path.exists(os.path.join(data_path, 'test.{}.obj'.format(model_type.replace('/', '.')))):
        test_data = convert_to_features(os.path.join(data_path, 'test.tsv'))
        dump_file(test_data, os.path.join(data_path, 'test.{}.obj'.format(model_type.replace('/', '.'))))
    if not os.path.exists(os.path.join(data_path, 'valid.{}.obj'.format(model_type.replace('/', '.')))):
        valid_data = convert_to_features(os.path.join(data_path, 'dev.tsv'))
        dump_file(valid_data, os.path.join(data_path, 'valid.{}.obj'.format(model_type.replace('/', '.'))))
    if not os.path.exists(os.path.join(data_path, 'train.{}.obj'.format(model_type.replace('/', '.')))):
        train_data = convert_to_features(os.path.join(data_path, 'train.tsv'))
        dump_file(train_data, os.path.join(data_path, 'train.{}.obj'.format(model_type.replace('/', '.'))))
