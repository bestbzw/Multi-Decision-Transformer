# -*- coding: utf-8 -*-
import json
import random
import csv
from transformers import BertTokenizer
from transformers import AutoTokenizer

from utils import *

tokenizer = None

import os

def get_one_sample_features(one):
    #one = one.strip().split("\t")
    
    title = one[1]
    sentence = one[2]
    label = int(one[0]) - 1

    tokens_1 = tokenizer.tokenize(title)
    tokens_2 = tokenizer.tokenize(sentence)
    seq = [tokenizer.cls_token] + tokens_1 + [tokenizer.sep_token]
    token_type_ids = [0] * len(seq)
    seq += tokens_2 + [tokenizer.sep_token]
    token_type_ids += [1] * (len(seq)-len(token_type_ids))

    seq_ids = tokenizer.convert_tokens_to_ids(seq)
    attention_mask = [1] * len(seq_ids)
    assert len(token_type_ids) == len(attention_mask)

    return [seq_ids,label, attention_mask, token_type_ids]


def convert_to_features(filename):
    #with open(filename, 'r') as f:
    #    lines = [line.strip().split('\t') for line in f.readlines() if len(line.split('\t')) == 2]
    lines = open(filename).readlines()
    lines = list(csv.reader(lines))
    data = multi_process(get_one_sample_features, lines)
    print('get {} with {} samples'.format(filename, len(data)))
    return data


def prepare_bert_data(model_type='bert-base-chinese', data_path="/home/bzw/data/douban"):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    if not os.path.exists(os.path.join(data_path, 'test.{}.obj'.format(model_type.replace('/', '.')))):
        test_data = convert_to_features(os.path.join(data_path, 'test.csv'))
        dump_file(test_data, os.path.join(data_path, 'test.{}.obj'.format(model_type.replace('/', '.'))))
    if not os.path.exists(os.path.join(data_path, 'valid.{}.obj'.format(model_type.replace('/', '.')))):
        valid_data = convert_to_features(os.path.join(data_path, 'test.csv'))
        dump_file(valid_data, os.path.join(data_path, 'valid.{}.obj'.format(model_type.replace('/', '.'))))
    if not os.path.exists(os.path.join(data_path, 'train.{}.obj'.format(model_type.replace('/', '.')))):
        train_data = convert_to_features(os.path.join(data_path, 'train.csv'))
        dump_file(train_data, os.path.join(data_path, 'train.{}.obj'.format(model_type.replace('/', '.'))))
