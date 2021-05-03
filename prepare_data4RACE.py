# -*- coding: utf-8 -*-
import json
import random

from transformers import AutoTokenizer

from utils import *

tokenizer = None

import os


def get_one_sample_features(one):
    
    alternatives = one['alternatives'].split('|')
    if len(alternatives) != 4:
        return []
    label = int(one['answer'])
    query = one['query']
    text_a = one['passage']
    choices_inputs = []
    for ending in alternatives:
        if query.find("_") != -1:
            text_b = query.replace("_", ending)
        else:
            text_b = query + " " + ending
        inputs = {}
        inputs["input_ids"] =  [tokenizer.cls_token_id] \
                            +  tokenizer.encode(text_b,add_special_tokens=False,max_length=128,truncation=True) \
                            +  [tokenizer.sep_token_id]
        inputs["token_type_ids"] = [0] * len(inputs["input_ids"])
        inputs["input_ids"] += tokenizer.encode(text_a, max_length=tokenizer.max_len - len(inputs["input_ids"])-1,truncation=True,add_special_tokens=False) +  [tokenizer.sep_token_id]
        inputs["attention_mask"] = [1] * len(inputs["input_ids"])
        inputs["token_type_ids"] += [1] * (len(inputs["input_ids"]) - len(inputs["token_type_ids"]))
#        inputs = tokenizer(
#            text_b,
#            text_a,
#            add_special_tokens=True,
#            max_length=tokenizer.max_len,
#            #padding="max_length",
#            truncation=True,
#            return_overflowing_tokens=True,
#        )
        
        choices_inputs.append(inputs)
    input_ids = [x["input_ids"] for x in choices_inputs]
    attention_mask = (
        [x["attention_mask"] for x in choices_inputs] if "attention_mask" in choices_inputs[0] else None)
    token_type_ids = (
        [x["token_type_ids"] for x in choices_inputs] if "token_type_ids" in choices_inputs[0] else None)
    
    global_attention_mask = None 
    return [input_ids, label, attention_mask, token_type_ids,global_attention_mask]


def convert_to_features(filename):
    with open(filename, encoding='utf-8') as f:
        raw = json.load(f)
        data = multi_process(get_one_sample_features, raw)
#        for data in raw:
#            get_one_sample_features(data)
    print('get {} with {} samples'.format(filename, len(data)))
    return data


def prepare_bert_data(model_type='bert-base-chinese', data_path="/home/bzw/data/ReCO"):
    assert data_path.split('/')[-1] == 'RACE'
    global tokenizer
    #tokenizer = BertTokenizer.from_pretrained(model_type)
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    if not os.path.exists(os.path.join(data_path,'test.{}.obj'.format(model_type.replace('/', '.')))):
        test_data = convert_to_features(os.path.join(data_path,'test.json'))
        dump_file(test_data, os.path.join(data_path,'test.{}.obj'.format(model_type.replace('/', '.'))))
    if not os.path.exists(os.path.join(data_path,'valid.{}.obj'.format(model_type.replace('/', '.')))):
        valid_data = convert_to_features(os.path.join(data_path,'dev.json'))
        dump_file(valid_data, os.path.join(data_path,'valid.{}.obj'.format(model_type.replace('/', '.'))))
    if not os.path.exists(os.path.join(data_path,'train.{}.obj'.format(model_type.replace('/', '.')))):
        train_data = convert_to_features(os.path.join(data_path,'train.json'))
        dump_file(train_data, os.path.join(data_path,'train.{}.obj'.format(model_type.replace('/', '.'))))
