# -*- coding: utf-8 -*-
import argparse

from transformers import AutoTokenizer
from transformers.configuration_bert import BertConfig
import torch
from utils import *
import json
import logging
from tqdm import tqdm
import os
import sys
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default='bert-base-chinese')
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument(
    "--fp16",
    action="store_true",
)
parser.add_argument("--data_path", type=str)
parser.add_argument(
    "--output_dir",
    type=str,
    help="save path of models"
)
parser.add_argument(
    '--split_layers',
    type=int,
    default=12
)
args = parser.parse_args()
logging.info(args)
data_path = args.data_path
model_type = args.model_type
batch_size = args.batch_size

test_data = load_file(os.path.join(data_path,'valid.{}.obj'.format(model_type.replace('/', '.'))))

dataset_type = data_path.split('/')[-1]
from model import Bert4cosine as bertmodel
model = bertmodel(model_type, layers=args.split_layers).cuda() 

tokenizer = AutoTokenizer.from_pretrained(model_type)

parameters = torch.load(os.path.join(args.output_dir,'checkpoint.{}.th'.format(model_type.replace('/', '.'))),map_location='cpu')
model.load_state_dict(parameters,strict=False)
model.cuda()
if args.fp16:
    try:
        from apex import amp
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    [model] = amp.initialize([model], opt_level='O1', verbosity=0)
model.eval()
total = len(test_data)
cosines = 0.
with torch.no_grad():
    for i in tqdm(range(0, total, batch_size)):
        seq = [x[0] for x in test_data[i:i + batch_size]]
        labels = [x[1] for x in test_data[i:i + batch_size]]
        
        attention_mask = [x[2] for x in test_data[i:i + batch_size]]
        token_type_ids = [x[3] for x in test_data[i:i + batch_size]]

        if dataset_type == "RACE":
            seq, attention_mask,token_type_ids = \
                RACE_padding(seq, pads=tokenizer.pad_token_id, max_len=512,attention_mask = attention_mask, token_type_ids = token_type_ids)
        else:    
            seq, attention_mask,token_type_ids = \
                padding(seq, pads=tokenizer.pad_token_id, max_len=512,attention_mask = attention_mask, token_type_ids = token_type_ids)
        
        seq = torch.LongTensor(seq).cuda()
        attention_mask = torch.LongTensor(attention_mask).cuda() if attention_mask is not None else None
        token_type_ids = torch.LongTensor(token_type_ids).cuda() if token_type_ids is not None else None
        if dataset_type == "RACE":
            shape = [-1] + list(seq.size())[2:]
            seq = seq.view(shape)
            attention_mask = attention_mask.view(shape) if attention_mask is not None else None
            token_type_ids = token_type_ids.view(shape) if token_type_ids is not None else None
        cosine = model([seq,None,attention_mask,token_type_ids])
        cosines += cosine.sum().item()


if dataset_type == "RACE":
    print("The similarity is {}".format(cosines/total/4))
else:
    print("The similarity is {}".format(cosines/total))
