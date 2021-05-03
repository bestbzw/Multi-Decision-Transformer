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
from thop import profile
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


class_num = {
    "ReCO":3,
    "RACE":4,
    "book_review":2,
    "lcqmc":2,
    "ag_news":4,
    "dbpedia":14
    }

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
    '--speed',
    type=float,
    default=0
)
parser.add_argument(
    '--split_layers',
    type=int,
    default=12
)
args = parser.parse_args()
print(args)
logging.info(args)
data_path = args.data_path
model_type = args.model_type
batch_size = args.batch_size

tokenizer = AutoTokenizer.from_pretrained(model_type)

dataset_type = data_path.split('/')[-1]
config = BertConfig.from_pretrained(model_type)
if dataset_type == 'ReCO':
    from modeling_bert import BertModel4ReCO as bertmodel
    model = bertmodel(config, class_num[dataset_type], tokenizer.cls_token_id,args.speed,args.split_layers).cuda()
elif dataset_type == 'RACE':
    from modeling_bert import BertModel4RACE as bertmodel
    model = bertmodel(config,class_num[dataset_type], args.speed,args.split_layers).cuda()
else:
    from modeling_bert import BertModel4Basic as bertmodel
    model = bertmodel(model_type, class_num[dataset_type], args.speed,args.split_layers).cuda()

test_data = load_file(os.path.join(data_path,'test.{}.obj'.format(model_type.replace('/', '.'))))


parameters = torch.load(os.path.join(args.output_dir,'checkpoint.{}.th'.format(model_type.replace('/', '.'))),map_location='cpu')
new_parameters = {}
for key,value in parameters.items():
    new_key = key.replace("encoder.encoder.","").replace("encoder.","")
    new_parameters[new_key] = value
del parameters
model.load_state_dict(new_parameters,strict=False)
model.cuda()
if args.fp16:
    try:
        from apex import amp
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    [model] = amp.initialize([model], opt_level='O1', verbosity=0)
model.eval()
total = len(test_data)
right = 0.0
total_flops = 0.

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
        
        inputs = (seq,attention_mask,token_type_ids)
        flops, params = profile(model, inputs, verbose=False)
        total_flops += flops

        probs,prediction = model(
                                    input_ids = seq,
                                    attention_mask = attention_mask,
                                    token_type_ids = token_type_ids)
        probs = [prob.cpu().numpy() for prob in probs]
        prediction = prediction.cpu()
        right += prediction.eq(torch.LongTensor(labels)).sum().item()

acc = 100 * right / total
print("The accuracy is {}%".format(acc))

flops = total_flops/total/1000**2
print("The flops is {}M".format(flops))
