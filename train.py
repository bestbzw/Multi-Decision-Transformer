# -*- coding: utf-8 -*-
import argparse
import torch

from transformers import AutoTokenizer

from utils import *
import torch.distributed as dist
import numpy as np
import logging
import gc
import math
from tqdm import tqdm
import os
import sys
from evaluate_metrics import evaluate
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

class_num = {
    "ReCO":3,
    "RACE":4,
    "book_review":2,
    "lcqmc":2,
    "ag_news":4,
    "dbpedia":14
    }

torch.manual_seed(0)
np.random.seed(0)
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--lr", type=float, default=2.0e-5)
parser.add_argument("--max_grad_norm", type=float, default=0.2)
parser.add_argument("--model_type", type=str, default="voidful/albert_chinese_base")
parser.add_argument("--data_path", type=str)
parser.add_argument(
    "--fp16",
    action="store_true",
#    default=True,
)
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
) 
parser.add_argument(
    "--output_dir",
    type=str,
    help="save path of models"
)
parser.add_argument(
    "--alpha",
    type=float,
    required=True
)
parser.add_argument(
    "--warmup_proportion",
    type=float,
    default = None
)
parser.add_argument(
    "--split_layer",
    type=int,
    default = 12
)


args = parser.parse_args()
logging.info(args)
data_path = args.data_path
model_type = args.model_type
local_rank = args.local_rank

dataset_type = data_path.split('/')[-1]
# import the data process code
exec("from prepare_data4{} import prepare_bert_data".format(dataset_type))

if local_rank >= 0:
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
    torch.cuda.set_device(args.local_rank)

if local_rank in [-1, 0]:
    prepare_bert_data(model_type, data_path)
if local_rank >= 0:
    dist.barrier()  # wait for the first gpu to load data

data = load_file(os.path.join(data_path,'train.{}.obj'.format(model_type.replace('/', '.'))))
valid_data = load_file(os.path.join(data_path,'valid.{}.obj'.format(model_type.replace('/', '.'))))
valid_data = sorted(valid_data, key=lambda x: len(x[0]))
batch_size = args.batch_size
tokenizer = AutoTokenizer.from_pretrained(model_type)

if dataset_type == 'ReCO':
    from model import Bert4ReCO as bertmodel
    model = bertmodel(model_type, class_num[dataset_type], tokenizer.cls_token_id,layers=args.split_layer).cuda()
elif dataset_type == 'RACE':
    from model import Bert4RACE as bertmodel
    model = bertmodel(model_type, layers=args.split_layer).cuda() 
else:
    from model import Bert_basic as bertmodel
    model = bertmodel(model_type, class_num[dataset_type],layers=args.split_layer).cuda()

layers = model.encoder.config.num_hidden_layers

# learning rate decay method
optimizer = optim(layers,args.lr,model,args.alpha,split_layers=args.split_layer)

# warm up
if args.warmup_proportion is not None and args.warmup_proportion!=0.0:
    total_steps = len(data) * args.epoch // (args.gradient_accumulation_steps * args.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer,int(args.warmup_proportion*total_steps),total_steps)

if args.fp16:
    try:
        from apex import amp
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

if local_rank >= 0:
    try:
        import apex
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use parallel training.")
    model = apex.parallel.DistributedDataParallel(model)


def get_shuffle_data():
    pool = {}
    for one in data:
        length = len(one[0]) // 5
        if length not in pool:
            pool[length] = []
        pool[length].append(one)
    for one in pool:
        np.random.shuffle(pool[one])
    length_lst = list(pool.keys())
    np.random.shuffle(length_lst)
    whole_data = [x for y in length_lst for x in pool[y]]
    if local_rank >= 0:
        remove_data_size = len(whole_data) % dist.get_world_size()
        thread_data = [whole_data[x + args.local_rank] for x in
                       range(0, len(whole_data) - remove_data_size, dist.get_world_size())]
        return thread_data
    return whole_data


def iter_printer(total, epoch):
    if local_rank >= 0:
        if local_rank == 0:
            return tqdm(range(0, total, batch_size), desc='epoch {}'.format(epoch))
        else:
            return range(0, total, batch_size)
    else:
        return tqdm(range(0, total, batch_size), desc='epoch {}'.format(epoch))

def train(epoch):
    model.train()
    train_data = get_shuffle_data()
    total = len(train_data)
    step = 0
    for i in iter_printer(total, epoch):
        seq = [x[0] for x in train_data[i:i + batch_size]]
        label = [x[1] for x in train_data[i:i + batch_size]]
        
        attention_mask = [x[2] for x in train_data[i:i + batch_size]]
       
        if "roberta" in model_type.lower() or "longformer" in model_type.lower():
            token_type_ids = None
        else:
            token_type_ids = [x[3] for x in train_data[i:i + batch_size]]
        if dataset_type == "RACE":
            seq, attention_mask,token_type_ids = \
                RACE_padding(seq, pads=tokenizer.pad_token_id, max_len=512,attention_mask = attention_mask, token_type_ids = token_type_ids)
        else:
            seq, attention_mask,token_type_ids = \
                padding(seq, pads=tokenizer.pad_token_id, max_len=512,attention_mask = attention_mask, token_type_ids = token_type_ids)
        seq = torch.LongTensor(seq).cuda()
        attention_mask = torch.LongTensor(attention_mask).cuda() if attention_mask is not None else None
        token_type_ids = torch.LongTensor(token_type_ids).cuda() if token_type_ids is not None else None
        label = torch.LongTensor(label).cuda()
        
        loss = model([seq, label, attention_mask, token_type_ids])

        loss = loss / args.gradient_accumulation_steps

        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()
            if args.warmup_proportion is not None and args.warmup_proportion!=0.0:
                scheduler.step()
        step += 1

def evaluation(epoch):
    model.eval()
    total = len(valid_data)
    
    all_prediction = []
    all_predictions = []
    for i in range(args.split_layer):
        all_predictions.append([])
    labels = [x[1] for x in valid_data]
    with torch.no_grad():
        for i in iter_printer(total, epoch):
            seq = [x[0] for x in valid_data[i:i + batch_size]]
            
            attention_mask = [x[2] for x in valid_data[i:i + batch_size]]
#            token_type_ids = [x[3] for x in valid_data[i:i + batch_size]]

            if "roberta" in model.encoder.config.model_type.lower() or "longformer" in model_type.lower():
                token_type_ids = None
            else:
                token_type_ids = [x[3] for x in valid_data[i:i + batch_size]]            

            if dataset_type == "RACE":
                seq, attention_mask,token_type_ids = \
                    RACE_padding(seq, pads=tokenizer.pad_token_id, max_len=512,attention_mask = attention_mask, token_type_ids = token_type_ids)
            else:
                seq, attention_mask,token_type_ids = \
                    padding(seq, pads=tokenizer.pad_token_id, max_len=512,attention_mask = attention_mask, token_type_ids = token_type_ids)
            
            seq = torch.LongTensor(seq).cuda()
            attention_mask = torch.LongTensor(attention_mask).cuda() if attention_mask is not None else None
            token_type_ids = torch.LongTensor(token_type_ids).cuda() if token_type_ids is not None else None
            
            predictions,probs = model([seq, None, attention_mask, token_type_ids])
            
            predictions = [prediction.cpu().tolist() for prediction in predictions]
            probs = [prob.cpu().numpy() for prob in probs]
            
            for j in range(probs[0].shape[0]):
                ans = -1
                max_prob = 0
                for k in range(len(probs)):
                    if max(probs[k][j]) >= max_prob:
                        ans = predictions[k][j]
                        max_prob = max(probs[k][j])
                all_prediction.append(ans)
            for k,prediction in enumerate(predictions):
                all_predictions[k].extend(prediction)
    rights,right = evaluate(all_prediction,all_predictions,labels,"acc") 
    logging.info('epoch {} eval acc is {}'.format(epoch, right))
    for k in range(len(probs)):
        logging.info('layer {} eval acc is {}'.format(k,rights[k]))
    return right

best_acc = evaluation(-1)
for epo in range(args.epoch):
    train(epo)
    if local_rank == -1 or local_rank == 0:
        accuracy = evaluation(epo)
        if accuracy > best_acc:
            best_acc = accuracy
            logging.info('---- new best_acc = {}'.format(best_acc))
            with open(os.path.join(args.output_dir,'checkpoint.{}.th'.format(model_type.replace('/', '.'))), 'wb') as f:
                state_dict = model.module.state_dict() if args.fp16 else model.state_dict()
                torch.save(state_dict, f)
