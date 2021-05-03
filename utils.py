# -*- coding: utf-8 -*-
import os
import pickle
import re
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import torch

def multi_process(func, lst, num_cores=multiprocessing.cpu_count(), backend='multiprocessing'):
    workers = Parallel(n_jobs=num_cores, backend=backend)
    output = workers(delayed(func)(one) for one in tqdm(lst))
    return [x for x in output if x]


def DBC2SBC(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 0x3000:
            inside_code = 0x0020
        else:
            inside_code -= 0xfee0
        if not (0x0021 <= inside_code <= 0x7e):
            rstring += uchar
            continue
        rstring += chr(inside_code)
    return rstring


def clean(txt):
    txt = DBC2SBC(txt)
    txt = txt.lower()
    return re.sub('\s*', '', txt)


def dump_file(obj, filename):
    f = open(filename, 'wb')
    pickle.dump(obj, f)


def load_file(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def get_dir_files(dirname):
    L = []
    for root, dirs, files in os.walk(dirname):
        for file in files:
            L.append(os.path.join(root, file))
    return L
def RACE_padding(sequence, pads=0, max_len=None, dtype='int32',attention_mask = None, token_type_ids = None, global_attention_mask = None,padding_int=True,num_choice=4):
    
    v_length = []
    for s in sequence:
        v_length+=[len(x) for x in s]# every sequence length
    
    seq_max_len = max(v_length)
    def round_to_power(x):
        return int((x-1)/512 + 1)*512

    if (max_len is None) or (max_len > seq_max_len) :
        if padding_int:
            max_len = round_to_power(seq_max_len) if max_len is None else min(max_len,round_to_power(seq_max_len))
        else:
            max_len = seq_max_len
    
    x = (np.ones((len(sequence),num_choice,max_len)) * pads).astype(dtype)
    x_attention_mask = (np.ones((len(sequence),num_choice,max_len)) * 0.).astype(dtype) if attention_mask is not None else None
    x_token_type_ids = (np.ones((len(sequence),num_choice,max_len)) * 0.).astype(dtype) if token_type_ids is not None else None
    x_global_attention_mask = (np.ones((len(sequence),num_choice, max_len)) * 0.).astype(dtype) if global_attention_mask is not None else None
    
    for idx, s in enumerate(sequence):
        for i in range(num_choice):
            trunc = s[i][:max_len]
            x[idx,i, :len(trunc)] = trunc
            if attention_mask is not None:
                x_attention_mask[idx,i,:len(trunc)] = attention_mask[idx][i][:max_len] 
            if token_type_ids is not None:
               x_token_type_ids[idx,i,:len(trunc)] = token_type_ids[idx][i][:max_len] 
            if global_attention_mask is not None:
                x_global_attention_mask[idx,i,:len(trunc)] = global_attention_mask[idx][i][:max_len]
    return x, x_attention_mask, x_token_type_ids

def padding(sequence, pads=0, max_len=None, dtype='int32',attention_mask = None, token_type_ids = None):
    v_length = [len(x) for x in sequence]  # every sequence length
    seq_max_len = max(v_length)
    if (max_len is None) or (max_len > seq_max_len):
        max_len = seq_max_len
    #max_len=128 
    x = (np.ones((len(sequence), max_len)) * pads).astype(dtype)
    x_attention_mask = (np.ones((len(sequence), max_len)) * 0.0).astype(dtype) if attention_mask is not None else None
    x_token_type_ids = (np.ones((len(sequence), max_len)) * 0.0).astype(dtype) if token_type_ids is not None else None
    
    for idx, s in enumerate(sequence):
        trunc = s[:max_len]
        x[idx, :len(trunc)] = trunc
        if attention_mask is not None:
           x_attention_mask[idx,:len(trunc)] = attention_mask[idx][:max_len] 
        if token_type_ids is not None:
           x_token_type_ids[idx,:len(trunc)] = token_type_ids[idx][:max_len] 
            
    return x, x_attention_mask, x_token_type_ids
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):


    lambda1 = (lambda epoch:float(epoch) / float(max(1, num_warmup_steps)) if epoch < num_warmup_steps \
                else max(0.0, float(num_training_steps - epoch) / float(max(1, num_training_steps - num_warmup_steps))))
    
    return LambdaLR(optimizer, lambda1, last_epoch)

def optim(n_layers,lr,model,alpha=5.,split_layers=12):
    params = dict(model.named_parameters())

    layer_params =  [] 
    for _ in range(n_layers):
        layer_params.append([])
    
    layer_params[0].extend(list(model.encoder.embeddings.parameters()))
    for key,value in params.items():
        layer = re.findall(r"encoder\.encoder\.layer\.(.*?)\.",key)
        if len(layer) == 1:    
            exec("layer_params[{}].append(value)".format(int(layer[0])))

    for key,value in params.items():
        if "predictions." in key:
            layer_params[-1].append(value)
        if "denses." in key:
            layer_params[-1].append(value)
    layer_params[-1].extend(list(model.encoder.pooler.parameters()))
   
    length = [len(p) for p in layer_params]
    assert sum(length) == len(params)

    params_list = []
    one_layer_block_num = n_layers/split_layers 
    for i,layer_param in enumerate(layer_params):
        m = alpha ** (split_layers-i//one_layer_block_num-1)
        params_list.append({'params':layer_param,'lr':lr/m})

    optimizer2 = torch.optim.AdamW(params_list,
                                  weight_decay=0.01)
    return optimizer2 
            
