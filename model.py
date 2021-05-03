# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
class Bert4ReCO(nn.Module):
    def __init__(self, model_type,num_class,cls_token_id,layers=12):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_type)
        self.n_hidden = self.encoder.config.hidden_size
        self.predictions = nn.ModuleList([nn.Linear(self.n_hidden, 1, bias=False) for _ in range(self.encoder.config.num_hidden_layers)])

        self.num_class = num_class
        self.cls_token_id = cls_token_id
        self.split_layers = layers

    def forward(self, inputs,layer=None):
        [seq, label,attention_mask, token_type_ids] = inputs
        
        layers = self.encoder.config.num_hidden_layers
        
        hiddens = self.encoder(
                            input_ids = seq,
                            attention_mask = attention_mask,
                            output_hidden_states=True,
                            token_type_ids = token_type_ids)[2]
        if self.split_layers == layers:
            hidden_list = hiddens[1:]
        else:
            assert layers%self.split_layers == 0
            hidden_list = [hiddens[(1+l)*int(layers/self.split_layers)] for l in range(self.split_layers)]

        mask_idx = torch.eq(seq, self.cls_token_id)  # 1 is the index in the seq we separate each candidates.
            
            
        losses = []
        output = []
        probs = []
        for layer,hidden in enumerate(hidden_list): 
            hidden = hidden.masked_select(mask_idx.unsqueeze(2).expand_as(hidden)).view(
            -1, self.num_class, self.n_hidden)  # (B, 3, hidden_dim)
            hidden = self.predictions[layer](hidden).squeeze(-1)  # (B, 3, 1) => (B, 3)
            if label is not None:
                loss =  F.cross_entropy(hidden, label)
                losses.append(loss)
            else:
                output.append(hidden.argmax(1))
                probs.append(hidden.softmax(dim=1))

        if label is None:
            return output, probs
        return (sum(losses)).sum()            

class Bert_basic(nn.Module):
    def __init__(self, model_type, num_class, layers=12):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_type)
        self.n_hidden = self.encoder.config.hidden_size
        self.predictions = nn.ModuleList([nn.Linear(self.n_hidden, num_class) for i in range(self.encoder.config.num_hidden_layers)])
        self.denses = nn.ModuleList([nn.Linear(self.n_hidden,self.n_hidden) for i in range(self.encoder.config.num_hidden_layers)])
        self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob)

        self.split_layers = layers

    def forward(self, inputs):
        [seq, label, attention_mask, token_type_ids] = inputs

        layers = self.encoder.config.num_hidden_layers
        hiddens = self.encoder(
            input_ids=seq,
            attention_mask=attention_mask,
            output_hidden_states=True,
            token_type_ids=token_type_ids)[2]
        if self.split_layers == layers:
            hidden_list = hiddens[1:]
        else:
            assert layers%self.split_layers == 0
            hidden_list = [hiddens[(1+l)*int(layers/self.split_layers)] for l in range(self.split_layers)]

        losses = []
        output = []
        probs = []
        for layer, hidden in enumerate(hidden_list):
            x = self.dropout(hidden[:, 0, :])  # (B, dim)
            x = torch.tanh(self.denses[layer](x))
            x = self.dropout(x)
            hidden = self.predictions[layer](x)
            if label is not None:
                loss = F.cross_entropy(hidden, label)
                losses.append(loss)
            else:
                output.append(hidden.argmax(1))
                probs.append(hidden.softmax(dim=1))
                
        if label is None:
            return output, probs
        else:
            return sum(losses).sum()     
class Bert4RACE(nn.Module):
    def __init__(self, model_type,layers=12):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_type)
        self.n_hidden = self.encoder.config.hidden_size
        self.split_layers = layers        
        
        self.predictions = nn.ModuleList([nn.Linear(self.n_hidden, 1, bias=False) for _ in range(layers)])

        self.denses = nn.ModuleList([nn.Linear(self.n_hidden,self.n_hidden) for _ in range(layers)])
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob)

    def forward(self, inputs,layer=None):
        [input_ids, label,attention_mask, token_type_ids] = inputs
        
        layers = self.encoder.config.num_hidden_layers
        
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        input_ids = input_ids.view(-1, input_ids.size(-1))

        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        
        hiddens = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            token_type_ids=token_type_ids)[2]
        if self.split_layers == layers:
            hidden_list = hiddens[1:]
        else:
            assert layers%self.split_layers == 0
            hidden_list = [hiddens[(1+l)*int(layers/self.split_layers)] for l in range(self.split_layers)]


        output = []
        probs = []
        losses = []
        for layer,hidden in enumerate(hidden_list): 
            pooled_output = self.activation(self.denses[layer](hidden[:,0]))
            pooled_output = self.dropout(pooled_output)
            logits = self.predictions[layer](pooled_output).squeeze(-1)  # (B, 3, 1) => (B, 3)
            reshaped_logits = logits.view(-1, num_choices)
            
            if label is not None:
                loss = F.cross_entropy(reshaped_logits, label)
                losses.append(loss)
            else:
                output.append(reshaped_logits.argmax(1))
                probs.append(reshaped_logits.softmax(dim=1))
        
        if label is None:
            return output, probs
        return (sum(losses)).sum()  

class Bert4cosine(nn.Module):
    def __init__(self, model_type,layers=12):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_type)
        self.n_hidden = self.encoder.config.hidden_size
        self.split_layers = layers        


    def forward(self, inputs):
        [seq, label,attention_mask, token_type_ids] = inputs
        
        layers = self.encoder.config.num_hidden_layers
        
        hiddens = self.encoder(
                            input_ids = seq,
                            attention_mask = attention_mask,
                            output_hidden_states=True,
                            token_type_ids = token_type_ids)[2]
        if self.split_layers == layers:
            hidden_list = hiddens[1:]
        else:
            assert layers%self.split_layers == 0
            hidden_list = [hiddens[(1+l)*int(layers/self.split_layers)] for l in range(self.split_layers)]

        hiddens = []
        cosines = []
        
        for i,hidden in enumerate(hidden_list):
            hidden = hidden[:,0,:]
            hiddens.append(hidden)
        for i,hidden1 in enumerate(hiddens):
            for hidden2  in hiddens[i+1:]:
                cosine =  (hidden1* hidden2).sum(1) / (torch.sqrt((hidden1* hidden1).sum(1)) * torch.sqrt((hidden2 * hidden2).sum(1)))
                cosines.append(cosine.sum(0)) # B
        return sum(cosines)/len(cosines)

