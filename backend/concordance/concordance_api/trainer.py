# Libraries
from __future__ import unicode_literals, print_function, division
from io import open
import math
import pickle
from os import listdir
from os.path import join, isfile
from datetime import datetime
from IPython.display import Image
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
import json
import os
import time
import random
# Models
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, BertModel

# Training
import torch.optim as optim

import re
# from GetWordMap import WordMapper
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = xm.xla_device()
#t1 = torch.ones(3, 3, device = device)
# print(t1)
BERT_NAME = "bert-base-multilingual-cased"
SRC_LANG = 'en_sent'
TRG_LANG = 'vi_sent'
THRESH = 0.4
LABEL = 'envialign'


def convert_single_example(tokenizer, example):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    max_vi_len = 128
    max_en_len = 128
    vi_tokens_a = example['vi_sent'].split()
    if len(vi_tokens_a) > max_vi_len - 2:
        vi_tokens_a = vi_tokens_a[0: (max_vi_len - 2)]

    # Token map will be an int -> int mapping between the `orig_tokens` index and
    # the `bert_tokens` index.

    # bert_tokens == ["[CLS]", "john", "johan", "##son", "'", "s", "house", "[SEP]"]
    # orig_to_tok_map == [1, 2, 4, 6]
    vi_orig_to_tok_map = []
    vi_tokens = []
    vi_segment_ids = []
    vi_head_mask = []

    vi_tokens.append("[CLS]")
    vi_segment_ids.append(0)
    vi_orig_to_tok_map.append(len(vi_tokens) - 1)
    vi_head_mask.append(1)
    # print(len(vi_tokens_a))
    for token in vi_tokens_a:
        if len(token) > 1:
            token = re.sub('_', ' ', token)
        subtokens = tokenizer.tokenize(token)
        vi_orig_to_tok_map.append(len(vi_tokens))
        vi_tokens.extend(subtokens)
        # vi_orig_to_tok_map.append(len(vi_tokens)-1)
        vi_segment_ids.extend([0 for _ in range(len(subtokens))])
        vi_head_mask.append(1)
        vi_head_mask.extend([0 for _ in range(len(subtokens) - 1)])
    vi_tokens.append("[SEP]")
    vi_segment_ids.append(0)
    vi_orig_to_tok_map.append(len(vi_tokens) - 1)
    vi_head_mask.append(1)
    vi_input_ids = tokenizer.convert_tokens_to_ids(vi_tokens)
    assert len(vi_orig_to_tok_map) == len(vi_tokens_a) + 2
    assert len(vi_input_ids) == len(vi_tokens)
    # print(len(orig_to_tok_map), len(tokens), len(input_ids), len(segment_ids)) #for debugging

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    vi_input_mask = [1] * len(vi_input_ids)
    en_tokens_a = example['en_sent'].split()
    if len(en_tokens_a) > max_en_len - 2:
        en_tokens_a = en_tokens_a[0: (max_en_len - 2)]

    en_orig_to_tok_map = []
    en_tokens = []
    en_segment_ids = []
    en_head_mask = []
    en_tokens.append("[CLS]")
    en_segment_ids.append(0)
    en_orig_to_tok_map.append(len(en_tokens) - 1)
    en_head_mask.append(1)
    # print(len(en_tokens_a))
    for token in en_tokens_a:
        subtokens = tokenizer.tokenize(token)
        en_orig_to_tok_map.append(len(en_tokens))
        en_tokens.extend(subtokens)
        # en_orig_to_tok_map.append(len(en_tokens)-1)
        en_segment_ids.extend([0 for _ in range(len(subtokens))])
        en_head_mask.append(1)
        en_head_mask.extend([0 for _ in range(len(subtokens) - 1)])
    en_tokens.append("[SEP]")
    en_segment_ids.append(1)
    en_orig_to_tok_map.append(len(en_tokens) - 1)
    en_head_mask.append(1)
    en_input_ids = tokenizer.convert_tokens_to_ids(en_tokens)
    assert len(en_orig_to_tok_map) == len(en_tokens_a) + 2
    assert len(en_input_ids) == len(en_tokens)
    en_input_mask = [1] * len(en_input_ids)
    vi_len = len(vi_tokens)
    en_len = len(en_tokens)
    envi_goldalign = None
    envi_goldtable = None
    if 'envialign' in example.keys():
        envi_goldalign = example['envialign']
    else:
        envi_goldalign = [0]*len(vi_tokens_a)
    envi_goldtable = np.zeros((vi_len, en_len), dtype=np.int)
    envi_goldtable[0][0] = 1
    envi_goldtable[vi_len - 1][en_len - 1] = 1
    # if (vi_len != len(envi_goldalign)):
    #     raise ValueError(vi_len, len(envi_goldalign))
    for j, a in enumerate(envi_goldalign):
        envi_goldtable[vi_orig_to_tok_map[j + 1],
                       en_orig_to_tok_map[a]] = 1
    # print(en_tokens)
    # print(en_orig_to_tok_map)
    # print(vi_tokens)
    # print(vi_orig_to_tok_map)
    en_sent = example['en_sent']
    vi_sent = example['vi_sent']
    envi_label = None
    if envi_goldtable is not None:
        # print(envi_goldtable)
        # print(envi_goldalign)
        envi_label = np.pad(envi_goldtable, ((0, max_vi_len - vi_len), (0, max_en_len - en_len)), 'constant',
                            constant_values=(0, 0))
    vien_goldalign = None
    vien_goldtable = None
    if 'vienalign' in example.keys():
        vien_goldalign = example['vienalign']
    else:
        vien_goldalign = [0]*len(en_tokens_a)
    vien_goldtable = np.zeros((en_len, vi_len), dtype=np.int)
    vien_goldtable[0, 0] = 1
    vien_goldtable[en_len - 1, vi_len - 1] = 1
    # if (en_len != len(vien_goldalign)):
    #    raise ValueError(en_len, len(vien_goldalign))
    for j, a in enumerate(vien_goldalign):
        vien_goldtable[en_orig_to_tok_map[j + 1],
                       vi_orig_to_tok_map[a]] = 1

    vien_label = None
    if vien_goldtable is not None:
        # print(vien_goldtable)
        # print(vien_goldalign)
        vien_label = np.pad(vien_goldtable,
                            ((0, max_en_len - en_len),
                             (0, max_vi_len - vi_len)), 'constant',
                            constant_values=(0, 0))
    # exit()
    # Zero-pad up to the sequence length.
    # print(vi_tokens)
    # print(en_tokens)
    while len(vi_input_ids) < max_vi_len:
        vi_input_ids.append(0)
        vi_input_mask.append(0)
        vi_segment_ids.append(0)
        vi_head_mask.append(0)
    while len(en_input_ids) < max_en_len:
        en_input_ids.append(0)
        en_input_mask.append(0)
        en_segment_ids.append(0)
        en_head_mask.append(0)
    assert len(vi_input_ids) == max_vi_len
    assert len(vi_input_mask) == max_vi_len
    # print(len(en_segment_ids), max_seq_length)
    assert len(en_segment_ids) == max_en_len
    assert len(en_head_mask) == max_en_len
    # print(vien_ProbIBM1Ex.shape, envi_ConesinEx.shape, enNerFeatureEx.shape, viNerFeatureEx.shape,
    #       viPosFeatureEx.shape, viNerFeatureEx.shape)
    return vi_input_ids, vi_input_mask, vi_segment_ids, vi_head_mask, \
        en_input_ids, en_input_mask, en_segment_ids, en_head_mask, \
        vien_label, envi_label,\
        vi_sent, en_sent, vi_orig_to_tok_map, en_orig_to_tok_map, envi_goldalign, vien_goldalign


def convert_examples_to_features(tokenizer, examples, max_seq_length=256):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    vi_input_ids_lst, vi_input_mask_lst, vi_segment_ids_lst, \
        vi_head_mask_lst, en_input_ids_lst, en_input_mask_lst, en_segment_ids_lst, \
        en_head_mask_lst, vien_label_lst, envi_label_lst, \
        vien_mask_lst, envi_mask_lst, enPos_lst, enNer_lst, viPos_lst, viNer_lst,\
        envi_Conesin_lst, vien_ProbIBM1_lst,\
        visent_lst, ensent_lst,\
        vi_orig_to_tok_map_lst, en_orig_to_tok_map_lst, envi_align_lst, vien_align_lst\
        = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    for example in examples:
        #convert_single_example(tokenizer, example, endict, vidict, envi_ProbDict1, Mapper, max_seq_length=128)
        #
        vi_input_ids, vi_input_mask, vi_segment_ids, vi_head_mask, \
            en_input_ids, en_input_mask, en_segment_ids, en_head_mask, \
            vien_label, envi_label, \
            vi_sent, en_sent,\
            vi_orig_to_tok_map, en_orig_to_tok_map, envi_align, vien_align = convert_single_example(
                tokenizer, example)
        vi_input_ids_lst.append(vi_input_ids)
        vi_input_mask_lst.append(vi_input_mask)
        vi_segment_ids_lst.append(vi_segment_ids)
        vi_head_mask_lst.append(vi_head_mask)
        en_input_ids_lst.append(en_input_ids)
        en_input_mask_lst.append(en_input_mask)
        en_segment_ids_lst.append(en_segment_ids)
        en_head_mask_lst.append(en_head_mask)
        vien_label_lst.append(vien_label)
        envi_label_lst.append(envi_label)
        visent_lst.append(vi_sent)
        ensent_lst.append(en_sent)
        vi_orig_to_tok_map_lst.append(vi_orig_to_tok_map)
        en_orig_to_tok_map_lst.append(en_orig_to_tok_map)
        envi_align_lst.append(envi_align)
        vien_align_lst.append(vien_align)
    return (
        vi_input_ids_lst,
        vi_input_mask_lst,
        vi_segment_ids_lst,
        vi_head_mask_lst,
        en_input_ids_lst,
        en_input_mask_lst,
        en_segment_ids_lst,
        en_head_mask_lst,
        vien_label_lst,
        envi_label_lst,
        visent_lst,
        ensent_lst, vi_orig_to_tok_map_lst,
        en_orig_to_tok_map_lst, envi_align_lst, vien_align_lst)


class BERT_WA(nn.Module):
    def __init__(self, max_en_len=128, max_vi_len=128, en_hidden_size=768, ext_hidden_size=64, out_dim=1):
        super(BERT_WA, self).__init__()
        self.en_hidden_size = en_hidden_size
        self.ext_hidden_size = ext_hidden_size
        self.max_en_len = max_en_len
        self.max_vi_len = max_vi_len
        options_name = BERT_NAME
        self.src_encoder = BertModel.from_pretrained(options_name)
        # self.trg_encoder = BertModel.from_pretrained(options_name)
        self.wa_layer = nn.Linear(
            2 * self.en_hidden_size, self.ext_hidden_size)
        nn.init.uniform_(self.wa_layer.weight, -0.7, 0.7)
        # self.enviext_layer = nn.Linear(2 * self.en_hidden_size, out_dim)
        # nn.init.uniform_(self.enviext_layer.weight, -0.7, 0.7)
        self.envi_out_layer = nn.Linear(self.ext_hidden_size, out_dim)
        nn.init.uniform_(self.envi_out_layer.weight, -0.7, 0.7)
        self.vien_out_layer = nn.Linear(self.ext_hidden_size, out_dim)
        nn.init.uniform_(self.vien_out_layer.weight, -0.7, 0.7)
        self.HTanh = torch.nn.Hardtanh()

    def is_train(self, trainable=True):
        if trainable:
            for param in self.parameters():
                param.requires_grad = True
            self.src_encoder.train()
        else:
            self.src_encoder.eval()
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, src_input_ids, src_attention_mask, src_token_type_ids,
                trg_input_ids, trg_attention_mask, trg_token_type_ids, en_head_mask, vi_head_mask, trainable=False):
        # print(src_attention_mask.shape)
        # print(src_head_mask.shape)
        # print(trg_attention_mask.shape)
        # print(trg_head_mask.shape)

        bert_src = self.src_encoder(input_ids=src_input_ids, attention_mask=src_attention_mask,
                                    token_type_ids=src_token_type_ids)
        bert_trg = self.src_encoder(input_ids=trg_input_ids, attention_mask=trg_attention_mask,
                                    token_type_ids=trg_token_type_ids)
        # bert_trg shape: batchsize x self.max_seq_length x 768
        bert_src = bert_src['last_hidden_state']
        bert_trg = bert_trg['last_hidden_state']

        bert_src = bert_src.unsqueeze(1)
        bert_trg = bert_trg.unsqueeze(2)
        # shape: batchsize x max_trg_len x max_src_len x 768
        out_src = bert_src.expand(-1, self.max_en_len, -1, -1)
        out_trg = bert_trg.expand(-1, -1, self.max_vi_len, -1)
        #print(vi_head_mask.shape, en_head_mask.shape)
        #assert vi_head_mask.shape == en_head_mask.shape
        # print(vi_head_mask[0,:20])
        # print(en_head_mask[0,:20])
        vi_head_mask = vi_head_mask.unsqueeze(1)
        en_head_mask = en_head_mask.unsqueeze(2)
        # shape: batchsize x max_trg_len x max_src_len x 768
        vi_mask = vi_head_mask.expand(-1, self.max_en_len, -1)
        en_mask = en_head_mask.expand(-1, -1, self.max_vi_len)
        # shape: batchsize x max_en_len x max_vi_len  -1x130x128
        label_mask = vi_mask * en_mask
        # shape: batchsize x max_trg_len x max_src_len x 2*768
        out_encoder = torch.cat([out_trg, out_src], -1)
        output = self.wa_layer(out_encoder)
        output = self.HTanh(output)

        envioutput = self.envi_out_layer(output)
        vienoutput = self.vien_out_layer(output)
        # print(envioutput.shape)
        envioutput = envioutput.squeeze(-1)
        vienoutput = vienoutput.squeeze(-1)
        vienoutput = torch.mul(vienoutput, label_mask)
        envioutput = torch.mul(envioutput, label_mask)
        # output = torch.softmax(output, dim=-1)
        assert len(vienoutput.shape) == 3
        assert len(envioutput.shape) == 3
        envioutput = envioutput.transpose(2, 1)
        vien_output = torch.softmax(vienoutput, dim=-1)
        envi_output = torch.softmax(envioutput, dim=-1)
        return vien_output, envi_output


class Synmng(nn.Module):
    def __init__(self):
        super(Synmng, self).__init__()
        self.model = BERT_WA()
        self.model.to(device)
        self.bestloss = 2.0
        self.max_seq_length = 128
        parameters = list(self.model.named_parameters())
        Linear_params = ['out_layer.weight', 'out_layer.bias']
        Ziplayer_params = ['zip_layer.weight', 'zip_layer.bias']
        choice_params = []
        lr_list = []
        for p in parameters:
            if p[0] in Linear_params:
                p[1].requires_grad = True
                choice_params.append(p[1])
                lr_list.append(1e-3)
            elif p[0] in Ziplayer_params:
                p[1].requires_grad = True
                choice_params.append(p[1])
                lr_list.append(1e-3)
            else:
                p[1].requires_grad = True
                choice_params.append(p[1])
                lr_list.append(1e-3)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=1e-5, amsgrad=True)
        #self.optimizer = optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9)
        self.traincount = 0
        self.losses = []
        # self.optimizer = AMSgrad([], lr=[])

    def save(self, filepath: str):
        save_dict = {}
        save_dict['model'] = self.state_dict()
        save_dict['bestloss'] = self.bestloss
        save_dict['optimizer'] = self.optimizer.state_dict()
        save_dict['traincount'] = self.traincount
        save_dict['losses'] = self.losses
        torch.save(save_dict, filepath)

    def load(self, filepath: str):
        with torch.no_grad():
            load_dict = torch.load(filepath, map_location=device)
            self.load_state_dict(load_dict['model'])
            # self.model.encoder.LSTM.flatten_parameters()
            self.bestloss = load_dict['bestloss']
            self.optimizer.load_state_dict(load_dict['optimizer'])
            self.traincount = load_dict['traincount']
            self.losses = load_dict['losses']
            self.to(device)
            print('WAmng Model has been loaded')

    def is_train(self, trainable=True):
        self.model.is_train(trainable)

    def OffTraining(self):
        parameters = list(self.model.named_parameters())
        for p in parameters:
            p[1].requires_grad = False

    def forward(self, inputs):
        torch.cuda.empty_cache()
        vi_input_ids_lst = inputs[0].detach()
        vi_input_mask_lst = inputs[1].detach()
        vi_segment_ids_lst = inputs[2].detach()
        vi_head_mask_lst = inputs[3].detach()
        en_input_ids_lst = inputs[4].detach()
        en_input_mask_lst = inputs[5].detach()
        en_segment_ids_lst = inputs[6].detach()
        en_head_mask_lst = inputs[7].detach()
        vien_label_lst = inputs[8].detach()
        envi_label_lst = inputs[9].detach()
        vienLabel = vien_label_lst
        enviLabel = envi_label_lst
        if not hasattr(self, 'traincount'):
            self.traincount = 0
            print('Has not traincount atribute')
        self.optimizer.zero_grad()
        '''
        src_input_ids, src_attention_mask, src_token_type_ids, src_head_mask,
                trg_input_ids, trg_attention_mask, trg_token_type_ids, trg_head_mask, label_mask
        '''
        vienout_matrix, enviout_matrix = self.model(src_input_ids=vi_input_ids_lst,
                                                    src_attention_mask=vi_input_mask_lst,
                                                    src_token_type_ids=vi_segment_ids_lst,
                                                    trg_input_ids=en_input_ids_lst,
                                                    trg_attention_mask=en_input_mask_lst,
                                                    trg_token_type_ids=en_segment_ids_lst,
                                                    en_head_mask=en_head_mask_lst,
                                                    vi_head_mask=vi_head_mask_lst,
                                                    trainable=True)
        #print(vienout_matrix.shape, enviout_matrix.shape)
        vi_head_mask = vi_head_mask_lst.unsqueeze(1)
        en_head_mask = en_head_mask_lst.unsqueeze(2)
        # shape: batchsize x max_trg_len x max_src_len x 768
        vi_mask = vi_head_mask.expand(-1, 128, -1)
        en_mask = en_head_mask.expand(-1, -1, 128)
        label_mask = vi_mask * en_mask
        # print(label_mask.shape)
        loss2 = MultiLossTable(
            vienout_matrix, enviout_matrix, vienLabel, enviLabel, label_mask)
        loss2.backward()
        self.optimizer.step()
        # xm.optimizer_step(self.optimizer)
        with torch.no_grad():
            # PredictLabel = torch.argmax(out_matrix, dim=-1)
            metric, vienmetric, envimetric = F1Score(
                vienout_matrix, enviout_matrix, vienLabel, enviLabel, label_mask)
            self.traincount += 1
        return loss2, metric, vienmetric, envimetric

    def evaluate(self, inputs):
        vi_input_ids_lst = inputs[0].detach()
        vi_input_mask_lst = inputs[1].detach()
        vi_segment_ids_lst = inputs[2].detach()
        vi_head_mask_lst = inputs[3].detach()
        en_input_ids_lst = inputs[4].detach()
        en_input_mask_lst = inputs[5].detach()
        en_segment_ids_lst = inputs[6].detach()
        en_head_mask_lst = inputs[7].detach()
        vien_label_lst = inputs[8].detach()
        envi_label_lst = inputs[9].detach()
        # enPos_lst = inputs[8].detach()
        # enNer_lst = inputs[9].detach()
        # viPos_lst = inputs[10].detach()
        # viNer_lst = inputs[11].detach()
        # envi_Conesin_lst = inputs[12].detach()
        # envi_ProbIBM1_lst = inputs[13].detach()
        vienLabel = vien_label_lst
        enviLabel = envi_label_lst
        with torch.no_grad():
            vienout_matrix, enviout_matrix = self.model(src_input_ids=vi_input_ids_lst,
                                                        src_attention_mask=vi_input_mask_lst,
                                                        src_token_type_ids=vi_segment_ids_lst,
                                                        trg_input_ids=en_input_ids_lst,
                                                        trg_attention_mask=en_input_mask_lst,
                                                        trg_token_type_ids=en_segment_ids_lst,
                                                        en_head_mask=en_head_mask_lst,
                                                        vi_head_mask=vi_head_mask_lst,
                                                        trainable=False)
            # PredictLabel = torch.argmax(out_matrix, dim=-1)
            vi_head_mask = vi_head_mask_lst.unsqueeze(1)
            en_head_mask = en_head_mask_lst.unsqueeze(2)
            # shape: batchsize x max_trg_len x max_src_len x 768
            vi_mask = vi_head_mask.expand(-1, 128, -1)
            en_mask = en_head_mask.expand(-1, -1, 128)
            Mask = vi_mask * en_mask
            eval_metric, vieneval_metric, envieval_metric = F1Score(
                vienout_matrix, enviout_matrix, vienLabel, enviLabel, Mask)
            eval_loss = MultiLossTable(
                vienout_matrix, enviout_matrix, vienLabel, enviLabel, Mask)
        return eval_loss, eval_metric, vieneval_metric, envieval_metric

    def Predict(self, inputs):
        vi_input_ids_lst = inputs[0].detach()
        vi_input_mask_lst = inputs[1].detach()
        vi_segment_ids_lst = inputs[2].detach()
        vi_head_mask_lst = inputs[3].detach()
        en_input_ids_lst = inputs[4].detach()
        en_input_mask_lst = inputs[5].detach()
        en_segment_ids_lst = inputs[6].detach()
        en_head_mask_lst = inputs[7].detach()
        with torch.no_grad():
            vienout_matrix, enviout_matrix = self.model(src_input_ids=vi_input_ids_lst,
                                                        src_attention_mask=vi_input_mask_lst,
                                                        src_token_type_ids=vi_segment_ids_lst,
                                                        trg_input_ids=en_input_ids_lst,
                                                        trg_attention_mask=en_input_mask_lst,
                                                        trg_token_type_ids=en_segment_ids_lst,
                                                        en_head_mask=en_head_mask_lst,
                                                        vi_head_mask=vi_head_mask_lst,
                                                        trainable=False)
        return vienout_matrix, enviout_matrix


def MultiLossTable(vienSigmoidTable, enviSigmoidTable, vienLabel, enviLabel, Mask):
    enviMask = Mask.transpose(1, 2)
    vienSigmoid = vienSigmoidTable * Mask
    enviSigmoid = enviSigmoidTable * enviMask
    vienDiffM = vienSigmoid - vienLabel
    enviDiffM = enviSigmoid - enviLabel
    viennegmask = Mask - vienLabel
    envinegmask = enviMask - enviLabel
    neg_vienDiffM = viennegmask * vienDiffM
    FL_neg_vienDiffM = neg_vienDiffM.detach() + 0.5
    neg_enviDiffM = envinegmask * enviDiffM
    FL_neg_enviDiffM = neg_enviDiffM.detach() + 0.5
    pos_vienDiffM = vienLabel * vienDiffM
    FL_pos_vienDiffM = -pos_vienDiffM.detach() + 0.5
    pos_enviDiffM = enviLabel * enviDiffM
    FL_pos_enviDiffM = -pos_enviDiffM.detach() + 0.5
    viennegloss = torch.sum(
        FL_neg_vienDiffM * neg_vienDiffM) / torch.sum((viennegmask == 1))
    envinegloss = torch.sum(
        FL_neg_enviDiffM * neg_enviDiffM) / torch.sum((envinegmask == 1))
    vienposloss = torch.sum(
        FL_pos_vienDiffM * pos_vienDiffM) / torch.sum((vienLabel == 1))
    enviposloss = torch.sum(
        FL_pos_enviDiffM * pos_enviDiffM) / torch.sum((enviLabel == 1))
    vienloss = torch.mean(viennegloss - vienposloss, dtype=torch.float32)
    enviloss = torch.mean(envinegloss - enviposloss, dtype=torch.float32)
    loss = float(enviloss.item() / (enviloss.item() + vienloss.item())) * vienloss + float(
        vienloss.item() / (enviloss.item() + vienloss.item())) * enviloss
    return loss


def Metric(PredictAlign, ylabel):
    Corret = 0
    Miss = 0
    Ylabel = torch.tensor(ylabel, device=device)
    # print(PredictAlign, Ylabel)
    for i in range(len(ylabel)):
        if (PredictAlign[i] == Ylabel[i]):
            Corret += 1
        else:
            Miss += 1
    # print(Corret, Miss)
    metric = max(0, 1 - Corret / len(ylabel) + Miss / len(ylabel))
    return metric


def F1Score(vienSigmoidTable, enviSigmoidTable, vienLabel, enviLabel, Mask):
    vienPredictTable = vienSigmoidTable * Mask
    enviPredictTable = enviSigmoidTable * Mask.transpose(1, 2)
    vienPredict = vienPredictTable > THRESH
    enviPredict = enviPredictTable > THRESH
    assert torch.sum(enviLabel) == torch.sum(enviLabel * Mask.transpose(1, 2))
    assert torch.sum(vienLabel) == torch.sum(vienLabel * Mask)
    vienCorrect = vienLabel * vienPredict
    enviCorrect = enviLabel * enviPredict
    vienTotal_common = torch.sum(vienCorrect, dtype=torch.float32)
    enviTotal_common = torch.sum(enviCorrect, dtype=torch.float32)
    vienTotal_sys = torch.sum(vienPredict, dtype=torch.float32)
    enviTotal_sys = torch.sum(enviPredict, dtype=torch.float32)
    vienTotal_ref = torch.sum(vienLabel, dtype=torch.float32)
    enviTotal_ref = torch.sum(enviLabel, dtype=torch.float32)
    vienrecall = vienTotal_common.item() / (vienTotal_ref.item())
    envirecall = enviTotal_common.item() / (enviTotal_ref.item())
    vienprec = vienTotal_common.item() / (vienTotal_sys.item() + 1)
    enviprec = enviTotal_common.item() / (enviTotal_sys.item() + 1)
    vienf1 = 0.0
    if (vienrecall + vienprec) > 0.0:
        vienf1 = 2 * vienrecall * vienprec / (vienrecall + vienprec)
    envif1 = 0.0
    if (envirecall + enviprec) > 0.0:
        envif1 = 2 * envirecall * enviprec / (envirecall + enviprec)
    recall = (vienrecall + envirecall) * 0.5
    prec = (vienprec + enviprec) * 0.5
    f1 = (vienf1 + envif1) * 0.5
    return (recall, prec, f1), (vienrecall, vienprec, vienf1), (envirecall, enviprec, envif1)


def GetBatch(inputs, tatoeba_indeces, i: int, batch_size=32, shuffle=False):
    # sample_indeces = [x for x in range(align.shape[0])]
    outputs = []
    with torch.no_grad():
        if (shuffle):
            random.shuffle(tatoeba_indeces)
        batch_half = int(batch_size)
        tatoeba_start = i * batch_half
        tatoeba_end = tatoeba_start + batch_half
        if (tatoeba_end > len(inputs[0])):
            tatoeba_end = len(inputs[0])
        tatoeba_selected_indeces = tatoeba_indeces[tatoeba_start: tatoeba_end]
        for m, fs in enumerate(inputs):
            if m >= 10:
                f = []
                for index in tatoeba_selected_indeces:
                    f.append(fs[index])
                outputs.append(f)
            else:
                outputs.append(fs[tatoeba_selected_indeces].clone())
    return tuple(outputs)


def GetPredBatch(inputs, tatoeba_indeces,
                 i: int, batch_size=32):
    # sample_indeces = [x for x in range(align.shape[0])]
    outputs = []
    with torch.no_grad():
        batch_half = int(batch_size)
        tatoeba_start = i * batch_half
        tatoeba_end = tatoeba_start + batch_half
        if (tatoeba_end > len(inputs[0])):
            tatoeba_end = len(inputs[0])
        tatoeba_selected_indeces = tatoeba_indeces[tatoeba_start: tatoeba_end]
        for m, fs in enumerate(inputs):
            if m >= 10:
                f = []
                for index in tatoeba_selected_indeces:
                    f.append(fs[index])
                outputs.append(f)
            else:
                outputs.append(fs[tatoeba_selected_indeces].clone())
    return tuple(outputs)


def GetBatchEval(inputs, tatoeba_indeces,
                 i: int, batch_size=32, shuffle=False):
    # sample_indeces = [x for x in range(align.shape[0])]
    src_sent_batch = []
    trg_sent_batch = []
    outputs = []
    with torch.no_grad():
        if (shuffle):
            random.shuffle(tatoeba_indeces)
        batch_half = int(batch_size)
        tatoeba_start = i * batch_half
        tatoeba_end = tatoeba_start + batch_half
        if (tatoeba_end > len(inputs[0])):
            tatoeba_end = len(inputs[0])
        tatoeba_selected_indeces = tatoeba_indeces[tatoeba_start: tatoeba_end]
        for m, fs in enumerate(inputs):
            if m >= 10:
                f = []
                for index in tatoeba_selected_indeces:
                    f.append(fs[index])
                outputs.append(f)
            else:
                outputs.append(fs[tatoeba_selected_indeces].clone())
    return tuple(outputs)


class ModelMng:
    def __init__(self, model_name: str, model_dir: str):
        # self.NUMBER_SENT_PAIR = 19842
        self.Name = model_name
        self.Model_dir = model_dir
        if (os.path.isfile(self.Model_dir + r'/The' + self.Name + '.pth')):
            self.rnn = Synmng()
            #self.rnn.load_state_dict(torch.load(self.Model_dir + r'/best' + self.Name + '.pth', map_location=device))
            self.rnn.load(self.Model_dir + r'/The' + self.Name + '.pth')
            #self.bestrnn = Synmng()
            #self.bestrnn.load(self.Model_dir + r'/TheBest' + self.Name + '.pth')
            print(list(self.rnn.model.named_parameters()))
            if not hasattr(self.rnn, 'bestloss'):
                self.rnn.bestloss = 1000.0
            # self.rnn.bestloss = 2.0
            self.loss = self.rnn.bestloss
            print('Model been loaded from disk')
        else:
            self.rnn = Synmng()
            self.bestrnn = Synmng()
            if not hasattr(self.rnn, 'bestloss'):
                self.rnn.bestloss = 1000.0
            print('The first model been build to')
            self.loss = 2.0
            self.bestloss = 2.0

    def is_train(self, trainable=True):
        self.rnn.is_train(trainable)

    def EpochTraining(self, inputs, batch_size=32):
        tatoeba_indeces = [x for x in range(len(inputs[0]))]
        random.shuffle(tatoeba_indeces)
        traindata_size = len(inputs[0])
        batch_num = math.ceil(traindata_size / batch_size)
        total_loss = 0.0
        total_recall = 0.0
        total_prec = 0.0
        total_f1 = 0.0
        vientotal_recall = 0.0
        vientotal_prec = 0.0
        vientotal_f1 = 0.0
        envitotal_recall = 0.0
        envitotal_prec = 0.0
        envitotal_f1 = 0.0
        for batch in range(int(batch_num)):
            batch_inputs = GetBatch(inputs, i=batch,
                                    tatoeba_indeces=tatoeba_indeces, batch_size=batch_size, shuffle=True)
            loss, metric, vienmetric, envimetric = self.rnn(batch_inputs)
            recall, prec, f1 = metric
            vienrecall, vienprec, vienf1 = vienmetric
            envirecall, enviprec, envif1 = envimetric
            total_loss += float(loss / batch_num)
            total_recall += float(recall / batch_num)
            total_prec += float(prec / batch_num)
            total_f1 += float(f1 / batch_num)
            vientotal_recall += float(vienrecall / batch_num)
            vientotal_prec += float(vienprec / batch_num)
            vientotal_f1 += float(vienf1 / batch_num)
            envitotal_recall += float(envirecall / batch_num)
            envitotal_prec += float(enviprec / batch_num)
            envitotal_f1 += float(envif1 / batch_num)
        return (total_loss, total_recall, total_prec, total_f1),\
               (vientotal_recall, vientotal_prec, vientotal_f1),\
               (envitotal_recall, envitotal_prec, envitotal_f1)

    def Eval(self, inputs, batch_size=200):
        testdata_size = len(inputs[0])
        test_indeces = [x for x in range(testdata_size)]
        batch_num = math.ceil(testdata_size / batch_size)
        # random.shuffle(test_indeces)
        total_loss = 0.0
        total_recall = 0.0
        total_prec = 0.0
        total_f1 = 0.0
        vientotal_recall = 0.0
        vientotal_prec = 0.0
        vientotal_f1 = 0.0
        envitotal_recall = 0.0
        envitotal_prec = 0.0
        envitotal_f1 = 0.0
        for batch in range(int(batch_num)):
            batch_inputs = GetBatchEval(inputs, tatoeba_indeces=test_indeces, i=batch,
                                        batch_size=batch_size)
            loss, metric, vienmetric, envimetric = self.rnn.evaluate(
                batch_inputs[:14])
            recall, prec, f1 = metric
            vienrecall, vienprec, vienf1 = vienmetric
            envirecall, enviprec, envif1 = envimetric
            total_loss += float(loss / batch_num)
            total_recall += float(recall / batch_num)
            total_prec += float(prec / batch_num)
            total_f1 += float(f1 / batch_num)
            vientotal_recall += float(vienrecall / batch_num)
            vientotal_prec += float(vienprec / batch_num)
            vientotal_f1 += float(vienf1 / batch_num)
            envitotal_recall += float(envirecall / batch_num)
            envitotal_prec += float(enviprec / batch_num)
            envitotal_f1 += float(envif1 / batch_num)
        if self.bestloss > total_loss:
            self.bestloss = total_loss
        self.loss = total_loss
        return (total_loss, total_recall, total_prec, total_f1), \
               (vientotal_recall, vientotal_prec, vientotal_f1), \
               (envitotal_recall, envitotal_prec, envitotal_f1)

    def Predict(self, inputs, batch_size=200):
        testdata_size = len(inputs[0])
        test_indeces = [x for x in range(testdata_size)]
        batch_num = math.ceil(testdata_size / batch_size)
        # random.shuffle(test_indeces)
        vienresult = []
        enviresult = []
        visent_lst = []
        ensent_lst = []
        vi_to_tok_lst = []
        en_to_tok_lst = []
        envi_align_lst = []
        vien_align_lst = []
        for batch in range(int(batch_num)):
            batch_inputs = GetBatchEval(inputs, tatoeba_indeces=test_indeces, i=batch,
                                        batch_size=batch_size)
            vienout_matrix, enviout_matrix = self.rnn.Predict(batch_inputs)
            for j in range(vienout_matrix.shape[0]):
                vienresult.append(vienout_matrix[j].cpu().numpy())
            for j in range(enviout_matrix.shape[0]):
                enviresult.append(enviout_matrix[j].cpu().numpy())
            visent_lst.extend(batch_inputs[-6])
            ensent_lst.extend(batch_inputs[-5])
            vi_to_tok_lst.extend(batch_inputs[-4])
            en_to_tok_lst.extend(batch_inputs[-3])
            envi_align_lst.extend(batch_inputs[-2])
            vien_align_lst.extend(batch_inputs[-1])
        return vienresult, enviresult, visent_lst, ensent_lst, vi_to_tok_lst, en_to_tok_lst, envi_align_lst, vien_align_lst


def Training():
    #train_folder = r'/content/drive/MyDrive/enViRawSuperData'
    #train_files = [join(train_folder, f) for f in listdir(train_folder) if isfile(join(train_folder, f))]
    train_files = r'/content/drive/MyDrive/AlignmentModel/train_data.json'
    test_files = r'/content/drive/MyDrive/AlignmentModel/test_data.json'

    Trainer = ModelMng(model_name='BERTBiDataModelV2_Adam_Softmax_BIDI_HTanh_64_1',
                       model_dir=r'/content/drive/MyDrive/AlignmentModel')
    Trainer.is_train(True)
    with open(train_files, "r", encoding="utf8") as json_file:
        train_data_raw = json.load(json_file)
    with open(test_files, "r", encoding="utf8") as json_file:
        dev_data_raw = json.load(json_file)
    tokenizer = BertTokenizer.from_pretrained(BERT_NAME)
    train_inputs = convert_examples_to_features(tokenizer, train_data_raw)
    dev_inputs = convert_examples_to_features(tokenizer, dev_data_raw)
    del tokenizer
    train_data = []
    m = 0
    for fs in train_inputs:
        with torch.no_grad():
            if m >= 10:
                f = fs
            elif m >= 8:
                f = torch.tensor(fs, dtype=torch.float, device=device)
            else:
                f = torch.tensor(fs, dtype=torch.int, device=device)
            train_data.append(f)
            m = m + 1
    train_data = tuple(train_data)
    dev_data = []
    m = 0
    for fs in dev_inputs:
        with torch.no_grad():
            if m >= 10:
                f = fs
            elif m >= 8:
                f = torch.tensor(fs, dtype=torch.float, device=device)
            else:
                f = torch.tensor(fs, dtype=torch.int, device=device)
            m = m + 1
            dev_data.append(f)
    dev_data = tuple(dev_data)
    get_time = time.time()
    get_time2 = time.time()
    count = 0
    while (Trainer.loss > 0.001):
        count = count + 1
        Trainer.is_train(True)
        total, vientotal, envitotal = Trainer.EpochTraining(
            train_data, batch_size=6)
        if count % 1 == 0:
            print(count, total)
            print('vien:', vientotal)
            print('envi:', envitotal)
        cur_time = time.time()
        if (cur_time - get_time) > 300:
            get_time = time.time()
            Trainer.is_train(False)
            total, vientotal, envitotal = Trainer.Eval(dev_data, batch_size=6)
            print('eval:', total)
            print('vien:', vientotal)
            print('envi:', envitotal)
            if Trainer.bestloss == Trainer.loss:
                Trainer.rnn.save(Trainer.Model_dir +
                                 r'/The' + Trainer.Name + '.pth')
                print('The model has Saved')


def Test(Trainer, test_data_raw):
    tokenizer = BertTokenizer.from_pretrained(BERT_NAME)
    test_inputs = convert_examples_to_features(tokenizer, test_data_raw)
    del tokenizer
    test_data = []
    m = 0
    for fs in test_inputs:
        with torch.no_grad():
            if m >= 10:
                f = fs
            elif m >= 8:
                f = torch.tensor(fs, dtype=torch.float, device=device)
            else:
                f = torch.tensor(fs, dtype=torch.int, device=device)
            test_data.append(f)
            m = m + 1
    test_data = tuple(test_data)
    vienmatrix, envimatrix, visent_lst, ensent_lst, \
        vi_to_tok_lst, en_to_tok_lst, envi_align_lst, vien_align_lst = Trainer.Predict(
            test_data, batch_size=24)
    vienword_matrix_lst = tok_to_word_matrix(
        vienmatrix, vi_to_tok_lst, en_to_tok_lst)
    enviword_matrix_lst = tok_to_word_matrix(
        envimatrix, vi_to_tok_lst, en_to_tok_lst, False)
    result = test_data_raw
    for i, enviword_matrix in enumerate(enviword_matrix_lst):
        # pred_matrix = enviword_matrix >= 0.5
        # pred_matrix2 = enviword_matrix * pred_matrix
        pred_align = np.argmax(enviword_matrix, axis=1)
        result[i]['envi_align'] = pred_align+1
    return result


def compare(label_matrix_lst, word_matrix_lst, envilabel_matrix_lst, enviword_matrix_lst,
            thresh=(0.001, 0.01, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.935, 0.945, 0.965, 0.975, 0.985, 0.99, 0.995)):
    n_sys = {}
    n_common = {}
    for th in thresh:
        n_sys[th] = 0.0
        n_common[th] = 0.0
    n_ref = 0.0
    for i, word_matrix in enumerate(word_matrix_lst):
        label_matrix = label_matrix_lst[i]
        n_ref = np.sum(label_matrix) + n_ref
        for th in thresh:
            #pred_align = np.argmax(word_matrix, axis=1)
            #label_align = np.argmax(label_matrix, axis=1)
            #print(pred_align, label_align)
            pred_matrix = word_matrix >= th
            pred_matrix2 = word_matrix * pred_matrix
            pred_align = np.argmax(pred_matrix2, axis=1)
            pred_matrix3 = np.zeros_like(pred_matrix)
            for r, c in enumerate(pred_align):
                #prinpt(r, c)
                if pred_matrix2[r, c] != 0:
                    pred_matrix3[r, c] = 1
            #label_align = np.argmax(label_matrix, axis=1)
            n_sys[th] = np.sum(pred_matrix3) + n_sys[th]
            n_common[th] = np.sum(np.multiply(
                label_matrix, pred_matrix3)) + n_common[th]
            #n_common[th] = np.sum(label_align == pred_align) + n_common[th]
            #n_sys[th] = n_ref
    for th in thresh:
        recall = n_common[th] / n_ref
        precision = 0.0
        if n_sys[th] > 0:
            precision = n_common[th] / n_sys[th]
        f1 = 2 * recall * precision / (recall + precision)
        print('{}: {:.3f} {:.3f} {:.3f}'.format(th, recall, precision, f1),
              end=' ')
        print('({} {} {})'.format(n_ref, n_sys[th], n_common[th]))

    envin_sys = {}
    envin_common = {}
    for th in thresh:
        envin_sys[th] = 0.0
        envin_common[th] = 0.0
    envin_ref = 0.0
    for i, word_matrix in enumerate(enviword_matrix_lst):
        label_matrix = envilabel_matrix_lst[i]
        envin_ref = np.sum(label_matrix) + envin_ref
        # print(word_matrix)
        for th in thresh:
            pred_matrix = word_matrix >= th
            pred_matrix2 = word_matrix * pred_matrix
            pred_align = np.argmax(pred_matrix2, axis=0)
            pred_matrix3 = np.zeros_like(pred_matrix)
            for r, c in enumerate(pred_align):
                #print(r, c)
                if pred_matrix2[c, r] != 0:
                    pred_matrix3[c, r] = 1
            # label_align = np.argmax(label_matrix, axis=1)
            envin_sys[th] = np.sum(pred_matrix3) + envin_sys[th]
            envin_common[th] = np.sum(np.multiply(
                label_matrix, pred_matrix3)) + envin_common[th]
    for th in thresh:
        recall = envin_common[th] / envin_ref
        precision = 0.0
        if envin_sys[th] > 0:
            precision = envin_common[th] / envin_sys[th]
        f1 = 2 * recall * precision / (recall + precision)
        print('{}: {:.3f} {:.3f} {:.3f}'.format(th, recall, precision, f1),
              end=' ')
        print('({} {} {})'.format(envin_ref, envin_sys[th], envin_common[th]))


def lst_to_vienlabel_matrix(label, vi_to_tok_lst, en_to_tok_lst):
    label_matrix_lst = []
    label_matrix = None
    for i, vi2tok in enumerate(vi_to_tok_lst):
        lst_align = label[i]
        en2tok = en_to_tok_lst[i]
        vi_len = len(vi2tok)
        en_len = len(en2tok)
        vien_a_matrix = np.zeros((en_len - 2, vi_len - 2), dtype=int)
        for j, vien_a_tok in enumerate(lst_align):
            if int(vien_a_tok) == 0:
                continue
            vien_a_matrix[int(j), int(vien_a_tok) - 1] = 1
        label_matrix_lst.append(vien_a_matrix)
    return label_matrix_lst


def lst_to_envilabel_matrix(label, vi_to_tok_lst, en_to_tok_lst):
    label_matrix_lst = []
    label_matrix = None
    for i, vi2tok in enumerate(vi_to_tok_lst):
        lst_align = label[i]
        en2tok = en_to_tok_lst[i]
        vi_len = len(vi2tok)
        en_len = len(en2tok)
        envi_a_matrix = np.zeros((en_len - 2, vi_len - 2), dtype=int)
        for j, envi_a_tok in enumerate(lst_align):
            if int(envi_a_tok) == 0:
                continue
            envi_a_matrix[int(envi_a_tok) - 1, int(j)] = 1
        label_matrix_lst.append(envi_a_matrix)
    return label_matrix_lst


def tok_to_word_matrix(matrix, vi_to_tok_lst, en_to_tok_lst, vien=True):
    word_matrix_lst = []
    word_matrix = None
    for i, vi2tok in enumerate(vi_to_tok_lst):
        tok_matrix = matrix[i]
        if not vien:
            tok_matrix = np.transpose(tok_matrix)
        en2tok = en_to_tok_lst[i]
        vi_len = len(vi2tok)
        en_len = len(en2tok)
        word_matrix = np.zeros((en_len - 2, vi_len - 2), dtype=np.float)
        for r in range(en_len - 1):
            if r == 0:
                continue
            for c in range(vi_len - 1):
                if c == 0:
                    continue
                small_matrix = np.nan_to_num(tok_matrix[en2tok[r], vi2tok[c]])
                word_matrix[r - 1, c - 1] = small_matrix
        word_matrix_lst.append(word_matrix)
    return word_matrix_lst


def tok_to_enviword_matrix(matrix, vi_to_tok_lst, en_to_tok_lst):
    word_matrix_lst = []
    word_matrix = None
    for i, vi2tok in enumerate(vi_to_tok_lst):
        tok_matrix = matrix[i]
        en2tok = en_to_tok_lst[i]
        vi_len = len(vi2tok)
        en_len = len(en2tok)
        word_matrix = np.zeros((en_len - 2, vi_len - 2), dtype=np.float)
        for c in range(vi_len - 1):
            if c == 0:
                continue
            for r in range(en_len - 1):
                if r == 0:
                    continue
                small_matrix = np.nan_to_num(
                    tok_matrix[en2tok[r]: en2tok[r + 1], vi2tok[c]: vi2tok[c + 1]])
                word_matrix[r - 1, c - 1] = small_matrix.mean()
        word_matrix_lst.append(word_matrix)
    return word_matrix_lst


def output_writer(filepath, ssent_lst, tsent_lst, filename_wa, vienlb_mx_lst, vienpred_mx_lst, envilb_mx_lst, envipred_mx_lst):
    data = {}
    for i, ssent in enumerate(ssent_lst):
        d = {}
        tsent = tsent_lst[i]
        #s_token_len, label_idx, t_len, id = info_lst[i]
        vienlb_mx = vienlb_mx_lst[i]
        envilb_mx = envilb_mx_lst[i]
        vienpred_mx = vienpred_mx_lst[i]
        envipred_mx = envipred_mx_lst[i]
        d['t_sent'] = ssent
        d['s_sent'] = tsent
        d['vienlb_mx'] = vienlb_mx.tolist()
        d['envilb_mx'] = envilb_mx.tolist()
        d['vienpred_mx'] = vienpred_mx.tolist()
        d['envipred_mx'] = envipred_mx.tolist()
        id = filename_wa + '_' + str(i)
        data[id] = d
    OutFile = open(filepath, "a", encoding='utf8')
    json.dump(data, OutFile, indent=2, ensure_ascii=False)
    OutFile.close()