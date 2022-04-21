#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Ryan Hu
# DATE: 2022/3/15 22:37
# DESCRIPTION:
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from transformers import AutoConfig, AutoModel
from allennlp.modules import ConditionalRandomField as CRF


class BertMLP(nn.Module):
    def __init__(self, num_classes):
        super(BertMLP, self).__init__()
        model_name = 'bert-base-chinese'
        config = BertConfig.from_pretrained(model_name, output_hidden_states=True)
        self.bert = BertModel(config)
        self.classifier = nn.Linear(in_features=768,
                                    out_features=num_classes)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask)
        hidden_state = output.last_hidden_state
        logits = self.classifier(hidden_state)
        return logits


class BertBiLSTM(nn.Module):
    def __init__(self, num_classes):
        super(BertBiLSTM, self).__init__()
        model_name = 'bert-base-chinese'
        config = BertConfig.from_pretrained(model_name, output_hidden_states=True)
        self.bert = BertModel(config)

        self.bilstm = nn.LSTM(input_size=768,
                              hidden_size=512,
                              num_layers=2,
                              batch_first=True,
                              bidirectional=True)

        self.classifier = nn.Linear(in_features=512*2,
                                    out_features=num_classes)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask)
        hidden_state = output.last_hidden_state
        output, _ = self.bilstm(hidden_state)
        logits = self.classifier(output)
        return logits


class BertBiLstmCRF(nn.Module):
    def __init__(self, num_classes):
        super(BertBiLstmCRF, self).__init__()
        model_name = 'peterchou/ernie-gram'
        config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.erine_gram = AutoModel.from_pretrained(model_name, config=config)

        self.bilstm = nn.LSTM(input_size=768,
                              hidden_size=512,
                              num_layers=2,
                              batch_first=True,
                              bidirectional=True)
        self.linear = nn.Linear(in_features=512 * 2,
                                out_features=num_classes)

        self.crf = CRF(num_tags=num_classes)

    def _featurize(self, input_ids, attention_mask):
        output = self.erine_gram(input_ids, attention_mask)
        hidden_states = output.hidden_states[1:]  # bert12层的输出 : list
        hidden_states = torch.stack(hidden_states, dim=2)  # (bacth, seq_len, 12, hidden_size)
        hidden_states = torch.max(hidden_states, dim=2)[0]  # (bacth, seq_len, hidden_size)
        output, _ = self.bilstm(hidden_states)
        output = self.linear(output)
        return output

    def forward(self, input_ids, attention_mask):
        output = self._featurize(input_ids, attention_mask)
        output = self.crf.viterbi_tags(output, attention_mask)
        return output

    def negative_log_like_hood(self, input_ids, attention_mask, labels):
        output = self._featurize(input_ids, attention_mask)
        output = self.crf.forward(
            inputs=output,
            tags=labels,
            mask=attention_mask
        )
        return -1 * output


class BertForABAS(nn.Module):
    def __init__(self, num_classes=3):
        super(BertForABAS, self).__init__()

        model_name = 'bert-base-uncased'
        config = BertConfig.from_pretrained(model_name, output_hidden_states=True)
        self.bert = BertModel(config)

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=256,
                               kernel_size=(3, 768),
                               stride=1)
        self.conv2 = nn.Conv2d(in_channels=1,
                               out_channels=256,
                               kernel_size=(4, 768),
                               stride=1)
        self.conv3 = nn.Conv2d(in_channels=1,
                               out_channels=256,
                               kernel_size=(5, 768),
                               stride=1)

        self.classifier = nn.Linear(in_features=3*256, out_features=num_classes)

    def forward(self, text_ids, attention_mask, aspect_boundary):
        # 获取bert中间层输出
        output = self.bert(text_ids, attention_mask)
        hidden_states = output.hidden_states[1:]  # size 12: (batch, seq_len, hidden_size)
        attention_states = torch.stack(hidden_states, dim=1)  # (batch, num_layers, seq_len, hidden_size)

        input4conv = []
        for s, a_b in zip(attention_states, aspect_boundary):
            aspect_start, aspect_end = a_b
            aspect = s[:, aspect_start:aspect_end]  # (num_layers, aspect_len, hidden_size)
            aspect = torch.max(aspect, dim=1)[0]  # (num_layers, hidden_size)
            input4conv.append(aspect)
        input4conv = torch.stack(input4conv, dim=0)  # (batch, num_layers, hidden_size)
        input4conv = input4conv.unsqueeze(dim=1)  # (batch, 1, num_layers, hidden_size)
        conv1_out = self.conv1(input4conv).squeeze().max(dim=-1)[0]
        conv2_out = self.conv2(input4conv).squeeze().max(dim=-1)[0]
        conv3_out = self.conv3(input4conv).squeeze().max(dim=-1)[0]
        # MLP
        input4mlp = torch.cat((conv1_out, conv2_out, conv3_out), dim=1)
        logits = self.classifier(input4mlp)
        return logits


class ErnieGramForTextSim(nn.Module):
    def __init__(self):
        super(ErnieGramForTextSim, self).__init__()

        model_name = 'peterchou/ernie-gram'
        self.ernie_gram = AutoModel.from_pretrained(model_name)
        self.cos_sim = nn.CosineSimilarity()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        # 句向量
        output1 = self.ernie_gram(input_ids1, attention_mask1)
        cls_embedding1 = output1.last_hidden_state[:, 0]  # (batch, hidden_size)
        output2 = self.ernie_gram(input_ids2, attention_mask2)
        cls_embedding2 = output2.last_hidden_state[:, 0]  # (batch, hidden_size)
        cos_sim = self.cos_sim(cls_embedding1, cls_embedding2)
        return self.sigmoid(cos_sim)
