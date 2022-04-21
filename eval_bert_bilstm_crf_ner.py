#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Ryan Hu
# DATE: 2022/3/15 14:02
# DESCRIPTION:
import os

import argparse
import time
import datetime
import logging
from pathlib import Path

import torch
import torch.nn as nn

from bert4nlp.model import BertMLP, BertBiLSTM, BertBiLstmCRF
from bert4nlp.utils.trainer import get_optimizer, load_model
from bert4nlp.utils.utils import AverageMeter, accuracy, set_seed
from bert4nlp.utils.logging import set_logging
from bert4nlp.dataset.ner_dataset import get_ner_loader, load_data


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default=os.path.basename(__file__).split('.')[0])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_classes', type=int, default=105)

    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--bs', type=int, default=16, help='batch size')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--model', type=str, default='bert-bilstm-crf')
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=1e-3)

    opt = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)
    return opt


def set_model(opt):
    """
    设置模型、损失函数、优化器
    :param opt:
    :return:
    """
    state_dict = torch.load('./exp_results/train_bert_bilstm_crf_ner-model_eirne-bilstm-crf-seed42/checkpoints/eirne-bilstm-crf_last.pth')
    model = BertBiLstmCRF(num_classes=opt.num_classes)

    _, model, _, _ = load_model(state_dict, model)
    model = model.cuda()

    return model


def test_data2txt(test_data):
    f = open(Path(root) / 'test.txt', 'w', encoding='utf-8')
    for text in test_data:
        for c in text:
            f.write(f"{c} O\n")
        f.write('\n')
    f.close()


if __name__ == '__main__':
    # setting
    opt = parse_option()
    model = set_model(opt)

    root = './rsrc/JD_NER'
    test_loader = get_ner_loader(root,
                                 dataset='jd',
                                 batch_size=opt.bs,
                                 max_seq_len=128,
                                 split='test')

    output_label = []
    for idx, (data, label) in enumerate(test_loader):
        input_ids = data['input_ids'].cuda()
        attention_mask = data['attention_mask'].cuda()
        label = label.cuda()

        bsz = label.shape[0]

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        for pred, _ in output:
            tags = test_loader.dataset.tz.idx_to_label(pred)
            output_label.append(tags[1:-1])

    data, _ = load_data('./rsrc/JD_NER/preliminary_test_a/word_per_line_preliminary_A.txt', mode='test')
    fout = open('res.txt', 'w', encoding='utf-8')
    for i in range(len(data)):
        for word, tag in zip(data[i], output_label[i]):
            fout.write(f"{word} {tag}\n")
        fout.write('\n')
