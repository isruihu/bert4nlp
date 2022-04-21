#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Ryan Hu
# DATE: 2022/3/20 10:40
# DESCRIPTION:
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import AutoTokenizer
from . import Subset


def load_data(fname, mode='train'):
    data, label = [], []
    f = open(fname, 'r', encoding='utf-8')
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if len(line) < 2:
            data.append((' ', ' '))
            label.append('1')
            continue
        if mode in ['train', 'val']:
            q1, q2, l = line.split('\t')
            data.append((q1, q2))
            label.append(l)
        else:
            q1, q2 = line.split('\t')
            data.append((q1, q2))
            label.append('0')
    return data, label


def pad_and_truncate(text,
                     max_len,
                     padding='post',
                     truncating='post',
                     padding_x=0,
                     mask=0):

    _len = len(text)
    attention_mask = [1 for _ in range(_len)]

    if _len > max_len:
        # 保留头尾的[CLS]和[SEP]
        if truncating == 'pre':
            text = [text[0]] + text[-(max_len - 2):] + [text[-1]]
        else:
            text = [text[0]] + text[:(max_len - 2)] + [text[-1]]
        attention_mask = attention_mask[:max_len]
    else:
        padding_num = max_len - _len
        if padding == 'post':
            text = text + [padding_x for _ in range(padding_num)]
            attention_mask = attention_mask + [mask for _ in range(padding_num)]
        else:
            text = [padding_x for _ in range(padding_num)] + text
            attention_mask = [mask for _ in range(padding_num)] + attention_mask

    return text, attention_mask


class Tokenizer:
    def __init__(self, max_seq_len):
        model_name = 'WangZeJun/simbert-base-chinese'
        self.tz = AutoTokenizer.from_pretrained(model_name)
        self.max_seq_len = max_seq_len

    def tokenize(self, text):
        text = self.tz.tokenize(text)
        text = ['[CLS]'] + text + ['[SEP]']
        text = self.tz.convert_tokens_to_ids(text)
        return pad_and_truncate(text, max_len=self.max_seq_len)


class SimDataset(Dataset):
    def __init__(self, root, split, tokenizer):
        super(SimDataset, self).__init__()
        if split == 'train':
            fname = Path(root) / 'train.tsv'
        elif split == 'val':
            fname = Path(root) / 'dev.tsv'
        else:
            fname = Path(root) / 'test.tsv'

        self.data, self.label = load_data(fname, split)
        self.tz = tokenizer

    def __getitem__(self, index):
        (text1, text2), label = self.data[index], self.label[index]
        input_ids1, attention_mask1 = self.tz.tokenize(text1)
        input_ids2, attention_mask2 = self.tz.tokenize(text2)
        input_ids1 = torch.LongTensor(input_ids1)
        attention_mask1 = torch.LongTensor(attention_mask1)
        input_ids2 = torch.LongTensor(input_ids2)
        attention_mask2 = torch.LongTensor(attention_mask2)
        label = torch.LongTensor([int(label)])
        data = {
            'input_ids1': input_ids1,
            'attention_mask1': attention_mask1,
            'input_ids2': input_ids2,
            'attention_mask2': attention_mask2,
        }
        return data, label

    def __len__(self):
        return len(self.data)


def get_sim_loader(root,
                   max_seq_len,
                   batch_size,
                   split='train',
                   num_workers=8,
                   limit=None):

    tokenizer = Tokenizer(max_seq_len)

    if split in ['train', 'val']:
        dataset_list = []
        data_names = ['bq_corpus', 'lcqmc', 'oppo', 'paws-x-zh']
        for name in data_names:
            path = Path(root) / name
            d = SimDataset(path, split, tokenizer)
            dataset_list.append(d)
        dataset = ConcatDataset(dataset_list)

    else:
        dataset = SimDataset(Path(root), split, tokenizer)

    if limit is not None:
        dataset = Subset(dataset, limit)

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True if split == 'train' else False
    )
    return loader
