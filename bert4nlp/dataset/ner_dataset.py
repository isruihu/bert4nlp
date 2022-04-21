#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Ryan Hu
# DATE: 2022/3/17 17:44
# DESCRIPTION:
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from . import Subset


def load_data(fname, mode='train'):
    data, label = [], []
    fin = open(fname, 'rb')
    lines = fin.readlines()

    words, tags = [], []
    for line in lines:
        line = line.decode('utf-8', 'ignore')
        if line == '\n':
            data.append(words)
            label.append(tags)
            words, tags = [], []
        else:
            line = line[:-1]
            if mode in ['train', 'val']:
                line = line.split()
                if len(line) < 2:
                    words.append(' ')
                    tags.append(line[0])
                else:
                    words.append(line[0])
                    tags.append(line[1])
            else:
                words.append(line)
                tags.append('O')
    return data, label


def get_labels(fname):
    labels = []
    fin = open(fname, 'rb')
    lines = fin.readlines()
    for line in lines:
        line = line.strip()
        if len(line) == 0: continue
        label = line.split()[-1]
        labels.append(label.decode('utf-8'))
    labels = sorted(list(set(labels)))
    labels = labels[:-1]
    labels.insert(0, 'O')

    d = {labels[i]: i for i in range(len(labels))}
    return d


def pad_and_truncate(text,
                     label,
                     max_len,
                     padding='post',
                     truncating='post',
                     padding_x=0,
                     padding_y=0,
                     mask=0):

    assert len(text) == len(label)
    _len = len(text)
    attention_mask = [1 for _ in range(_len)]

    if _len > max_len:
        # 保留头尾的[CLS]和[SEP]
        if truncating == 'pre':
            text = [text[0]] + text[-(max_len - 2):] + [text[-1]]
            label = [label[0]] + label[-(max_len - 2):] + [label[-1]]
        else:
            text = [text[0]] + text[:(max_len - 2)] + [text[-1]]
            label = [label[0]] + label[:(max_len - 2)] + [label[-1]]
        attention_mask = attention_mask[:max_len]
    else:
        padding_num = max_len - _len
        if padding == 'post':
            text = text + [padding_x for _ in range(padding_num)]
            label = label + [padding_y for _ in range(padding_num)]
            attention_mask = attention_mask + [mask for _ in range(padding_num)]
        else:
            text = [padding_x for _ in range(padding_num)] + text
            label = [padding_y for _ in range(padding_num)] + label
            attention_mask = [mask for _ in range(padding_num)] + attention_mask

    return text, attention_mask, label


class Tokenizer4Bert:
    def __init__(self, label2idx, max_seq_len):
        model_name = 'peterchou/ernie-gram'
        self.tz = AutoTokenizer.from_pretrained(model_name)
        self.label2idx = label2idx
        self.idx2label = {v: k for k, v in label2idx.items()}
        self.max_seq_len = max_seq_len

    def tokenize(self, text, label):
        text = ['[CLS]'] + text + ['[SEP]']
        label = ['O'] + label + ['O']
        text = self.tz.convert_tokens_to_ids(text)
        label = self.label_to_idx(label)
        return pad_and_truncate(text, label, max_len=self.max_seq_len)

    def label_to_idx(self, label):
        return [self.label2idx[l] for l in label]

    def idx_to_label(self, idx):
        return [self.idx2label[i] for i in idx]


class NERDataset(Dataset):
    def __init__(self, data, label, label2idx, max_seq_len):
        """
        :param data: (num_samples, seq_len)
        :param label: (num_samples, seq_len)
        :param label2idx: 标签和序号的映射字典
        :param max_seq_len: 每个样本的最大长度
        :param split: 训练/验证/测试
        """
        super(NERDataset, self).__init__()
        self.data = data
        self.label = label
        self.label2idx = label2idx
        self.tz = Tokenizer4Bert(label2idx, max_seq_len=max_seq_len)

    def __getitem__(self, index):
        text, label = self.data[index], self.label[index]
        text, attention_mask, label = self.tz.tokenize(text, label)
        data = {
            'input_ids': torch.LongTensor(text),
            'attention_mask': torch.LongTensor(attention_mask)
        }
        label = torch.LongTensor(label)
        return data, label

    def __len__(self):
        return len(self.data)


class WeiboNERDataset(NERDataset):
    def __init__(self, root, split, max_seq_len):
        self.label2idx = get_labels(Path(root) / 'weiboNER.conll.train')

        if split == 'train':
            fname = Path(root) / 'weiboNER.conll.train'
        elif split == 'val':
            fname = Path(root) / 'weiboNER.conll.dev'
        else:
            fname = Path(root) / 'weiboNER.conll.test'

        self.data, self.label = load_data(fname)
        super(WeiboNERDataset, self).__init__(self.data, self.label, self.label2idx, max_seq_len=max_seq_len)


class JdNERDataset(NERDataset):
    def __init__(self, root, split, max_seq_len):
        self.label2idx = get_labels(Path(root) / 'train_data' / 'train.txt')

        if split in ('train', 'val'):
            path = Path(root) / 'train_data' / 'train.txt'
            data, label = load_data(path, mode=split)
            indices = torch.randperm(len(data))
            split_num = int(len(data) * 0.8)
            if split == 'train':
                indices = indices[:split_num]
            elif split == 'val':
                indices = indices[split_num:]
            self.data = [data[i] for i in indices]
            self.label = [label[i] for i in indices]

        else:
            path = Path(root) / 'preliminary_test_a' / 'word_per_line_preliminary_A.txt'
            self.data, self.label = load_data(path, mode=split)

        super(JdNERDataset, self).__init__(self.data, self.label, self.label2idx, max_seq_len=max_seq_len)


def get_ner_loader(root,
                   dataset: str,
                   max_seq_len,
                   batch_size,
                   split='train',
                   num_workers=8,
                   limit=None):

    if dataset == 'weibo':
        dataset = WeiboNERDataset(root, split, max_seq_len)
    elif dataset == 'jd':
        dataset = JdNERDataset(root, split, max_seq_len)
    else:
        raise NotImplementedError

    if limit is not None:
        dataset = Subset(dataset, limit)

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True if split == 'train' else False
    )
    return loader
