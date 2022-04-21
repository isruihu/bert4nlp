#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Ryan Hu
# DATE: 2022/3/15 14:02
# DESCRIPTION:
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import time
import datetime
import logging
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn

from bert4nlp.model import ErnieGramForTextSim
from bert4nlp.utils.trainer import get_optimizer, load_model
from bert4nlp.dataset.similarity_dataset import get_sim_loader


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default=os.path.basename(__file__).split('.')[0])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_classes', type=int, default=2)

    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--bs', type=int, default=64, help='batch size')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--nw', type=int, default=4)

    parser.add_argument('--model', type=str, default='erine')
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=2e-5)

    opt = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)
    return opt


def set_model(opt):
    """
    设置模型、损失函数、优化器
    :param opt:
    :return:
    """
    model = ErnieGramForTextSim()
    state_dict = torch.load('./exp_results/train_erine_gram_text_sim-model_erine-seed42-bs32/checkpoints/erine_best.pth')
    _, model, _, _ = load_model(state_dict, model)
    model = model.cuda()

    criterion = nn.BCELoss()
    optimizer = get_optimizer(
        opt.optim,
        filter(lambda p: p.requires_grad, model.parameters()),
        opt.lr
    )
    decay_epochs = [opt.epochs * 2 // 3, opt.epochs * 4 // 5]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_epochs, gamma=0.1)
    return model, criterion, optimizer, scheduler


def eval(test_loader, model):
    model.eval()

    res = []
    with torch.no_grad():
        loader = test_loader.__iter__()
        for _ in tqdm(range(len(test_loader))):
            data, label = next(loader)
            input_ids1 = data['input_ids1'].cuda()
            attention_mask1 = data['attention_mask1'].cuda()
            input_ids2 = data['input_ids2'].cuda()
            attention_mask2 = data['attention_mask2'].cuda()
            labels = label.squeeze().cuda()

            bsz = labels.shape[0]

            output = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
            output = torch.as_tensor(output > 0.5, dtype=torch.long)
            # print(output.shape)
            res.append(output)
    res = torch.cat(res, dim=0).cpu().numpy().tolist()
    return res


if __name__ == '__main__':
    # setting
    opt = parse_option()
    # data
    root = './rsrc/qianyan'
    test_loader = get_sim_loader(root,
                                 max_seq_len=128,
                                 batch_size=512,
                                 split='test',
                                 num_workers=8,
                                 limit=opt.limit)
    # criterion
    model, criterion, optimizer, scheduler = set_model(opt)
    # eval
    res = eval(test_loader, model)
    res = [f"{str(i)}\n" for i in res]
    fout = open('predict.csv', 'w')
    fout.writelines(res)
