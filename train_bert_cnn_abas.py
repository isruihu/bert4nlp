#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Ryan Hu
# DATE: 2022/3/15 14:02
# DESCRIPTION:
import argparse
import os
import time
import datetime
import logging
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn

from bert4nlp.model import BertForABAS
from bert4nlp.utils.trainer import get_optimizer, save_model
from bert4nlp.utils.utils import AverageMeter, accuracy, set_seed
from bert4nlp.utils.logging import set_logging
from bert4nlp.dataset.abas_dataset import get_abas_loader


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default=os.path.basename(__file__).split('.')[0])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_classes', type=int, default=3)

    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--bs', type=int, default=32, help='batch size')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--aug', type=bool, default=True, help='data augmentation')

    parser.add_argument('--model', type=str, default='bert-cnn')
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
    model = BertForABAS(num_classes=opt.num_classes).cuda()
    model.bert.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(
        opt.optim,
        filter(lambda p: p.requires_grad, model.parameters()),
        opt.lr,
    )
    decay_epochs = [opt.epochs * 2 // 3, opt.epochs * 4 // 5]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_epochs, gamma=0.1)
    return model, criterion, optimizer, scheduler


def train_epoch(train_loader, model, criterion, optimizer, epoch):
    model.train()
    avg_loss = AverageMeter()

    for idx, x in tqdm(enumerate(train_loader)):
        input_ids = x['text_bert_indices'].cuda()
        attention_mask = x['attention_mask'].cuda()
        aspect_boundary = x['aspect_boundary'].cuda()
        labels = x['polarity'].cuda()

        bsz = labels.shape[0]

        output = model(input_ids, attention_mask, aspect_boundary)
        loss = criterion(output, labels)
        avg_loss.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return avg_loss.avg


def validate(val_loader, model):
    model.eval()
    top1 = AverageMeter()

    with torch.no_grad():
        for idx, x in enumerate(val_loader):
            input_ids = x['text_bert_indices'].cuda()
            attention_mask = x['attention_mask'].cuda()
            aspect_boundary = x['aspect_boundary'].cuda()
            labels = x['polarity'].cuda()

            bsz = labels.shape[0]

            output = model(input_ids, attention_mask, aspect_boundary)
            acc1 = accuracy(output, labels)
            top1.update(acc1[0], bsz)

    return top1.avg


if __name__ == '__main__':
    # setting
    opt = parse_option()
    exp_name = f"{opt.exp_name}-model_{opt.model}-seed{opt.seed}"
    opt.exp_name = exp_name

    output_dir = f'exp_results/{exp_name}'
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    set_logging(exp_name, 'INFO', str(save_path))
    set_seed(opt.seed)
    logging.info(f'Set seed: {opt.seed}')

    # data
    root = './rsrc/semeval14'
    train_loader = get_abas_loader(root,
                                   max_seq_len=256,
                                   batch_size=opt.bs,
                                   split='train')
    val_loader = get_abas_loader(root,
                                 max_seq_len=256,
                                 batch_size=opt.bs,
                                 split='val')

    logging.info('train size: {} validation size: {} [batch size: {}]'.format(
        len(train_loader.dataset),
        len(val_loader.dataset),
        opt.bs
    ))

    # criterion
    model, criterion, optimizer, scheduler = set_model(opt)

    # training
    start_time = time.time()
    for epoch in range(1, opt.epochs + 1):
        logging.info(f'[{epoch} / {opt.epochs}]')
        loss = train_epoch(train_loader, model, criterion, optimizer, epoch)
        logging.info(f'train loss: {loss:.3f}')
        scheduler.step()

        (save_path / 'checkpoints').mkdir(parents=True, exist_ok=True)
        save_file = save_path / 'checkpoints' / f'{opt.model}_last.pth'
        save_model(model, optimizer, opt, opt.epochs, save_file)

        val_acc = validate(val_loader, model)

        logging.info(
            f'val/acc: {val_acc.item():.3f}'
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info(f'Total training time: {total_time_str}')
