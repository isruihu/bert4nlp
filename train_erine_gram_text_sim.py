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

from bert4nlp.model import ErnieGramForTextSim
from bert4nlp.utils.trainer import get_optimizer, save_model
from bert4nlp.utils.utils import AverageMeter, accuracy, set_seed
from bert4nlp.utils.logging import set_logging
from bert4nlp.dataset.similarity_dataset import get_sim_loader


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default=os.path.basename(__file__).split('.')[0])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_classes', type=int, default=2)

    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--bs', type=int, default=196, help='batch size')
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
    model = ErnieGramForTextSim().cuda()
    # for param in model.parameters():
    #     param.requires_grad = False
    # for name, param in model.named_parameters():
    #     if '9' in name or '10' in name or '11' in name or 'pooler' in name:
    #         param.requires_grad = True
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    criterion = nn.BCELoss()
    optimizer = get_optimizer(
        opt.optim,
        filter(lambda p: p.requires_grad, model.parameters()),
        opt.lr
    )
    decay_epochs = [opt.epochs * 2 // 3, opt.epochs * 4 // 5]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_epochs, gamma=0.1)
    return model, criterion, optimizer, scheduler


def train_epoch(train_loader, model, criterion, optimizer, epoch):
    model.train()
    avg_loss = AverageMeter()

    loader = train_loader.__iter__()
    for idx in tqdm(range(len(train_loader))):
        data, label = next(loader)
        input_ids1 = data['input_ids1'].cuda()
        attention_mask1 = data['attention_mask1'].cuda()
        input_ids2 = data['input_ids2'].cuda()
        attention_mask2 = data['attention_mask2'].cuda()
        labels = label.squeeze().float().cuda()

        bsz = labels.shape[0]

        output = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
        loss = criterion(output, labels)
        avg_loss.update(loss.item(), bsz)
        if idx % 1000 == 0:
            print(f"training loss: {avg_loss.avg}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return avg_loss.avg


def validate(val_loader, model):
    model.eval()
    top1 = AverageMeter()

    with torch.no_grad():
        for idx, (data, label) in enumerate(val_loader):
            input_ids1 = data['input_ids1'].cuda()
            attention_mask1 = data['attention_mask1'].cuda()
            input_ids2 = data['input_ids2'].cuda()
            attention_mask2 = data['attention_mask2'].cuda()
            labels = label.squeeze().cuda()

            bsz = labels.shape[0]

            output = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
            output = torch.as_tensor(output > 0.5, dtype=torch.long)
            acc1 = accuracy(output, labels)
            top1.update(acc1[0], bsz)

    return top1.avg


if __name__ == '__main__':
    # setting
    opt = parse_option()
    exp_name = f"{opt.exp_name}-model_{opt.model}-seed{opt.seed}-bs{opt.bs}"
    opt.exp_name = exp_name

    output_dir = f'exp_results/{exp_name}'
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    set_logging(exp_name, 'INFO', str(save_path))
    set_seed(opt.seed)
    logging.info(exp_name)
    logging.info(f'Set seed: {opt.seed}')

    # data
    root = './rsrc/qianyan'
    train_loader = get_sim_loader(root,
                                  max_seq_len=128,
                                  batch_size=opt.bs,
                                  split='train',
                                  num_workers=opt.nw,
                                  limit=opt.limit)
    val_loader = get_sim_loader(root,
                                max_seq_len=128,
                                batch_size=opt.bs,
                                split='val',
                                num_workers=opt.nw,
                                limit=opt.limit)

    logging.info('train size: {} validation size: {} [batch size: {}]'.format(
        len(train_loader.dataset),
        len(val_loader.dataset),
        opt.bs
    ))

    # criterion
    model, criterion, optimizer, scheduler = set_model(opt)

    # training
    best_val_acc = {'acc': 0, 'epoch': 0}
    start_time = time.time()
    for epoch in range(1, opt.epochs + 1):
        logging.info(f'[{epoch} / {opt.epochs}] Learning rate: {scheduler.get_last_lr()[0]}')
        loss = train_epoch(train_loader, model, criterion, optimizer, epoch)
        logging.info(f'train loss: {loss:.3f}')
        scheduler.step()

        (save_path / 'checkpoints').mkdir(parents=True, exist_ok=True)
        save_file = save_path / 'checkpoints' / f'{opt.model}_last.pth'
        save_model(model, optimizer, opt, opt.epochs, save_file)

        val_acc = validate(val_loader, model)
        if val_acc > best_val_acc['acc']:
            best_val_acc['acc'] = val_acc.item()
            best_val_acc['epoch'] = epoch
            save_file = save_path / 'checkpoints' / f'{opt.model}_best.pth'
            save_model(model, optimizer, opt, opt.epochs, save_file)

        logging.info(f'\t val_acc: {val_acc.item():.3f}\n'
                     f'\t best_val_acc: {best_val_acc["acc"]:.3f} best_epoch: {best_val_acc["epoch"]}')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info(f'Total training time: {total_time_str}')
