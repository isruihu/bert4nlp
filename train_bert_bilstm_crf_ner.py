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
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn

from bert4nlp.model import BertMLP, BertBiLSTM, BertBiLstmCRF
from bert4nlp.utils.trainer import get_optimizer, save_model
from bert4nlp.utils.utils import AverageMeter, accuracy, set_seed
from bert4nlp.utils.logging import set_logging
from bert4nlp.dataset.ner_dataset import get_ner_loader


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default=os.path.basename(__file__).split('.')[0])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_classes', type=int, default=105)

    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--bs', type=int, default=16, help='batch size')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--aug', type=bool, default=True, help='data augmentation')

    parser.add_argument('--model', type=str, default='eirne-bilstm-crf')
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
    model = BertBiLstmCRF(num_classes=opt.num_classes).cuda()
    criterion = nn.CrossEntropyLoss()

    bert_para, other_para = [], []
    for name, para in model.named_parameters():
        if 'bert' in name:
            bert_para.append(para)
        else :
            other_para.append(para)
    optimizer = torch.optim.Adam([
        {'params': bert_para, 'lr': 2e-5},
        {'params': other_para, 'lr': 1e-3, 'weight_decay': 1e-4}
    ])

    decay_epochs = [opt.epochs * 2 // 3, opt.epochs * 4 // 5]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_epochs, gamma=0.1)
    return model, criterion, optimizer, scheduler


def train_epoch(train_loader, model, criterion, optimizer, epoch):
    model.train()
    avg_loss = AverageMeter()

    loader = train_loader.__iter__()
    for _ in tqdm(range(len(loader))):
        data, label = next(loader)
        input_ids = data['input_ids'].cuda()
        attention_mask = data['attention_mask'].cuda()
        label = label.cuda()

        bsz = label.shape[0]

        loss = model.negative_log_like_hood(input_ids, attention_mask, label)
        avg_loss.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return avg_loss.avg


def validate(val_loader, model):
    model.eval()
    top1 = AverageMeter()

    tp, fn, fp, tn = 1e-10, 1e-10, 1e-10, 1e-10
    with torch.no_grad():
        for idx, (data, label) in enumerate(val_loader):
            input_ids = data['input_ids'].cuda()
            attention_mask = data['attention_mask'].cuda()
            label = label.cuda()

            bsz = label.shape[0]

            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            for (pred, _), tags in zip(output, label):
                pred = torch.LongTensor(pred).cuda()
                tags = tags[:len(pred)]

                true_positive = tags > 0
                true_negative = tags == 0

                tp = tp + torch.sum(true_positive[pred == tags])
                fn = fn + torch.sum(true_positive[pred != tags])
                fp = fp + torch.sum(true_negative[pred != tags])
                tn = tn + torch.sum(true_negative[pred == tags])

                acc1, = accuracy(pred, tags, topk=(1,))
                top1.update(acc1[0], bsz)

        print(f"tp:{tp.item():.3f} fn:{fn.item():.3f}")
        print(f"fp:{fp.item():.3f} tn:{tn.item():.3f}")
        precision, recall = tp / (tp + fp), tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

    return f1, precision, recall


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
    root = './rsrc/JD_NER'
    train_loader = get_ner_loader(root,
                                  dataset='jd',
                                  max_seq_len=128,
                                  batch_size=opt.bs,
                                  split='train')
    val_loader = get_ner_loader(root,
                                dataset='jd',
                                max_seq_len=128,
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
    best_val = {'f1': 0, 'precision': 0, 'recall': 0, 'epoch': 0}
    start_time = time.time()
    for epoch in range(1, opt.epochs + 1):
        logging.info(f'[{epoch} / {opt.epochs}]')
        loss = train_epoch(train_loader, model, criterion, optimizer, epoch)
        logging.info(f'train loss: {loss:.3f}')
        scheduler.step()

        (save_path / 'checkpoints').mkdir(parents=True, exist_ok=True)
        save_file = save_path / 'checkpoints' / f'{opt.model}_last.pth'
        save_model(model, optimizer, opt, opt.epochs, save_file)

        f1, precision, recall = validate(val_loader, model)
        if f1 > best_val['f1']:
            best_val['f1'] = f1.item()
            best_val['precision'] = precision.item()
            best_val['recall'] = recall.item()
            best_val['epoch'] = epoch
            save_file = save_path / 'checkpoints' / f'{opt.model}_best.pth'
            save_model(model, optimizer, opt, opt.epochs, save_file)

        logging.info(
            f'\t val_f1: {f1.item():.3f} '
            f'val_precision: {precision.item():.3f} '
            f'val_recall: {recall.item():.3f}\n'
            f'\t best_val_f1: {best_val["f1"]:.3f} '
            f'best_val_precision: {best_val["precision"]:.3f} '
            f'best_val_recall: {best_val["recall"]:.3f} '
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info(f'Total training time: {total_time_str}')
