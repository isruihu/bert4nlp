#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Ryan Hu
# DATE: 2022/3/15 14:21
# DESCRIPTION:
import torch
import torch.optim as optim


def get_model():
    pass


def save_model(model, optimizer, opt, epoch, save_file):
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def load_model(state_dict, model, optimizer=None):
    opt = state_dict['opt']
    model.load_state_dict(state_dict['model'])
    if optimizer is not None:
        optimizer.load_state_dict(state_dict['optimizer'])
    epoch = state_dict['epoch']

    return opt, model, optimizer, epoch


def get_optimizer(name, parameters, lr):
    if name == 'adam':
        optimizer = optim.Adam(parameters, lr=lr, weight_decay=1e-4)
    elif name == 'sgd':
        optimizer = optim.SGD(parameters, weight_decay=.0005, momentum=.9, lr=lr)
    else:
        raise ValueError()

    return optimizer
