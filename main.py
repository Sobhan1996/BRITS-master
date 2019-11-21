import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import numpy as np

import time
import utils
import models
import argparse
import data_loader
import pandas as pd
import ujson as json

import matplotlib.pyplot as plt

from sklearn import metrics

from ipdb import set_trace

import json

with open('./models/settings.txt') as json_file:
    settings = json.load(json_file)

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--model', type=str)
parser.add_argument('--hid_size', type=int)
parser.add_argument('--impute_weight', type=float)
parser.add_argument('--label_weight', type=float)
args = parser.parse_args()


def train(model):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    data_iter = data_loader.get_loader(batch_size=args.batch_size)

    all_evals = []
    all_imputations = []
    all_eval_masks = []

    for epoch in range(args.epochs):
        model.train()

        run_loss = 0.0

        for idx, data in enumerate(data_iter):
            data = utils.to_var(data)
            ret = model.run_on_batch(data, optimizer, epoch)

            run_loss += ret['loss'].item()

            print '\r Progress epoch {}, {:.2f}%, average loss {}'.format(epoch, (idx + 1) * 100.0 / len(data_iter), run_loss / (idx + 1.0)),

        all_evals, all_imputations, all_eval_masks = evaluate(model, data_iter)

    plt.plot(all_evals[0:200, :], 'r')
    plt.plot(all_imputations[0:200, :], 'b')
    plt.plot(all_eval_masks[0:200, :], 'g.')
    plt.subplots_adjust(wspace=2)
    plt.show()

def evaluate(model, val_iter):
    model.eval()

    all_eval_masks = []
    all_imputations = []
    all_evals = []
    flag = 1

    for idx, data in enumerate(val_iter):
        data = utils.to_var(data)
        ret = model.run_on_batch(data, None)

        eval_masks = ret['eval_masks'].data.cpu().numpy()
        eval_ = ret['evals'].data.cpu().numpy()
        imputation = ret['imputations'].data.cpu().numpy()

        first_d = eval_masks.shape[0]
        second_d = eval_masks.shape[1]
        third_d = eval_masks.shape[2]
        if flag == 1:
            all_evals = eval_.reshape(first_d * second_d, third_d)
            all_imputations = imputation.reshape(first_d * second_d, third_d)
            all_eval_masks = eval_masks.reshape(first_d * second_d, third_d)
            flag = 0
        else:
            all_evals = np.concatenate((all_evals, eval_.reshape(first_d * second_d, third_d)), axis=0)
            all_imputations = np.concatenate((all_imputations, imputation.reshape(first_d * second_d, third_d)), axis=0)
            all_eval_masks = np.concatenate((all_eval_masks, eval_masks.reshape(first_d * second_d, third_d)), axis=0)

    columns_ones = count_ones_in_columns(all_eval_masks)
    columns_ones = [float(all_evals.shape[0]) / x for x in columns_ones]

    all_evals = np.add(np.multiply(all_evals, settings['std']), np.asarray(settings['mean']))
    all_imputations = np.add(np.multiply(all_imputations, settings['std']), np.asarray(settings['mean']))

    print '\n'
    print 'MAE', np.multiply(np.abs(all_evals - all_imputations).mean(axis=0), columns_ones)
    print 'MRE', np.abs(all_evals - all_imputations).sum(axis=0) / np.abs(all_evals[np.where(all_eval_masks == 1)]).sum(axis=0)

    return all_evals, all_imputations, all_eval_masks


def count_ones_in_columns(masks):
    counts = []
    for i in range(0, masks.shape[1]):
        count = np.count_nonzero(masks[:, i])
        counts.append(count)
    return counts


def run():
    model = getattr(models, args.model).Model(args.hid_size, args.impute_weight, args.label_weight)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total params is {}'.format(total_params))

    if torch.cuda.is_available():
        model = model.cuda()

    train(model)


if __name__ == '__main__':
    run()

