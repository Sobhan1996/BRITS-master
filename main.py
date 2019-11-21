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

    evals = []
    imputations = []
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

        evals += eval_[np.where(eval_masks == 1)].tolist()
        imputations += imputation[np.where(eval_masks == 1)].tolist()

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

    evals = np.asarray(evals)
    imputations = np.asarray(imputations)
    print '\n'

    print 'MAE', np.abs(evals - imputations).mean()

    print 'MRE', np.abs(evals - imputations).sum() / np.abs(evals).sum()


    return all_evals, all_imputations, all_eval_masks


def run():
    model = getattr(models, args.model).Model(args.hid_size, args.impute_weight, args.label_weight)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total params is {}'.format(total_params))

    if torch.cuda.is_available():
        model = model.cuda()

    train(model)


if __name__ == '__main__':
    run()

