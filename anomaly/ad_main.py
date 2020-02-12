import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import utils
import models
import argparse
import data_loader
import matplotlib.pyplot as plt
import json
import os

from ad_process import UCIDataset, parse_id

def train(model, epochs):
    with open('../models/settings.txt') as json_file:
        settings = json.load(json_file)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    data_iter = data_loader.get_loader(batch_size=args.batch_size, shuffle=False)

    all_evals = []
    all_imputations = []
    all_eval_masks = []

    for epoch in range(epochs):
        model.train()

        run_loss = 0.0

        for idx, data in enumerate(data_iter):
            data = utils.to_var(data)
            ret = model.run_on_batch(data, optimizer, epoch)

            run_loss += ret['loss'].item()

            # print '\r Progress epoch {}, {:.2f}%, average loss {}'.format(epoch, (idx + 1) * 100.0 / len(data_iter), run_loss / (idx + 1.0)),
            print ' Progress epoch {}, {:.2f}%, average loss {}'.format(epoch, (idx + 1) * 100.0 / len(data_iter),
                                                                          run_loss / (idx + 1.0)),

        all_evals, all_imputations, all_eval_masks = evaluate(model, data_iter)

    imputing_columns = settings['imputing_columns']

    # delta_matrix = all_evals[:, imputing_columns[0]] - all_imputations[:, imputing_columns[0]]

    return all_evals, all_imputations, all_eval_masks, imputing_columns
    # print(delta_matrix)
    # print('delta_matrix shape is {}'.format(delta_matrix.shape))
    # print(list(delta_matrix))
    # print(list(all_eval_masks[:, imputing_columns[0]]))
    # print(list(np.multiply(delta_matrix, all_eval_masks[:, imputing_columns[0]])))

    # plot figures
    # plt.plot(all_evals[:, imputing_columns[0]], 'r', label="True")
    # plt.plot(all_imputations[:, imputing_columns[0]], 'b', label="Imputed")
    # plt.plot(all_eval_masks[:, imputing_columns[0]], 'g.')
    # plt.subplots_adjust(wspace=2)
    # plt.legend(loc="upper left")
    # plt.title('BRITS')
    # plt.show()

#    np.savetxt(settings['eval_masks_output'], all_eval_masks)


def evaluate(model, val_iter):
    model.eval()

    with open('../models/settings.txt') as json_file:
        settings = json.load(json_file)

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


    all_evals = np.add(np.multiply(all_evals, settings['std']), np.asarray(settings['mean']))
    all_imputations = np.add(np.multiply(all_imputations, settings['std']), np.asarray(settings['mean']))

    # columns_ones = count_ones_in_columns(all_eval_masks)
    #
    # eval_imputed_diff = np.multiply(np.abs(all_evals - all_imputations), all_eval_masks)
    # eval_imputed_diff_squared = [x ** 2 for x in eval_imputed_diff]
    # attributes_variance = np.nanvar(all_evals, axis=0)
    # nrms_denominator = np.multiply(attributes_variance, columns_ones)
    # nrms_vector = np.sqrt(np.divide(np.nansum(eval_imputed_diff_squared), nrms_denominator))
    # nrms_vector[nrms_vector == np.inf] = 0
    #
    # print('NRMS Vector', nrms_vector)
    #
    # nrms_vector_squared = [x ** 2 for x in nrms_vector]
    # nrms = np.sqrt(np.dot(nrms_vector_squared, columns_ones) / np.sum(columns_ones))
    #
    # print('NRMS', nrms)

    return all_evals, all_imputations, all_eval_masks


def count_ones_in_columns(masks):
    counts = []
    for i in range(0, masks.shape[1]):
        count = np.count_nonzero(masks[:, i])
        counts.append(count)
    return counts


def run(model_name, hid_size, impute_weight, label_weight, epochs):
    model = getattr(models, model_name).Model(hid_size, impute_weight, label_weight)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total params is {}'.format(total_params))

    if torch.cuda.is_available():
        model = model.cuda()

    all_evals, all_imputations, all_eval_masks, imputing_columns = train(model, epochs)
    return all_evals, all_imputations, all_eval_masks, imputing_columns


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)  # 1000
    # parser.add_argument('--epochs', type=int, default=1)  # 1000
    # parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model', type=str, default='brits')
    # parser.add_argument('--model', type=str, default='rits_i')
    parser.add_argument('--hid_size', type=int, default=108)
    # parser.add_argument('--hid_size', type=int, default=16)
    parser.add_argument('--impute_weight', type=float, default=0.3)
    parser.add_argument('--label_weight', type=float, default=1.0)

    parser.add_argument('--window', type=int, default=50)
    args = parser.parse_args()

    # use for generating sub-sequences
    window = args.window
    # use for training
    epochs = args.epochs
    batch_size = args.batch_size
    model_name = args.model
    hid_size = args.hid_size
    impute_weight = args.impute_weight
    label_weight = args.label_weight

    num_tuple = 100
    ds_name = 'air_all'
    # ds_name = 'air_all_0.1symmetry1'
    # ds_name = 'air_{}_0.1symmetry1'.format(num_tuple)
    raw_fpath = '../raw/{}.csv'.format(ds_name)
    df = pd.read_csv(raw_fpath)
    print('df shape is {} '.format(df.shape))
    df_dummy = pd.get_dummies(df.drop('timestamp', axis=1))
    df_timestamp = df[['timestamp']]
    print('df dummy shape is {} '.format(df_dummy.shape))
    torch.manual_seed(0)
    np.random.seed(0)
    ### debug
    # target_cols = [0, 1, 2, 3]
    ### in theory, only one will be detected and imputed
    target_cols = [0]
    nrow = window
    # nrow = 1
    output_ad = 'result/{}_delta_e{}.csv'.format(ds_name, epochs)
    output_imp = 'result/{}_impute_e{}.csv'.format(ds_name, epochs)

    delta_matrix = np.zeros(df_dummy.shape)
    impute_matrix = np.zeros(df_dummy.shape)

    for col in target_cols:
        for index in range(nrow):
            print('Processing target col {} with missing index {}'.format(df_dummy.columns[col], index))
            json_fpath = '../json/{}_{}_{}'.format(ds_name, col, index)
            dataset = UCIDataset(window, raw_fpath, json_fpath, [col])

            for id_ in dataset.ids:
                try:
                    # generate dataset
                    parse_id(id_, dataset, index)
                except Exception as e:
                    print(e)
                    continue
            print('parse id over')
            # write params into dataset
            del dataset
            # read in the dataset and handle it
            all_evals, all_imputations, all_eval_masks, imputing_columns = \
                run(model_name, hid_size, impute_weight, label_weight, epochs)

            current_delta = (all_evals - all_imputations) * all_eval_masks
            delta_matrix += current_delta
            current_impute = all_imputations * all_eval_masks
            impute_matrix += current_impute

            # print(delta_matrix)
            # after all index are set missing one by one, the final delta_matrix should be full on all target_cols
            df_delta = pd.DataFrame(delta_matrix, columns=df_dummy.columns)
            df_delta['timestamp'] = df_timestamp
            df_delta.to_csv(output_ad, index=False)

            df_impute = pd.DataFrame(impute_matrix, columns=df_dummy.columns)
            df_impute['timestamp'] = df_timestamp
            df_impute.to_csv(output_imp, index=False)
