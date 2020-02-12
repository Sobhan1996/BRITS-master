import os
import re
import numpy as np
import pandas as pd
import ujson as json
import json as js
import sys
import argparse


class UCIDataset:
    def __init__(self, window, source_dataset, output_json, imputing_columns):
        self.read_dataset(source_dataset)
        self.window = window
        self.set_ids()
        self.imputing_columns = imputing_columns
        self.data_frame = pd.get_dummies(self.data_frame)
        self.columns = self.data_frame.shape[1]

        self.mean = np.asarray(list(self.data_frame.mean(axis=0)))
        self.std = np.asarray(list(self.data_frame.std(axis=0, skipna=True)))

        ### modify the std to 1
        self.std[self.std == 0.] = 1

        self.fs = open(output_json, 'w')

    def read_dataset(self, source_dataset):
        raw_df = pd.read_csv(source_dataset)
        ### todo(aoqian): seperate time and main body
        self.timestamp = raw_df[['timestamp']]
        self.data_frame = raw_df.drop('timestamp', axis=1)

    def set_ids(self):
        """
        the id of sub sequences
        :return:
        """
        self.ids = range(0, self.data_frame.shape[0] / self.window)

    ### todo(aoqian): might be removed then, since no need to get the label
    def get_label(self, id_):
        return 0

    def update_evals(self, evals, id_):
        return evals

    def read_data(self, id_):
        frame = self.data_frame.loc[id_ * self.window: (id_+1) * self.window - 1, :]
        evals = []
        for i in range(self.window):
            evals.append(list(frame.iloc[i, :]))

        evals = (np.array(evals) - self.mean) / self.std

        return evals

    def __del__(self):
        print("destroy UCI dataset")
        if not isinstance(self.mean, list):
            self.mean = self.mean.tolist()
        if not isinstance(self.std, list):
            self.std = self.std.tolist()
        data = {
            "SEQ_LEN": self.window,
            "COLUMNS": self.columns,
            "JsonFile": self.fs.name,
            "mean": self.mean,
            "std": self.std,
            "imputing_columns": self.imputing_columns
        }
        with open('../models/settings.txt', 'w') as outfile:
            js.dump(data, outfile)
        self.fs.close()


def parse_delta(masks, window, columns, dir_):
    """
    todo(aoqian): 1. do not normalize time column; 2. compute the correct delta matrix, may follow GURI-GAN
    compute the delta matrix: the time gaps, but should focus
    :param masks:
    :param window:
    :param columns:
    :param dir_:
    :return:
    """
    if dir_ == 'backward':
        masks = masks[::-1]

    deltas = []

    for h in range(window):
        if h == 0:
            deltas.append(np.ones(columns))
        else:
            deltas.append(np.ones(columns) + (1 - masks[h]) * deltas[-1])

    return np.array(deltas)


def parse_rec(values, masks, evals, eval_masks, window, columns, dir_):
    deltas = parse_delta(masks, window, columns, dir_)

    # only used in GRU-D
    forwards = pd.DataFrame(values).fillna(method='ffill').fillna(0.0).as_matrix()

    rec = {}

    rec['values'] = np.nan_to_num(values).tolist()
    rec['masks'] = masks.astype('int32').tolist()
    # imputation ground-truth
    rec['evals'] = np.nan_to_num(evals).tolist()
    rec['eval_masks'] = eval_masks.astype('int32').tolist()
    rec['forwards'] = forwards.tolist()
    rec['deltas'] = deltas.tolist()

    return rec


def parse_id(id_, ds, index):
    evals = ds.read_data(id_)
    shp = evals.shape
    evals = evals.reshape(-1)       # 5 flattens the data in a 1d list

    # randomly eliminate 10% values as the imputation ground-truth
    indices = np.where(~np.isnan(evals))[0].tolist()    # 6 getting indices of the flat evals list that are not nan
    # find all indices on the imputing columns
    indices = list(filter(lambda x: (x % ds.columns in ds.imputing_columns), indices))
    indices = [indices[index]]

    values = evals.copy()
    values[indices] = np.nan    # 8 setting 10 percent indices to nan in the new list values which has been copied from evals

    masks = ~np.isnan(values)   # 9 bool matrix which is true for not nan indices of values
    # for the indices that we randomly selected, eval_masks is true in those indices, others are all false
    eval_masks = (~np.isnan(values)) ^ (~np.isnan(evals))

    evals = ds.update_evals(evals, id_)

    evals = evals.reshape(shp)
    values = values.reshape(shp)

    masks = masks.reshape(shp)
    eval_masks = eval_masks.reshape(shp)    # 10 reshaping everything to its original shape

    label = ds.get_label(id_)

    rec = {'label': label}

    # prepare the model for both directions
    rec['forward'] = parse_rec(values, masks, evals, eval_masks, ds.window, ds.columns, dir_='forward')
    rec['backward'] = parse_rec(values[::-1], masks[::-1], evals[::-1], eval_masks[::-1], ds.window, ds.columns, dir_='backward')
    rec = json.dumps(rec)

    ds.fs.write(rec + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', type=int, default=50)
    parser.add_argument('--percent', type=int, default=10)
    parser.add_argument('--imputing', type=int, default=1)
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()

    window = args.window
    imputing = args.imputing
    percent = args.percent
    # index = args.index

    raw_fpath = '../raw/air_100.csv'
    df = pd.read_csv(raw_fpath)
    df_dummy = pd.get_dummies(df.drop('timestamp', axis=1))

    # make it integer
    nrow = window
    # target_cols = [0, 1, 2, 3]

    ### debug,
    ### in theroy, only one will be detected and imputed
    target_cols = [0]
    nrow = 1
    for col in target_cols:
        for index in range(nrow):
            dataset = UCIDataset(window, raw_fpath, '../json/jsonAir100_process', [col])
            for id_ in dataset.ids:
                print('Processing sub series {}'.format(id_))
                try:
                    parse_id(id_, dataset, index)
                except Exception as e:
                    print(e)
                    continue
