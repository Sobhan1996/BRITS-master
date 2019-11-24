# coding: utf-8

import os
import re
import numpy as np
import pandas as pd
import ujson as json
import json as js

class BRITSDataset:
    def __init__(self, window):
        self.ids = []
        self.window = window
        self.set_ids()
        self.columns = 0
        self.fs = None
        self.mean = []
        self.std = []
        self.imputing_columns = []

    def set_ids(self):
        pass

    def read_data(self, id_):
        pass

    def get_label(self, id_):
        pass

    def update_evals(self, evals, id_):
        return evals

    def __del__(self):
        self.mean = self.mean.tolist()
        self.std = self.std.tolist()
        data = {
            "SEQ_LEN": self.window,
            "COLUMNS": self.columns,
            "JsonFile": self.fs.name,
            "mean": self.mean,
            "std": self.std
        }
        with open('./models/settings.txt', 'w') as outfile:
            js.dump(data, outfile)
        self.fs.close()


class PhysioNetDataset(BRITSDataset):
    def __init__(self):
        BRITSDataset.__init__(self, 48)
        self.attributes = ['DiasABP', 'HR', 'Na', 'Lactate', 'NIDiasABP', 'PaO2', 'WBC', 'pH', 'Albumin', 'ALT', 'Glucose', 'SaO2',
                           'Temp', 'AST', 'Bilirubin', 'HCO3', 'BUN', 'RespRate', 'Mg', 'HCT', 'SysABP', 'FiO2', 'K', 'GCS',
                           'Cholesterol', 'NISysABP', 'TroponinT', 'MAP', 'TroponinI', 'PaCO2', 'Platelets', 'Urine', 'NIMAP',
                           'Creatinine', 'ALP']

        self.mean = np.array([59.540976152469405, 86.72320413227443, 139.06972964987443, 2.8797765291788986, 58.13833409690321,
                              147.4835678885565, 12.670222585415166, 7.490957887101613, 2.922874149659863, 394.8899400819931,
                              141.4867570064675, 96.66380228136883, 37.07362841054398, 505.5576196473552, 2.906465787821709,
                              23.118951553526724, 27.413004968675743, 19.64795551193981, 2.0277491155660416, 30.692432164676188,
                              119.60137167841977, 0.5404785381886381, 4.135790642787733, 11.407767149315339, 156.51746031746032,
                              119.15012244292181, 1.2004983498349853, 80.20321011673151, 7.127188940092161, 40.39875518672199,
                              191.05877024038804, 116.1171573535279, 77.08923183026529, 1.5052390166989214, 116.77122488658458])

        self.std = np.array(
            [13.01436781437145, 17.789923096504985, 5.185595006246348, 2.5287518090506755, 15.06074282896952, 85.96290370390257,
             7.649058756791069, 8.384743923130074, 0.6515057685658769, 1201.033856726966, 67.62249645388543, 3.294112002091972,
             1.5604879744921516, 1515.362517984297, 5.902070316876287, 4.707600932877377, 23.403743427107095, 5.50914416318306,
             0.4220051299992514, 5.002058959758486, 23.730556355204214, 0.18634432509312762, 0.706337033602292,
             3.967579823394297, 45.99491531484596, 21.97610723063014, 2.716532297586456, 16.232515568438338, 9.754483687298688,
             9.062327978713556, 106.50939503021543, 170.65318497610315, 14.856134327604906, 1.6369529387005546,
             133.96778334724377])

        self.fs = open('./json/json', 'w')

        self.label_df = pd.read_csv('./raw/Outcomes-a.txt').set_index('RecordID')['In-hospital_death']

        self.columns = 35

    def set_ids(self):
        for file_name in os.listdir('./raw'):
            # the patient data in PhysioNet contains 6-digits
            matched = re.search('\d{6}', file_name)
            if matched:
                file_id = matched.group()
                self.ids.append(file_id)

    def get_label(self, id_):
        return self.label_df.loc[int(id_)]

    def read_data(self, id_):

        def to_time_bin(x):
            h, m = map(int, x.split(':'))
            return h

        def parse_data(x):
            x = x.set_index('Parameter').to_dict()['Value']

            values = []

            for attr in self.attributes:
                if x.has_key(attr):
                    values.append(x[attr])
                else:
                    values.append(np.nan)
            return values

        data = pd.read_csv('./raw/{}.txt'.format(id_))  # 1 read data from one file which is one time series for one person
        data['Time'] = data['Time'].apply(lambda x: to_time_bin(x))  # 2 removing minutes from time column
        evals = []
        for h in range(48):
            evals.append(parse_data(data[data['Time'] == h]))  # 3 for each timestamp(hour) if an attribute is present we put it otherwise nan
        evals = (np.array(evals) - self.mean) / self.std    # 4 no need to explain

        return evals


class UCIDataset(BRITSDataset):
    def __init__(self, window, source_dataset, output_json, imputing_columns):
        self.data_frame = pd.read_csv(source_dataset)
        BRITSDataset.__init__(self, window)
        self.imputing_columns = imputing_columns
        self.data_frame = pd.get_dummies(self.data_frame)
        self.columns = self.data_frame.shape[1]

        self.mean = np.asarray(list(self.data_frame.mean(axis=0)))
        self.std = np.asarray(list(self.data_frame.std(axis=0, skipna=True)))

        self.fs = open(output_json, 'w')

    def set_ids(self):
        self.ids = range(0, self.data_frame.shape[0] / self.window)

    def get_label(self, id_):
        return 0

    def read_data(self, id_):
        frame = self.data_frame.loc[id_ * self.window: (id_+1) * self.window - 1, :]
        evals = []
        for i in range(self.window):
            evals.append(list(frame.iloc[i, :]))

        evals = (np.array(evals) - self.mean) / self.std

        return evals


class StockDataset(UCIDataset):
    def __init__(self, window, source_dataset, output_json):
        self.data_frame = pd.read_csv(source_dataset)
        self.data_frame = self.data_frame.drop(['timestamp'], axis=1)
        BRITSDataset.__init__(self, window)

        self.evals_data_frame = self.data_frame[[' truth']]
        self.data_frame = self.data_frame.drop([' truth'], axis=1)

        self.data_frame = pd.get_dummies(self.data_frame)
        self.columns = self.data_frame.shape[1]

        self.mean = np.asarray(list(self.data_frame.mean(axis=0)))
        self.std = np.asarray(list(self.data_frame.std(axis=0, skipna=True)))

        self.fs = open(output_json, 'w')

    def update_evals(self, orig_evals, id_):
        frame = self.evals_data_frame.loc[id_ * self.window: (id_+1) * self.window - 1, :]
        evals = []
        for i in range(self.window):
            evals.append(list(frame.iloc[i, :]))

        evals = (np.array(evals) - self.mean) / self.std

        return evals



def parse_delta(masks, window, columns, dir_):
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


def parse_id(id_, ds):

    evals = ds.read_data(id_)

    shp = evals.shape

    evals = evals.reshape(-1)       # 5 flattens the data in a 1d list

    # randomly eliminate 10% values as the imputation ground-truth
    indices = np.where(~np.isnan(evals))[0].tolist()    # 6 getting indices of the flat evals list that are not nan
    if (not ds.imputing_columns) == False:
        indices = list(filter(lambda x: (x % ds.columns in ds.imputing_columns), indices))
    if len(indices) > 10:
        indices = np.random.choice(indices, len(indices) // 10)     # 7 randomly selecting 10 percent of the non nan indices

    values = evals.copy()
    values[indices] = np.nan    # 8 setting 10 percent indices to nan in the new list values which has been copied from evals

    masks = ~np.isnan(values)   # 9 bool matrix which is true for not nan indices of values
    eval_masks = (~np.isnan(values)) ^ (~np.isnan(evals))  # for the 10 percent indices that we randomly selected, eval_masks is true in those indices

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
    # print(rec)

    ds.fs.write(rec + '\n')


# dataset = PhysioNetDataset()
dataset = UCIDataset(50, './PRSA_data_2010.1.1-2014.12.31.csv', './json/jsonAir', [5])
# dataset = StockDataset(30, './stock10k.data', './json/jsonStock')

for id_ in dataset.ids:
    print('Processing data point {}'.format(id_))
    try:
        parse_id(id_, dataset)
    except Exception as e:
        print(e)
        continue

