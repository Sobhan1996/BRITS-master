"""
check and evaluate how the delta matrix looks like, whether there is a mask showing there are real anomalies
"""
import numpy as np
import pandas as pd


def evaluate_delta(df_delta, attrs=None):
    if attrs is None:
        attrs = list(df_delta.columns)
    df_delta[attrs] = abs(df_delta[attrs])
    mean = df_delta[attrs].mean(axis=0)

    print(mean)


def evaluate_dirty_delta(df_delta, df_mask, attrs=None):
    """

    :param df_delta:
    :param df_mask:
    :param attrs:
    :return: attr_mean_dict, group by dirty/clean
    """
    if attrs is None:
        attrs = list(df_delta.columns)

    df_delta[attrs] = abs(df_delta[attrs])
    attr_mean_dict = {}

    for attr in attrs:
        df_attr = df_delta[['timestamp', attr]]
        df_attr_long = df_attr.melt(id_vars='timestamp', var_name='attribute', value_name='delta')
        df_mask['is_dirty'] = True
        df_all = df_attr_long.merge(df_mask, on=['timestamp', 'attribute'], how='left')
        df_all['is_dirty'].fillna(False, inplace=True)

        df_group = df_all.groupby('is_dirty')
        means = df_group['delta'].mean()
        attr_mean_dict[attr] = means

    return attr_mean_dict


def evaluate_anomaly(df_raw, df_delta, df_mask, attrs=None):
    if attrs is None:
        attrs = list(df_raw.columns)
    df_delta[attrs] = abs(df_delta[attrs])

    def precision():
        numerator = fil_cor.copy()
        denominator = fil_ad.copy()

        if denominator.sum() == 0:
            return np.nan
        return numerator.sum() / denominator.sum()

    def recall():
        numerator = fil_cor.copy()
        denominator = fil_dirty.copy()

        if denominator.sum() == 0:
            return np.nan
        return numerator.sum() / denominator.sum()

    df_result = pd.DataFrame()

    for attr in attrs:
        min_val = min(df_delta[attr])
        max_val = max(df_delta[attr])
        taus = np.linspace(min_val, max_val, 10)

        df_attr_res = pd.DataFrame()
        df_attr_res['tau'] = taus
        df_attr_res['attr'] = attr

        df_attr = df_delta[['timestamp', attr]]
        df_attr_long = df_attr.melt(id_vars='timestamp', var_name='attribute', value_name='delta')
        df_mask['is_dirty'] = True
        df_all = df_attr_long.merge(df_mask, on=['timestamp', 'attribute'], how='left')

        df_all['is_dirty'].fillna(False, inplace=True)
        fil_dirty = df_all['is_dirty'] == True

        precs, recs, f1s = [], [], []
        for tau in taus:
            fil_ad = df_all['delta'] > tau

            fil_cor = fil_dirty & fil_ad

            prec = precision()
            rec = recall()
            f1 = 2 * prec * rec / (prec + rec)

            precs.append(prec)
            recs.append(rec)
            f1s.append(f1)
        df_attr_res['precision'] = precs
        df_attr_res['recall'] = recs
        df_attr_res['f1'] = f1s

        df_result = pd.concat([df_result, df_attr_res], axis=1, sort=False)


if __name__ == '__main__':
    # ds_name = 'air_100'
    # ds_name = 'air_all'
    # frac_err = 0.1
    # err_type = 'symmetry'
    # error_seed = 1
    #
    # raw_fpath = '../raw/{}_{}{}{}.csv'.format(ds_name, frac_err, err_type, error_seed)
    # delta_fpath = 'result/{}_{}{}{}_delta.csv'.format(ds_name, frac_err, err_type, error_seed)
    # mask_fpath = '../raw/{}_{}{}{}_mask.csv'.format(ds_name, frac_err, err_type, error_seed)
    # df_raw = pd.read_csv(raw_fpath)
    # df_delta = pd.read_csv(delta_fpath)
    # df_mask = pd.read_csv(mask_fpath)
    #
    # attrs = ['pm2.5']
    # evaluate_anomaly(df_raw, df_delta, df_mask, attrs=attrs)


    # for normal dataset
    # ds_name = 'air_all'
    # epochs = 1
    # raw_fpath = '../raw/{}.csv'.format(ds_name)
    # delta_fpath = 'result/{}_delta_e{}.csv'.format(ds_name, epochs)
    #
    # df_raw = pd.read_csv(raw_fpath)
    # df_delta = pd.read_csv(delta_fpath)
    # attrs = ['pm2.5']
    # evaluate_delta(df_delta, attrs=attrs)

    # for dirty dataset
    ds_name = 'air_all'
    frac_err = 0.1
    err_type = 'symmetry'
    error_seed = 1
    epochs = 10

    raw_fpath = '../raw/{}_{}{}{}.csv'.format(ds_name, frac_err, err_type, error_seed)
    delta_fpath = 'result/{}_{}{}{}_delta_e{}.csv'.format(ds_name, frac_err, err_type, error_seed, epochs)
    mask_fpath = '../raw/{}_{}{}{}_mask.csv'.format(ds_name, frac_err, err_type, error_seed)
    df_raw = pd.read_csv(raw_fpath)
    df_delta = pd.read_csv(delta_fpath)
    df_mask = pd.read_csv(mask_fpath)

    attrs = ['pm2.5']
    evaluate_delta(df_delta, attrs=attrs)
    attr_mean_dict = evaluate_dirty_delta(df_delta, df_mask, attrs=attrs)
    for attr in attrs:
        print(attr)
        print(attr_mean_dict)
