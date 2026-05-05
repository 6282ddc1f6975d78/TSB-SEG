# Taken from: https://github.com/softlab-unimore/time2feat

from collections import defaultdict

import itertools
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import scipy.spatial.distance as dist
import time
import warnings
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.metrics import euclidean_distances
from sklearn.metrics.cluster import adjusted_mutual_info_score, adjusted_rand_score, homogeneity_score, \
    normalized_mutual_info_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.preprocessing import StandardScaler
from tsfresh import extract_features
from tsfresh.feature_selection import relevance


class PFA(object):
    def __init__(self, q=None):
        self.q = q

    def fit(self, X, expl_var_value: float):
        if not self.q:
            self.q = X.shape[1]

        sc = StandardScaler()
        X_trans = sc.fit_transform(X)
        # Choice of the Explained Variance
        pca = PCA(expl_var_value)
        pca.fit(X_trans)
        princComp = len(pca.explained_variance_ratio_)
        A_q = pca.components_.T

        kmeans = KMeans(n_clusters=princComp, n_init=10)
        kmeans.fit(A_q)
        clusters = kmeans.predict(A_q)
        cluster_centers = kmeans.cluster_centers_

        dists = defaultdict(list)
        for i, c in enumerate(clusters):
            dist = euclidean_distances([A_q[i, :]], [cluster_centers[c, :]])[0][0]
            dists[c].append((i, dist))

        self.indices_ = [sorted(f, key=lambda x: x[1])[0][0] for f in dists.values()]
        self.features_ = X_trans[:, self.indices_]
        list_feat = []
        for x in self.indices_:
            list_feat.append(X.columns[x])

        # print("Features Selected: " + str(list_feat))
        return list_feat, pca.explained_variance_ratio_


def _define_model(model_type: str, num_cluster: int):
    """ Define the clustering model """
    if model_type == 'Hierarchical':
        model = AgglomerativeClustering(n_clusters=num_cluster)
    elif model_type == 'KMeans':
        model = KMeans(n_clusters=num_cluster, n_init=10)
    elif model_type == 'Spectral':
        model = SpectralClustering(n_clusters=num_cluster)
    else:
        raise ValueError('{} is not supported'.format(model_type))
    return model


def cluster_metrics(y_true: np.array, y_pred: np.array):
    """ Compute main clustering metrics """
    return {
        'ami': adjusted_mutual_info_score(y_true, y_pred),
        'nmi': normalized_mutual_info_score(y_true, y_pred),
        'rand': adjusted_rand_score(y_true, y_pred),
        'homogeneity': homogeneity_score(y_true, y_pred),
    }


class ClusterWrapper(object):
    """ Wrapper for several clustering algorithms """

    def __init__(self, n_clusters: int, model_type: str, transform_type: str = None, normalize: bool = False):
        self.num_cluster = n_clusters
        self.model_type = model_type
        self.normalize = normalize

        self.model = _define_model(model_type, n_clusters)
        self.transform_type = transform_type

    def _normalize(self, x: np.array):
        x_mean = np.mean(x, axis=1, keepdims=True)
        x_std = np.std(x, axis=1, keepdims=True)

        x = (x - x_mean) / x_std
        return x

    def remove_nan(self, x: np.array):
        if len(x.shape) == 2:
            count = np.isnan(x).any(axis=0).sum()
            if count > 0:
                print('Remove {} nan columns for clustering step'.format(count))
                cond = np.logical_not(np.isnan(x).any(axis=0))
                x = x[:, cond]

            cond = ((x == float('inf')) | (x == float('-inf'))).any(axis=0)

        return x

    def fit_predict(self, x: np.array):
        x = self.remove_nan(x)
        if self.normalize:
            x = self._normalize(x)
        elif self.transform_type:
            transformer = get_transformer(self.transform_type)
            x = transformer.fit_transform(x)
        x = x.reshape((len(x), -1))
        return self.model.fit_predict(x)


def get_transformer(transform_type: str):
    if transform_type == 'std':
        transformer = StandardScaler()
    elif transform_type == 'minmax':
        transformer = MinMaxScaler()
    elif transform_type == 'robust':
        transformer = RobustScaler()
    else:
        raise ValueError('Select the wrong transformer: ', transform_type)

    return transformer


def apply_transformation(x_train, x_test, transform_type: str):
    np.seterr(divide='ignore', invalid='ignore')
    transformer = get_transformer(transform_type)
    x_train = transformer.fit_transform(x_train)
    x_test = transformer.transform(x_test)

    return x_train, x_test


def create_fc_parameter(features: list):
    single_features = [x.split('single__', 1)[1] for x in features if x.startswith('single__')]
    pair_features = [x for x in features if x.startswith('pair__')]

    single_features = feature_extraction.settings.from_columns(single_features)
    features_dict = {'single': single_features, 'pair': pair_features}
    return features_dict


def feature_selection(df_feats: pd.DataFrame, labels: dict = None, context: dict = None):
    if labels:
        train_idx = list(labels.keys())
        test_idx = [i for i in range(len(df_feats)) if i not in train_idx]
        y_train = list(labels.values())

        df_train_features = df_feats.iloc[train_idx, :].reset_index(drop=True)
        df_test_features = df_feats.iloc[test_idx, :]

        df_test_features = pd.concat([df_train_features, df_test_features], axis=0, ignore_index=True)
        params = {
            'transform_type': context['transform_type'],
            'model_type': context['model_type'],
            'score_mode': 'simple',
            'strategy': 'sk_base',
        }
        new_params = simple_grid_search(df_train_features, y_train, df_test_features, params)
        top_k = new_params['top_k']
        score_mode = new_params['score_mode']

        top_features = features_scoring_selection(df_train_features, y_train, mode=score_mode, top_k=top_k,
                                                  strategy='sk_base')
    else:
        top_features = features_scoring_selection(df_feats, [], mode='simple', top_k=1,
                                                  strategy='none')
    return top_features


def features_simple_selection(df: pd.DataFrame, labels: list, top_k: int = 20):
    df = df.dropna(axis='columns')
    sk = SelectKBest(k=min(len(df.columns), top_k))
    sk.fit(df, labels)
    return list(sk.get_feature_names_out())


def features_scoring_selection(df: pd.DataFrame, labels: list, mode: str = 'simple', top_k: int = 20,
                               strategy: str = 'multi'):
    """ Scoring the importance for each feature for the given labels """
    df = df.dropna(axis='columns')
    labels = pd.Series((i for i in labels))
    top_k = min(len(df.columns), top_k)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if strategy == 'tsfresh':
        relevance_features = relevance.calculate_relevance_table(df, labels, ml_task='classification', multiclass=False)
        relevance_features = relevance_features.sort_values(by='p_value')
        relevance_features = relevance_features.dropna(subset=['p_value'])
        ex = pd.Series(relevance_features['p_value'], index=relevance_features['feature'])

    elif strategy == 'multi':
        relevance_features = relevance.calculate_relevance_table(df, labels, ml_task='classification', multiclass=True)
        p_columns = [col for col in relevance_features.columns if col.startswith('p_value')]
        relevance_features = relevance_features.dropna(subset=p_columns)
        relevance_features['p_value'] = relevance_features[p_columns].mean(axis=1)
        relevance_features = relevance_features.sort_values(by='p_value')
        ex = pd.Series(relevance_features['p_value'], index=relevance_features['feature'])

    elif strategy == 'sk_base':
        # ToDo: manage all null case
        cond = ((df == float('inf')) | (df == float('-inf'))).any(axis=0)
        df = df.drop(df.columns[cond], axis=1)

        sk = SelectKBest(k='all')
        sk.fit(df, labels)
        df_sk = pd.DataFrame({'score': sk.scores_, 'feature': df.columns})
        df_sk.dropna(inplace=True)
        df_sk.sort_values('score', ascending=False, inplace=True)
        ex = pd.Series(df_sk['score'].values[:top_k], index=df_sk['feature'].values[:top_k])

    elif strategy == 'sk_pvalue':
        cond = ((df == float('inf')) | (df == float('-inf'))).any(axis=0)
        df = df.drop(df.columns[cond], axis=1)

        sk = SelectKBest(k='all')
        sk.fit(df, labels)
        df_sk = pd.DataFrame({'pvalues': sk.pvalues_, 'feature': df.columns})
        df_sk.dropna(inplace=True)
        df_sk.sort_values('pvalues', ascending=True, inplace=True)
        ex = pd.Series(df_sk['pvalues'].values[:top_k], index=df_sk['feature'].values[:top_k])

    elif strategy == 'none':
        cond = ((df == float('inf')) | (df == float('-inf'))).any(axis=0)
        df = df.drop(df.columns[cond], axis=1)
        selector = VarianceThreshold()
        selector.fit(df)
        features = selector.get_feature_names_out()
        top_k = len(features)
        ex = pd.Series([1] * top_k, index=features)

    else:
        raise ValueError('Strategy {} is not supported'.format(strategy))

    top_k_feat = {}
    feats_value = {}
    if mode == 'simple':
        for x in ex[:top_k].index:
            top_k_feat[x] = list(df[x])
        featOrd = pd.DataFrame(top_k_feat)
        # print(featOrd)
        feats_choosed, weight_feats = pfa_scoring(featOrd, 0.9)
        for x in feats_choosed:
            feats_value[x] = list(df[x])
    elif mode == 'domain':
        feat_domain = {}
        feats_choosed = {}
        for x in ex.index:
            domain = x.split('__')[0]
            if not domain in feat_domain.keys():
                feat_domain[domain] = {}
            if len(feat_domain[domain]) < top_k:
                feat_domain[domain][x] = list(df[x])
        # print(feat_domain.keys())
        for key in feat_domain.keys():
            featOrd = pd.DataFrame(feat_domain[key])
            feats_choosed[key], weight_feats = pfa_scoring(featOrd, 0.9)
        for type_feat in feats_choosed.keys():
            for key in feats_choosed[type_feat]:
                feats_value[key] = list(df[key])
    # print(pfa_scoring(featOrd, 0.9))
    # return pd.DataFrame(feats_value)
    return list(feats_value.keys())


def pfa_scoring(df: pd.DataFrame, expl_var_selection: float):
    pfa = PFA()
    feat_PFA, expl_variance_ration = pfa.fit(df, expl_var_selection)
    x = pfa.features_
    column_indices = pfa.indices_
    return feat_PFA, expl_variance_ration


def simple_grid_search(df_base: pd.DataFrame, y_base: list, df_complete: pd.DataFrame, params: dict):
    cond = 'model_type' in params and 'transform_type' in params and 'score_mode' in params
    assert cond, 'Please provide all mandatory params (model, transform and score mode)'

    cond = (pd.isnull(df_base)) & (pd.isnull(df_complete.iloc[:len(df_base)]))
    mask = (df_base == df_complete.iloc[:len(df_base)])
    mask = np.where(cond, True, mask)
    assert np.all(mask), 'df_base must be in the first record of df_complete'

    # Optional params
    k_best = params.get('k_best', False)
    strategy = params.get('strategy', 'sk_base')
    pre_transform = params.get('pre_transform', False)

    grid_params = {
        'top_k': [1],
        'score_mode': [params['score_mode']],
        'transform_type': [params['transform_type']],
    }

    if strategy.startswith('sk'):
        grid_params['top_k'] = [10, 25, 50, 100, 200, 300] * 2
        # grid_params['transform_type'] = ['minmax', 'std']
        grid_params['score_mode'] = ['simple', 'domain']
    elif strategy.startswith('none'):
        grid_params['transform_type'] = ['minmax', 'std', None]
        grid_params['score_mode'] = ['simple']

    else:
        grid_params['top_k'] = [10, 25, 50, 100, 200, 300] * 2

    results = []
    time.sleep(0.1)
    for new_params in ParameterGrid(grid_params):
        if not k_best:
            max_rows = len(df_base)  # 800
            top_features = features_scoring_selection(df_base.iloc[:max_rows, :], y_base[:max_rows],
                                                      mode=new_params['score_mode'], top_k=new_params['top_k'],
                                                      strategy=strategy)
        else:
            top_features = features_simple_selection(df_base, y_base, top_k=new_params['top_k'])

        # Extract train and test
        x_train = df_base[top_features].values
        x_test = df_complete[top_features].values

        # Transformation step
        if new_params['transform_type'] and not pre_transform:
            _, x_test = apply_transformation(x_train, x_test, new_params['transform_type'])

        # Clustering step
        num_labels = len(set(y_base))
        model = ClusterWrapper(model_type=params['model_type'], n_clusters=num_labels)
        y_pred = model.fit_predict(x_test)

        # Compute results
        res_metrics = cluster_metrics(y_base, y_pred[:len(y_base)])
        res_metrics.update(new_params)
        results.append(res_metrics)

    df_res = pd.DataFrame(results)
    df_res = df_res.fillna('None')
    df_res = df_res.groupby(['top_k', 'score_mode', 'transform_type'])['nmi'].mean().reset_index()
    df_res = df_res.replace(['None'], [None])
    df_res.sort_values('nmi', ascending=False, inplace=True)
    # print(df_res)
    new_params = {
        'top_k': df_res['top_k'].iloc[0],
        'score_mode': df_res['score_mode'].iloc[0],
        'transform_type': df_res['transform_type'].iloc[0],
    }
    return new_params


def distance(ts1: np.array, ts2: np.array, distance_feat: str = None):
    metrics = {
        'braycurtis': dist.braycurtis,
        'canberra': dist.canberra,
        'chebyshev': dist.chebyshev,
        'cityblock': dist.cityblock,
        'correlation': dist.correlation,
        'cosine': dist.cosine,
        'euclidean': dist.euclidean,
        'minkowski': dist.minkowski,
        # 'mahalanobis': dist.mahalanobis,
        # 'seuclidean': dist.seuclidean,
        # 'sqeuclidean': dist.sqeuclidean,
    }
    if distance_feat is None:
        distances = {k: f(ts1, ts2) for k, f in metrics.items()}
    else:
        distances = {distance_feat: metrics[distance_feat](ts1, ts2)}
    return distances


def extract_pair_features(ts1: np.array, ts2: np.array, distance_feat: str = None):
    assert len(ts1.shape) == 1, 'ts1 is not a univariate time series'
    assert len(ts2.shape) == 1, 'ts2 is not a univariate time series'

    features = {}
    distances = distance(ts1, ts2, distance_feat)
    features.update(distances)

    return features


# Adapt Multivariate time series for extracting features.
def adapt_time_series(ts, sensors_name):
    list_multi_ts = {
        k: ts[:, i].tolist()
        for i, k in enumerate(sensors_name)
    }

    list_id = [1 for _ in ts]
    list_time = [i for i in range(len(ts))]

    dict_df = {'id': list_id, 'time': list_time}
    for sensor in sensors_name:
        dict_df[sensor] = list_multi_ts[sensor]

    df_time_series = pd.DataFrame(dict_df)
    return df_time_series


def extract_univariate_features(ts: np.array, sensors_name: list, feats_select: dict = None):
    dict_ts = adapt_time_series(ts, sensors_name)
    features_extracted = extract_features(dict_ts, column_id='id', column_sort='time', n_jobs=0,
                                          kind_to_fc_parameters=feats_select, disable_progressbar=True)
    # print('End feature extraction')
    # ToDo: Optimize following cycle
    features = {}
    for feat in features_extracted.columns:
        assert len(features_extracted[feat]) == 1
        features[feat] = float(features_extracted[feat].iloc[0])
    return features


def padding_series(ts_list: list):
    def padding(ts: np.array, new_length: int):
        ts_padded = np.empty((new_length, ts.shape[1]))
        ts_padded[:] = None
        ts_padded[:len(ts), :] = ts
        return ts_padded

    max_length = np.max([len(ts) for ts in ts_list])
    ts_list = [padding(ts, max_length) for ts in ts_list]
    return ts_list


def extract_single_series_features(record: np.array, sensors_name: list):
    # Extract intra-signal features
    # Extract univariate features for each signal in the multivariate time series
    features_extracted = extract_univariate_features(record, sensors_name)
    # Rename key by inserting the predefined name
    features_single = {'single__{}'.format(k): val for k, val in features_extracted.items()}

    return features_single


def extract_single_series_features_batch(ts_list: np.array, batch_size: int = -1):
    """ Extract features for each signal in each time series in batch mode """
    pid = os.getpid()
    sensors_name = [str(i) for i in range(ts_list.shape[2])]
    ts_list = [arr for arr in ts_list]

    if batch_size == -1:
        batch_size = len(ts_list)

    num_batch = int(np.ceil(len(ts_list) / batch_size))

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    curr_dir = os.path.join(curr_dir, '../../tmp/')

    # print('Start univariate feature extraction in batch : pid {}'.format(pid))
    for i in range(num_batch):
        a = i * batch_size
        b = (i + 1) * batch_size
        ts_batch = ts_list[a:b]

        # Concatenate record and rename sensor names
        record = np.hstack(ts_batch)
        tmp_sensors_name = ["{}aaa{}".format(i, name) for i in range(len(ts_batch)) for name in sensors_name]

        # Extract univariate features for each signal
        # print('Start univariate feature extraction: pid {} batch {}'.format(pid, i))
        features_extracted = extract_univariate_features(record, tmp_sensors_name)
        # print('End univariate feature extraction: pid {} batch {}'.format(pid, i))

        # Reshape extracted features in each signals and format them correctly for each multivariate time series
        df_features = [{} for _ in range(len(ts_batch))]
        for k, val in features_extracted.items():
            pos, name = k.split('aaa', 1)
            pos = int(pos)
            name = 'single__{}'.format(name)
            df_features[pos][name] = val

        filename = os.path.join(curr_dir, 'fe_{}_{}.csv'.format(pid, i))
        pd.DataFrame(df_features).to_csv(filename, index=False)

    # print('End univariate feature extraction in batch : pid {}'.format(pid))

    # Construct the complete feature extracted dataset
    # print('Start construction univariate dataset : pid {}'.format(pid))

    df_features = []
    for i in range(num_batch):
        filename = os.path.join(curr_dir, 'fe_{}_{}.csv'.format(pid, i))
        df = pd.read_csv(filename)
        df_features.append(df)
        os.remove(filename)

    df_features = pd.concat(df_features, axis=0, ignore_index=True)

    # print('End construction univariate dataset : pid {}'.format(pid))

    return df_features


def extract_pair_series_features(mts: np.array):
    """ Extract features for each time series pair """
    features_pair = {}  # Initialize a empty dictionary to save extracted features

    # Extract each possible combination pair
    indexes = np.arange(mts.shape[1])
    combs = list(itertools.combinations(indexes, r=2))
    for i, j in combs:
        # Extract pair features for each pair in the multivariate time series
        feature = extract_pair_features(mts[:, i], mts[:, j])

        # Rename key by inserting the feature name
        feature = {'pair__{}__{}__{}'.format(k, i, j): val for k, val in feature.items()}

        features_pair.update(feature)
    return features_pair


def feature_extraction_simple(ts_list: (list, np.array), batch_size: int = -1):
    """ Extract intra and inter-signals features for each multivariate time series """
    ts_features_list = []

    # Extract features based on pair feature functions
    for ts_record in ts_list:
        features_pair = extract_pair_series_features(ts_record)
        ts_features_list.append(features_pair)

    # Create dataframe for pair features
    df_pair_features = pd.DataFrame(ts_features_list)

    # Extract features based on functions for univariate signals in batch mode
    df_single_features = extract_single_series_features_batch(ts_list, batch_size=batch_size)

    # Create time series feature dataFrame and return
    df_features = pd.concat([df_single_features, df_pair_features], axis=1)

    return df_features


def get_balanced_job(number_pool, number_job):
    """ Define number of records to assign to each processor """
    list_num_job = []
    if number_job <= number_pool:
        for i in range(number_job):
            list_num_job.append(1)
    else:
        for i in range(number_pool):
            list_num_job.append(int(number_job / number_pool))
        for i in range(number_job % number_pool):
            list_num_job[i] = list_num_job[i] + 1

    return list_num_job


def feature_extraction(ts_list: (list, np.array), batch_size: int = -1, p: int = 1):
    """ Multiprocessing implementation of the feature extraction step """
    # Define the number of processors to use
    max_pool = mp.cpu_count() if p == -1 else p
    num_batch = (len(ts_list) // batch_size) + 1
    max_pool = num_batch if num_batch < max_pool else max_pool

    # Balance records between jobs
    balance_job = get_balanced_job(number_pool=max_pool, number_job=len(ts_list))
    # print('Feature extraction with {} processor and {} batch size'.format(max_pool, batch_size))

    index = 0
    list_arguments = []
    for i, size in enumerate(balance_job):
        list_arguments.append((ts_list[index:index + size], batch_size))
        index += size

    # Multi processing script execution
    pool = mp.Pool(max_pool)
    extraction_feats = pool.starmap(feature_extraction_simple, list_arguments)
    pool.close()
    pool.join()

    # Concatenate all results
    df_features = pd.concat(extraction_feats, axis=0, ignore_index=True)
    return df_features
