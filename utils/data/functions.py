import numpy as np
import pandas as pd
import torch

def load_features(feat_path, dtype=np.float32):
    feat_df = pd.read_csv(feat_path, header=None)
    feat = np.array(feat_df, dtype=dtype)
    return feat

def load_adjacency_matrix(adj_path, dtype=np.float32):
    adj_df = pd.read_csv(adj_path, header=None)
    adj = np.array(adj_df, dtype=dtype)
    return adj


def min_max_normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)


def min_max_denormalize(normalized_data, original_min, original_max):
    return normalized_data * (original_max - original_min) + original_min

def wmin_max_normalize(data):
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array")

    normalized_data = np.zeros_like(data, dtype=np.float32)

    for i in range(data.shape[1]):
        col_min = np.min(data[:, i])
        col_max = np.max(data[:, i])
        normalized_data[:, i] = (data[:, i] - col_min) / (col_max - col_min)

    return normalized_data
def combine_data(case, weather):
    # batch_size, num_node, feat_num
    result = None
    case_flatten = case.reshape(-1)
    weather_flatten = weather.reshape(-1)
    i = 0
    j = 0
    while i < case_flatten.shape[0] and j < weather_flatten.shape[0]:
        case_value = case_flatten[i]
        weather_values = weather_flatten[j:j + 3]
        if case_value.ndim == 0:
            case_value = case_value.reshape(1)
        if weather_values.ndim == 0:
            weather_values = weather_values.reshape(1)

        combined_values = np.concatenate((case_value, weather_values))

        if result is None:
            result = combined_values
        else:
            result = np.concatenate((result, combined_values))
        i += 1
        j += 3

    result = result.reshape(case.shape[0], case.shape[1], -1)

    return result

def spearman_correlation_matrix(data):
    sum = None
    for i in range(data.shape[1]):
        ranks = np.apply_along_axis(lambda x: np.argsort(np.argsort(x)), axis=0, arr=data[:, i, :])
        covariance_matrix = np.cov(ranks, rowvar=False)
        standard_deviations = np.sqrt(np.diag(covariance_matrix))
        correlation_matrix = covariance_matrix / (standard_deviations[:, None] * standard_deviations[None, :])
        sum = correlation_matrix if i == 0 else sum + correlation_matrix
    sum = sum / 47
    print('Spearman Correlation Matrix:\n', sum)

def cal_spearman(train_data, _weather, weekend):
    x = np.array(train_data)
    w = np.array(_weather)

    com = combine_data(x, w)
    week = np.array(weekend)
    week = week.reshape(week.shape[0], week.shape[1], 1)
    com = np.concatenate((com, week), axis=2)
    correlation_matrix = spearman_correlation_matrix(com)
    return com
weather = None
weather_train = None
weather_val = None
weather_test = None
def generate_dataset(
    data, seq_len, pre_len, time_len=None, split_ratio=0.7, normalize=True
):
    _weather = pd.read_csv(f"/home/yn/cx/wzk/yuce/data/weather.csv", header=None)
    weekend = pd.read_csv(f"/home/yn/cx/wzk/yuce/data/weekend.csv", header=None)

    _weather = np.array(_weather, dtype=np.float32)

    new_columns = np.zeros(235, dtype=bool)
    for i in range(0, 235, 5):
        new_columns[i:i + 5] = [True, True, False, True, False]
    _weather = _weather[:, new_columns]

    if time_len is None:
        time_len = data.shape[0]

    if normalize:
        data = min_max_normalize(data)
        weekend = weekend / 52
        _weather = min_max_normalize(_weather)

    train_size = int(time_len * split_ratio)
    val_size = int(time_len * 0.1)
    test_size = int(time_len - train_size - val_size)

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:time_len]

    train_X, train_Y, val_X, val_Y, test_X, test_Y = [], [], [], [], [], []
    _weather_train, _weather_val, _weather_test = [], [], []
    combine_train, combine_val, combine_test = [], [], []

    print("data: train process....")
    for i in range(0, len(train_data) - seq_len - pre_len):
        x = np.array(train_data[i: i + seq_len])
        w = np.array(_weather[i:i + seq_len])
        train_X.append(x)
        _weather_train.append(w)
        com = combine_data(x, w)
        week = np.array(weekend[i:i + seq_len])
        week = week.reshape(week.shape[0], week.shape[1], 1)
        com = np.concatenate((com, week), axis=2)
        combine_train.append(com)
        train_Y.append(np.array(train_data[i + seq_len: i + seq_len + pre_len]))

    print("data: val process....")
    val_offset = train_size
    for i in range(0, len(val_data) - seq_len - pre_len, pre_len):#
        x = np.array(val_data[i: i + seq_len])
        w = np.array(_weather[val_offset + i: val_offset + i + seq_len])
        val_X.append(x)
        _weather_val.append(w)
        com = combine_data(x, w)
        week = np.array(weekend[val_offset + i: val_offset + i + seq_len])
        week = week.reshape(week.shape[0], week.shape[1], 1)
        com = np.concatenate((com, week), axis=2)
        combine_val.append(com)
        val_Y.append(np.array(val_data[i + seq_len: i + seq_len + pre_len]))

    print("data: test process....")
    test_offset = train_size + val_size
    for i in range(0, len(test_data) - seq_len - pre_len, pre_len):
        x = np.array(test_data[i: i + seq_len])
        w = np.array(_weather[test_offset + i: test_offset + i + seq_len])
        test_X.append(x)
        _weather_test.append(w)
        com = combine_data(x, w)
        week = np.array(weekend[test_offset + i: test_offset + i + seq_len])
        week = week.reshape(week.shape[0], week.shape[1], 1)
        com = np.concatenate((com, week), axis=2)
        combine_test.append(com)
        test_Y.append(np.array(test_data[i + seq_len: i + seq_len + pre_len]))

    combine_train = np.array(combine_train)
    combine_val = np.array(combine_val)
    combine_test = np.array(combine_test)

    train_X = np.array(train_X).reshape(len(train_X), seq_len, 47, 1)
    train_Y = np.array(train_Y)
    val_X = np.array(val_X).reshape(len(val_X), seq_len, 47, 1)
    val_Y = np.array(val_Y)
    test_X = np.array(test_X).reshape(len(test_X), seq_len, 47, 1)
    test_Y = np.array(test_Y)

    _train_X = np.concatenate((train_X, combine_train), axis=3)
    _val_X = np.concatenate((val_X, combine_val), axis=3)
    _test_X = np.concatenate((test_X, combine_test), axis=3)

    return (_train_X, train_Y,
            _val_X, val_Y,
            _test_X, test_Y)


def generate_torch_datasets(
    data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
):
    train_X, train_Y, val_X,val_Y,test_X, test_Y = generate_dataset(
        data,
        seq_len,
        pre_len,
        time_len=time_len,
        split_ratio=split_ratio,
        normalize=normalize,
    )
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_X), torch.FloatTensor(train_Y)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(val_X), torch.FloatTensor(val_Y)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_X), torch.FloatTensor(test_Y)
    )
    return train_dataset, val_dataset,test_dataset











