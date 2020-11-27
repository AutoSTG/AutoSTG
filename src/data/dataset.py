import logging
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler

from setting import data_path
from utils.helper import Scaler


class TrafficDataset:
    def __init__(self, path, train_prop, valid_prop,
                 num_sensors, in_length, out_length,
                 batch_size_per_gpu, num_gpus):
        logging.info('initialize %s DataWrapper', path)
        self._path = path
        self._train_prop = train_prop
        self._valid_prop = valid_prop
        self._num_sensors = num_sensors
        self._in_length = in_length
        self._out_length = out_length
        self._batch_size_per_gpu = batch_size_per_gpu
        self._num_gpus = num_gpus

        self.build_graph()
        self.build_data_loader()

    def build_graph(self):
        logging.info('initialize graph')

        # normalize adj_mat
        self.adj_mats = self.read_adj_mat()
        for dim in range(self.adj_mats.shape[-1]):
            values = self.adj_mats[:, :, dim][self.adj_mats[:, :, dim] != np.inf].flatten()
            self.adj_mats[:, :, dim] = np.exp(-np.square(self.adj_mats[:, :, dim] / (values.std() + 1e-8)))

        # normalize node_ft
        self.node_fts = self.read_loc()
        self.node_fts = (self.node_fts - self.node_fts.mean(axis=0)) / (self.node_fts.std(axis=0) + 1e-8)

    def build_data_loader(self):
        logging.info('initialize data loader')
        train, valid, test = self.read_traffic()
        self.scaler = Scaler(train.values, missing_value=0)
        # data for search
        self.search_train = self.get_data_loader(train, shuffle=True, tag='search train',
                                                 num_gpus=self._num_gpus)  # for weight update
        self.search_valid = self.get_data_loader(valid, shuffle=True, tag='search valid',
                                                 num_gpus=self._num_gpus)  # for arch update
        # data for training & evaluation
        self.train = self.get_data_loader(train, shuffle=True, tag='train', num_gpus=1)
        self.valid = self.get_data_loader(valid, shuffle=False, tag='valid', num_gpus=1)
        self.test = self.get_data_loader(test, shuffle=False, tag='test', num_gpus=1)

    def get_data_loader(self, data, shuffle, tag, num_gpus):
        logging.info('load %s inputs & labels', tag)

        num_timestamps = data.shape[0]

        # fill missing value
        data_f = self.fill_traffic(data)

        # transform data distribution
        data_f = np.expand_dims(self.scaler.transform(data_f.values), axis=-1)  # [T, N, 1]

        # time in day
        time_ft = (data.index.values - data.index.values.astype('datetime64[D]')) / np.timedelta64(1, 'D')
        time_ft = np.tile(time_ft, [1, self._num_sensors, 1]).transpose((2, 1, 0))  # [T, N, 1]

        # day in week
        # day_ft = np.zeros(shape=(num_timestamps, self._num_sensors, 7)) # [T, N, 7]
        # day_ft[np.arange(num_timestamps), :, data.index.dayofweek] = 1

        # put all input features together
        in_data = np.concatenate([data_f, time_ft, ], axis=-1)  # [T, N, D]
        out_data = np.expand_dims(data.values, axis=-1)  # [T, N, 1]

        # create inputs & labels
        inputs, labels = [], []
        for i in range(self._in_length):
            inputs += [in_data[i: num_timestamps + 1 - self._in_length - self._out_length + i]]
        for i in range(self._out_length):
            labels += [out_data[self._in_length + i: num_timestamps + 1 - self._out_length + i]]
        inputs = np.stack(inputs).transpose((1, 3, 2, 0))
        labels = np.stack(labels).transpose((1, 3, 2, 0))

        # logging info of inputs & labels
        logging.info('load %s inputs & labels [ok]', tag)
        logging.info('input shape: %s', inputs.shape)  # [num_timestamps, c, n, input_len]
        logging.info('label shape: %s', labels.shape)  # [num_timestamps, c, n, output_len]

        # create dataset
        dataset = TensorDataset(
            torch.from_numpy(inputs).to(dtype=torch.float),
            torch.from_numpy(labels).to(dtype=torch.float)
        )

        # create sampler
        sampler = RandomSampler(dataset, replacement=True,
                                num_samples=self._batch_size_per_gpu * num_gpus) if shuffle else SequentialSampler(
            dataset)

        # create dataloader
        data_loader = DataLoader(dataset=dataset, batch_size=self._batch_size_per_gpu * num_gpus, sampler=sampler,
                                 num_workers=4, drop_last=False)
        return data_loader

    def read_idx(self):
        with open(os.path.join(data_path, self._path, 'sensor_graph/graph_sensor_ids.txt')) as f:
            ids = f.read().strip().split(',')
        idx = {}
        for i, id in enumerate(ids):
            idx[id] = i
        return idx

    def read_loc(self):
        idx = self.read_idx()
        loc = np.loadtxt(os.path.join(data_path, self._path, 'sensor_graph/graph_sensor_locations.csv'), delimiter=',',
                         skiprows=1)
        loc_ft = np.zeros((self._num_sensors, 2))
        for i in range(loc.shape[0]):
            id = str(int(loc[i, 1]))
            loc_ft[idx[id], :] = loc[i, 2:4]
        return loc_ft

    def read_adj_mat(self):
        cache_file = os.path.join(data_path, self._path, 'sensor_graph/adjacent_matrix_cached.npz')
        try:
            arrays = np.load(cache_file)
            g = arrays['g']
            logging.info('load adj_mat from the cached file [ok]')
        except:
            logging.info('load adj_mat from the cached file [fail]')
            logging.info('load adj_mat from scratch')
            idx = self.read_idx()
            graph_csv = pd.read_csv(os.path.join(data_path, self._path, 'sensor_graph/distances.csv'),
                                    dtype={'from': 'str', 'to': 'str'})

            g = np.zeros((self._num_sensors, self._num_sensors, 2))
            g[:] = np.inf

            for k in range(self._num_sensors): g[k, k] = 0
            for row in graph_csv.values:
                if row[0] in idx and row[1] in idx:
                    g[idx[row[0]], idx[row[1]], 0] = row[2]  # distance
                    g[idx[row[0]], idx[row[1]], 1] = 1  # hop

            g = np.concatenate([g, np.transpose(g, (1, 0, 2))], axis=-1)
            np.savez_compressed(cache_file, g=g)
            logging.info('save graph to the cached file [ok]')
        return g

    def read_traffic(self):
        data = pd.read_hdf(os.path.join(data_path, self._path, 'traffic.h5'))
        num_train = int(data.shape[0] * self._train_prop)
        num_eval = int(data.shape[0] * self._valid_prop)
        num_test = data.shape[0] - num_train - num_eval
        return data[:num_train].copy(), data[num_train: num_train + num_eval].copy(), data[-num_test:].copy()

    def fill_traffic(self, data):
        data = data.copy()
        data[data < 1e-5] = float('nan')
        data = data.fillna(method='pad')
        data = data.fillna(method='bfill')
        return data

    @property
    def batch_size_per_gpu(self):
        return self._batch_size_per_gpu
