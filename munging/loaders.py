__author__ = 'mateuszopala'
import pandas as pd
import os
import numpy as np
import itertools
import json
import theano
from sklearn.preprocessing import StandardScaler
import pickle


class UnsupervisedLoader(object):
    def __init__(self, data_path, dump_parameters=False):
        train = pd.read_csv(os.path.join(data_path, 'train.csv'))
        test = pd.read_csv(os.path.join(data_path, 'test.csv'))
        columns = list(test.columns)
        self.data = train.append(test)
        self.data = self.data[columns].values
        # self.convert_string_columns_to_binary_membership()
        for i in xrange(self.data.shape[0]):
            for j in xrange(self.data.shape[1]):
                if type(self.data[i, j]) == str:
                    self.data[i, j] = ord(self.data[i, j]) - ord('A')
        self.data = self.data.astype(theano.config.floatX)
        np.random.shuffle(self.data)
        min_val = self.data.min(axis=0)
        max_val = self.data.max(axis=0)
        self.data = (self.data - min_val) / (max_val - min_val)
        if dump_parameters:
            with open(os.path.join(data_path, 'parameters.pkl'), 'wb') as f:
                pickle.dump((max_val, min_val), f)

    def update_data(self, preprocessor):
        self.data = preprocessor.tf_hidden(self.data)

    def generate(self, mini_batch_size=128, preprocessor=None):
        self.data = preprocessor.tf_hidden(self.data) if preprocessor else self.data
        for i in itertools.cycle(xrange(0, self.data.shape[0], mini_batch_size)):
            yield self.data[i: i + mini_batch_size, :], self.data[i: i + mini_batch_size, :]


class SupervisedLoader(object):
    @staticmethod
    def load(data_path):
        train = pd.read_csv(os.path.join(data_path, 'train.csv'))
        test = pd.read_csv(os.path.join(data_path, 'test.csv'))
        columns = list(test.columns)
        np.random.shuffle(train.values)
        x = train[columns].values
        y = train['Hazard'].values.astype(theano.config.floatX)
        for i in xrange(x.shape[0]):
            for j in xrange(x.shape[1]):
                if type(x[i, j]) == str:
                    x[i, j] = ord(x[i, j]) - ord('A')
        x = x.astype(theano.config.floatX)
        x /= x.max(axis=0)
        return x, y


class TestSetLoader(object):
    @staticmethod
    def load(data_path):
        test = pd.read_csv(os.path.join(data_path, 'test.csv'))
        ids = test['Id'].values
        x = test.values
        for i in xrange(x.shape[0]):
            for j in xrange(x.shape[1]):
                if type(x[i, j]) == str:
                    x[i, j] = ord(x[i, j]) - ord('A')
        x = x.astype(theano.config.floatX)
        x /= x.max(axis=0)
        return x, ids


class ConfigHolder(dict):
    def __init__(self, init, name=None):
        name = name or self.__class__.__name__.lower()
        dict.__init__(self, init)
        dict.__setattr__(self, "_name", name)

    def __getstate__(self):
        return self.__dict__.items()

    def __setstate__(self, items):
        for key, val in items:
            self.__dict__[key] = val

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, dict.__repr__(self))

    def __str__(self):
        n = self._name
        s = ["{}(name={!r}):".format(self.__class__.__name__, n)]
        s = s + ["  {}.{} = {!r}".format(n, it[0], it[1]) for it in self.items()]
        s.append("\n")
        return "\n".join(s)

    def __setitem__(self, key, value):
        return super(ConfigHolder, self).__setitem__(key, value)

    def __getitem__(self, name):
        return super(ConfigHolder, self).__getitem__(name)

    def __delitem__(self, name):
        return super(ConfigHolder, self).__delitem__(name)


def load_configuration(path):
    with open(path, 'r') as f:
        json_conf = json.load(f)
    return ConfigHolder(json_conf, os.path.basename(path))
