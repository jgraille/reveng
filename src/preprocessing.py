import numpy
import pandas
import os
from utils.utils import categorical_var, get_position
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from utils.utils import update_indexes
import itertools


class PreProcessing:

    def __init__(self, config_file):
        self.initial_categorical_features = config_file['categorical_features']
        self._excludes = config_file['exclude']
        self._header = numpy.genfromtxt(fname=os.getcwd() + config_file['path'], delimiter=',', dtype='str', max_rows=1)
        print('\nLoading data...', config_file['path'])
        self.data = numpy.genfromtxt(fname=os.getcwd() + config_file['path'], delimiter=',', dtype='str', skip_header=1)
        self.index_target = get_position(self._header, config_file['target_output'])
        self._categorical_features = update_indexes(self.initial_categorical_features, self._excludes, self._header)
        self._class_names = None
        self._categorical_names = None
        print('\nInitialising parameters done...')

    @property
    def categorical_features(self):
        return self._categorical_features

    @categorical_features.setter
    def categorical_features(self, value):
        if isinstance(value, list):
            self._categorical_features = value
        else:
            raise TypeError("Value is not a list")

    @property
    def excludes(self):
        return self._excludes

    @excludes.setter
    def excludes(self, value):
        if isinstance(value, list):
            self._excludes = value
        else:
            raise TypeError("Value is not a list")

    @property
    def header(self):
        return self._header

    @header.setter
    def header(self, value):
        if isinstance(value, list):
            self._header = value
        else:
            raise TypeError("Value is not a list")

    @property
    def class_names(self):
        return self._class_names

    @class_names.setter
    def class_names(self, value):
        self._class_names = value

    @property
    def categorical_names(self):
        return self._categorical_names

    @categorical_names.setter
    def categorical_names(self, value):
        self._categorical_names = value

    def encoding(self):
        try:
            y = self.data[:, self.index_target]
            """
            le = LabelEncoder()
            le.fit(y)
            y = le.transform(y)
            class_names = le.classes_
            """
            self._class_names = 'temp'
            x = self.data
            for i in self._excludes:
                x = numpy.delete(x, get_position(self._header, i), axis=1)
            self._categorical_names, data = categorical_var(x, self._categorical_features)
            x = x.astype(float)
            return x, y
        except Exception as e:
            print('\nFailed to apply PreProcessing.encoding method: ', e.__repr__())

    def encoding_and_split(self):
        try:
            x, y = self.encoding()
            feature_names = [x for x in list(self._header) if x not in self._excludes]
            numpy.random.seed(1)
            train, test, labels_train, labels_test = train_test_split(x, y, train_size=0.75)
            return (train, test, labels_train, labels_test), feature_names
        except Exception as e:
            print('\nFailed to apply PreProcessing.encoding_and_split method: ', e.__repr__())

    def frequency(self):
        try:
            data = pandas.DataFrame(self.data)
            for i in range(0, len(self._header)):
                if i in self.initial_categorical_features:
                    data[i] = data[i].map(data[i].value_counts())
            return data.to_numpy()
        except Exception as e:
            print('\nFailed to apply PreProcessing.frequency method: ', e.__repr__())

    def frequency_and_scale(self):
        try:
            data = self.frequency()
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
            return data
        except Exception as e:
            print('\nFailed to apply PreProcessing.frequency_and_scale method: ', e.__repr__())

    def outliers(self):
        try:
            print('\nCalculating outliers')
            data = pandas.DataFrame(data=self.frequency(), columns=self._header)
            features = self._header.tolist()
            features = [a for a in features if a not in self._excludes]
            data = data[features].apply(pandas.to_numeric, errors='coerce')
            n = int(0.70*len(features))

            Q1 = data.quantile(q=0.25, axis=0)
            Q3 = data.quantile(q=0.75, axis=0)

            IQR = Q3 - Q1
            outlier_step = 1.5*IQR
            indexes = []

            for i in features:
                print('Processing column name: ', i)
                if (data[i] < Q1[i] - outlier_step[i]).any() or (data[i] > Q3[i] + outlier_step[i]).any():
                    res = data[i].loc[(data[i] < Q1[i] - outlier_step[i]) | (data[i] > Q3[i] + outlier_step[i])].index.values
                    indexes.append(res.tolist())
            indexes = list(itertools.chain(*indexes))
            result = pandas.DataFrame(data=indexes, columns=['indexes'])
            result = result['indexes'].value_counts()
            result = result[result > n].index.values
            return result, pandas.DataFrame(data=self.data, columns=self._header)
        except Exception as e:
            print('\nFailed to apply PreProcessing.outliers method: ', e.__repr__())







































