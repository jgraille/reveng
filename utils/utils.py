from sklearn.preprocessing import LabelEncoder
import json
import os


def init_params():
    try:
        with open(os.getcwd() + '/config/config.json') as f:
            config = json.loads(f.read())
            f.close()
        config_db = config.get('database', {})
        config_automl = config.get('automl_param', {})
        config_file1 = config.get('file1', {})
        config_file2 = config.get('file2', {})
        config_file3 = config.get('file3', {})
        return config_db, config_automl, config_file1, config_file2, config_file3
    except Exception as e:
        print('\nFailed to load parameters...', e.__repr__())


def categorical_var(numpydf, categorical_features):
    """
    from https://marcotcr.github.io/lime/tutorials/Tutorial_H2O_continuous_and_cat.html
    """
    categorical_names = {}
    for i in categorical_features:
        le = LabelEncoder()
        le.fit(numpydf[:, i])
        numpydf[:, i] = le.transform(numpydf[:, i])
        categorical_names[i] = le.classes_
    return categorical_names, numpydf


def get_position(header, target):
    i = 0
    while header[i] != target:
        i = i + 1
    return i


def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


def update_indexes(original_list, excludes, header):
    res = original_list
    for value in excludes:
        index_value = get_position(header, value)
        if index_value not in res:
            upper_values = [i for i in res if i > index_value]
            under_values = [i for i in res if i < index_value]
            upper_values = [(lambda x: x - 1)(x) for x in upper_values]
            res = under_values + upper_values
            header = [s for s in header if s != value]
        else:
            raise Exception('\nupdate_indexes(), the target index can not be in the selected values list.')
    return res
