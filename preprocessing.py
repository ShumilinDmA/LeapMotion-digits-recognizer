import numpy as np


class StandardScaler:
    """
    Class perform standartization of the data.
    Methods:
        self.fit - fit scaler to get means and std
        self.transform - transform given data by using means and std from fit methods
        self.fit_transform - perform sequence of methods : fit and transform
    """
    def __init__(self):
        self.mean = None
        self.std = None
        self.is_fitted = False

    def fit(self, data):
        """
        Calculate mean and std
        :param data: Dataset to transform
        :return:
        """
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)
        self.is_fitted = True

    def transform(self, data):
        """
        Perfom standartization by using means and std
        :param data: Dataset to transform
        :return:
        """
        assert self.is_fitted, "Scaler don't fit yet"
        return (data - self.mean) / (self.std + 1e-8)

    def fit_transform(self, data):
        """
        Sequence of methods to apply
        :param data: Dataset to transform
        :return:
        """
        self.fit(data)
        return self.transform(data)


def train_test_split(data, labels, shuffle=True, test_proportion=0.2):
    """
    Split given dataset and labels with given proportions
    :param data: Datset to split
    :param labels: Labels to split
    :param shuffle: Shuffle data and labels
    :param test_proportion: ratio test size to all data
    :return:
    """
    n_sample = len(data)
    n_test = int(n_sample*test_proportion)
    if shuffle:
        index_list = set(range(0, n_sample))
        index_test = set(np.random.choice(n_sample, n_test, replace=False))
        index_train = index_list ^ index_test
        return data[list(index_train)], data[list(index_test)], labels[list(index_train)], labels[list(index_test)]
    else:
        return data[:-n_test], data[-n_test:], labels[:-n_test], labels[-n_test:]


def k_fold(X, y, folds=5):
    """
    Function is generator to get chunks of data
    :param X: Dataset
    :param y: Labels
    :param folds: Number of folds
    :return: Chunks of data in generator
    """
    n_sample = len(X)
    n_fold = int(n_sample/folds)
    index_list = np.random.permutation(n_sample)
    for fold in range(folds):
        val_index = index_list[fold*n_fold:(fold+1)*n_fold]
        train_index = list(set(index_list) ^ set(val_index))
        yield X[train_index], X[val_index], y[train_index], y[val_index]


def one_hot_encoding(labels):
    """
    Function perform one-hot encoding of given categorical data
    :param labels: Categorical data
    :return: One-hot encoding matrix
    """
    marks = np.unique(labels)
    ohe = np.zeros((len(labels), len(marks)))
    for indx, mark in enumerate(marks):
        ohe[labels == mark, indx] = 1
    return ohe
