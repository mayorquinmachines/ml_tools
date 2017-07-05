""" Classes to preprocess data for mercedes kaggle contest """
#!/usr/bin/env
import pandas as pd
import tensorflow as tf
import numpy as np


def load_data(path, drop_cols=None):
    """ Helper function to load in data from csv"""
    if drop_cols:
        dataframe = pd.read_csv(path, header=0)
        dataframe = dataframe.drop(drop_cols, axis=1)
        return dataframe
    else:
        return pd.read_csv(path, header=0)

def get_cols(dfm):
    """ Helper function to get keep uniform columns across all sets """
    dataframe = pd.get_dummies(dfm)
    return list(dataframe.columns)

def group_list(l, group_size=100):
    """ Generator to chunk data into batches """
    for i in range(0, len(l), group_size):
        yield l[i:i+group_size]

def reset_graph(seed=42):
    """ Reset tensorflow graph """
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def apply_pipe(train,val,test, pipe):
    """ Apply a pipe method from the Pipe() class"""
    train = pipe.fit_transform(train)
    val = pipe.transform(val)
    test = pipe.transform(test)
    return train,val,test

def cat_cols(train, test):
    """ Get common columns when there are one-hot encodings"""
    train_cols = get_cols(train)
    test_cols = get_cols(test)
    columns = [x for x in train_cols if x in test_cols]
    return columns

def common_cols(train, test):
    """ Get common columns for test and train columns"""
    train_cols = train.columns.tolist()
    test_cols = test.columns.tolist()
    columns = [x for x in train_cols if x in test_cols]
    return columns

