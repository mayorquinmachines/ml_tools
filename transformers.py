""" Classes to preprocess data for mercedes kaggle contest """
#!/usr/bin/env

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameImputer(BaseEstimator, TransformerMixin):
    def __init__(self, median=False):
        """ 
        columns with dtype == object are inputed with most frequent value
        columns with dtype == 'other' can be inputed with mean or median if flag is true.
        """
        self.median = median

    def fit(self, X, y=None):
        if self.median:
            self.fill = pd.Series([X[c].value_counts().index[0] if X[c].dtype == 'object' \
                        else X[c].median() for c in X], index = X.columns)
        else:
            self.fill = pd.Series([X[c].value_counts().index[0] if X[c].dtype == 'object' \
                        else X[c].mean() for c in X], index = X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

class FactorFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, feature_pairs):
    """ Multiplies features specified as a list of tuples """
        self.feature_pairs = feature_pairs

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        new_cols = {}
        for i,pair in enumerate(self.feature_pairs):
            new_feat = X.pair[0] * X.pair[1]
            add_col = {'Xpair'+str(i): new_feat}
            new_cols.update(add_col)
        new_cols_df = pd.Dataframe(new_cols)
        final_df = pd.concat([X, new_cols_df], axis=1)
        return final_df
        

class DataFrameSelector(BaseEstimator, TransformerMixin):
    """ Transforms a pandas dataframe to a numpy array 
    to feed to other transformations. """
    def __init__(self, attribute_names=None):
        """ Initializing with column names of interest """
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        """ Takes a pandas dataframe, checks if columns have
        been specified and transform to numpy array """
        if self.attribute_names:
            return X[self.attribute_names].values
        else:
            return X.values

class Dummifier(BaseEstimator, TransformerMixin):
    """ One hot encoding for categorical columns """
    def __init__(self, cat=True):
        """ Initializing with a flag for 
        categorical columns """
        self.cat = cat
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        """ Returns dataframe with dummy columns"""
        dataframe = pd.get_dummies(X)
        return dataframe

