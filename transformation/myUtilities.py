# Import libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import re
from sklearn.preprocessing import LabelBinarizer

TITANIC_PATH = "source"

# This function loads the train and test dataset from file.
def load_titanic_datasets(titanic_path=TITANIC_PATH):
    train_csv_path = os.path.join(titanic_path, "train.csv")
    test_csv_path = os.path.join(titanic_path, "test.csv")
    return pd.read_csv(train_csv_path), pd.read_csv(test_csv_path)

# This function plots a correlation matrix.
def plot_corr_matrix(corr_matrix, fig_size=(10, 5), color_map='Greys'):
    fig, ax = plt.subplots(figsize=fig_size)
    cax = ax.matshow(corr_matrix, cmap=color_map)
    fig.colorbar(cax)
    plt.title("Correlation Matrix")

    labels = list(corr_matrix)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for (i, j), z in np.ndenumerate(corr_matrix):
        ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

    plt.show()

# This class is a custom transformer which adds a Title attribute to the
name_idx = 3
class TitleAttrAdder(BaseEstimator, TransformerMixin):
    def __init__(self, miss_idx=None, mrs_idx=None):
        self.miss_idx = miss_idx
        self.mrs_idx = mrs_idx
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        # Initialise the Title array to 'Mr'
        title = np.full(len(X), 'Mr', dtype='U6')

        # Set the Title attribute with Mrs, Miss or Master
        l = len(X.shape)
        i = 0

        if (len(X.shape) > 1):
            for row in X:
                m = re.search('Mrs|Miss|Master', row[0])
                if m:
                    title[i] = m.group(0)
                i+=1
        else:
            for row in X:
                m = re.search('Mrs|Miss|Master', row)
                if m:
                    title[i] = m.group(0)
                i += 1

        # Set the following designated rows to Title = Miss or Mrs according to the data exploration.
        title[self.miss_idx] = 'Miss'
        title[self.mrs_idx] = 'Mrs'

        return title.reshape((len(title),1))

# This class selects the desired attributes and drops the rest, and converts the DataFrame to a Numpy array.
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
        return None
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

class LabelBinarizerForPipeline(LabelBinarizer):
    def fit(self, X, y=None):
        return super(LabelBinarizerForPipeline, self).fit(X)
    def transform(self, X):
        return super(LabelBinarizerForPipeline, self).transform(X)
    def fit_transform(self, X, Y=None):
        result = super(LabelBinarizerForPipeline, self).fit(X).transform(X)
        return result