'''
Miscellaneous utilities for facial emotion recognition
'''

# Necessary imports
import os, sys
import shutil
import pandas as pd
import numpy as np
from skimage.io import imread, imshow, imsave
from skimage.transform import resize
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import ipywidgets as widgets
from pathlib import Path
from itertools import product, combinations


def train_test_split(data, labels=None, ratio=0.8, shuffle=False):
    '''
    Split data into train and test sets.
    
    Parameters
    ----------
    data : array or ndarray
    labels : array or ndarray(optional)
        Array with corresponding to data labels.
    ratio : float 
        Denote fraction of the train set.
    
    Returns
    -------
    train_x, [train_y], test_x, [train_y] : tuple of arrays
        Tuple of train/test data or (data, labels) pairs.
    '''
    
    size = len(data)
    train_size = round(size * ratio)
    indicies = np.arange(size)
    if shuffle:
        np.random.shuffle(indicies)
    
    # Get train and test indicies
    train_indicies = indicies[:train_size]
    test_indicies = indicies[train_size: ]
    
    train_x = data[train_indicies]
    test_x = data[test_indicies]
    if labels is not None:
        train_y = labels[train_indicies]
        test_y = labels[test_indicies]
        return train_x, train_y, test_x, test_y
    else:
        return train_x, test_x


def array_to_str(points, sep=', '):
    '''
    Convert np.array into flat string with values, delimited with sep.
    
    Parameters
    ----------
    points : ndarray
        Array with ndim > 1.

    Returns
    s : str
        String with flat values of the input array.
    -------
    '''
    
    return sep.join(points.ravel().astype(str))


def str_to_array(s, sep=', ', n_cols=2, dtype=np.int):
    '''
    Convert string with delimeter sep into numpy array of shape (.., n_cols)
    
    Parameters
    ----------
    s : str
        Input string with values, delimited with sep.
    sep: str
        Delimiter in the input string.
    n_cols : int
        Number of columns in output array.
    dtype : type
        Values type of the output array.
    
    Returns
    -------
    arr : ndarray
        Numpy array of shape (-1, n_cols)
    '''
    
    return np.array(s.split(sep), dtype=dtype).reshape(-1, n_cols)


def labels_to_categorical(labels):
    '''
    Convert binary independent labels in two arrays into categorical labela:
        (0, 0) -> 0
        (0, 1) -> 1
        (1, 0) -> 2
        (1, 1) -> 3
    
    Parameters
    ----------
    labels : DataFrame with two label columns.
    
    Returns
    -------
    categorcial : Series
    '''
    
    mapping = list(product((0, 1), (0, 1)))
    categorical = labels.apply(lambda x: mapping.index(tuple(x)), axis=1)
    return categorical


def points_to_distances(points, normalize=True):
    '''
    Calculate elementwise euclidian distance between facial landmarks coordinates.
    
    Parameters
    ----------
    points : ndarray
        Numpy array of landmarks coordinates of shape (?, 2)
    normalize : bool
        Denotes output normalization.
    
    Returns
    -------
    distances : ndarray
    '''
    # Get all pairs of points
    pairs = combinations(points, 2)
    # Calculate distances
    distances = np.array([np.linalg.norm(x1-x2) for x1, x2 in pairs])
    if normalize:
        # Normalize
        distances = (distances - distances.mean()) / distances.std()
    return distances


def show_landmarks(image, landmarks=None, labels=None):
    '''
    Show image with landmarks.
    
    Parameters
    ----------
    image : ndarray or PIL image
    landmarks : ndarray
        Array with coordinates of facial landmarks.
    labels : str (optional)
        Optional string for the title of the plot.
    '''
    
    plt.imshow(image)
    if landmarks is not None:
        plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='lime')
    if labels:
        plt.title('Smile: %d, open mouth: %d' % labels)
    plt.axis('off')
    plt.show()


def plot_history(history, figsize=(16, 6)):
    '''
    Plot keras model train history.
    
    Parameters
    ----------
    histoty : dict
        History, returned by Keras model.fit() method.
    figsize : tuple
        Optional plots figure size.
    '''
    
    # Get data from history object
    loss = history.history['loss']
    acc = history.history['acc']
    if 'val_loss' in history.history.keys():
        plot_validation = True
        val_loss = history.history['val_loss']
        val_acc = history.history['val_acc']
    else:
        plot_validation = False
    epoch = history.epoch
    
    # Create figure for the plots
    f = plt.figure(figsize=figsize)
    
    # Plot losses
    ax1 = f.add_subplot(121)
    ax1.plot(epoch, loss, label='Train loss')
    if plot_validation:
        ax1.plot(epoch, val_loss, label='Validation loss')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.legend()
    ax1.set_title('Losses')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Plot accuracy
    ax2 = f.add_subplot(122)
    ax2.plot(epoch, acc, label='Train accuracy')
    if plot_validation:
        ax2.plot(epoch, val_acc, label='Validation accuracy')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('accuracy')
    ax2.legend()
    ax2.set_title('Accuracy')
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.show()

    
def load_annotations(path_to_csv, filename_index=False):
    '''
    Read annotations from csv file into DataFrame().
    
    Parameters
    ----------
    path_to_csv : str
        Path to csv file with annotaions.
    filename_index : bool
        If True, read filename column as index for the DataFrame().
        Otherwise index would be np.arange(..)
    
    Returns
    -------
    df : DataFrame()
    '''
    
    # Types of the columns
    dtype = {
        'smile': np.int, 
        'mouth_open': np.int, 
        'labeled': np.bool, 
        'points': np.str
    }
    
    # Parameters for pd.read_csv() method.
    params = {
        'header': 0
    }
    
    if filename_index:
        dtype['filename'] = np.str
        params['index_col'] = 0
    
    params['dtype'] : dtype
    df = pd.read_csv(path_to_csv, **params)
    return df
    
def main():
    pass
    

if __name__  == '__main__':
    main()
    
