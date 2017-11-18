# -*- coding: utf-8 -*-
"""
Utility functions

"""
import numpy as np
import matplotlib.pyplot as plt


def create_feature_tensor(segments, labels):
    """Create tensor X with the features for the experiments

    Parameters
    ----------
    segments: list with n_jackets elements. Each element is a list with
        the n_segments objects of type Segment which defines a jacket contour.

    labels: the labels for each segment in segments, as a ndarray of int with
        shape (n_jackets, n_segments)

    Returns
    -------
    The X tensor of features with shape (n_jackets, n_segments, n_features),
    where the features axis is composed by 7 features in this order:
    x0norm, y0norm, x1norm, y1norm, xMidNorm, yMidNorm, angle

    """
    n_features = 7
    n_jackets, n_segments = labels.shape
    X = np.zeros([n_jackets, n_segments, n_features])

    for jacket_segments, i in zip(segments, range(n_jackets)):
        for s, j in zip(jacket_segments, range(n_segments)):
            X[i, j] = (s.x0norm, s.y0norm, s.x1norm, s.y1norm,
                       (s.x0norm + s.x1norm) / 2.,
                       (s.y0norm + s.y1norm) / 2.,
                       s.angle / (2*np.pi))

    return X


def add_gaussian_noise(X, mu=0, sigma=0.1):
    """Add gaussian noise to each element in X"""
    noise = np.random.normal(0.0, sigma, size=X.size)
    return X + noise.reshape(X.shape)


def plot_segments(segments, caption='', labels_segments=None):
    """Plot a figure with the segments of a jacket

    Parameters
    ----------
    segments (list): list of Segment objects
    caption (string): the title of the figure
    label_segments: sequence-like object with the label given for each segment

    Returns
    -------
    The figure to plot.

    """
    colors = 'rgbcmyk'*2
    fig = plt.figure()
    num_segments = len(segments)
    for s, n in zip(segments, range(num_segments)):
        if labels_segments is None:
            color = 'blue'
        else:
            color = colors[labels_segments[n]]

        plt.plot([s.y0, s.y1], [s.x0, s.x1], 'o-', color=color, linewidth=2)
        plt.text((s.y0+s.y1)/2, (s.x0+s.x1)/2, str(n), color='k')

    plt.axis('equal')
    plt.gca().invert_yaxis()
    plt.title(caption)
    plt.show(block=False)

    return fig


def save_segments(dst_path, segments, caption='', labels_segments=None):
    """Save a figure of jacket segments in disk

    Parameters
    ----------
    dst_path (string): path to the new image
    segments (list): list of Segment objects
    caption (string): the title of the figure
    label_segments: sequence-like object with the label given for each segment

    Returns
    -------
    The figure saved to disk.

    """
    colors = 'rgbcmyk'*2
    fig = plt.figure()
    num_segments = len(segments)
    for s, n in zip(segments, range(num_segments)):
        if labels_segments is None:
            color = 'blue'
        else:
            color = colors[labels_segments[n]]

        plt.plot([s.y0, s.y1], [s.x0, s.x1], 'o-', color=color, linewidth=2)
        plt.text((s.y0+s.y1)/2, (s.x0+s.x1)/2, str(n), color='k')

    plt.axis('equal')
    plt.gca().invert_yaxis()
    plt.title(caption)
    plt.savefig(dst_path)

    return fig
