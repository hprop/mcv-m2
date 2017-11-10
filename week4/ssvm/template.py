# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 20:15:45 2015

@author: joans

= Modified by Team 5 =

"""
import os
import matplotlib.pyplot as plt
import numpy as np

from pandas import ExcelFile

from pystruct.models import ChainCRF
from pystruct.learners import FrankWolfeSSVM  # OneSlackSSVM, NSlackSSVM
from sklearn.cross_validation import KFold
from sklearn.svm import LinearSVC

from plot_segments import plot_segments


num_segments_per_jacket = 40
add_gaussian_noise_to_features = False
sigma_noise = 0.1
plot_labeling = False
plot_coefficients = False


# Load the segments and the groundtruth for all jackets
path_measures = 'man_jacket_hand_measures.xls'
xl = ExcelFile(path_measures)
sheet = xl.parse(xl.sheet_names[0])
# Be careful, parse() just reads literals, does not execute formulas
xl.close()

it = sheet.iterrows()
labels_segments = []
segments = []
for row in it:
    ide = row[1]['ide']
    segments.append(np.load(os.path.join('segments', ide + '_front.npy')))
    labels_segments.append(list(row[1].values[-num_segments_per_jacket:]))

labels_segments = np.array(labels_segments).astype(int)


# Show groundtruth for 3rd jacket
n = 2
plot_segments(segments[n], sheet.ide[n], labels_segments[n])


# Make matrices X of shape (number of jackets, number of features)
# and Y of shape (number of jackets, number of segments) where,
# for all jackets,
#     X = select the features for each segment
#     Y = the grountruth label for each segment
Y = labels_segments
num_jackets = labels_segments.shape[0]
num_labels = np.unique(np.ravel(labels_segments)).size

# CHANGE THIS IF YOU CHANGE NUMBER OF FEATURES
num_features = 7
X = np.zeros((num_jackets, num_segments_per_jacket, num_features))

for jacket_segments, i in zip(segments, range(num_jackets)):
    for s, j in zip(jacket_segments, range(num_segments_per_jacket)):
        """ set the features """
        X[i, j, 0:num_features] = \
            s.x0norm, s.y0norm, s.x1norm, s.y1norm, \
            (s.x0norm + s.x1norm) / 2., (s.y0norm + s.y1norm) / 2., \
            s.angle / (2*np.pi)

print('X, Y done')

# You can add some noise to the features
if add_gaussian_noise_to_features:
    print('Noise sigma {}'.format(sigma_noise))
    X = X + np.random.normal(0.0, sigma_noise, size=X.size).reshape(X.shape)

# Define the graphical model
model = ChainCRF()
crf = FrankWolfeSSVM(model=model, C=2)  # , max_iter=11)


# Compare SVM with S-SVM doing k-fold cross validation, see scikit-learn.org.
# With k=5, in each fold we have 4 jackets for testing, 19 for training,
# with k=23 we have leave one out: 22 for training, 1 for testing.
n_folds = 5
scores_crf = np.zeros(n_folds)
scores_svm = np.zeros(n_folds)
wrong_segments_crf = []
wrong_segments_svm = []

kf = KFold(num_jackets, n_folds=n_folds)
fold = 0
for train_index, test_index in kf:
    print(' ')
    print('train index {}'.format(train_index))
    print('test index {}'.format(test_index))
    print('{} jackets for training, {} for testing'
          .format(len(train_index), len(test_index)))

    X_train = X[train_index]
    Y_train = Y[train_index]
    X_test = X[test_index]
    Y_test = Y[test_index]

    # Train the S-SVM
    crf.fit(X_train, Y_train)

    # Label the testing set and print results
    Y_pred = crf.predict(X_test)
    Y_pred = np.vstack(Y_pred)

    fails = np.sum(Y_pred != Y_test)
    score = 1 - (float(fails) / (Y_test.shape[0] * Y_test.shape[1]))
    wrong_segments_crf.append(fails)
    scores_crf[fold] = score

    print('CRF - total wrong segments: {}, score: {}'.format(fails, score))

    # Figure showing the result of classification of segments for each jacket
    # in the testing part of present fold
    if plot_labeling:
        for ti, pred in zip(test_index, Y_pred):
            print(ti)
            print(pred)
            s = segments[ti]
            plot_segments(s,
                          caption='SSVM predictions for jacket ' + str(ti+1),
                          labels_segments=pred)

    # LINEAR SVM TRAINING AND TESTING

    # Prepare data for svm (expected matrix with row-vector features)
    X_train = np.vstack(X_train)
    X_test = np.vstack(X_test)
    Y_train = Y_train.flatten()
    Y_test = Y_test.flatten()

    # Create and train
    svm = LinearSVC(dual=False, C=.1)
    svm.fit(X_train, Y_train)

    # Use test dataset to meassure the svm's accuracy
    Y_pred = svm.predict(X_test)

    fails = np.sum(Y_pred != Y_test)
    score = 1 - float(fails) / Y_test.size
    wrong_segments_svm.append(fails)
    scores_svm[fold] = score

    print('SVM - total wrong segments: {}, score: {}'.format(fails, score))

    fold += 1


"""
Global results
"""
total_segments = num_jackets*num_segments_per_jacket
wrong_segments_crf = np.array(wrong_segments_crf)
wrong_segments_svm = np.array(wrong_segments_svm)
print('Results per fold ')
print('Scores CRF : {}'.format(scores_crf))
print('Scores SVM : {}'.format(scores_svm))
print('Wrongs CRF : {}'.format(wrong_segments_crf))
print('Wrongs SVM : {}'.format(wrong_segments_svm))
print(' ')
print('Final score CRF: {}, {} wrong labels in total out of {}'
      .format(1.0 - wrong_segments_crf.sum() / float(total_segments),
              wrong_segments_crf.sum(),
              total_segments))
print('Final score SVM: {}, {} wrong labels in total out of {}'
      .format(1.0 - wrong_segments_svm.sum() / float(total_segments),
              wrong_segments_svm.sum(),
              total_segments))


if plot_coefficients:
    name_of_labels = [
        'neck',
        'left shoulder',
        'outer left sleeve',
        'left wrist',
        'inner left sleeve',
        'left chest',
        'waist',
        'right chest',
        'inner right sleeve',
        'right wrist',
        'outer right sleeve',
        'right shoulder',
    ]

    # Show image of the learned unary coefficients, size (num_labels,
    # num_features). Use matshow() and colorbar()
    unary_coef = crf.w[:num_labels * num_features].reshape(num_labels,
                                                           num_features)
    plt.matshow(unary_coef)
    plt.colorbar()
    plt.xticks(range(num_features),
               ['x0', 'y0', 'x1', 'y1', 'xC', 'yC', 'angle'])
    plt.yticks(range(len(name_of_labels)), name_of_labels)
    plt.title("Unary coefficients")
    plt.show()

    # Show image of pairwise coefficients size (num_labels, num_labels)"""
    pairwise_coef = crf.w[num_labels * num_features:].reshape(num_labels,
                                                              num_labels)
    plt.matshow(pairwise_coef)
    plt.colorbar()
    plt.tick_params(labeltop=False, labelbottom=True)
    plt.xticks(range(len(name_of_labels)), name_of_labels, rotation='vertical')
    plt.yticks(range(len(name_of_labels)), name_of_labels)
    plt.title("Pairwise coefficients")
    plt.show()
