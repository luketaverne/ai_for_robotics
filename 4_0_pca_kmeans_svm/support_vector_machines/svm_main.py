#!/usr/bin/env python2
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn import preprocessing

def visualize_2D(features, labels, clf_a, clf_b, clf_c):
    h = 0.02
    x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
    y_min, y_max = features[:, 1].min() - 1, features[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    for i, clf in enumerate((clf_a, clf_b, clf_c)):
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        if clf.probability == False:
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
            Z = Z[:,1].reshape(xx.shape)
        plt.subplot(2, 2, i + 1)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        plt.scatter(features[:, 0], features[:, 1], c=labels, cmap=plt.cm.coolwarm)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xlabel('principal component 1')
        plt.ylabel('principal component 2')
        plt.title('SVMs')
    plt.show()

# Load features and labels from file.
# WARNING: These are different from the results of the previous exercises! Use the provided file!
features = np.genfromtxt(open("features_svm.txt"))
n_samples, n_features = np.shape(features)
labels = np.genfromtxt(open("labels_svm.txt"))
features_evaluation = np.genfromtxt(open("features_evaluation.txt"))
visualization = True

# Normalize your input feature data (the evaluation data is already normalized).
features = preprocessing.scale(features)

# TODO: Train 3 different classifiers as specified in the exercise sheet (exchange the value for None).
classifier_linear = svm.SVC(kernel='linear', C=2e10).fit(features, labels)
classifier_g_perfect = svm.SVC(kernel='rbf', C=2e10).fit(features, labels)
classifier_g_slack = svm.SVC(kernel='rbf', C=1.0).fit(features, labels)

# Compute the scores for each one
linear_score = classifier_linear.score(features, labels)
g_perfect_score = classifier_g_perfect.score(features, labels)
g_slack_score = classifier_g_slack.score(features, labels)

#Print scores
print('Linear score: ', linear_score)
print('g_perfect_score: ', g_perfect_score)
print('g_slack_score: ', g_slack_score)

# TODO (optional): Train 3 different classifiers only on first 2 dimensions for visualization (exchange the value for None).
if visualization:
    classifier_linear_viz = svm.SVC(kernel='linear', C=2e10).fit(features[:,0:2], labels)
    classifier_g_perfect_viz = svm.SVC(kernel='rbf', C=2e10).fit(features[:,0:2], labels)
    classifier_g_slack_viz = svm.SVC(kernel='rbf', C=1.0).fit(features[:,0:2], labels)
    visualize_2D(features, labels, classifier_linear_viz, classifier_g_perfect_viz, classifier_g_slack_viz)

# TODO: classify evaluation data and store classifications to file (exchange the value for None).
Y_linear = classifier_linear.predict(features_evaluation)
Y_g_perfect = classifier_g_perfect.predict(features_evaluation)
Y_g_slack = classifier_g_slack.predict(features_evaluation)

# Save probability results to file.
# Y_linear, Y_g_perfect and Y_g_slack are of the form N x 1 dimensions,
#  with number of features N.
np.savetxt('results_svm_Y_linear.txt', Y_linear)
np.savetxt('results_svm_Y_g_perfect.txt', Y_g_perfect)
np.savetxt('results_svm_Y_g_slack.txt', Y_g_slack)

# TODO: Train the same 3 classifiers as specified in the exercise sheet with additional probability estimates (exchange the value for None).
classifier_linear = svm.SVC(kernel='linear', C=2e10, probability=True).fit(features, labels)
classifier_g_perfect = svm.SVC(kernel='rbf', C=2e10, probability=True).fit(features, labels)
classifier_g_slack = svm.SVC(kernel='rbf', C=1.0, probability=True).fit(features, labels)

# TODO (optional): Train 3 different classifiers with probability estimates only on first 2 dimensions for visualization (exchange the value for None).
if visualization:
    classifier_linear_viz = svm.SVC(kernel='linear', C=2e10, probability=True).fit(features[:,0:2], labels)
    classifier_g_perfect_viz = svm.SVC(kernel='rbf', C=2e10, probability=True).fit(features[:,0:2], labels)
    classifier_g_slack_viz = svm.SVC(kernel='rbf', C=1.0, probability=True).fit(features[:,0:2], labels)
    visualize_2D(features, labels, classifier_linear_viz, classifier_g_perfect_viz, classifier_g_slack_viz)

# TODO: classify newly loaded features and store classification probabilities to file (exchange the value for None).
P_linear = classifier_linear.predict_proba(features_evaluation)
P_g_perfect = classifier_g_perfect.predict_proba(features_evaluation)
P_g_slack = classifier_g_slack.predict_proba(features_evaluation)

# Save probability results to file.
# P_linear, P_g_perfect and P_g_slack are of the form N x 2 dimensions,
#  with number of features N and classification probabilities for the two classes.
np.savetxt('results_svm_P_linear.txt', P_linear)
np.savetxt('results_svm_P_g_perfect.txt', P_g_perfect)
np.savetxt('results_svm_P_g_slack.txt', P_g_slack)
