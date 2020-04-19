"""Build a language detector model

The goal of this exercise is to train a linear classifier on text features
that represent sequences of up to 3 consecutive characters so as to be
recognize natural languages by using the frequencies of short character
sequences as 'fingerprints'.

"""
# Author: Olivier Grisel <olivier.grisel@ensta.org>
# License: Simplified BSD

import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_mat(y_test, y_pred, ind_plot, target_names, f):
    n_classes = len(target_names)
    conf_mat = metrics.confusion_matrix(y_test, y_pred, range(n_classes))
    ax = f.add_subplot(1, 3, ind_plot+1)
    sns.heatmap(conf_mat, annot=True, fmt="d", ax=ax, cmap="YlGn")
    ax.set(ylim=[n_classes, 0],
           ylabel="True label",
           xlabel="Predicted label")
    ax.set_xticklabels(target_names)
    ax.set_yticklabels(target_names)
    ax.set_title('Confusion Matrix')


# The training data folder must be passed as first argument
languages_data_folder = r'C:\Users\yyahe\Google Drive\ML training\ML part\subject10-SKLEARN\data\short_paragraphs'
dataset = load_files(languages_data_folder)
classes_num = len(dataset.target_names)

# Split the dataset in training and test set:
docs_train, docs_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, test_size=0.5)

# TASK: Build a an analyzer that splits strings into sequence of 1 to 3 characters instead of word tokens
analyzer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))

# TASK: Build a vectorizer / classifier pipeline using the previous analyzer
# the pipeline instance should stored in a variable named clf
classifiers = [('perceptron', Perceptron()), ('svc', SVC()), ('linear_svc', SVC(kernel='linear'))]
fig = plt.figure(figsize=(15, 5))

for ind_classifier in range(len(classifiers)):
    clf = Pipeline([('analyze', analyzer),  classifiers[ind_classifier]])
    clf.fit(docs_train, y_train)
    y_predicted = clf.predict(docs_test)
    print(metrics.classification_report(y_test, y_predicted,
                                    target_names=dataset.target_names))
    cm = metrics.confusion_matrix(y_test, y_predicted)
    plot_confusion_mat(y_test, y_predicted, ind_classifier, dataset.target_names, fig)
    plt.title(f'{classifiers[ind_classifier][0]}, score: {clf.score(docs_test, y_test):.2f}')

# Predict the result on some short new sentences:
    sentences = [
        u'This is a language detection test.',
        u'Ceci est un test de d\xe9tection de la langue.',
        u'Dies ist ein Test, um die Sprache zu erkennen.',
    ]
    predicted = clf.predict(sentences)

    for s, p in zip(sentences, predicted):
        print(u'The language of "%s" is "%s"' % (s, dataset.target_names[p]))
