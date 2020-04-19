"""Build a sentiment analysis / polarity model

Sentiment analysis can be casted as a binary text classification problem,
that is fitting a linear classifier on features extracted from the text
of the user messages so as to guess weather the opinion of the author is
positive or negative.

In this examples we will use a movie review dataset.

"""
# Author: Olivier Grisel <olivier.grisel@ensta.org>
# License: Simplified BSD

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_mat(y_test, y_pred, target_names):
    n_classes = len(target_names)
    conf_mat = metrics.confusion_matrix(y_test, y_pred, range(n_classes))
    ax = plt.subplot()
    sns.heatmap(conf_mat, annot=True, fmt="d", ax=ax, cmap="YlGn")
    ax.set(ylim=[n_classes, 0],
           ylabel="True label",
           xlabel="Predicted label")
    ax.set_xticklabels(target_names)
    ax.set_yticklabels(target_names)
    ax.set_title('Confusion Matrix')


if __name__ == "__main__":
    # NOTE: we put the following in a 'if __name__ == "__main__"' protected
    # block to be able to use a multi-core grid search that also works under
    # Windows, see: http://docs.python.org/library/multiprocessing.html#windows
    # The multiprocessing module is used as the backend of joblib.Parallel
    # that is used when n_jobs != 1 in GridSearchCV

    # the training data folder must be passed as first argument
    movie_reviews_data_folder = r'C:\Users\yyahe\Google Drive\ML training\ML part\subject10-SKLEARN\data\txt_sentoken'
    dataset = load_files(movie_reviews_data_folder, shuffle=False)
    print("n_samples: %d" % len(dataset.data))

    # split the dataset in training and test set:
    docs_train, docs_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.25, random_state=None)

    # TASK: Build a vectorizer / classifier pipeline that filters out tokens that are too rare or too frequent
    pipeline = Pipeline([('analyze', TfidfVectorizer(min_df=0.01, max_df=0.9)), ('LinearSVC', LinearSVC())])

    # TASK: Build a grid search to find out whether unigrams or bigrams are more useful.
    # Fit the pipeline on the training set using grid search for the parameters
    parameters = {'analyze__ngram_range': ((1, 1), (2, 2))}  # unigrams or bigrams
    grid_search = GridSearchCV(pipeline, parameters)
    grid_search.fit(docs_train, y_train)
    best_parameters = grid_search.best_estimator_.get_params()

    # TASK: print the cross-validated scores for the each parameters set explored by the grid search
    mean_cv_scores = grid_search.cv_results_['mean_test_score']
    std_cv_scores = grid_search.cv_results_['std_test_score']
    param_names = ['unigrams', 'bigrams']
    for ind in range(len(mean_cv_scores)):
        print(f'score: mean-{mean_cv_scores[ind]:.2f}, std-{std_cv_scores[ind]:.2f} for {param_names[ind]}')

    # TASK: Predict the outcome on the testing set and store it in a variable named y_predicted
    clf = grid_search.best_estimator_.fit(docs_train, y_train)
    y_predicted = clf.predict(docs_test)

    # Print the classification report
    print(metrics.classification_report(y_test, y_predicted, target_names=dataset.target_names))

    # Print and plot the confusion matrix
    plot_confusion_mat(y_test, y_predicted, dataset.target_names)
