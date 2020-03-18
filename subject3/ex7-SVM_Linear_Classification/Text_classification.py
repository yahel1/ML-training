from sklearn.datasets import load_svmlight_file
from sklearn import svm

if __name__ == '__main__':
    test_features, test_labels = load_svmlight_file('email_test.txt')
    for set_size in ['50', '100', '400', 'all']:
        file_name = 'email_train-'+set_size+'.txt'
        train_features, train_labels = load_svmlight_file(file_name, n_features=test_features.shape[1])
        clf = svm.SVC(kernel='linear', C=1)
        clf.fit(train_features, train_labels)
        accuracy = 100*clf.score(test_features, test_labels)
        print(set_size + f' documents: Accuracy = {accuracy:.4}%')
