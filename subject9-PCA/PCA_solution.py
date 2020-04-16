from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm
from sklearn.metrics import confusion_matrix
import seaborn as sns


def plot_one_image(indices, sub_plot_row, x, y, name):
    i = 1
    for ind in indices:
        ax = plt.subplot(2, len(indices), sub_plot_row * len(indices) + i)
        plt.imshow(np.reshape(x[:, ind], [8, 8]), cmap=plt.cm.Greys)
        ax.set_title(y[ind], fontsize=16)
        if i == 1:
            ax.set_ylabel(name, fontsize=16)
        i += 1


def plot_comparable(column_num, Xdata_and_name, digits_label):
    m = len(digits_label)
    indices = np.random.randint(0, m, column_num)
    for sub_plot_row, (x, name) in enumerate(Xdata_and_name):
        plot_one_image(indices, sub_plot_row, x, digits_label, name)


def plot_confusion_mat(y_test, y_pred):
    n_classes = 10
    conf_mat = confusion_matrix(y_test, y_pred, range(n_classes))
    ax = plt.subplot()
    sns.heatmap(conf_mat, annot=True, ax=ax, cmap="YlGn")
    ax.set(ylim=[n_classes, 0],
           ylabel="True label",
           xlabel="Predicted label")
    ax.set_title('Confusion Matrix')


if __name__ == '__main__':
    X, y = load_digits(return_X_y=True)
    m, n = np.shape(X)
    X_zero_mean = X.T - np.mean(X, axis=1)
    U, s, vh = np.linalg.svd(X_zero_mean)
    xrot = U.T @ X_zero_mean

    Sigma = 1/m * sum([np.expand_dims(x, axis=0)*np.expand_dims(x, axis=1) for x in xrot.T])
    w, U1 = np.linalg.eig(Sigma)
    plt.imshow(np.log(Sigma))
    plt.colorbar()

    # 2: Find k, the number of components to retain
    computed_var_percentage = np.cumsum(w)/np.sum(w)*100
    wanted_var_percentage = [50, 60, 70, 80, 90, 95, 99]
    ind = 0
    k = np.zeros(np.shape(wanted_var_percentage))
    for wanted_percentage in wanted_var_percentage:
        k[ind] = np.min([idx for idx, computed_percentage in enumerate(computed_var_percentage)
                         if computed_percentage > wanted_percentage])
        ind += 1
    plt.plot(k, wanted_var_percentage, 'o')
    plt.xlabel('k', fontsize=16)
    plt.ylabel('Variance percentage', fontsize=16)
    plt.grid()
    plt.plot(range(1, n+1), computed_var_percentage)

    # 3: Implement PCA with dimension reduction
    percentage = 90
    k = np.min([idx for idx, computed_percentage in enumerate(computed_var_percentage) if computed_percentage > percentage])
    U_k = U[:, :k]
    x_rot_k = U_k.T @ X_zero_mean
    X_Recovered = U_k @ x_rot_k

    example_num = 7
    Xdata_and_name = [(X_zero_mean, 'original'), (X_Recovered, f'PCA- {percentage}%')]
    plot_comparable(example_num, Xdata_and_name, y)

    # 4: Train a classification model
    train_accuracy = np.zeros(n)
    test_accuracy = np.zeros(n)
    for k in range(n):
        U_k = U[:, :k]
        X_Recovered = U_k @  U_k.T @ X_zero_mean
        # scores = cross_val_score(classifier, X_Recovered.T, y, cv=5)
        X_train, X_test, y_train, y_test = train_test_split(X_Recovered.T, y, test_size=0.2)
        classifier = svm.SVC(gamma=0.001)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        test_accuracy[k] = classifier.score(X_test, y_test)
        train_accuracy[k] = classifier.score(X_train, y_train)

    plt.plot(range(n), test_accuracy, 'o-')
    plt.xlabel('k', fontsize=16)
    plt.ylabel('Test accuracy', fontsize=16)
    best_k = np.argmax(test_accuracy)
    print(f'The highest accuracy value is {100*test_accuracy[best_k]}% for k={best_k}')

    plt.figure()
    plot_confusion_mat(y_test, y_pred)


