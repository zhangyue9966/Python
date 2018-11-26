import numpy as np
import sklearn.metrics as metrics
import math
import matplotlib.pyplot as plt
import scipy.io


def load_dataset():
    mat = scipy.io.loadmat('./data/spam.mat')
    labels_train = mat['ytrain']  # 3450*1
    Xtrain = mat['Xtrain']  # 3450*57
    Xtest = mat['Xtest']  # 1151*57
    return (Xtrain, labels_train), Xtest


def standardize(X):
    m = np.mean(X, axis=0)
    s = np.std(X, axis=0)
    result = (X - m) / s
    return result


def transform(X):
    X = np.log(X + 0.1)
    return X


def binarize(X):
    X[X > 0] = 1
    X[X < 0] = 0
    return X


def sigmod(theta, X):
    return float(1) / (1 + math.e ** (-X.dot(theta)))


def gradient(theta, X, labels, reg):
    num_train = X.shape[0]
    tmp = sigmod(theta, X) - labels
    grad = tmp.T.dot(X)
    grad = grad.T
    grad += 2 * reg * theta
    grad /= num_train
    return grad


def get_loss(theta, X, labels, reg):
    tmp = sigmod(theta, X)
    one = labels * np.log(tmp)
    two = (1 - labels) * np.log(1 - tmp)
    loss = np.sum(-one - two) + 2 * np.sum(theta * theta)
    return loss


def train_gd(X, labels, learning_rate, reg, num_iter):
    np.random.seed(0)
    theta = np.zeros((X.shape[1], 1))
    loss = []
    current_loss = get_loss(theta, X, labels, reg)
    loss.append(current_loss)
    for iter in range(num_iter):
        theta -= learning_rate * gradient(theta, X, labels, reg)
        current_loss = get_loss(theta, X, labels, reg)
        if iter % 1000 == 0:
            print current_loss
            #     predict_labels = predict(theta, X)
            #     print metrics.accuracy_score(labels, predict_labels)
        loss.append(current_loss)
    return theta, loss


def train_sgd(X, labels, learning_rate, reg, num_iter):
    np.random.seed(0)
    theta = np.zeros((X.shape[1], 1))
    num_train = X.shape[0]
    loss = []
    current_loss = get_loss(theta, X, labels, reg)
    loss.append(current_loss)

    for iter in range(num_iter):
        index = np.random.choice(num_train, 2)
        X_batch = X[index, :]
        labels_batch = labels_train[index, :]
        learning=learning_rate/(iter/100 + 1)
        theta -= learning * gradient(theta, X_batch, labels_batch, reg)
        current_loss = get_loss(theta, X, labels, reg)
        if iter % 1000 == 0:
            print current_loss
        loss.append(current_loss)
    return theta, loss


def predict(theta, X):
    tmp = sigmod(theta, X)
    tmp[tmp >= 0.5] = 1
    tmp[tmp < 0.5] = 0
    return tmp


if __name__ == "__main__":
    (Xtrain, labels_train), Xtest = load_dataset()

    # Question 1#
    # num_iter = 10000
    # learning_rate = 3e-3
    # reg = 0.5
    # Xtrain_std = standardize(Xtrain)
    # _, std_loss = train_gd(Xtrain_std, labels_train, learning_rate=learning_rate, reg=reg, num_iter=num_iter)
    # Xtrain_tran = transform(Xtrain)
    # _, tran_loss = train_gd(Xtrain_tran, labels_train, learning_rate=learning_rate, reg=reg, num_iter=num_iter)
    # Xtrain_bin = binarize(Xtrain)
    # _, bin_loss = train_gd(Xtrain_bin, labels_train, learning_rate=learning_rate, reg=reg, num_iter=num_iter)
    # plt.subplot(311)
    # plt.plot(std_loss)
    # plt.title('Standardize')
    # plt.subplot(312)
    # plt.plot(tran_loss)
    # plt.title('Transform')
    # plt.subplot(313)
    # plt.plot(bin_loss)
    # plt.title('Binarize')
    # plt.show()

    # Question2,3#
    num_iter =50000
    learning_rate = 3e-3
    reg = 0.5
    Xtrain_std = standardize(Xtrain)
    _, std_loss = train_sgd(Xtrain_std, labels_train, learning_rate=learning_rate, reg=reg, num_iter=num_iter)
    Xtrain_tran = transform(Xtrain)
    _, tran_loss = train_sgd(Xtrain_tran, labels_train, learning_rate=learning_rate, reg=reg, num_iter=num_iter)
    Xtrain_bin = binarize(Xtrain)
    _, bin_loss = train_sgd(Xtrain_bin, labels_train, learning_rate=learning_rate, reg=reg, num_iter=num_iter)
    plt.subplot(311)
    plt.plot(std_loss)
    plt.title('Standardize')
    plt.subplot(312)
    plt.plot(tran_loss)
    plt.title('Transform')
    plt.subplot(313)
    plt.plot(bin_loss)
    plt.title('Binarize')
    plt.show()


    # question 4
    # Xtrain_tran = transform(Xtrain)
    # Xtest_tran=transform(Xtest)
    # theta, _ = train_gd(Xtrain_tran, labels_train, learning_rate=3e-3, reg=0.5, num_iter=10000)
    # result = predict(theta, Xtest_tran)
    # index = np.arange(1,result.shape[0]+1)
    # result=result.reshape(result.shape[0],)
    # file = np.column_stack((index, result))
    # np.savetxt('result.csv', file, delimiter=',',fmt='%d', header='Id,Category')
