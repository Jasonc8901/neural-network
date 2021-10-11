import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import scipy.io


def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))
    sig = np.minimum(sig, 0.95)  # Set upper bound
    sig = np.maximum(sig, 0.0001)  # Set lower bound
    return sig


def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))


def relu(x):
    return max(0.0, x)


def first_nn():
    feature_set = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]])

    labels = np.array([[1, 0, 0, 1, 1]])
    labels = labels.reshape(5, 1)

    np.random.seed(1000)
    weights = np.random.rand(3, 1)
    bias = np.random.rand(1)
    lr = 0.05

    es = []
    eps = []

    for epoch in range(500):
        inputs = feature_set

        # feedforward step1
        XW = np.dot(feature_set, weights) + bias

        # feedforward step2
        z = sigmoid(XW)

        # backpropagation step 1
        error = z - labels

        print(error.sum())

        # backpropagation step 2
        dcost_dpred = error
        dpred_dz = sigmoid_der(z)

        z_delta = dcost_dpred * dpred_dz

        inputs = feature_set.T
        weights -= lr * np.dot(inputs, z_delta)

        for num in z_delta:
            bias -= lr * num

        es.append(error.sum())
        eps.append(epoch)

    single_point = np.array([0, 1, 0])  # network test
    result = sigmoid(np.dot(single_point, weights) + bias)
    print(result)
    points = np.linspace(-10, 10, 20)

    plt.plot(es, eps, c="b")
    # plt.plot(points, sigmoid(points), c="b")

    plt.show(block=True)
    plt.interactive(False)


if __name__ == '__main__':
    # first_nn()
    mat = scipy.io.loadmat('training.mat')
    feature_set = mat['X']
    s = feature_set.shape
    feature_set = feature_set.reshape(s[3], s[0] * s[1] * s[2])[:50]

    labels = mat['y'][:50]

    np.random.seed(1000)
    weights = np.random.rand(3072, 1)
    bias = np.random.rand(1)
    lr = 0.05

    for epoch in range(500):
        inputs = feature_set

        # feedforward step1
        XW = np.dot(feature_set, weights) + bias

        # feedforward step2
        z = relu(XW)

        # backpropagation step 1
        error = z - labels

        print(error.sum())

        # backpropagation step 2
        dcost_dpred = error
        dpred_dz = sigmoid_der(z)

        z_delta = dcost_dpred * dpred_dz

        inputs = feature_set.T
        weights -= lr * np.dot(inputs, z_delta)

        for num in z_delta:
            bias -= lr * num



