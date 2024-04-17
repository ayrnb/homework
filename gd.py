import os
import pickle
import numpy as np
from dataloader import load

current_path = os.path.abspath(__file__)


"""
gradien _descent: Calculate the gradient

"""


def gradient(x, y, w):
    N, D = x.shape
    g = np.zeros(D)
    for i in range(N):
        l = np.exp(-np.dot(w, x[i]))
        mu = 1 / (1 + l)
        g += x[i] * (y[i] - mu)
    return g


"""
accelerated_gradient_descent
"""


def accelerated_gradient_descent(x, y, w_init, learning_rate, momentum, num_iterations):
    w = w_init
    velocity = np.zeros_like(w)

    for _ in range(num_iterations):
        gt = gradient(x, y, w)
        velocity = momentum * velocity - learning_rate * gt
        w = w + velocity

    return w


def log_likelihood(x, y, w, lamb=0.0):
    ll = 0.0
    N = x.shape[0]
    for i in range(N):
        l = 1 + np.exp(np.dot(w, x[i]))
        ll += y[i] * np.dot(w, x[i]) - np.log(l)
    ll -= 0.5 * lamb * np.linalg.norm(w) ** 2
    return ll


def evaluate(x, y, w):
    N = x.shape[0]
    prediction = []
    for i in range(N):
        l = np.exp(-np.dot(w, x[i]))
        p = 1 / (1 + l)
        if p > 0.5:
            prediction.append(1)
        else:
            prediction.append(0)
    return np.mean(np.array(prediction) == y)


"""
 Gradient Descent & Accelerated Gradient Descent: Parameter selection-name
"""


def gd(train, test, lr, patience, name="accelerated", momentum=0.9):
    D = train["feature"].shape[1]
    # w = np.random.random(D)
    w = np.random.uniform(0, 0.1, D)
    acc_list = []
    ll_list = []
    best = None
    wait = 0
    step = 0
    while wait < patience:
        if name == "accelerated":
            w = accelerated_gradient_descent(
                train["feature"], train["label"], w, lr, momentum
            )
        else:
            g = gradient(train["feature"], train["label"], w)
            w += lr * g
        # print("w: %f" % np.mean(w))
        acc = evaluate(test["feature"], test["label"], w)
        ll = log_likelihood(train["feature"], train["label"], w)
        if best is None or acc > best:
            best = acc
            wait = 0
        else:
            wait += 1
        print("step=%d, acc=%f, ll=%f" % (step, acc, ll))
        acc_list.append(acc)
        ll_list.append(ll)
        step += 1

    return acc_list, ll_list


if __name__ == "__main__":
    patience = 10
    lr = 0.0001
    momentum = 0.95
    train = {}
    test = {}
    dir_path = os.path.dirname(current_path)
    output_file = os.path.join(dir_path, "agd_lr_0.0001.pickle")

    train["feature"], train["label"] = load(os.path.join(dir_path, "a9a\\a9a.txt"))
    test["feature"], test["label"] = load(os.path.join(dir_path, "a9a\\a9a.t"))
    # add for w_0
    ones = np.ones([train["feature"].shape[0], 1])
    train["feature"] = np.append(train["feature"], ones, axis=1)
    ones = np.ones([test["feature"].shape[0], 1])
    test["feature"] = np.append(test["feature"], ones, axis=1)

    # acc_list, ll_list = gd(train, test, lr, patience, name="no_accelerated")
    acc_list, ll_list = gd(train, test, lr, patience, momentum)

    with open(output_file, "wb") as f:
        d = {"acc": acc_list, "ll": ll_list}
        pickle.dump(d, f)
