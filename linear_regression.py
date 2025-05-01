import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression1:
    def __init__(self, epochs=10000, alpha=0.01, normalize=True, plot_cost=False, tol=1e-9):
        self.epochs = epochs
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None
        self.costH = []
        self.mean_ = 0
        self.std_ = 0
        self.normalize = normalize
        self.plot_cost = plot_cost
        self.tol = tol

    def _fit_transform(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0) + 1e-8
        return (X - self.mean_) / self.std_

    def _transform(self, X):
        return (X - self.mean_) / self.std_

    def costFunction(self, X, Y, w):
        return np.sum((X @ w - Y) ** 2) / (2 * len(Y))

    def fit(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y).ravel()
        if self.normalize:
            X = self._fit_transform(X)

        X_ = np.c_[np.ones((X.shape[0], 1)), X]
        w = np.zeros(X_.shape[1])

        prev_cost = float('inf')
        for i in range(self.epochs):
            h = X_ @ w
            dif = h - Y
            gradient = (X_.T @ dif) / len(Y)
            w -= self.alpha * gradient

            cost = self.costFunction(X_, Y, w)
            self.costH.append(cost)

            if abs(prev_cost - cost) < self.tol:
                print(f"Early stopping at epoch {i}, cost difference: {abs(prev_cost - cost)}")
                break
            prev_cost = cost

        self.intercept_ = w[0]
        self.coef_ = w[1:]

        if self.plot_cost:
            plt.plot(self.costH)
            plt.xlabel("Epoch")
            plt.ylabel("Cost")
            plt.title("Cost History")
            plt.show()
        return self

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.normalize:
            X = self._transform(X)
        return X @ self.coef_ + self.intercept_

    def score(self, X, Y):
        Y = np.asarray(Y).ravel()
        Y_pred = self.predict(X)
        ss_total = np.sum((Y - np.mean(Y)) ** 2)
        ss_res = np.sum((Y - Y_pred) ** 2)
        r2 = 1 - ss_res / ss_total
        return r2