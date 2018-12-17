import numpy as np


def ridge(X, y, lamb=0):
    return np.linalg.inv(X.T @ X + lamb * np.eye(np.shape(X)[1])) @ X.T @ y


def least_squares(X, y):
	return ridge(X, y)