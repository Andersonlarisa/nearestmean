from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import scipy.spatial
from collections import Counter

class KNN(BaseEstimator, ClassifierMixin):
    def __init__(self, k=1):
        """
            Parameters
            ----------
            k: Number of nearest self.neighbors
            problem: Type of learning
                0 = Regression, 1 = Classification
            metric: Distance metric to be used.
                0 = Euclidean, 1 = Manhattan
        """
        self.k = k
        # metric: int = 0
        #self.metric = metric

    def fit(self, X_train, y):
        self.X_train = X_train
        self.y_train = y
        return self

    def distance(self, X1, X2):
        distance = scipy.spatial.distance.euclidean(X1, X2)

    def predict(self, X_test):
        import numpy as np
        from scipy import stats

        m = self.X_train.shape[0]
        n = X_test.shape[0]
        y_pred = []

        # Calculating distances
        for i in range(n):  # for every sample in X_test
            distance = []  # To store the distances
            for j in range(m):  # for every sample in X_train
                if self.metric == 0:
                    d = (np.sqrt(np.sum(np.square(X_test[i, :] - self.X_train[j, :]))))  # Euclidean distance
                else:
                    d = (np.absolute(X_test[i, :] - self.X_train[j, :]))  # Manhattan distance
                distance.append((d, self.y_train[j]))
            distance = sorted(distance)  # sorting distances in ascending order

            # Getting k nearest neighbors
            neighbors = []
            for item in range(self.k):
                neighbors.append(distance[item][1])  # appending K nearest neighbors

            # Making predictions
            if self.problem == 0:
                y_pred.append(np.mean(neighbors))  # For Regression
            else:
                y_pred.append(stats.mode(neighbors)[0][0])  # For Classification
        return y_pred

    def score(self, X_test, y):
        predictions = self.predict(X_test)
        return (predictions == y).sum() / len(y)

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"k": self.k}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self