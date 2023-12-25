import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class logistic_regression:
    def __init__(self, x, y, order=1):
        poly_features = PolynomialFeatures(degree=order, include_bias=False)
        self.X = poly_features.fit_transform(x)
        self.theta = (self.X.T @ self.X) ** (-1) @ self.X.T @ y
        self.y = y


    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


    def logistic_regression_sgdmb(self, X, y, max_iter, ucenje, n_epochs=50, batch_size=10):
        p, q = X.shape
        self.theta = np.random.randn(q, 1)
        progress = []
        for epoch in range(n_epochs):
            shuffled_indices = np.random.permutation(p)
            X_shuffled = X[shuffled_indices]
            y_shuffled = y[shuffled_indices]

            for i in range(0, p, batch_size):
                X_mini_batch = X_shuffled[i:i + batch_size]
                y_mini_batch = y_shuffled[i:i + batch_size]

                logits = X_mini_batch @ self.theta

                predictions = self.sigmoid(logits)

                rez = 1 / batch_size * X_mini_batch.T.dot(predictions - y_mini_batch)

                self.theta = self.theta - ucenje * rez

            pom= self.loss_fun()
            #print(f'Epoch{epoch+1}/{n_epochs}, loss:{pom}')
            progress.append(pom)

        return progress

    def logistic_fun(self, x):
        return 1 / (1 + np.exp(-np.dot(self.theta.T, x)))

    def loss_fun(self):
        l = 0
        for i in range(np.shape(self.X)[0]):
            if self.y[i] == 1:
                l += np.log(self.logistic_fun(self.X[i, :]) + 1e-10)
            else:
                l += np.log(1 - self.logistic_fun(self.X[i, :]) + 1e-10)
        return l

    #def logistic_regression_sgd(self, X, y, max_iter, ucenje):
        #p, q = X.shape
        #self.theta = np.random.randn(q, 1)
        #progress = []

        #for iteration in range(max_iter):
            #for i in range(p):
                #x_i = X[i, :].reshape(-1, 1)
                #prediction = self.sigmoid(np.dot(self.theta.T, x_i))
                #gradient = np.dot(x_i, (y[i] - prediction))
                #self.theta = self.theta + ucenje * gradient

            #loss = self.loss_fun()
            #progress.append(loss)
            #print(f'Iteration {iteration + 1}/{max_iter}, Loss: {loss}')

        #return progress

    #def loss_fun(self):
        #logits = np.dot(self.X, self.theta)
        #predictions = self.sigmoid(logits)
        #loss = -np.mean(self.y * np.log(predictions) + (1 - self.y) * np.log(1 - predictions + 1e-10))
        #return loss

    #def mini_batch(self, max_iter, mu, batch_size):
        #progress = np.zeros(max_iter)
        #for iteration in range(max_iter):
            #perm = np.random.permutation(np.shape(self.X)[0])
            #self.X = self.X[perm]
            #self.y = self.y.T
            #self.y = (self.y[perm]).T
            #mb = self.X[:batch_size, :]
            #mby = self.y[:, :batch_size]
            #dl = (mby - self.h(mb.T)) @ mb
            #self.theta += mu * dl
            #progress[iteration] = self.loss_fcn()
        #return progress


class gnb:
    def __init__(self, X, y, num_classes, nb=True):
        self.num_classes = num_classes
        self.priors = np.zeros(num_classes)
        self.num_features = np.shape(X)[1]
        self.m = np.zeros((self.num_features, num_classes))
        self.sigma = np.zeros(((self.num_features, self.num_features, num_classes)))
        for i in range(num_classes):
            self.priors[i] = np.sum(y == i)
            self.m[:, i] = np.mean(X[np.reshape(y, -1) == i], axis=0)
            self.sigma[:, :, i] = np.cov((X[np.reshape(y, -1) == i]).T) + np.eye(self.num_features) * 0.0001
            if nb == True:
                self.sigma[:, :, i] = np.diag(np.diag(self.sigma[:, :, i]))
        self.priors = self.priors / np.sum(self.priors)

    def h(self, x):
        test_result = np.zeros((self.num_classes))
        for i in range(self.num_classes):
            x_m = np.reshape(x - self.m[:, i], (self.num_features, 1))
            sigma_inv = np.linalg.inv(self.sigma[:, :, i])
            test_result[i] = np.exp(-0.5 * (x_m).T @ sigma_inv @ (x_m))
            test_result[i] /= (((2 * np.pi) ** self.num_features) * np.linalg.det(self.sigma[:, :, i])) ** 0.5
            test_result[i] *= self.priors[i]
        return test_result