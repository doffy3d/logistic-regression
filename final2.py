import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



# Set the random seed for reproducibility
np.random.seed(42)

class LogisticRegression:
    def __init__(self, x, y, order=1):
        self.X = x
        self.y = y
        self.theta = np.random.randn(x.shape[1], 1)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def logistic_regression_sgdmb(self, max_iter, learning_rate, n_epochs=50, batch_size=10):
        p, q = self.X.shape
        progress = []

        for epoch in range(n_epochs):
            shuffled_indices = np.random.permutation(p)
            X_shuffled = self.X[shuffled_indices]
            y_shuffled = self.y[shuffled_indices]

            for i in range(0, p, batch_size):
                X_mini_batch = X_shuffled[i:i + batch_size]
                y_mini_batch = y_shuffled[i:i + batch_size]

                logits = X_mini_batch @ self.theta
                predictions = self.sigmoid(logits)

                #gradient = 1 / batch_size * X_mini_batch.T.dot(predictions - y_mini_batch)
                gradient = X_mini_batch.T @ (predictions - y_mini_batch)  # Updated gradient calculation

                self.theta = self.theta - learning_rate * gradient

            loss = self.loss_fun()
            progress.append(loss)
            print(f'Epoch {epoch + 1}/{n_epochs}, Loss: {loss}')

        return progress

    #def logistic_fun(self, x):
        #return self.sigmoid(np.dot(x, self.theta))
    def logistic_fun(self, x):
        return self.sigmoid(np.dot(x.T, self.theta))

    """""
    def loss_fun(self):
        logits = self.X @ self.theta
        predictions = self.sigmoid(logits)
        loss = -np.mean(self.y * np.log(predictions) + (1 - self.y) * np.log(1 - predictions + 1e-10))
        return loss

    """""
    def loss_fun(self):
        l = 0
        for i in range(np.shape(self.X)[0]):
            if self.y[i] == 1:
                l += np.log(self.logistic_fun(self.X[i, :]) + 1e-10)
            else:
                l += np.log(1 - self.logistic_fun(self.X[i, :]) + 1e-10)
        return l


    def predict(self, x):
        probabilities = self.logistic_fun(x)
        return np.argmax(probabilities, axis=1).reshape(-1, 1)


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
            test_result[i] = np.exp(-0.5 * (x_m).T @ sigma_inv @ (x_m))[0,0]
            test_result[i] /= (((2 * np.pi) ** self.num_features) * np.linalg.det(self.sigma[:, :, i])) ** 0.5
            test_result[i] *= self.priors[i]
        return test_result

# Load and preprocess the data
data = read_csv('multiclass_data.csv', header=None)
data = np.array(data)

X = data[:, :-1]
y = data[:, -1:]

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Function to calculate acc
def acc(a, b):
    return sum(np.equal(a, b)) / len(a)

# Logistic Regression
cl = []

learning_rates = [0.02, 0.1, 0.001]
minibatch_sizes = [10, 100, 5]

# Train for each class
for i in range(3):
    temp = np.zeros((len(y_train), 1))
    temp[y_train == i] = [1];
    cl.append(LogisticRegression(X_train, temp))

# Plot
plt.figure()

for lr in learning_rates:
    for mmb in minibatch_sizes:
        model = LogisticRegression(X_train, y_train)
        progress = model.logistic_regression_sgdmb(max_iter=100, learning_rate=lr, n_epochs=50, batch_size=mmb)
        plt.plot(np.arange(mmb, len(progress) * mmb + 1, mmb), progress, label=f'LR - α={lr}, m={mmb}')

plt.title('Convergence for different learning rates and minibatch sizes')
plt.xlabel('Number of training examples')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate on the test set
test_results_lr = np.zeros((len(y_test), 3))
final_match_lr = np.zeros((len(y_test), 1))

for i in range(len(y_test)):
    for j in range(3):
        reshaped = np.reshape(X_test[i, :], (X_test.shape[1], 1))
        test_results_lr[i, j] = cl[j].logistic_fun(reshaped)[0, 0]
    final_match_lr[i] = np.argmax(test_results_lr[i, :])

print('Logistic regression accuracy')
print(acc(final_match_lr, y_test))



# GNB
gnb = gnb(X_train, y_train, 3, nb=True)
test_results_gnb = np.zeros((len(y_test), 3))
final_match_gnb = np.zeros((len(y_test), 1))
for i in range(len(y_test)):
    test_results_gnb[i, :] = gnb.h(X_test[i, :].T)
    final_match_gnb[i] = np.argmax(test_results_gnb[i, :])
print('GNB accuracy')
print(acc(final_match_gnb, y_test))