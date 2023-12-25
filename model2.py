import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from klasa2 import *


data = read_csv('multiclass_data.csv', header=None)
data = np.array(data)

X = data[:, :-1]
y = data[:, -1:]
scaler = StandardScaler()
X = scaler.fit_transform(X)
cl = []

X_train, X_test, y_train, y_test = train_test_split(X, y)

learning_rate = 0.02
max_iter = 100
mmb = [0.1, 0.02, 0.00002]

def acc(a, b):
    return sum(np.equal(a, b)) / len(a)

# LOGISTIC REGRESSION
for i in range(3):
    temp = np.zeros((len(y_train), 1))
    temp[y_train == i] = [1];
    cl.append(logistic_regression(X_train, temp))
    progress = cl[i].logistic_regression_sgdmb(X_train, y_train,max_iter, ucenje=mmb[i], n_epochs=50, batch_size=10)

test_results_lr = np.zeros((len(y_test), 3))
final_match_lr = np.zeros((len(y_test), 1))

for i in range(len(y_test)):
    for j in range(3):
        reshaped = np.reshape(X_test[i, :], (5, 1))
        test_results_lr[i, j] = cl[j].logistic_fun(reshaped)[0, 0]
    final_match_lr[i] = np.argmax(test_results_lr[i, :])
print('Logistic regression accuracy')
print(acc(final_match_lr, y_test))


max_iter = 100
alpha_optimal = 0.02
minibatch_size_optimal = 10
alphas = [0.1, 0.001]
minibatch_sizes = [100, 5]


plt.figure(figsize=(10, 5))

plt.subplot(3, 2, 1)
model_optimal = logistic_regression(X_train, y_train, order=1)
y_optimal = model_optimal.logistic_regression_sgdmb(X_train, y_train, max_iter, ucenje=alpha_optimal, n_epochs=50, batch_size=minibatch_size_optimal)
x_optimal = np.arange(minibatch_size_optimal, len(y_optimal) * minibatch_size_optimal + 1, minibatch_size_optimal)
plt.plot(x_optimal, y_optimal, label=f'Optimalno (α={alpha_optimal}, m={minibatch_size_optimal})')
plt.title('Optimalno')
plt.xlabel('broj iteracija')
plt.ylabel('funkcija gubitka')
plt.legend()


plt.subplot(3, 2, 2)
for alpha_large in alphas:
    model_large_alpha = logistic_regression(X_train, y_train, order=1)
    y_large_alpha = model_large_alpha.logistic_regression_sgdmb(X_train, y_train, max_iter, ucenje=alpha_large, n_epochs=50, batch_size=minibatch_size_optimal)
    x_large_alpha = np.arange(minibatch_size_optimal, len(y_large_alpha) * minibatch_size_optimal + 1, minibatch_size_optimal)
    plt.plot(x_large_alpha, y_large_alpha, label=f'veliko α={alpha_large}')
plt.title('veliko α')
plt.xlabel('broj iteracija')
plt.ylabel('funkcija gubitka')
plt.legend()


plt.subplot(3, 2, 3)
for alpha_small in alphas:
    model_small_alpha = logistic_regression(X_train, y_train, order=1)
    y_small_alpha = model_small_alpha.logistic_regression_sgdmb(X_train, y_train, max_iter, ucenje=alpha_small, n_epochs=50, batch_size=minibatch_size_optimal)
    x_small_alpha = np.arange(minibatch_size_optimal, len(y_small_alpha) * minibatch_size_optimal + 1, minibatch_size_optimal)
    plt.plot(x_small_alpha, y_small_alpha, label=f'malo α={alpha_small}')
plt.title('malo α')
plt.xlabel('broj iteracija')
plt.ylabel('funkcija gubitka')
plt.legend()


plt.subplot(3, 2, 4)
for mmb_large in minibatch_sizes:
    model_large_mmb = logistic_regression(X_train, y_train, order=1)
    y_large_mmb = model_large_mmb.logistic_regression_sgdmb(X_train, y_train, max_iter, ucenje=alpha_optimal, n_epochs=50, batch_size=mmb_large)
    x_large_mmb = np.arange(mmb_large, len(y_large_mmb) * mmb_large + 1, mmb_large)
    plt.plot(x_large_mmb, y_large_mmb, label=f'veliko m={mmb_large}')
plt.title('veliko m')
plt.xlabel('broj iteracija')
plt.ylabel('funkcija gubitka')
plt.legend()


plt.subplot(3, 2, 5)
for mmb_small in minibatch_sizes:
    model_small_mmb = logistic_regression(X_train, y_train, order=1)
    y_small_mmb = model_small_mmb.logistic_regression_sgdmb(X_train, y_train, max_iter, ucenje=alpha_optimal, n_epochs=50, batch_size=mmb_small)
    x_small_mmb = np.arange(mmb_small, len(y_small_mmb) * mmb_small + 1, mmb_small)
    plt.plot(x_small_mmb, y_small_mmb, label=f'malo m={mmb_small}')
plt.title('malo m')
plt.xlabel('broj iteracija')
plt.ylabel('funkcija gubitka')
plt.legend()

plt.tight_layout()
plt.show()










# GNB
gnb = gnb(X_train, y_train, 3, nb=True)
test_results_gnb = np.zeros((len(y_test), 3))
final_match_gnb = np.zeros((len(y_test), 1))
for i in range(len(y_test)):
    test_results_gnb[i, :] = gnb.h(X_test[i, :].T)
    final_match_gnb[i] = np.argmax(test_results_gnb[i, :])
print('GNB accuracy')
print(acc(final_match_gnb, y_test))

