"""
basic_model.py:
Ich wollte da den output von so pytorch modellen testen, aber das heir ist ein haufen scheisse, wobei ich nicht verstehe wieso
die angepassten daten so schlecht ausgegeben werden.

egal ob sigmoid oder tanh, die daten muessten eig auch ohne skalierung ansatzweise richtig sein?
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# np.set_printoptions(suppress=True,
#                     formatter={'float_kind': '{:0.3f}'.format})

# A = [-5, -5, 0, 5, 5, -5]
# B = [-5, -5, -5, 5, 5, 0]
# C = [-4, 0, 4, 4, 4, -4]

# A = [-5, -5, 0]
# B = [-5, -5, -5]
# C = [-4, 0, 4]

# A = [-5, 0]
# B = [-5, -5]
# C = [0, 4]

A = [0]
B = [.4]
C = [.7]

y = torch.tensor((A, B, C, [-.4]), dtype=torch.float)  # 3 X 2 tensor
X = torch.tensor(([0], [.4], [.7], [-.4]), dtype=torch.float)  # 3 X 1 tensor
x0Predicted = torch.tensor(([0]), dtype=torch.float)  # 1 X 2 tensor
x1Predicted = torch.tensor(([.4]), dtype=torch.float)  # 1 X 2 tensor
x2Predicted = torch.tensor(([.7]), dtype=torch.float)  # 1 X 2 tensor

# X = torch.tensor(([2, 9], [1, 5], [3, 6]), dtype=torch.float) # 3 X 2 tensor
# y = torch.tensor(([92], [100], [89]), dtype=torch.float) # 3 X 1 tensor
# x0Predicted = torch.tensor(([4, 8]), dtype=torch.float) # 1 X 2 tensor

print(X.size())
print(y.size())

# scale units
X_max, _ = torch.max(X, 0)
xPredicted_max, _ = torch.max(x0Predicted, 0)

# X = torch.div(X, X_max)
# x0Predicted = torch.div(x0Predicted, xPredicted_max)
print(X)
print(x0Predicted)


# y = y / 100  # max test score is 100


class Neural_Network(nn.Module):
    def __init__(self, ):
        super(Neural_Network, self).__init__()
        # parameters
        # TODO: parameters can be parameterized instead of declaring them here
        self.inputSize = 1
        self.hiddenSize = 5
        self.outputSize = 1

        # weights
        self.W1 = torch.randn(self.inputSize, self.hiddenSize)  # 2 X hiddenSize tensor
        self.W2 = torch.randn(self.hiddenSize, self.hiddenSize)  # hiddenSize X hiddenSize tensor
        self.W3 = torch.randn(self.hiddenSize, self.outputSize)  # hiddenSize X 1 tensor

    def forward(self, X):
        self.z = torch.matmul(X, self.W1)  # 3 X 3 ".dot" does not broadcast in PyTorch
        self.z2 = self.tanh(self.z)  # activation function
        self.z3 = torch.matmul(self.z2, self.W2)
        self.z4 = self.tanh(self.z3)  # activation function
        self.z5 = torch.matmul(self.z4, self.W3)
        o = self.tanh(self.z5)  # final activation function
        return o

    def sigmoid(self, s):
        return 1 / (1 + torch.exp(-s))

    def sigmoidPrime(self, s):
        # derivative of sigmoid
        return s * (1 - s)

    def tanh(self, x):
        return torch.tanh(x)

    def tanhPrime(self, x):
        return 1.0 - np.tanh(x) ** 2

    def backward(self, X, y, o):
        self.o_error = y - o  # error in output
        self.o_delta = self.o_error * self.tanhPrime(o)  # derivative of sig to error
        self.z4_error = torch.matmul(self.o_delta, torch.t(self.W3))
        self.z4_delta = self.z4_error * self.tanhPrime(self.z4)
        self.z2_error = torch.matmul(self.z4_delta, torch.t(self.W2))
        self.z2_delta = self.z2_error * self.tanhPrime(self.z2)

        self.W1 += torch.matmul(torch.t(X), self.z2_delta)
        self.W2 += torch.matmul(torch.t(self.z2), self.z4_delta)
        self.W2 += torch.matmul(torch.t(self.z4), self.o_delta)

    def train(self, X, y):
        # forward + backward pass for training
        o = self.forward(X)
        self.backward(X, y, o)

    def saveWeights(self, model):
        # we will use the PyTorch internal storage functions
        torch.save(model, "NN")
        # you can reload model with all the weights and so forth with:
        # torch.load("NN")

    def predict(self):
        print("Predicted data based on trained weights: ")
        print("Input (scaled): \n" + str(x0Predicted))
        print("Output: \n" + str(np.array(self.forward(x0Predicted))))

        print("Input (scaled): \n" + str(x1Predicted))
        print("Output: \n" + str(np.array(self.forward(x1Predicted))))

        print("Input (scaled): \n" + str(x2Predicted))
        print("Output: \n" + str(np.array(self.forward(x2Predicted))))

        print("Input (scaled): \n" + str(-.4))
        print("Output: \n" + str(np.array(self.forward(torch.tensor(([-.4]), dtype=torch.float)))))

        print("Input (scaled): \n" + str(1))
        print("Output: \n" + str(np.array(self.forward(torch.tensor(([1]), dtype=torch.float)))))

        print("Input (scaled): \n" + str(-1))
        print("Output: \n" + str(np.array(self.forward(torch.tensor(([-1]), dtype=torch.float)))))

        print("Input (scaled): \n" + str(.1))
        print("Output: \n" + str(np.array(self.forward(torch.tensor(([0.1]), dtype=torch.float)))))

        fig, ax = plt.subplots()
        # ax.plot(-.4, self.forward(torch.tensor(([-.4]))))

        results = []
        values = [-5.0, -1.0, -.3, 0.0, .2, .4, 1.0, 5.0]
        for element in values:
            results.append(self.forward(torch.tensor(([element]))))

        ax.plot(values, results, label="prediction")
        ax.plot(values, values, label="ground truth")

        ax.set(xlabel='time (s)', ylabel='voltage (mV)',
               title='About as simple as it gets, folks')
        ax.grid()

        fig.savefig("test.png")
        plt.show()


NN = Neural_Network()
for i in range(20):  # trains the NN 1,000 times
    print("#" + str(i) + " Loss: " + str(torch.mean((y - NN(X)) ** 2).detach().item()))  # mean sum squared loss
    NN.train(X, y)
# NN.saveWeights(NN)
NN.predict()
