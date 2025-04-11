import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def loadData():
    x = pd.read_csv("../data/linearX.csv")
    y = pd.read_csv("../data/linearY.csv")
    return x, y


class MinMaxScaler:
    def __init__(self):
        pass

    def fitTransform(self, x):
        xMin, xMax = np.min(x), np.max(x)
        xNorm = (x - xMin) / (xMax - xMin)
        return xNorm


class LinearRegression:
    def __init__(self):
        self.w = 0
        self.b = 0
        self.costs = []

    def modelFunc(self, x_i):
        yHat_i = self.w * x_i + self.b
        return yHat_i

    def lossFunc(self, x_i, y_i):
        yHat_i = self.modelFunc(x_i)
        l_i = (yHat_i - y_i) ** 2
        return l_i

    def costFunc(self, x, y):
        m = x.shape[0]
        J_wb = 0
        for i in range(m):
            l_i = self.lossFunc(x[i], y[i])
            J_wb += l_i
        J_wb *= 1 / (2 * m)
        return J_wb

    def error(self, x_i, y_i):
        yHat_i = self.modelFunc(x_i)
        e_i = yHat_i - y_i
        return e_i

    def fit(self, x, y, epochs, a):
        x = x.to_numpy()
        y = y.to_numpy()
        m = x.shape[0]
        dJ_dw, dJ_db = 0, 0
        for epoch in range(epochs):
            e, ex = 0, 0
            for i in range(m):
                e_i = self.error(x[i], y[i])
                e += e_i
                e_i_x_i = e_i * x[i]
                ex += e_i_x_i
            dJ_dw = (1 / m) * ex
            dJ_db = (1 / m) * e
            self.w -= a * dJ_dw
            self.b -= a * dJ_db
            self.costs.append(self.costFunc(x, y))
        return dJ_dw, dJ_db

    def predict(self, x):
        yHat = self.modelFunc(x)
        return yHat

    def sgdFit(self, x, y, epochs, a):
        x = x.to_numpy()
        y = y.to_numpy()
        m = x.shape[0]
        for epoch in range(epochs):
            indices = np.random.permutation(m)
            x, y = x[indices], y[indices]
            for i in range(m):
                e_i = self.error(x[i], y[i])
                e_i_x_i = e_i * x[i]
                dJ_dw_i = e_i_x_i
                dJ_db_i = e_i
                self.w -= a * dJ_dw_i
                self.b -= a * dJ_db_i
            self.costs.append(self.costFunc(x, y))

    def mbgdFit(self, x, y, batchSize, epochs, a):
        x = x.to_numpy()
        y = y.to_numpy()
        m = x.shape[0]
        for epoch in range(epochs):
            indices = np.random.permutation(m)
            x, y = x[indices], y[indices]
            for i in range(0, m, batchSize):
                dJ_dw, dJ_db = 0, 0
                xBatch = x[i:i + batchSize]
                yBatch = y[i:i + batchSize]
                batchSizeActual = len(xBatch)
                e, ex = 0, 0
                for j in range(batchSizeActual):
                    e_i = self.error(xBatch[j], yBatch[j])
                    e += e_i
                    e_i_x_i = e_i * xBatch[j]
                    ex += e_i_x_i
                dJ_dw = (1 / batchSizeActual) * ex
                dJ_db = (1 / batchSizeActual) * e
                self.w -= a * dJ_dw
                self.b -= a * dJ_db
            self.costs.append(self.costFunc(x, y))


    def plotCost(self, epochs):
        fig, ax = plt.subplots()
        ax.plot(list(range(epochs)), self.costs)
        ax.set_title("Cost vs Epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cost")
        plt.show()

    def scatterPlot(self, x, y):
        _, ax = plt.subplots()
        ax.scatter(x, y)
        ax.set_title("Regression")
        ax.set_xlabel("Features")
        ax.set_ylabel("Labels")
        plt.plot(x, self.predict(x), color="red", label="Best fit line")
        plt.show()
