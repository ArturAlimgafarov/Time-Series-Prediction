import numpy as np
import math
from sklearn import linear_model

# INPUT: data - array of nums (for every month), k - count of predictions
# OUTPUT: array of k predicted values (next k months)

class TimeSeries:
    def __init__(self, data, k):
        self.data = data
        self.count = len(data)
        self.predictCount = k
        self.mean = np.mean(data)
        self.yearsCount = math.ceil(self.count / 12)
        self.timeAxis = list(range(1, self.count + 1))

    def addModel(self):
        y = self.data.copy()
        totalYear = [sum(y[i:i + 12]) for i in range(self.count - 11)]
        movingAvg = [x / 12 for x in totalYear]
        movingAvgMid = [(movingAvg[i] + movingAvg[i + 1]) / 2 for i in range(len(movingAvg) - 1)]
        S = [y[i + 2] - movingAvgMid[i] for i in range(len(movingAvgMid))]

        _S = [0, 0] + [x for x in S] + [0 for _ in range(self.count - len(S) - 2)]
        table = np.matrix([_S[12 * i: 12 * i + 12] for i in range(self.yearsCount)])
        total = [float(sum(table[:, i])) for i in range(12)]
        Ssr = [t / 3 for t in total]
        k = sum(Ssr) / 12
        Sadj = [s - k for s in Ssr]

        S = self.yearsCount * Sadj
        TE = [(y[i] - S[i]) for i in range(self.count)]
        b = ((sum([TE[i] * self.timeAxis[i] for i in range(self.count)]) / self.count) -
             (np.mean(TE) * np.mean(self.timeAxis))) / ((sum([i ** 2 for i in self.timeAxis]) / self.count) - (np.mean(self.timeAxis) ** 2))
        a = (sum(TE) / self.count) - b * (sum(self.timeAxis) / self.count)
        T = [(a + b * i) for i in self.timeAxis]
        ST = np.array(S) + np.array(T)

        absErr = np.mean([abs((y[i] - ST[i]) * 100 / y[i]) for i in range(self.count)])
        sigma = math.sqrt(np.mean([((y[i] - ST[i]) * 100 / y[i]) ** 2 for i in range(self.count)]))
        rSqr = (1 - sum([(y[i] - ST[i]) ** 2 for i in range(self.count)]) / sum([(y[i] - self.mean) ** 2 for i in range(self.count)])) * 100
        residuals = [(y[i] - ST[i]) for i in range(self.count)]
        dw = (sum([(residuals[i] - residuals[i - 1]) ** 2 for i in range(1, self.count)])) / sum([r ** 2 for r in residuals])

        predt = [(self.count + (self.count % 12) + i + 1) for i in range(self.predictCount)]
        predS = [S[i] for i in range(self.predictCount)]
        predT = [(a + b * i) for i in predt]

        predict = np.array(predS) + np.array(predT)

        return {
            'trend': (a, b),
            'MAPE': absErr,
            'MSE': sigma,
            'R-square': rSqr,
            'DW': dw,
            'predict': predict
        }

    def multModel(self):
        y = self.data.copy()

        totalYear = [sum(y[i:i + 12]) for i in range(len(y) - 11)]
        movingAvg = [x / 12 for x in totalYear]
        movingAvgMid = [(movingAvg[i] + movingAvg[i + 1]) / 2 for i in range(len(movingAvg) - 1)]
        S = [y[i + 2] / movingAvgMid[i] for i in range(len(movingAvgMid))]

        _S = [0, 0] + [x for x in S] + [0 for _ in range(self.count - len(S) - 2)]
        table = np.matrix([_S[12 * i: 12 * i + 12] for i in range(self.yearsCount)])
        total = [float(sum(table[:, i])) for i in range(12)]
        Ssr = [t / 3 for t in total]
        k = sum(Ssr) / 12
        Sadj = [s * k for s in Ssr]

        S = self.yearsCount * Sadj
        TE = [(y[i] / S[i]) for i in range(self.count)]
        b = ((sum([TE[i] * self.timeAxis[i] for i in range(self.count)]) / self.count) -
             (np.mean(TE) * np.mean(self.timeAxis))) / ((sum([i ** 2 for i in self.timeAxis]) / self.count) - (np.mean(self.timeAxis) ** 2))
        a = (sum(TE) / self.count) - b * (sum(self.timeAxis) / self.count)
        T = [(a + b * i) for i in self.timeAxis]
        ST = np.array(S) * np.array(T)

        absErr = np.mean([abs((y[i] - ST[i]) * 100 / y[i]) for i in range(self.count)])
        sigma = math.sqrt(np.mean([((y[i] - ST[i]) * 100 / y[i]) ** 2 for i in range(self.count)]))
        rSqr = (1 - sum([(y[i] - ST[i]) ** 2 for i in range(self.count)]) / sum([(y[i] - self.mean) ** 2 for i in range(self.count)])) * 100
        residuals = [(y[i] / ST[i]) for i in range(self.count)]
        dw = sum([(residuals[i] - residuals[i - 1]) ** 2 for i in range(1, self.count)]) / sum([r ** 2 for r in residuals])

        predt = [(self.count + (self.count % 12) + i + 1) for i in range(self.predictCount)]
        predS = [S[i] for i in range(self.predictCount)]
        predT = [(a + b * i) for i in predt]

        predict = np.array(predS) * np.array(predT)

        return {
            'trend': (a, b),
            'MAPE': absErr,
            'MSE': sigma,
            'R-square': rSqr,
            'DW': dw,
            'predict': predict
        }

    def dvModel(self):
        pass