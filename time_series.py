import numpy as np
import math
from sklearn import linear_model

# INPUT: data - array of nums (for every month), k - count of predictions
# OUTPUT: array of k predicted values (next k months)

class TimeSeries:
    def __init__(self, data, k):
        self.data = data
        self.predictCount = k

    def addModel(self):
        n = len(self.data)
        t = list(range(1, n + 1)) # time axis
        y = self.data.copy()
        avg = np.mean(y) # average value of raw data
        yearsCount = math.ceil(n / 12)
        totalYear = [sum(y[i:i + 12]) for i in range(n - 11)]
        movingAvg = [x / 12 for x in totalYear]
        movingAvgMid = [(movingAvg[i] + movingAvg[i + 1]) / 2 for i in range(len(movingAvg) - 1)]
        S = [y[i + 2] - movingAvgMid[i] for i in range(len(movingAvgMid))]

        _S = [0, 0] + [x for x in S] + [0 for _ in range(n - len(S) - 2)]
        table = np.matrix([_S[12 * i: 12 * i + 12] for i in range(yearsCount)])
        total = [float(sum(table[:, i])) for i in range(12)]
        Ssr = [t / 3 for t in total]
        k = sum(Ssr) / 12
        Sadj = [s - k for s in Ssr]

        S = yearsCount * Sadj
        TE = [(y[i] - S[i]) for i in range(n)]
        b = ((sum([TE[i] * t[i] for i in range(n)]) / n) - (np.mean(TE) * np.mean(t))) / ((sum([i ** 2 for i in t]) / n)
                                                                                          - (np.mean(t) ** 2))
        a = (sum(TE) / n) - b * (sum(t) / n)
        T = [(a + b * i) for i in t]
        ST = np.array(S) + np.array(T)

        absErr = np.mean([abs((y[i] - ST[i]) * 100 / y[i]) for i in range(n)])
        sigma = math.sqrt(np.mean([((y[i] - ST[i]) * 100 / y[i]) ** 2 for i in range(n)]))
        rSqr = (1 - sum([(y[i] - ST[i]) ** 2 for i in range(n)]) / sum([(y[i] - avg) ** 2 for i in range(n)])) * 100
        residuals = [(y[i] - ST[i]) for i in range(n)]
        dw = (sum([(residuals[i] - residuals[i - 1]) ** 2 for i in range(1, n)])) / sum([r ** 2 for r in residuals])

        predt = [(n + (n % 12) + i + 1) for i in range(self.predictCount)]
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
        pass

    def dvModel(self):
        pass