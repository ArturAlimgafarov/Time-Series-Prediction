import numpy as np
import matplotlib.pyplot as plt
import time_series as ts

def main():
    data = np.loadtxt('dataset.txt')[:, 1]
    predictionCount = 3 # months

    timeSeries = ts.TimeSeries(data, predictionCount)

    am = data[-1] + timeSeries.addModel()['predict']
    mm = data[-1] + timeSeries.multModel()['predict']
    dv = data[-1] + timeSeries.dvModel()['predict']

    n = len(data)
    dataTime = list(range(1, n + 1))
    predictTime = list(range(n, n + 1 + predictionCount))

    print(dataTime)
    print(predictTime)

if __name__ == '__main__':
    main()