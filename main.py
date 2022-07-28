import numpy as np
import matplotlib.pyplot as plt
import time_series as ts

def main():
    data = np.loadtxt('dataset.txt')[:, 1]
    predictionCount = 3 # months

    timeSeries = ts.TimeSeries(data, predictionCount)

    am = [data[-1]] + list(timeSeries.addModel()['predict'])
    mm = [data[-1]] + list(timeSeries.multModel()['predict'])
    dv = [data[-1]] + list(timeSeries.dvModel()['predict'])

    n = len(data)
    dataTime = list(range(1, n + 1))
    predictTime = list(range(n, n + 1 + predictionCount))

    plt.plot(dataTime, data, '-b')
    plt.plot(predictTime, am, '--r')
    plt.plot(predictTime, mm, '--g')
    plt.plot(predictTime, dv, '--k')
    plt.show()


if __name__ == '__main__':
    main()