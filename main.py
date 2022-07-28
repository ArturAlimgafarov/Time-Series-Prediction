import numpy as np
import matplotlib.pyplot as plt
import time_series as ts

def main():
    data = np.loadtxt('dataset.txt')[:, 1]
    predictionCount = 3 # months

    timeSeries = ts.TimeSeries(data, predictionCount)

    am = timeSeries.addModel()
    mm = timeSeries.multModel()
    dv = timeSeries.dvModel()

    print(f'{am}\n{mm}\n{dv}')

if __name__ == '__main__':
    main()