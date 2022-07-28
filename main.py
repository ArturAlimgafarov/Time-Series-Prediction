import numpy as np
import matplotlib.pyplot as plt
import time_series as ts

def main():
    data = np.loadtxt('dataset.txt')[:, 1]
    predictionCount = 3 # months

    model = ts.TimeSeries(data, predictionCount)
    print(model.addModel())

if __name__ == '__main__':
    main()