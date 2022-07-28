import numpy as np
import math
from sklearn import linear_model

# загрузка исходных данных
data = np.loadtxt('dataset.txt')
predCount = int(input('Введите количество прогнозов: '))

print('\nШАГ 1.')
t = data[:, 0]
Y = data[:, 1]
meanY = np.mean(Y) # среднее значение наблюдаемой величины
N = len(Y) # количество наблюдений
M = math.ceil(N / 12) # количество лет

# матрица исходных данных
dataMatrix = np.zeros((N, 12), dtype=float)
for i in range(N):
    dataMatrix[i, 0] = i + 1
for i in range(M):
    for j in range(11):
        dataMatrix[12 * i + j][j + 1] = 1
print(dataMatrix)


print('\nШАГ 2.')
clf = linear_model.LinearRegression()
clf.fit(dataMatrix, Y)
txt = ['(intercept)', 't', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11']
cfs = [clf.intercept_] + list(clf.coef_)
summary = zip(txt, cfs)
for row in summary:
    print('{: <11}: {: >15}'.format(*row))

print('\nШАГ 3.')
_Y = [(np.dot(xRow, clf.coef_) + clf.intercept_) for xRow in dataMatrix]
absErr = np.mean([abs((Y[i] - _Y[i]) / Y[i]) * 100 for i in range(N)]) # абсолютная ошибка прогноза
sigma = math.sqrt(np.mean([((Y[i] - _Y[i]) * 100 / Y[i]) ** 2 for i in range(N)])) # среднеквадратическая ошибка
rSqr = (1 - sum([(Y[i] - _Y[i]) ** 2 for i in range(N)]) / sum([(Y[i] - meanY) ** 2 for i in range(N)])) * 100
residuals = [(Y[i] - _Y[i]) for i in range(N)]
dw = (sum([(residuals[i] - residuals[i - 1]) ** 2 for i in range(1, N)]) + residuals[0] ** 2) / sum([r ** 2 for r in residuals])
print('Абсолютная ошибка прогноза: ', absErr)
print('Среднеквадратическая ошибка: ', sigma)
print('Коэффициент детерминации: ', rSqr)
print('Коэффициент Дарбина-Уотсона: ', dw)

newN = N + 12 * (predCount // 12 + 1)
M = math.ceil(newN / 12)
dataMatrix = np.zeros((newN, 12), dtype=float)
for i in range(newN):
    dataMatrix[i, 0] = i + 1
for i in range(M):
    for j in range(11):
        dataMatrix[12 * i + j][j + 1] = 1
predY = [(np.dot(dataMatrix[i], clf.coef_) + clf.intercept_) for i in range(N, N + predCount)]
predY = [round(item, 4) for item in predY]
print('Спрогнозированные значения на ' + str(predCount) + ' месяца: ', predY)
