import numpy as np
import math
import matplotlib.pyplot as plt

# загрузка исходных данных
data = np.loadtxt('dataset.txt')
predCount = int(input('Введите количество прогнозов: '))

plt.plot(data[:, 0], data[:, 1])
plt.show()
exit(0)

print('\nШАГ 1.')
t = data[:, 0]
Y = data[:, 1]
meanY = np.mean(Y) # среднее значение наблюдаемой величины
N = len(Y) # количество наблюдений
M = math.ceil(N / 12) # количество лет
# print(t)
# print(Y)

totalYear = [sum(Y[i:i+12]) for i in range(len(Y) - 11)]
movingAvg = [x / 12 for x in totalYear]
movingAvgMid = [(movingAvg[i] + movingAvg[i + 1]) / 2 for i in range(len(movingAvg) - 1)]
S = [Y[i + 2] - movingAvgMid[i] for i in range(len(movingAvgMid))]
print('Итого за год: ', totalYear) # итого за год
print('Скользящая средняя: ', movingAvg) # скользящая средняя
print('Центр. скользящая средняя: ', movingAvgMid) # центр. скользящая средняя
print('Оценка сезонной компоненты: ', S) # оценка сезонной компоненты


print('\nШАГ 2.')
_S = [0, 0] + [x for x in S] + [0 for _ in range(N - len(S) - 2)]
table = np.matrix([_S[12 * i : 12 * i + 12] for i in range(M)])
total = [float(sum(table[:, i])) for i in range(12)] # итого
Ssr = [t / 3 for t in total] # средняя оценка сезонной компоненты
k = sum(Ssr) / 12 # корректирующий коэффициент
Sadj = [s - k for s in Ssr] # скорректированный S
print('Итого(по месяцам): ', total)
print('Средняя оценка сезонной компоненты: ', Ssr)
print('Корректирующий коэффициент: ', k)
print('Скорректированная сезонная компонента: ', Sadj)

print('\nШАГ 3.')
S = M * Sadj
TE = [(Y[i] - S[i]) for i in range(N)]
b = ((sum([TE[i] * t[i] for i in range(N)]) / N) - (np.mean(TE) * np.mean(t))) / ((sum([i ** 2 for i in t]) / N)
                                                                                  - (np.mean(t) ** 2))
a = (sum(TE) / N) - b * (sum(t) / N)
T = [(a + b * i) for i in t]
ST = np.array(S) + np.array(T)
print('Коэффициенты тренда Т(a, b): ', a, b) # T = a + b * t
print('Расчетные значения: ', ST) # расчетные значения

print('\nШАГ 4.')
absErr = np.mean([abs((Y[i] - ST[i]) * 100 / Y[i]) for i in range(N)]) # абсолютная ошибка прогноза
sigma = math.sqrt(np.mean([((Y[i] - ST[i]) * 100 / Y[i]) ** 2 for i in range(N)])) # среднеквадратическая ошибка
rSqr = (1 - sum([(Y[i] - ST[i]) ** 2 for i in range(N)]) / sum([(Y[i] - meanY) ** 2 for i in range(N)])) * 100
residuals = [(Y[i] - ST[i]) for i in range(N)]
dw = (sum([(residuals[i] - residuals[i - 1]) ** 2 for i in range(1, N)])) / sum([r ** 2 for r in residuals])
print('Абсолютная ошибка прогноза: ', absErr)
print('Среднеквадратическая ошибка: ', sigma)
print('Коэффициент детерминации: ', rSqr)
print('Коэффициент Дарбина-Уотсона: ', dw)

print('\nШАГ 5.')
predt = [(N + (N % 12) + i + 1) for i in range(predCount)]
predS = [S[i] for i in range(predCount)]
predT = [(a + b * i) for i in predt]
predST = np.array(predS) + np.array(predT)
predST = [round(item, 4) for item in predST]
print('Спрогнозированные значения на ' + str(predCount) + ' месяца: ', predST)
