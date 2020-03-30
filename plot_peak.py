import heartpy as hp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import resample
import numpy as np
from math import pi
from heartpy import visualizeutils

df = pd.read_csv('data.csv', header=0, sep=',', quotechar="'",)
print(df.head(10))
#print(df['Abdomen_4'])
print(df.iloc[:, 0:2])

#data = hp.get_data('data.csv')


data = df['Abdomen_4']
sample_rate = 300
#первоначальный вид графика по точкам
plt.figure(figsize=(12, 4))
plt.title('Start data')
plt.plot(data)
plt.show()

filtered = hp.remove_baseline_wander(data, sample_rate)
#фильтр Notch для удаления базового отклонения от (особенно) сигналов ЭКГ
plt.figure(figsize=(12, 4))
plt.title('Without baseline wander')
plt.plot(filtered)
plt.show()


filtered = hp.filter_signal(filtered, cutoff=0.05, sample_rate=sample_rate, filtertype='notch')
#T-волна + QRS, но мы заинтересованы только в  QRS
#фильтр - избавляемся от ненужных компонентов, не задевая QRS
plt.figure(figsize=(12, 4))
plt.title('Original and Filtered signal')
plt.plot(data, label='original data')
plt.plot(filtered, alpha=0.5, label='filtered data')
plt.legend()
plt.show()


resampled_data = resample(data, len(filtered))
#scale_data - функция, которая масштабирует передаваемые данные так, чтобы они указали границы
#в любом другом исполнении не работает
wd, m = hp.process(hp.scale_data(resampled_data), sample_rate * 2)
#получение пиков в сигнале
plt.figure(figsize=(12, 4))
hp.plotter(wd, m)


# for measure in m.keys():
#
#     print('%s: %f' % (measure, m[measure]))
#for w in wd.keys():

for sure in wd.keys():

    print(sure, wd[sure])


print('RR_indices', wd['RR_indices'][0])
print(len(wd["RR_indices"]))

plt.plot(data[180:648])
plt.show()

y = list(data[180:648])

y = np.array(y)
y = y + 100
T = len(y) - 1
t = np.linspace(0, len(y), num=len(y))
y_t = y * np.sin((2*pi/T)*t)


x = np.array(range(180, 648, 1))
T = len(x) - 1
t = np.linspace(0, len(x), num=len(x))
x_t = y * np.cos((2*pi/T)*t)


plt.plot(x_t, y_t)
plt.show()