import heartpy as hp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import resample

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

for measure in m.keys():

    print('%s: %f' % (measure, m[measure]))
print(wd)
