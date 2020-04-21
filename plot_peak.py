import heartpy as hp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.signal import resample
from math import pi
from heartpy import visualizeutils

df = pd.read_csv('data.csv', header=0, sep=',', quotechar="'",)
data = df['Abdomen_3']
sample_rate = 300
num = 0

my_path = "ECG/Abdomen_3"


def started(data):

#первоначальный вид графика по точкам
    plt.figure(figsize=(12, 4))
    plt.title('Start data')
    plt.plot(data)


def baseline(data, sample_rate):

    # фильтр Notch для удаления базового отклонения от (особенно) сигналов ЭКГ
    filtered = hp.remove_baseline_wander(data, sample_rate)
    plt.figure(figsize=(12, 4))
    plt.title('Without baseline wander')
    plt.plot(filtered)

    return filtered


def filter_signal(filtered, sample_rate, data):

    filtered = hp.filter_signal(filtered, cutoff=0.05, sample_rate=sample_rate, filtertype='notch')
    #T-волна + QRS, но мы заинтересованы только в  QRS
    #фильтр - избавляемся от ненужных компонентов, не задевая QRS
    plt.figure(figsize=(12, 4))
    plt.title('Original and Filtered signal')
    plt.plot(data, label='original data')
    plt.plot(filtered, alpha=0.5, label='filtered data')
    plt.legend()

    return filtered


def resamples(data, filtered, sample_rate):

    resampled_data = resample(data, len(filtered))
    #scale_data - функция, которая масштабирует передаваемые данные так, чтобы они указали границы
    #в любом другом исполнении не работает
    wd, m = hp.process(hp.scale_data(resampled_data), sample_rate * 2)
    plt.figure(figsize=(12, 4))
    # hp.plotter(wd, m)

    return wd, m


def get_image(wd, data, num):

    for item in wd['RR_indices']:

        y = list(data[item[0]:item[1]])
        y = np.array(y)
        y = y + 90

        T = len(y) - 1
        t = np.linspace(0, len(y), num=len(y))
        y_t = y * np.sin((2 * pi / T) * t)

        x = np.array(range(item[0], item[1], 1))
        T = len(x) - 1
        t = np.linspace(0, len(x), num=len(x))
        x_t = y * np.cos((2 * pi / T) * t)

        plt.figure(figsize=(8, 8))
        plt.plot(x_t, y_t)
        # plt.show()
        plt.savefig(os.path.join(my_path, "{0:04d}.png".format(num)))

        num += 1


if __name__ == '__main__':
    started(data)
    filtered = baseline(data, sample_rate)
    filtered = filter_signal(filtered, sample_rate, data)
    wd, m = resamples(data, filtered, sample_rate)
    get_image(wd, data, num)
