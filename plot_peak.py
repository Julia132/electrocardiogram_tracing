import heartpy as hp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.signal import resample
from math import pi
from os import listdir
from os.path import isfile, join
from heartpy import visualizeutils
from heartpy import exceptions

sample_rate = 300
num = 0

path_csv = 'patients'
path_image = "results"


def started(data):
    """
    функция для отрисовки исходного сигнала, для наглядности. может пригодится при публикации
    :param data: исходные данные
    :return:
    """
    plt.figure(figsize=(12, 4))
    plt.title('Start data')
    plt.plot(data)
    plt.show()


def baseline(data, sample_rate):
    """
    для фильтрации с фильтром Notch -  для удаления базового отклонения от (особенно) сигналов ЭКГ
    :param data: начальные данные
    :param sample_rate: шаг
    :return: отфильтрованный сигнал
    """

    filtered = hp.remove_baseline_wander(data, sample_rate)
    # plt.figure(figsize=(12, 4))
    # plt.title('Without baseline wander')
    # plt.plot(filtered)

    return filtered


def filter_signal(filtered, sample_rate, data):
    """
    функция для избавляения от ненужных компонентов, не задевая QRS
    :param filtered: отфильтрованный ранее сигнал
    :param sample_rate: шаг
    :param data: первоначальные данные, используются для сранении при прописовке графиков
    :return: отфильтрованный сигнал
    """

    filtered = hp.filter_signal(filtered, cutoff=0.05, sample_rate=sample_rate, filtertype='notch')
    # plt.figure(figsize=(12, 4))
    # plt.title('Original and Filtered signal')
    # plt.plot(data, label='original data')
    # plt.plot(filtered, alpha=0.5, label='filtered data')
    # plt.legend()

    return filtered


def resamples(data, filtered, sample_rate):
    """
    функция для нормализации сигнала
    :param data: исходные данные
    :param filtered: отфильтрованный сигнал
    :param sample_rate: шаг
    :return: wd -содержит пары пиков и множество других характеристик сигнала, m - статистика по сигналу
    """

    resampled_data = resample(data, len(filtered))
    #scale_data - функция, которая масштабирует передаваемые данные так, чтобы они указали границы
    #в любом другом исполнении не работает
    wd, m = hp.process(hp.scale_data(resampled_data), sample_rate * 2)
    # plt.figure(figsize=(12, 4))
    # hp.plotter(wd, m)

    return wd, m


def get_image(wd, data, num, st_name, name_patient, path_image):
    """
    функция для преобразования сигнала в изображение
    :param wd: переменная, содержащая пары пиков и множество других характеристик сигнала
    :param data: исходные данные
    :param num: счетчик, для нумерации изображений
    :param st_name: имя датчика, нужно для дополнения путя
    :param name_patient: название пациента, нужно для дополнения путя
    :param path_image: путь до общей папки
    :return:
    """

    for item in wd['RR_indices']:

        y = list(data[item[0]:item[1]])
        y = np.array(y)
        y = y + 2

        if len(y) > 1:
            T = len(y) - 1
        else:
            T = len(y)
        t = np.linspace(0, len(y), num=len(y))
        y_t = y * np.sin((2 * pi / T) * t)

        x = np.array(range(item[0], item[1], 1))
        if len(x) > 1:
            T = len(x) - 1
        else:
            T = len(x)
        t = np.linspace(0, len(x), num=len(x))
        x_t = y * np.cos((2 * pi / T) * t)

        # plt.figure(figsize=(8, 8))
        plt.figure()
        plt.plot(x_t, y_t)
        # plt.show()

        path = os.path.join(path_image, name_patient, st_name)
        plt.savefig(os.path.join(path, "{0:04d}.png".format(num)))
        plt.cla()
        plt.clf()
        plt.close('all')

        num += 1


if __name__ == '__main__':

    for patient in [f for f in listdir(path_csv) if isfile(join(path_csv, f))]:

        name_patient = os.path.splitext(join(patient))[0]

        print(name_patient)

        data_patient = pd.read_csv(join(path_csv, patient), header=0, sep=',', quotechar="'")

        list_datch = [data_patient['i'], data_patient['ii'], data_patient['iii'], data_patient['avr'],
                      data_patient['avl'], data_patient['avf'], data_patient['v1'], data_patient['v2'],
                      data_patient['v3'], data_patient['v4'], data_patient['v5'], data_patient['v6']]

        for item in list_datch:
            # started(item)
            filtered = baseline(item, sample_rate)
            filtered = filter_signal(filtered, sample_rate, item)
            try:
                wd, m = resamples(item, filtered, sample_rate)
                get_image(wd, item, num, item.name, name_patient, path_image)
            except exceptions.BadSignalWarning as exception:
                print("Warning! Exception occured!")
                print(exception)
