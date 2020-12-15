import keras
import matplotlib.pyplot as plt
import cv2
import numpy as np

seed_value = 0
np.random.seed(seed_value)

from keras.layers import Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.layers.core import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, \
    classification_report, confusion_matrix, accuracy_score, f1_score
from imblearn.metrics import sensitivity_specificity_support
from sklearn import metrics
from keras.optimizers import Adam
from os.path import join, isfile
from os import listdir

path_to_data = 'for_CNN/all'


def get_optimal_threshold(prediction, testY):

    j = -1e-7
    best_item = []

    for item in np.linspace(np.min(prediction), np.max(prediction), 1000):

        pred = np.where(prediction > item, 1, 0)
        metric = f1_score(testY, pred)

        if metric > j:

            j = metric
            best_item.append(item)

    return best_item[-1]


def get_data(path):

    m = []

    for patient in listdir(path):

        matrix_image = []

        for filename in [f for f in listdir(join(path, patient)) if isfile(join(path, patient, f))]:

            image = cv2.imread(join(path, patient, filename))

            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            image = cv2.resize(image, (64, 64))

            image = image / 255.

            matrix_image.append(image)

        matrix_output = np.stack([x for l in matrix_image for x in l])

        matrix_output = [([x for l in matrix_output for x in l])]

        m.append(matrix_output[0])

    return m


def model(data):

    y = [0]*14 + [1]*11
    # y = [0] * 168 + [1] * 132

    trainX, testX, trainY, testY = train_test_split(data, y, test_size=0.25, random_state=100)

    model = keras.models.Sequential()

    model.add(Conv2D(32, (3, 3),  padding='same', input_shape=(64, 64, 12)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    model.add(Conv2D(filters=256, kernel_size=(9, 9), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2,  2), strides=(2, 2), padding='valid'))

    model.add(Flatten())

    model.add(Dense(4096, input_shape=(4096,)))
    model.add(Activation('relu'))

    model.add(Dropout(0.4))

    model.add(Dense(4096))
    model.add(Activation('relu'))

    model.add(Dropout(0.4))

    model.add(Dense(100))
    model.add(Activation('relu'))

    model.add(Dropout(0.4))

    model.add(Dense(17))
    model.add(Activation('relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(optimizer=Adam(lr=1e-4), loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

    EPOCHS = 20

    trainX = np.expand_dims(np.array(trainX), axis=3)
    testX = np.expand_dims(np.array(testX), axis=3)

    print(trainX.shape)

    n_samples, num, chanel = trainX.shape
    trainX = trainX.reshape((n_samples, num // (12*64), num // (12*64), num // (64**2)))

    n_samples, num, chanel = testX.shape
    testX = testX.reshape((n_samples, num // (12*64), num // (12*64), num // (64**2)))

    print(testY)

    H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=32, verbose=2)
    print("[INFO] evaluating network...")

    predictions = model.predict(testX, batch_size=32)

    # predictions = np.where(np.array(predictions) > 0.5, 1, 0)

    optimal_threshold = get_optimal_threshold(np.array(predictions), testY)

    print('optimal_threshold', optimal_threshold)

    predictions = np.where(np.array(predictions) > optimal_threshold, 1, 0)

    print(predictions)

    precision, recall, fscore, support = precision_recall_fscore_support(testY, predictions)
    _, specificity, _ = sensitivity_specificity_support(testY, predictions)
    print('Accuracy', accuracy_score(testY, predictions))
    print('binary precision value', precision[1])
    print('binary recall value', recall[1])
    print('binary fscore value', fscore[1])
    print('binary specificity value', specificity[1])

    print(classification_report(testY, predictions))

    draw_loss(EPOCHS, H)

    plt.figure(figsize=(5, 5))
    fpr, tpr, thresholds = roc_curve(testY, predictions)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('CNN', roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Correctly defined pixels')
    plt.ylabel('Fallaciously defined pixels')
    plt.legend(loc=0, fontsize='small')
    plt.title("ROC - curve")
    plt.show()


    plt.figure(figsize=(8, 8))
    precision, recall, thresholds = metrics.precision_recall_curve(testY, predictions)
    plt.plot(recall, precision)
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.title("Curve dependent Precision и Recall of threshold")
    plt.legend(loc='best')
    plt.show()


def draw_loss(EPOCHS, H):
    N = np.arange(0, EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["loss"], label="train_loss", linestyle='--')
    plt.plot(N, H.history["val_loss"], label="val_loss", linestyle='-.')
    plt.plot(N, H.history["acc"], label="train_acc", linestyle= ':')
    plt.plot(N, H.history["val_acc"], label="val_acc",  linestyle='-')
    plt.title("Training Loss and Accuracy (Simple NN)")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()
    # plt.savefig("Training Loss and Accuracy.pdf")


def main():
    matrix_output = get_data(path_to_data)
    model(matrix_output)


if __name__ == '__main__':
    main()

