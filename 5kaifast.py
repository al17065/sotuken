import soundfile as sf
import numpy as np
from matplotlib import pyplot as plt
import wave
import glob
import os
from sklearn import svm
from sklearn.model_selection import KFold
import joblib


files = glob.glob("./5kaifast/*.wav")
# 0:0, 1:25, 2:50, 3:75, 4:100
DIM_OF_FEATURE = 17640
NUM_OF_FILES = len(files)
data = np.zeros((NUM_OF_FILES, DIM_OF_FEATURE))
label = np.zeros((NUM_OF_FILES))

for i, file in enumerate(files):
    wdata, samplerate = sf.read(file)
    peak5 = []
    peaksum = []

    x = 17640
    for j in range(5):
        peak5.append(wdata[x-4410:x+13230])
        x = x + 17640

    peaksum = (peak5[0] + peak5[1] + peak5[2] + peak5[3] + peak5[4])

    x = np.fft.fft(peaksum)
    data[i, :] = np.abs(x)

    if "100-" in file:
        label[i] = 4
    elif "25-" in file:
        label[i] = 1
    elif "50-" in file:
        label[i] = 2
    elif "75-" in file:
        label[i] = 3
    elif "0-" in file:
        label[i] = 0

kf = KFold(n_splits = NUM_OF_FILES)
correctNum = 0
for train_index, test_index in kf.split(data, label):
    train_data = data[train_index]
    train_label = label[train_index]
    test_data = data[test_index]
    test_label = label[test_index]
    # 0:kara, 1:25%, 2:50%, 3:75% 4:full
    clf = svm.SVC()
    clf.fit(train_data, train_label)
    pred = clf.predict(test_data)
    for i in range(len(pred)):
        if pred[i] == test_label[i] : correctNum += 1
    print("seikai ritsu = ", correctNum / (i+1))
    correctNum = 0
