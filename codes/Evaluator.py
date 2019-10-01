from __future__ import print_function, division, absolute_import
import os
import numpy as np
np.random.seed(1)
from tensorflow import set_random_seed
set_random_seed(1)
import pandas as pd
from keras.callbacks import  Callback, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from utils import log_macc, results_log




def eval(y_val,y_pred,y_predDom,val_parts,val_files,val_wav_files,foldname):
    true = []
    pred = []
    col = ["Wave file", "True", "Prediction", "Beats","Wrong predictions", "False Confidense"]

    start_idx = 0
    df = []
    predvalue = 0
    truevalue = 0
    y_pred = np.argmax(y_pred, axis=-1)
    y_val = np.transpose(np.argmax(y_val, axis=-1))

    for s,w in zip(val_parts,val_wav_files):
        files = []
        files.append(w)
        temp_T = y_val[start_idx:start_idx + int(s)]
        temp = y_pred[start_idx:start_idx + int(s)]
        normal = sum(temp==0)
        abnormal = sum(temp==1)
        if(sum(temp_T == 0) > sum(temp_T == 1)):
            true.append(0)
            truevalue = 0
        else:
            true.append(1)
            truevalue = 1

        files.append("T" if(truevalue) else "F")

        if(sum(temp == 0) > sum(temp == 1)):
            pred.append(0)
            predvalue = 0
        else:
            pred.append(1)
            predvalue = 1
        files.append("T" if(predvalue) else "F")
        files.append(s)
        files.append(normal if(truevalue) else abnormal)	#Wrong predictions
        files.append(int(100*normal/s) if(truevalue) else int(100*abnormal/s))	#"False Confidense"
        df.append(files)
        start_idx = start_idx + int(s)

    df = pd.DataFrame(df,columns = col)
    df.set_index('Wave file')
    df.to_csv('../../Adversarial Heart Sound Results/confidence/'+foldname+'.csv',index=False)
    eps = 1.1e-5
    TN, FP, FN, TP = confusion_matrix(true, pred, labels=[0,1]).ravel()
    sensitivity = TP / (TP + FN + eps)
    specificity = TN / (TN + FP + eps)
    precision = TP / (TP + FP + eps)
    F1 = 2 * (precision * sensitivity) / (precision + sensitivity + eps)
    Macc = (sensitivity + specificity) / 2

    print("Macc - ",Macc)
    print("F1 - ",F1)
    print("sensitivity - ",sensitivity)
    print("specificity - ",specificity)
    print("precision - ",precision)