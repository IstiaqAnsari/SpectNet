from __future__ import print_function, division, absolute_import
import os
import numpy as np
np.random.seed(1)
from tensorflow import set_random_seed
set_random_seed(1)
import pandas as pd
from keras.callbacks import  Callback, ReduceLROnPlateau,LearningRateScheduler
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from keras import backend as K
import tensorflow as tf
from keras.utils import to_categorical

class log_macc(Callback):

    def __init__(self, val_parts,decision='majority',verbose=0, val_files=None,checkpoint_name=None,wav_files=None):
        super(log_macc, self).__init__()
        self.val_parts = val_parts
        self.decision = decision
        self.verbose = verbose
        self.val_files = np.asarray(val_files)
        self.wav_files = wav_files
        self.checkpoint_name = checkpoint_name
        print("Check point ", checkpoint_name)
        # self.x_val = x_val
        # self.y_val = y_val

    def on_epoch_end(self, epoch, logs):
        eps = 1.1e-5
        if logs is not None:
            y_pred = self.model.predict(self.validation_data[0], verbose=self.verbose)

            # Handling multiple outputs of model and taking into account only the classifier
            if isinstance(y_pred, list):
                y_pred_domain = y_pred[1]
                y_pred = y_pred[0]
            true = []
            pred = []
            files = []
            start_idx = 0

            if self.decision == 'majority':
                
                y_pred = np.argmax(y_pred, axis=-1)
                y_val = np.transpose(np.argmax(self.validation_data[1], axis=-1))

                
                for j,s in enumerate(self.val_parts):

                    if not s:  ## for e00032 in validation0 there was no cardiac cycle
                        continue
                    # ~ print "part {} start {} stop {}".format(s,start_idx,start_idx+int(s)-1)

                    temp_ = y_val[start_idx:start_idx + int(s)]
                    temp = y_pred[start_idx:start_idx + int(s)]

                    if (sum(temp == 0) > sum(temp == 1)):
                        pred.append(0)
                    else:
                        pred.append(1)

                    if (sum(temp_ == 0) > sum(temp_ == 1)):
                        true.append(0)
                    else:
                        true.append(1)

                    if self.val_files is not None:
                        files.append(self.val_files[start_idx])

                    start_idx = start_idx + int(s)

            if self.decision =='confidence':
                y_val = np.transpose(np.argmax(self.validation_data[1], axis=-1))
                for s in self.val_parts:
                    if not s:  ## for e00032 in validation0 there was no cardiac cycle
                        continue
                    # ~ print "part {} start {} stop {}".format(s,start_idx,start_idx+int(s)-1)
                    temp_ = y_val[start_idx:start_idx + int(s) - 1]
                    if (sum(temp_ == 0) > sum(temp_ == 1)):
                        true.append(0)
                    else:
                        true.append(1)
                    temp = np.sum(y_pred[start_idx:start_idx + int(s) - 1],axis=0)
                    pred.append(int(np.argmax(temp)))
                    start_idx = start_idx + int(s)
            if self.decision == 'match':
                y_val = np.transpose(np.argmax(self.validation_data[1], axis=-1))
                y_pred = np.argmax(y_pred, axis=-1)
                for v in y_val:
                  true.append(v)
                for v in y_pred:
                  pred.append(v)

            TN, FP, FN, TP = confusion_matrix(true, pred, labels=[0,1]).ravel()
            # TN = float(TN)
            # TP = float(TP)
            # FP = float(FP)
            # FN = float(FN)
            sensitivity = TP / (TP + FN + eps)
            specificity = TN / (TN + FP + eps)
            precision = TP / (TP + FP + eps)
            F1 = 2 * (precision * sensitivity) / (precision + sensitivity + eps)
            Macc = (sensitivity + specificity) / 2
            logs['learning_rate'] = K.get_value(self.model.optimizer.lr)

            #logs['lambda_rate'] = K.get_value(self.model.get_layer('grl').hp_lambda)

            logs['val_sensitivity'] = np.array(sensitivity)
            logs['val_specificity'] = np.array(specificity)
            logs['val_precision'] = np.array(precision)
            logs['val_F1'] = np.array(F1)
            logs['val_macc'] = np.array(Macc)
            logs['acc_wav'] = np.array((TN+TP)/(TN+TP+FP+FN))
            # print(logs.keys())
            if('val_class_acc' in logs.keys()):
                logs['model_path'] = self.checkpoint_name.format(epoch=epoch+1, val_acc=logs['val_class_acc'])   ## added one with epoch for correct indexing
            elif('val_acc' in logs.keys()):
                logs['model_path'] = self.checkpoint_name.format(epoch=epoch+1, val_acc=logs['val_acc'])   ## added one with epoch for correct indexing
            if self.verbose:
                print("TN:{},FP:{},FN:{},TP:{},Macc:{},F1:{}".format(TN, FP, FN, TP,Macc,F1))

            #### Learning Rate for Adam ###
            # if self.model.optimizer == 'Adam':
            #
            #     lr = self.model.optimizer.lr
            #     if self.model.optimizer.initial_decay > 0:
            #         lr *= (1. / (1. + self.model.optimizer.decay * K.cast(self.model.optimizer.iterations,
            #                                                               K.dtype(self.model.optimizer.decay))))
            #     t = K.cast(self.model.optimizer.iterations, K.floatx()) + 1
            #     lr_t = lr * (
            #             K.sqrt(1. - K.pow(self.model.optimizer.beta_2, t)) / (1. - K.pow(self.model.optimizer.beta_1, t)))
            #     logs['lr'] = np.array(float(K.get_value(lr_t)))

            if self.val_files is not None:
                true = np.asarray(true)
                pred = np.asarray(pred)
                files = np.asarray(files)
                tpn = true == pred
                for dataset in set(files):
                    mask = files == dataset
                    logs['acc_'+dataset] = np.sum(tpn[mask])/np.sum(mask)
                    # mask = self.val_files=='x'
                    # TN, FP, FN, TP = confusion_matrix(np.asarray(true)[mask], np.asarray(pred)[mask], labels=[0, 1]).ravel()
                    # sensitivity = TP / (TP + FN + eps)
                    # specificity = TN / (TN + FP + eps)
                    # logs['ComParE_UAR'] = (sensitivity + specificity) / 2

def results_log(results_path,log_dir,log_name,activation_function,addweights,kernel_size,maxnorm,
                dropout_rate,dropout_rate_dense,l2_reg,l2_reg_dense,batch_size,lr,bn_momentum,lr_decay,num_dense,comment,num_filt,opt=Adam,outlayer='',):
    df = pd.read_csv(results_path)
    df1 = pd.read_csv(log_dir + log_name + '/training.csv')
    max_idx = df1['val_macc'].idxmax()
    print("#"*200)
    print(df1.columns)
    val_acc = "val_"+outlayer+"acc"
    acc = outlayer + "acc"
    new_entry = {'Filename': log_name, 'Weight Initialization': 'he_normal',
                 'Activation': activation_function + '-softmax', 'Class weights': addweights,
                 'Kernel Size': kernel_size, 'Max Norm': maxnorm,
                 'Dropout -filters': dropout_rate,
                 'Dropout - dense': dropout_rate_dense,
                 'L2 - filters': l2_reg, 'L2- dense': l2_reg_dense,
                 'Batch Size': batch_size, 'Optimizer': opt.__name__, 'Learning Rate': lr,
                 'BN momentum': bn_momentum, 'Lr decay': lr_decay,
                 'Best Val Acc Per Cardiac Cycle': df1.loc[max_idx][val_acc] * 100,
                 'Epoch': df1.loc[[max_idx]]['epoch'].values[0],
                 'Training Acc per cardiac cycle': df1.loc[max_idx][acc] * 100,
                 'Specificity': df1.loc[max_idx]['val_specificity'] * 100,
                 'Macc': df1.loc[max_idx]['val_macc'] * 100,
                 'Precision': df1.loc[max_idx]['val_precision'] * 100,
                 'Sensitivity': df1.loc[max_idx]['val_sensitivity'] * 100,
                 'Number of filters': str(num_filt),
                 'F1': df1.loc[max_idx]['val_F1'] * 100,
                 'Number of Dense Neurons': num_dense,
                 'Comment': comment}

    index, _ = df.shape
    new_entry = pd.DataFrame(new_entry, index=[index])
    df2 = pd.concat([df, new_entry], axis=0)
    # df2 = df2.reindex(df.columns)
    df2.to_csv(results_path, index=False)
    df2.tail()
    print("Saving to results.csv")

def Confused_Crossentropy(y_true, y_pred):
    y_predfused = tf.multiply(y_pred,0)+.5
    #y_predfused = tf.convert_to_tensor((np.ones((batch,num_class),dtype=np.float32)*0.5))
    #y_truefused = tf.convert_to_tensor( to_categorical(np.ones(batch),num_class) )
    return K.abs(K.categorical_crossentropy(y_true, y_pred)-K.categorical_crossentropy(y_true,y_predfused))

def Confused_Crossentropy(y_true, y_pred):
    y_predfused = tf.multiply(y_pred,0)+.5
    #y_predfused = tf.convert_to_tensor((np.ones((batch,num_class),dtype=np.float32)*0.5))
    #y_truefused = tf.convert_to_tensor( to_categorical(np.ones(batch),num_class) )
    return K.abs(K.categorical_crossentropy(y_true, y_pred)-K.categorical_crossentropy(y_true,y_predfused))

