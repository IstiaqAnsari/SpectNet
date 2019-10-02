from __future__ import print_function, division, absolute_import
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# set_session(tf.Session(config=config))
# from clr_callback import CyclicLR
# import dill
from BalancedDannAudioDataGenerator import BalancedAudioDataGenerator
import os,time
from scipy.io import loadmat
import numpy as np
np.random.seed(1)
from tensorflow import set_random_seed
set_random_seed(1)
import math
import pandas as pd
import tables,h5py
from datetime import datetime
import argparse
from keras.callbacks import TensorBoard, Callback, ReduceLROnPlateau
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger
from keras import backend as K
from keras.utils import plot_model
from Heartnet import heartnet 
from utils import log_macc, results_log
from dataLoader import reshape_folds
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
import Evaluator
import dataLoader
sns.set()

if __name__ == '__main__':
    try:
        ########## Parser for arguments (foldname, random_seed, load_path, epochs, batch_size)
        parser = argparse.ArgumentParser(description='Specify fold to process')
        parser.add_argument("test_domains",
                            help="which fold to use from balanced folds generated in /media/taufiq/Data/"
                                 "heart_sound/feature/potes_1DCNN/balancedCV/folds/")
        parser.add_argument("--train_domains",
                            help = "trainer domain ")
        parser.add_argument("--tune", type=float,
                            help="Tuner or data split test_split")
        parser.add_argument("--dann",type=float,
                            help = "if given dann is activated else zero")
        parser.add_argument("--seed", type=int,
                            help="Random seed for the random number generator (defaults to 1)")
        parser.add_argument("--loadmodel",
                            help="load previous model checkpoint for retraining (Enter absolute path)")
        parser.add_argument("--epochs", type=int,
                            help="Number of epochs for training")
        parser.add_argument("--batch_size", type=int,
                            help="number of minibatches to take during each backwardpass preferably multiple of 2")
        parser.add_argument("--verbose", type=int, choices=[1, 2],
                            help="Verbosity mode. 1 = progress bar, 2 = one line per epoch (default 2)")
        parser.add_argument("--classweights", type=bool,
                            help="if True, class weights are added according to the ratio of the "
                                 "two classes present in the training data")
        parser.add_argument("--comment",
                            help = "Add comments to the log files")
        parser.add_argument("--optim",
                            help = "Add comments to the log files")
        parser.add_argument("--type", type=int)
        parser.add_argument("--lr", type=float)
        parser.add_argument("--eval",type=bool)

        args = parser.parse_args()
        if args.tune:
            test_split = args.tune
        else:
            test_split = 0
        if args.eval:evaluate = args.eval
        else: evaluate = False
        print(evaluate)
        
        domain_list = 'abcdefghi'
        test_domains = args.test_domains
        train_domains = domain_list
        for c in test_domains:
            train_domains = train_domains.replace(c,"")
        if args.train_domains:
            train_domains = args.train_domains
        if(test_split==0):
            foldname = train_domains+"_"+test_domains
        else:
            foldname = train_domains+"_"+test_domains+"_tune_"+str(test_split)
        print("%s selected for training" % (train_domains))
        print("%s selected for validation" % (test_domains))
        print(foldname)

        optim = 'Adam'
        if args.optim:
            optim = args.optim
        if args.seed:  # if random seed is specified
            print("Random seed specified as %d" % (args.seed))
            random_seed = args.seed
        else:
            random_seed = 2

        if args.loadmodel:  # If a previously trained model is loaded for retraining
            load_path = args.loadmodel  #### path to model to be loaded

            idx = load_path.find("weights")
            initial_epoch = int(load_path[idx + 8:idx + 8 + 4])

            print("%s model loaded\nInitial epoch is %d" % (args.loadmodel, initial_epoch))
        else:
            print("no model specified, using initializer to initialize weights")
            initial_epoch = 0
            load_path = False

        if args.epochs:  # if number of training epochs is specified
            print("Training for %d epochs" % (args.epochs))
            epochs = args.epochs
        else:
            epochs = 400
            print("Training for %d epochs" % (epochs))

        if args.batch_size:  # if batch_size is specified
            print("Training with %d samples per minibatch" % (args.batch_size))
            batch_size = args.batch_size
        else:
            batch_size = 1020
            print("Training with %d minibatches" % (batch_size))

        if args.verbose:
            verbose = args.verbose
            print("Verbosity level %d" % (verbose))
        else:
            verbose = 1
        if args.classweights:
            addweights = True
        else:
            addweights = False
        if args.comment:
            comment = args.comment
        else:
            comment = None
        if args.type:
            type = args.type
        else:
            type = 3
        print("Type %d FIR selected as front-end" % type)
        if args.lr:
            lr = args.lr
        else:
            lr = 0.0012843784
        if args.dann:
            hp_lambda = np.float32(args.dann)
        else:
            hp_lambda = np.float32(0)

        #########################################################

        foldname = foldname
        random_seed = random_seed
        load_path = load_path
        initial_epoch = initial_epoch
        epochs = epochs
        batch_size = batch_size
        verbose = verbose
        type = type


        model_dir = '../../Adversarial Heart Sound Results/models/'
        fold_dir = '../../feature/potes_1DCNN/balancedCV/folds/folds_phys_compare_pascal/'
        fold_dir = '../../feature/potes_1DCNN/balancedCV/folds/all_folds_wav_name/'

        if(test_split>0):
            log_name = foldname +' Tuned '+str(int(test_split*100))+' '+ str(datetime.now())
        else:
            log_name = foldname + ' ' + str(datetime.now())

        log_dir = '../../Adversarial Heart Sound Results/logs/'
        if(args.dann):
            if(args.dann>0):
                model_dir = model_dir + 'dann/'
                log_dir = log_dir + 'dann/'
        if not os.path.exists(model_dir + log_name):
            if not evaluate:
                os.makedirs(model_dir + log_name)
        checkpoint_name = model_dir + log_name + "/" + 'weights.{epoch:04d}-{val_class_acc:.4f}.hdf5'
        results_path = '../../results_2class.csv'
        outlayer = 'class'

        num_filt = (8, 4)
        num_dense = 20

        bn_momentum = 0.99
        eps = 1.1e-5
        bias = False
        l2_reg = 0.04864911065093751
        l2_reg_dense = 0.
        kernel_size = 5
        maxnorm = 10000.
        dropout_rate = 0.5
        dropout_rate_dense = 0.
        padding = 'valid'
        activation_function = 'relu'
        subsam = 2
        FIR_train= True
        trainable = True
        decision = 'majority'  # Decision algorithm for inference over total recording ('majority','confidence','match')

        # lr =  0.0012843784 ## After bayesian optimization

        ###### lr_decay optimization ######
        lr_decay =0.0001132885
        # lr_decay =3.64370733503E-06
        # lr_decay =3.97171548784E-08
        ###################################



        lr_reduce_factor = 0.5
        patience = 4  # for reduceLR
        cooldown = 0  # for reduceLR
        res_thresh = 0.5  # threshold for turning probability values into decisions

        ############## Importing data ############

        num_class_domain = len(set(train_domains + test_domains))
        num_class = 2

        x_train, y_train, y_domain, train_parts,x_val, y_val, val_domain, val_parts, val_wav_files = dataLoader.getData(fold_dir,train_domains,test_domains,test_split)


        val_files = val_domain
        #Create meta labels and domain labels
        domains = train_domains
        if(test_split>0):
            domains = domains + test_domains
        domainClass = [(cls,dfc) for cls in range(2) for dfc in domains]
        
        print(domainClass)
        meta_labels = [domainClass.index((cl,df)) for (cl,df) in zip(y_train,y_domain)]

        y_domain = np.array([list(set(train_domains+test_domains)).index(lab) for lab in y_domain])
        val_domain = np.array([list(set(train_domains+test_domains)).index(lab) for lab in val_domain])

        ################### Reshaping ############
        [x_train, x_val], [y_train,y_domain,y_val] = reshape_folds([x_train,x_val],[y_train,y_domain,y_val])
        y_train = to_categorical(y_train, num_classes=num_class)
        y_domain = to_categorical(y_domain,num_classes=num_class_domain)
        y_val = to_categorical(y_val, num_classes=num_class)
        val_domain = to_categorical(val_domain,num_classes=num_class_domain)
        print("Train  files ", y_train.shape, "  Domain ", y_domain.shape)










        
        ############## Create a model ############







        ###  if evaluate is not selected  ####
        if(evaluate):
            print("Testing")
            print("Testing")
            print("Testing")
            print("Testing")

            weight = 'weights.0155-0.5229.hdf5'
            logname = 'abcdefgh_i_tune_0.3 Tuned 30 2019-09-26 14:24:11.245075'
            load_path = '../../Adversarial Heart Sound Results/models/'
            load_path = load_path +logname + '/' + weight
            model = heartnet(load_path,activation_function, bn_momentum, bias, dropout_rate, dropout_rate_dense,
                         eps, kernel_size, l2_reg, l2_reg_dense, lr, lr_decay, maxnorm,
                         padding, random_seed, subsam, num_filt, num_dense, FIR_train, trainable, type,num_class=num_class,num_class_domain=num_class_domain,hp_lambda=hp_lambda)
            y_pred,y_predDom = model.predict(x_val, verbose=verbose)
            Evaluator.eval(y_val,y_pred,y_predDom,val_parts,val_files,val_wav_files,foldname)
        else:
            print("Training")
            print("Training")
            print("Training")
            print("Training")

            model = heartnet(load_path,activation_function, bn_momentum, bias, dropout_rate, dropout_rate_dense,
                             eps, kernel_size, l2_reg, l2_reg_dense, lr, lr_decay, maxnorm,
                             padding, random_seed, subsam, num_filt, num_dense, FIR_train, trainable, type,num_class=num_class,num_class_domain=num_class_domain,hp_lambda=hp_lambda)
            model.summary()
            plot_model(model, to_file='model.png', show_shapes=True)
            model_json = model.to_json()
            with open(model_dir + log_name+"/model.json", "w") as json_file:
                json_file.write(model_json)

        ####### Define Callbacks ######

            modelcheckpnt = ModelCheckpoint(filepath=checkpoint_name,
                                            monitor='val_class_acc',save_best_only=False, mode='max') 
            tensbd = TensorBoard(log_dir=log_dir + log_name,
                                 batch_size=batch_size, histogram_freq = 3,
                                 write_grads=True,
                                 # embeddings_freq=99,
                                 # embeddings_layer_names=embedding_layer_names,
                                 # embeddings_data=x_val,
                                 # embeddings_metadata=metadata_file,
                                 write_images=False)
            csv_logger = CSVLogger(log_dir + log_name + '/training.csv')
            ######### Scheduler #################
            ## learning rate 
            def step_decay(epoch):
                
                lr0 = .00128437
                #print("learning rate , lr 0 ", lr, lr0)
                a = 1
                b = 1
                p = epoch/epochs
                lrate = lr0/math.pow((1+a*p),b)
                return lrate
            lrate = LearningRateScheduler(step_decay,verbose = 1)
            # Lambda for gradient reversal layer
            def f_hp_decay(epoch):
                minEpoch = 100
                if hp_lambda == 0:
                    return hp_lambda
                if epoch<minEpoch:
                    return np.float32(hp_lambda)
                gamma =  1
                p = (epoch-200) / (epochs)
                lam =  (2 / (1 + 1*(math.e ** (- gamma * p)))) - 1+.01  # 3 porjonto jaabe
                # hp_lambda = hp_lambda * (params['hp_decay_const'] ** global_epoch_counter)
                return np.float32(lam)
            class hpRateScheduler(Callback):
                """Learning rate scheduler.
                # Arguments
                    schedule: a function that takes an epoch index as input
                        (integer, indexed from 0) and current learning rate
                        and returns a new learning rate as output (float).
                    verbose: int. 0: quiet, 1: update messages.
                """
                def __init__(self, schedule, verbose=0):
                    super(hpRateScheduler, self).__init__()
                    self.schedule = schedule
                    self.verbose = verbose
                def on_epoch_begin(self, epoch,logs=None):
                    hp_lambda = self.schedule(epoch)
                    if not isinstance(hp_lambda, (float, np.float32, np.float64)):
                        raise ValueError('The output of the "schedule" function '
                                         'should be float.')
                    # K.set_value(self.model.layers[-3].hp_lambda, hp_lambda)
                    self.model.get_layer('grl').hp_lambda  = hp_lambda
                    if self.verbose > 0:
                        print('\nEpoch %05d: HP setting hp_lambda '
                              'rate to %s.' % (epoch + 1, hp_lambda))
            hprate = hpRateScheduler(f_hp_decay,verbose = 1)
            class MyCallback(Callback):
                def on_epoch_begin(self, epoch, logs=None):
                    lr = self.model.optimizer.lr
                    # If you want to apply decay.
                    decay = self.model.optimizer.decay
                    iterations = self.model.optimizer.iterations
                    lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
                    print(epoch, " Setting Learning rate: " , K.eval(lr_with_decay))
            trackLr = MyCallback()
            class TimeHistory(Callback):
                def on_train_begin(self, logs={}):
                    self.times = []

                def on_epoch_begin(self, epoch, logs={}):
                    self.epoch_time_start = time.time()

                def on_epoch_end(self, epoch, logs={}):
                    self.times.append(time.time() - self.epoch_time_start)
                    print("Epoch Time : ",time.time() - self.epoch_time_start)
            time_callback = TimeHistory()
            ######### Data Generator ############
            datagen = BalancedAudioDataGenerator(
                                         shift=.1,
                                         # roll_range=.1,
                                         # fill_mode='reflect',
                                         # featurewise_center=True,
                                         # zoom_range=.1,
                                         # zca_whitening=True,
                                         # samplewise_center=True,
                                         # samplewise_std_normalization=True,
                                         )
            # valgen = AudioDataGenerator(
            #     # fill_mode='reflect',
            #     # featurewise_center=True,
            #     # zoom_range=.2,
            #     # zca_whitening=True,
            #     # samplewise_center=True,
            #     # samplewise_std_normalization=True,
            # )
            flow = datagen.flow(x_train, [y_train,y_domain],
                                meta_label=meta_labels,
                                batch_size=batch_size, shuffle=True,
                                seed=random_seed)
            model.fit_generator(flow,
                                #steps_per_epoch=len(x_train) // batch_size,
                                steps_per_epoch=flow.steps_per_epoch,
                                # max_queue_size=20,
                                use_multiprocessing=False,
                                epochs=epochs,
                                verbose=verbose,
                                shuffle=True,
                                callbacks=[modelcheckpnt,hprate,trackLr,time_callback,
                                           log_macc(val_parts, decision=decision,verbose=verbose,val_files=val_files,wav_files=val_wav_files,checkpoint_name = checkpoint_name),
                                           tensbd, csv_logger],
                                validation_data=(x_val, [y_val,val_domain]),
                                initial_epoch=initial_epoch,
                                )


        

            ############### log results in csv ###############
            plot_model(model, to_file=log_dir + log_name + '/model.png', show_shapes=True)
            results_log(results_path=results_path, log_dir=log_dir, log_name=log_name,
                        activation_function=activation_function, addweights=addweights,
                        kernel_size=kernel_size, maxnorm=maxnorm,
                        dropout_rate=dropout_rate, dropout_rate_dense=dropout_rate_dense, l2_reg=l2_reg,
                        l2_reg_dense=l2_reg_dense, batch_size=batch_size,
                        lr=lr, bn_momentum=bn_momentum, lr_decay=lr_decay,
                        num_dense=num_dense, comment=comment,num_filt=num_filt,outlayer=outlayer)
            # print(model.layers[1].get_weights())
            # with K.get_session() as sess:
            #     impulse_gammatone = sess.run(model.layers[1].impulse_gammatone())
            # plt.plot(impulse_gammatone)
            # plt.show()

    except KeyboardInterrupt:
        ############ If ended in advance ###########
        if not evaluate:
            plot_model(model, to_file=log_dir + log_name + '/model.png', show_shapes=True)
            results_log(results_path=results_path, log_dir=log_dir, log_name=log_name,
                        activation_function=activation_function, addweights=addweights,
                        kernel_size=kernel_size, maxnorm=maxnorm,
                        dropout_rate=dropout_rate, dropout_rate_dense=dropout_rate_dense, l2_reg=l2_reg,
                        l2_reg_dense=l2_reg_dense, batch_size=batch_size,
                        lr=lr, bn_momentum=bn_momentum, lr_decay=lr_decay,
                        num_dense=num_dense, comment=comment,num_filt=num_filt,outlayer=outlayer)
        else:
            print("why would you interrupt evaluation -_- ")