from custom_layers import Conv1D_zerophase_linear, Conv1D_linearphase, Conv1D_zerophase,\
    DCT1D, Conv1D_gammatone, Conv1D_linearphaseType, Attention
from keras.layers import Input, Conv1D, MaxPooling1D, Dense, Dropout, Flatten, Activation, AveragePooling1D, Add
from keras import initializers
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.layers.core import Lambda
from keras.regularizers import l2
from keras.constraints import max_norm
from keras.optimizers import Adam, SGD # Nadam, Adamax
import numpy as np
import tables,h5py
from Gradient_Reverse_Layer import GradientReversal
from ResultAnalyser import Result
from utils import Confused_Crossentropy
from keras.utils import plot_model
from keras import backend as K
def branch(input_tensor,num_filt,kernel_size,random_seed,padding,bias,maxnorm,l2_reg,
           eps,bn_momentum,activation_function,dropout_rate,subsam,trainable):

    num_filt1, num_filt2 = num_filt
    t = Conv1D(num_filt1, kernel_size=kernel_size,
                kernel_initializer=initializers.he_normal(seed=random_seed),
                padding=padding,
                use_bias=bias,
                kernel_constraint=max_norm(maxnorm),
                trainable=trainable,
                kernel_regularizer=l2(l2_reg))(input_tensor)
    t = BatchNormalization(epsilon=eps, momentum=bn_momentum, axis=-1)(t)
    t = Activation(activation_function)(t)
    t = Dropout(rate=dropout_rate, seed=random_seed)(t)
    return t
def zeropad(x):
    y = K.zeros_like(x)
    return K.concatenate([x, y], axis=2)

def zeropad_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 3
    shape[2] *= 2
    return tuple(shape)

def res_block(input_tensor,num_filt,kernel_size,stride,padding,random_seed,bias,maxnorm,l2_reg,
           eps,bn_momentum,activation_function,dropout_rate,subsam,trainable,cat=True):

    t = Conv1D(num_filt, kernel_size=kernel_size,
                kernel_initializer=initializers.he_normal(seed=random_seed),
                padding=padding,
                strides=stride,
                use_bias=bias,
                kernel_constraint=max_norm(maxnorm),
                trainable=trainable,
                kernel_regularizer=l2(l2_reg))(input_tensor)
    t = BatchNormalization(epsilon=eps, momentum=bn_momentum, axis=-1)(t)
    t = Activation(activation_function)(t)
    t = Dropout(rate=dropout_rate, seed=random_seed)(t)
    t = Conv1D(num_filt, kernel_size=kernel_size,
                kernel_initializer=initializers.he_normal(seed=random_seed),
                padding=padding,
                strides=1,
                use_bias=bias,
                kernel_constraint=max_norm(maxnorm),
                trainable=trainable,
                kernel_regularizer=l2(l2_reg))(t)
    t = BatchNormalization(epsilon=eps, momentum=bn_momentum, axis=-1)(t)
    t = Activation(activation_function)(t)
    t = Dropout(rate=dropout_rate, seed=random_seed)(t)
    
    p = MaxPooling1D(pool_size=stride)(input_tensor)
    if(stride>1):
        if(cat):
            p = Lambda(zeropad, output_shape=zeropad_output_shape)(p)
    
    t = Add()([t,p])
    return t

def heartnet(load_path,activation_function='relu', bn_momentum=0.99, bias=False, dropout_rate=0.5, dropout_rate_dense=0.0,
             eps=1.1e-5, kernel_size=5, l2_reg=0.0, l2_reg_dense=0.0,lr=0.0012843784, lr_decay=0.0001132885, maxnorm=10000.,
             padding='valid', random_seed=1, subsam=2, num_filt=(8, 4), num_dense=20,FIR_train=False,trainable=True,type=1,
             num_class=2, num_class_domain=1,hp_lambda=0,batch_size=1024):
    
    #num_dense = 20 default 
    input = Input(shape=(2500, 1))

    coeff_path = '../data/filterbankcoeff60.mat'
    coeff = tables.open_file(coeff_path)
    b1 = coeff.root.b1[:]
    b1 = np.hstack(b1)
    b1 = np.reshape(b1, [b1.shape[0], 1, 1])

    b2 = coeff.root.b2[:]
    b2 = np.hstack(b2)
    b2 = np.reshape(b2, [b2.shape[0], 1, 1])

    b3 = coeff.root.b3[:]
    b3 = np.hstack(b3)
    b3 = np.reshape(b3, [b3.shape[0], 1, 1])

    b4 = coeff.root.b4[:]
    b4 = np.hstack(b4)
    b4 = np.reshape(b4, [b4.shape[0], 1, 1])

    ## Conv1D_linearphase

    # input1 = Conv1D_linearphase(1 ,61, use_bias=False,
    #                 # kernel_initializer=initializers.he_normal(random_seed),
    #                 weights=[b1[30:]],
    #                 padding='same',trainable=FIR_train)(input)
    # input2 = Conv1D_linearphase(1, 61, use_bias=False,
    #                 # kernel_initializer=initializers.he_normal(random_seed),
    #                 weights=[b2[30:]],
    #                 padding='same',trainable=FIR_train)(input)
    # input3 = Conv1D_linearphase(1, 61, use_bias=False,
    #                 # kernel_initializer=initializers.he_normal(random_seed),
    #                 weights=[b3[30:]],
    #                 padding='same',trainable=FIR_train)(input)
    # input4 = Conv1D_linearphase(1, 61, use_bias=False,
    #                 # kernel_initializer=initializers.he_normal(random_seed),
    #                 weights=[b4[30:]],
    #                 padding='same',trainable=FIR_train)(input)

    ## Conv1D_linearphase Anti-Symmetric
    #
    input1 = Conv1D_linearphaseType(1 ,60, use_bias=False,
                    # kernel_initializer=initializers.he_normal(random_seed),
                    weights=[b1[31:]],
                    padding='same',trainable=FIR_train, type = type)(input)
    input2 = Conv1D_linearphaseType(1, 60, use_bias=False,
                    # kernel_initializer=initializers.he_normal(random_seed),
                    weights=[b2[31:]],
                    padding='same',trainable=FIR_train, type = type)(input)
    input3 = Conv1D_linearphaseType(1, 60, use_bias=False,
                    # kernel_initializer=initializers.he_normal(random_seed),
                    weights=[b3[31:]],
                    padding='same',trainable=FIR_train, type = type)(input)
    input4 = Conv1D_linearphaseType(1, 60, use_bias=False,
                    # kernel_initializer=initializers.he_normal(random_seed),
                    weights=[b4[31:]],
                    padding='same',trainable=FIR_train, type = type)(input)
    t1 = branch(input1,num_filt,kernel_size,random_seed,padding,bias,maxnorm,l2_reg,
           eps,bn_momentum,activation_function,dropout_rate,subsam,trainable)
    t2 = branch(input2,num_filt,kernel_size,random_seed,padding,bias,maxnorm,l2_reg,
           eps,bn_momentum,activation_function,dropout_rate,subsam,trainable)
    t3 = branch(input3,num_filt,kernel_size,random_seed,padding,bias,maxnorm,l2_reg,
           eps,bn_momentum,activation_function,dropout_rate,subsam,trainable)
    t4 = branch(input4,num_filt,kernel_size,random_seed,padding,bias,maxnorm,l2_reg,
           eps,bn_momentum,activation_function,dropout_rate,subsam,trainable)
    #Conv1D_gammatone

    # input1 = Conv1D_gammatone(kernel_size=81,filters=1,fsHz=1000,use_bias=False,padding='same')(input)
    # input2 = Conv1D_gammatone(kernel_size=81,filters=1,fsHz=1000,use_bias=False,padding='same')(input)
    # input3 = Conv1D_gammatone(kernel_size=81,filters=1,fsHz=1000,use_bias=False,padding='same')(input)
    # input4 = Conv1D_gammatone(kernel_size=81,filters=1,fsHz=1000,use_bias=False,padding='same')(input)
    
    xx = Concatenate(axis=-1)([t1,t2,t3,t4])
    
    xx = res_block(xx,64,kernel_size,2,'same',random_seed,bias,maxnorm,l2_reg,
           eps,bn_momentum,activation_function,dropout_rate,subsam,trainable)
    xx = res_block(xx,64,kernel_size,1,'same',random_seed,bias,maxnorm,l2_reg,
           eps,bn_momentum,activation_function,dropout_rate,subsam,trainable)
    
    xx = res_block(xx,128,kernel_size,3,'same',random_seed,bias,maxnorm,l2_reg,
           eps,bn_momentum,activation_function,dropout_rate,subsam,trainable)
    xx = res_block(xx,128,kernel_size,1,'same',random_seed,bias,maxnorm,l2_reg,
           eps,bn_momentum,activation_function,dropout_rate,subsam,trainable)
    
    xx = res_block(xx,128,kernel_size,2,'same',random_seed,bias,maxnorm,l2_reg,
           eps,bn_momentum,activation_function,dropout_rate,subsam,trainable,cat=False)
    xx = res_block(xx,128,kernel_size,2,'same',random_seed,bias,maxnorm,l2_reg,
           eps,bn_momentum,activation_function,dropout_rate,subsam,trainable,cat=False)
    
    xx = Conv1D(128, kernel_size=kernel_size,
                kernel_initializer=initializers.he_normal(seed=random_seed),
                padding=padding,
                strides=2,
                use_bias=bias,
                kernel_constraint=max_norm(maxnorm),
                trainable=trainable,
                kernel_regularizer=l2(l2_reg))(xx)
    
    merged = Flatten()(xx)
    
    dann_in = GradientReversal(hp_lambda=hp_lambda,name='grl')(merged)
    dsc = Dense(50,
                   activation=activation_function,
                   kernel_initializer=initializers.he_normal(seed=random_seed),
                   use_bias=bias,
                   kernel_constraint=max_norm(maxnorm),
                   kernel_regularizer=l2(l2_reg_dense),
                   name = 'domain_dense')(dann_in)   
    dsc = Dense(num_class_domain, activation='softmax', name = "domain")(dsc)          
    merged = Dense(num_dense,
                   activation=activation_function,
                   kernel_initializer=initializers.he_normal(seed=random_seed),
                   use_bias=bias,
                   kernel_constraint=max_norm(maxnorm),
                   kernel_regularizer=l2(l2_reg_dense),
                   name = 'class_dense')(merged)
    merged = Dense(num_class, activation='softmax', name="class")(merged)
    
    model = Model(inputs=input, outputs=[merged,dsc])
    
    if load_path:
        model.load_weights(filepath=load_path, by_name=False)
    
    #if load_path:  # If path for loading model was specified
    #model.load_weights(filepath='../../models_dbt_dann/fold_a_gt 2019-09-09 16:53:52.063276/weights.0041-0.6907.hdf5', by_name=True)
    # models/fold_a_gt 2019-09-04 17:36:52.860817/weights.0200-0.7135.hdf5
    
    #if optim=='Adam':
    #    opt = Adam(lr=lr, decay=lr_decay)
    #else:  
    opt = SGD(lr=lr,decay=lr_decay)
    model.compile(optimizer=opt, loss={'class':'categorical_crossentropy','domain':'categorical_crossentropy'}, metrics=['accuracy'])
    return model