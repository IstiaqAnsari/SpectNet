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
    t = Conv1D(16, kernel_size=kernel_size,
               kernel_initializer=initializers.he_normal(seed=random_seed),
               padding=padding,
               use_bias=bias,
               trainable=trainable,
               kernel_constraint=max_norm(maxnorm),
               kernel_regularizer=l2(l2_reg))(t)
    t = BatchNormalization(epsilon=eps, momentum=bn_momentum, axis=-1)(t)
    t = Activation(activation_function)(t)
    t = Dropout(rate=dropout_rate, seed=random_seed)(t)
    # t = Flatten()(t)
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
             num_class=2, num_class_domain=1,hp_lambda=0,batch_size=1024,optim='SGD',segments='0101'):
    
    #num_dense = 20 default 
    input = Input(shape=(2500, 1))

    
    xx = branch(input,num_filt,kernel_size,random_seed,padding,bias,maxnorm,l2_reg,
           eps,bn_momentum,activation_function,dropout_rate,subsam,trainable)
    xx = branch(input,num_filt,kernel_size,random_seed,padding,bias,maxnorm,l2_reg,
           eps,bn_momentum,activation_function,dropout_rate,subsam,trainable)
    xx = res_block(xx,32,kernel_size,2,'same',random_seed,bias,maxnorm,l2_reg,
           eps,bn_momentum,activation_function,dropout_rate,subsam,trainable)
    xx = res_block(xx,64,kernel_size,2,'same',random_seed,bias,maxnorm,l2_reg,
           eps,bn_momentum,activation_function,dropout_rate,subsam,trainable)
    
    xx = MaxPooling1D(pool_size=2)(xx)
    
#     xx = res_block(xx,128,kernel_size,2,'same',random_seed,bias,maxnorm,l2_reg,
#            eps,bn_momentum,activation_function,dropout_rate,subsam,trainable)
#     xx = res_block(xx,128,kernel_size,1,'same',random_seed,bias,maxnorm,l2_reg,
#            eps,bn_momentum,activation_function,dropout_rate,subsam,trainable)
    
#     xx = res_block(xx,128,kernel_size,2,'same',random_seed,bias,maxnorm,l2_reg,
#            eps,bn_momentum,activation_function,dropout_rate,subsam,trainable,cat=False)
#     xx = res_block(xx,128,kernel_size,2,'same',random_seed,bias,maxnorm,l2_reg,
#            eps,bn_momentum,activation_function,dropout_rate,subsam,trainable,cat=False)
    
    xx = Conv1D(128, kernel_size=kernel_size,
                kernel_initializer=initializers.he_normal(seed=random_seed),
                padding=padding,
                strides=2,
                use_bias=bias,
                kernel_constraint=max_norm(maxnorm),
                trainable=trainable,
                kernel_regularizer=l2(l2_reg))(xx)
    xx = MaxPooling1D(pool_size=2)(xx)
    merged = Flatten()(xx)
    merged = Dropout(rate=dropout_rate, seed=random_seed)(merged)
    
    
    dann_in = GradientReversal(hp_lambda=hp_lambda,name='grl')(merged)
    dsc = Dense(80,
                   activation=activation_function,
                   kernel_initializer=initializers.he_normal(seed=random_seed),
                   use_bias=bias,
                   kernel_constraint=max_norm(maxnorm),
                   kernel_regularizer=l2(l2_reg_dense),
                   name = 'domain_dense')(dann_in)   
    dsc = Dense(num_class_domain, activation='softmax', name = "domain")(dsc)          
    merged = Dense(50,
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
    
    if optim=='Adam':
        opt = Adam(lr=lr, decay=lr_decay)
    else:  
        opt = SGD(lr=lr,decay=lr_decay)
    if(num_class_domain>1):
        domain_loss_function = 'categorical_crossentropy'
    else:
        domain_loss_function = 'binary_crossentropy'
    model.compile(optimizer=opt, loss={'class':'categorical_crossentropy','domain':domain_loss_function}, loss_weights=[1,0], metrics=['accuracy'])
    #model.compile(optimizer=opt, loss={'class':'categorical_crossentropy','domain':'categorical_crossentropy'}, metrics=['accuracy'])
    return model


def getAttentionModel(model,foldname,lr,lr_decay):
    load_path = Result(foldname, find = True).df['model_path']
    load_path = load_path.replace(load_path[-16:-12],str(int(load_path[-16:-12])+1).rjust(4,'0'))

    model.load_weights(load_path)
    layers = {x.name:x for x in model.layers[-5:]}
    layers
    while('flatten' not in model.layers[-1].name):
        model.layers.pop()
    merged = Attention(name='att')(model.layers[-1].output)
    dann_in = layers['grl'](merged)
    dsc = layers['domain_dense'](dann_in)
    dsc = layers['domain'](dsc)
    merged = layers['class_dense'](merged)
    merged = layers['class'](merged)
    model = Model(inputs=model.layers[0].input, outputs=[merged,dsc])
    for layer in model.layers:
        if 'flatten' in layer.name:
            break
        else:
            layer.trainable = False
    opt = SGD(lr=lr,decay=lr_decay)

    model.compile(optimizer=opt, loss={'class':'categorical_crossentropy','domain':'categorical_crossentropy'}, metrics=['accuracy'])
    return model