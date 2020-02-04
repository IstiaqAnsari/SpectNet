from __future__ import print_function, absolute_import, division
from keras.initializers import Initializer
from custom_layers import Conv1D_zerophase_linear, Conv1D_linearphase, Conv1D_zerophase,\
    DCT1D, Conv1D_gammatone, Conv1D_linearphaseType, Attention
from keras.layers import Input, Conv1D, MaxPooling1D,MaxPooling2D, Dense, Dropout, Flatten, Activation, AveragePooling1D, Add,Reshape
from keras import initializers
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.layers.core import Lambda
from keras.utils import conv_utils
from keras.engine.topology import InputSpec
from keras.engine.base_layer import Layer
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
from keras import backend as K
from keras.engine.base_layer import Layer
from keras.engine.topology import InputSpec
import tensorflow as tf,keras
from keras.layers.merge import Concatenate
from keras.utils import conv_utils
from keras.layers import Input, MaxPooling1D ,Conv1D,Activation,Dense, activations, initializers,Conv2D
from keras.layers import Flatten, regularizers, constraints,Dropout,Multiply
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD
from keras.models import Model
from keras.utils import plot_model
import numpy as np
from scipy.fftpack import dct
from keras.backend.common import normalize_data_format
from keras.layers.merge import Multiply
from keras.utils import to_categorical
from keras.callbacks import LearningRateScheduler
from Gradient_Reverse_Layer import GradientReversal
from custom_layers import Conv1D_gammatone, Conv1D_zerophase
import scipy
from keras import backend as K
from keras.engine.topology import Layer
from keras.engine.topology import InputSpec
import tensorflow as tf
from keras.utils import conv_utils
from keras.layers import activations, initializers, regularizers, constraints
import numpy as np
from scipy.fftpack import dct
from keras.backend.common import normalize_data_format
from keras.layers.merge import Multiply

class mfcc_kernel_init(Initializer):
#     def __init__(self):
        
    def __call__(self, shape, dtype=K.floatx()):
        self.shape = shape
        (kernel_size,in_channels,out_channels) = shape
        if(in_channels!=out_channels):
            raise ValueError("Input and Output Channels must be same. Got {0} input channels and {0} output channels".format(in_channels,out_channels))
        mat = K.eye(in_channels,dtype=dtype)
        mat_n = [mat for i in range(kernel_size)]
        return K.stack(mat_n)
    def get_config(self):
        return {
            "shape":self.shape
        }
class Freq_Init(Initializer):
    def __init__(self, minf=0., maxf=500):
        self.minf = minf
        self.maxf = maxf
    def __call__(self, shape, dtype=K.floatx()):
        (kernel_size,in_channels) = shape
        start = self.hz2mel(self.minf)
        end = self.hz2mel(self.maxf)
        n = ( end-start)/(kernel_size-1)
        mel =  K.expand_dims(K.arange(start,end+1,n,dtype=dtype),axis=1)
        return self.mel2hz(mel)
    def hz2mel(self,hz):
        return 2595 * (K.log(1.0+(hz*1.0)/700.)/K.log(10.0))
    def mel2hz(self,mel):
        return 700*(10**(mel/2595.0)-1)
    def get_config(self):
        return {
            'minf': self.minf,
            'maxf': self.maxf
        }
def hz2mel(hz):
    return 2595 * np.log10(1+hz/700.)
def mel2hz(mel):
    return 700*(10**(mel/2595.0)-1)
def erb(f):
    return 24.7*(4.37*10**-3*f+1)
class beta_init(Initializer):
    def __init__(self, val = 100):
        self.val = val
    def __call__(self,shape,dtype=K.floatx()):
        (kernel_size,in_channels) = shape
        beta_weights = tf.convert_to_tensor(np.ones((kernel_size,1))*self.val,dtype=K.floatx())
        return beta_weights
    def get_config(self):
        return {
            'val': self.val
        }


def expand(x):
    return K.expand_dims(x, axis=-1)
def expand_output_shape(input_shape):
    return tuple(list(input_shape)+[1])
def zeropad_len(x):
    return x[:,:-1,:]
def zeropad_len_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 3
    shape[1] -= 1
    return tuple(shape)
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
        if(t.shape[1]!=p.shape):
            t = Lambda(zeropad_len, output_shape=zeropad_len_output_shape)(t)

    t = Add()([t,p])
    return t
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
    t = Conv1D(num_filt2, kernel_size=kernel_size,
               kernel_initializer=initializers.he_normal(seed=random_seed),
               padding=padding,
               use_bias=bias,
               trainable=trainable,
               kernel_constraint=max_norm(maxnorm),
               kernel_regularizer=l2(l2_reg))(t)
    t = BatchNormalization(epsilon=eps, momentum=bn_momentum, axis=-1)(t)
    t = Activation(activation_function)(t)
    t = Dropout(rate=dropout_rate, seed=random_seed)(t)
    
    return t
def zeropad_len_Ax1(x):
    print(x.shape)
    return x[:,:-1,:,:]
def zeropad_len_Ax2(x):
    return x[:,:,:-1,:]
def zeropad_len_output_shape_Ax1(input_shape):
    shape = list(input_shape)
    assert len(shape) == 4
    shape[1] -= 1
    return tuple(shape)
def zeropad_len_output_shape_Ax2(input_shape):
    shape = list(input_shape)
    assert len(shape) == 4
    shape[2] -= 1
    return tuple(shape)
def zeropad2d(x):
    y = K.zeros_like(x)
    return K.concatenate([x, y], axis=3)
def zeropad_output_shape2d(input_shape):
    shape = list(input_shape)
    assert len(shape) == 4
    shape[-1] *= 2
    return tuple(shape)
def branch2d(input_tensor,num_filt,kernel_size,random_seed,padding,bias,maxnorm,l2_reg,
           eps,bn_momentum,activation_function,dropout_rate,subsam,trainable):
    
    pad1,pad2 = padding
    num_filt1, num_filt2 = num_filt
    t = Conv2D(num_filt1, kernel_size=kernel_size,
                kernel_initializer=initializers.he_normal(seed=random_seed),
                padding=pad1,
                use_bias=bias,
                kernel_constraint=max_norm(maxnorm),
                trainable=trainable,
                kernel_regularizer=l2(l2_reg))(input_tensor)
    t = BatchNormalization(epsilon=eps, momentum=bn_momentum, axis=-1)(t)
    t = Activation(activation_function)(t)
    t = Dropout(rate=dropout_rate, seed=random_seed)(t)
    t = Conv2D(num_filt2, kernel_size=kernel_size,
               kernel_initializer=initializers.he_normal(seed=random_seed),
               padding=pad2,
               use_bias=bias,
               trainable=trainable,
               kernel_constraint=max_norm(maxnorm),
               kernel_regularizer=l2(l2_reg))(t)
    t = BatchNormalization(epsilon=eps, momentum=bn_momentum, axis=-1)(t)
    t = Activation(activation_function)(t)
    t = Dropout(rate=dropout_rate, seed=random_seed)(t)
    
    return t
def res_block2d(input_tensor,num_filt,kernel_size,stride,padding,random_seed,bias,maxnorm,l2_reg,
           eps,bn_momentum,activation_function,dropout_rate,subsam,trainable,cat=True):

    t = Conv2D(num_filt, kernel_size=kernel_size,
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
    t = Conv2D(num_filt, kernel_size=kernel_size,
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
    
    p = MaxPooling2D(pool_size=stride)(input_tensor)
    if(stride>1):
        if(cat):
            p = Lambda(zeropad2d, output_shape=zeropad_output_shape2d)(p)
#         print("T", t.shape, "P", p.shape)
        if(t.shape[1]!=p.shape[1]):
            t = Lambda(zeropad_len_Ax1, output_shape=zeropad_len_output_shape_Ax1)(t)
        if(t.shape[2]!=p.shape[2]):
            t = Lambda(zeropad_len_Ax2, output_shape=zeropad_len_output_shape_Ax2)(t)
#         print("T", t.shape, "P", p.shape, "PORe")
    t = Add()([t,p])
    return t
def heartnet(kernel_size=5,fs=1000,winlen=0.025,winstep=0.01,filters=26,random_seed=1,padding='valid',bias=False,
           lr=0.0012843784,lr_decay=0.0001132885,subsam=2,num_filt=(26,32),num_dense=20,trainable=True,batch_size=1024,
           l2_reg=0.0,l2_reg_dense=0.0,bn_momentum=0.99,dropout_rate=0.5,dropout_dense=0.0,eps = 1.1e-5,maxnorm=10000,
           activation_function='relu'):
    input = Input(shape=(2500, 1))
    t = Conv1D_gammatone(kernel_size=81,strides=1,filters=filters,
                         fsHz=fs,use_bias=False,padding='same',
                         fc_initializer=Freq_Init(minf=50.0,maxf=fs/2),
                         amp_initializer=initializers.constant(10**4),
                        beta_initializer=beta_init(val=100),name="gamma"
                        )(input)
    t = BatchNormalization(epsilon=eps, momentum=bn_momentum, axis=-1)(t)
    t = MFCC(rank = 1,filters=filters,kernel_size=int(winlen*fs),output_format='signal',strides=int(winstep*fs),
              kernel_initializer = mfcc_kernel_init(), name="mfcc")(t)
    t = BatchNormalization(epsilon=eps, momentum=bn_momentum, axis=-1)(t)
    
    t = branch(t,num_filt,kernel_size,random_seed,padding,bias,maxnorm,l2_reg,
           eps,bn_momentum,activation_function,dropout_rate,subsam,trainable)

    t = res_block(t,32,kernel_size,1,'same',random_seed,bias,maxnorm,l2_reg,
           eps,bn_momentum,activation_function,dropout_rate,subsam,trainable)

    t = res_block(t,64,kernel_size,2,'same',random_seed,bias,maxnorm,l2_reg,
           eps,bn_momentum,activation_function,dropout_rate,subsam,trainable)
    t = Flatten()(t)
    t = Dense(20,
                   activation=activation_function,
                   kernel_initializer=initializers.he_normal(seed=random_seed),
                   use_bias=bias,
                   kernel_constraint=max_norm(maxnorm),
                   kernel_regularizer=l2(l2_reg_dense),
                   name = 'class_dense')(t)
    t = Dense(2, activation='softmax', name="class")(t)
    opt = SGD(lr=.001,decay=.001)
    
    model = Model(inputs=input, outputs=t)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def heartnet2D(kernel_size=5,fs=1000,winlen=0.025,winstep=0.01,filters=26,random_seed=1,padding='valid',bias=False,
           lr=0.0012843784,lr_decay=0.0001132885,subsam=2,num_filt=(16,32),num_dense=20,trainable=True,batch_size=1024,
           l2_reg=0.0,l2_reg_dense=0.0,bn_momentum=0.99,dropout_rate=0.5,dropout_dense=0.0,eps = 1.1e-5,maxnorm=10000,
           activation_function='relu'):
    input = Input(shape=(2500, 1))
    t = Conv1D_gammatone(kernel_size=81,strides=1,filters=filters,
                         fsHz=fs,use_bias=False,padding='same',
                         fc_initializer=Freq_Init(minf=50.0,maxf=fs/2),
                         amp_initializer=initializers.constant(10**4),
                        beta_initializer=beta_init(val=100),name="gamma"
                        )(input)
    t = BatchNormalization(epsilon=eps, momentum=bn_momentum, axis=-1)(t)
    t = MFCC(rank = 1,filters=filters,kernel_size=int(winlen*fs),output_format='signal',strides=int(winstep*fs),
              kernel_initializer = mfcc_kernel_init(), name="mfcc")(t)
    # t = Lambda(expand,output_shape=expand_output_shape)(t)
    t = BatchNormalization(epsilon=eps, momentum=bn_momentum, axis=-1)(t)
    t = Reshape(target_shape=(-1,filters,1))(t)

    t = Conv2D(32, kernel_size=kernel_size,
                kernel_initializer=initializers.he_normal(seed=random_seed),
                padding='valid',
                use_bias=bias,
                kernel_constraint=max_norm(maxnorm),
                trainable=trainable,
                kernel_regularizer=l2(l2_reg))(t)
    # t = BatchNormalization(epsilon=eps, momentum=bn_momentum, axis=-1)(t)
    # t = Activation(activation_function)(t)
    # t = Dropout(rate=dropout_rate, seed=random_seed)(t)

    # t = branch2d(t,num_filt,kernel_size,random_seed,('valid','valid'),bias,maxnorm,l2_reg,
    #        eps,bn_momentum,activation_function,dropout_rate,subsam,trainable)
    # t = MaxPooling2D((2,1))(t)
    # t = branch2d(t,(32,64),(kernel_size,3),random_seed,('valid','valid'),bias,maxnorm,l2_reg,
    #        eps,bn_momentum,activation_function,dropout_rate,subsam,trainable)
    # t = MaxPooling2D((2,1))(t)
    # t = branch2d(t,(64,128),kernel_size,random_seed,('valid','valid'),bias,maxnorm,l2_reg,
    #        eps,bn_momentum,activation_function,dropout_rate,subsam,trainable)
    # t = MaxPooling2D((2,1))(t)
    # t = branch2d(t,(128,256),kernel_size,random_seed,('same','same'),bias,maxnorm,l2_reg,
    #        eps,bn_momentum,activation_function,dropout_rate,subsam,trainable)
#     t = res_block2d(t,32,5,1,'valid',random_seed,bias,maxnorm,l2_reg,
#            eps,bn_momentum,activation_function,dropout_rate,subsam,trainable)

#     t = res_block2d(t,64,5,2,'same',random_seed,bias,maxnorm,l2_reg,
#            eps,bn_momentum,activation_function,dropout_rate,subsam,trainable)
    
#     t = res_block2d(t,128,5,2,'same',random_seed,bias,maxnorm,l2_reg,
#            eps,bn_momentum,activation_function,dropout_rate,subsam,trainable)
#     t = MaxPooling2D((2,1))(t)
    t = Flatten()(t)
    t = Dense(20,
                   activation=activation_function,
                   kernel_initializer=initializers.he_normal(seed=random_seed),
                   use_bias=bias,
                   kernel_constraint=max_norm(maxnorm),
                   kernel_regularizer=l2(l2_reg_dense),
                   name = 'class_dense')(t)
    t = Dense(2, activation='softmax', name="class")(t)
    opt = Adam(lr=.001,decay=.001,epsilon=eps)
    
    model = Model(inputs=input, outputs=t)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

class MFCC(Layer): 
    def __init__(self, rank,
                 filters,
                 kernel_size,
                 output_format='signal',
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(MFCC, self).__init__(**kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank,
                                                      'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = K.normalize_data_format(data_format)
        self.kernel_initializer = kernel_initializer
        self.output_format = output_format
        self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)
    
        
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',trainable=False)
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        outputs = K.pow(K.abs(inputs),2)
        outputs = K.conv1d(
            outputs,
            self.kernel,
            strides=self.strides[0],
            padding=self.padding,
            data_format=self.data_format)
        outputs = K.log(outputs)
#         outputs = tf.signal.dct(outputs,type=2,axis=-1,norm='ortho')
        if(self.output_format=='image'):
            outputs = K.expand_dims(outputs,axis=-1)
        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i])
                new_space.append(new_dim)
            if(self.output_format=='image'):
                return (input_shape[0],) + tuple(new_space) + (self.filters,) + (1,)
            else:
                return (input_shape[0],) + tuple(new_space) + (self.filters,)
            
        if self.data_format == 'channels_first':
            raise NotImplementedError("Output formate image/signal not handled")
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i])
                new_space.append(new_dim)
            return (input_shape[0], self.filters) + tuple(new_space)
    
    def get_config(self):
        config = {
            'rank': self.rank,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'output_format':self.output_format,
            'kernel_initializer' : initializers.serialize(self.kernel_initializer)
        }
        base_config = super(MFCC, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

