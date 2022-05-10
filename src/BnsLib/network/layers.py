import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Layer
from tensorflow.keras import backend as K
from tensorflow.python.ops import nn
from tensorflow.python.keras.engine.base_layer import input_spec
import numpy as np
import copy
from functools import partial

InputSpec = input_spec.InputSpec


class BaseInception1D(object):
    """A class that implements a 1D inception module. Strictly speaking,
    it is not a layer but a set of layers. However, it can be used like
    a Keras layer.
    
    Arguments
    ---------
    See the notes section
    
    Notes
    -----
    -Arguments when initilizing the object are passed to the according
     layers. There are three different types of inputs you may give.
         -Arguments to the convolutional layers (3)
         -Arguments to the dimensional reduction layers (3)
         -Arguments to the maximum pooling layer (1)
    -General comments on arguments:
         As the object summarizes multiple layers, arguments that would
         typically be used for a single layer now act on all layers of
         that kind. If you want to specify different values for the
         different options you can pass a list of arguments of the
         appropriate length as the value to the keyword.
         E.g. if you wanted to set the kernel sizes of the main
         convolutional layers to (4, 6, 8), you could call
         BaseInception1D(kernel_size=[4,6,8])
         If the kernel sizes should all be 6 however, you could use
         BaseInception1D(kernel_size=6)
         or
         BaseInception1D(kernel_size=[6,6,6])
         
         As multiple layers share the same namespace for their keywords,
         you have to prepend the keywords with a prefix.
         
         These prefixes are:
             -Convolutional layers: 'conv_'
             -Dimensional reduction layers: 'dimred_'
             -Max Pooling layer: 'pooling_'
         
         If no prefix is given and the keyword can be found in
         BaseInception1D.conv_defaults it is understood as an argument
         to the convolutional layers. Therefore, the example of setting
         the kernel sizes of the main convolutional filters works.
         Equivalently one could write
         BaseInception1D(conv_kernel_size=[4,6,8])
         If for some reason you wanted to change the kernel sizes of the
         dimensional reduction layers to (4, 6, 8), you would use
         BaseInception1D(dimred_kernel_size=[4,6,8]) instead.
          
         Finally, there is a third way of setting the kernel sizes and
         number of filters of the main convolutional layers. As this is
         the most common difference between the different
         implementations of inception modules one can omit the keyword
         altogether. The order is equivalent to those of the Conv1D
         layers (filters, kernel_size). Therefore, one could set the
         BaseInception1D to have (16, 32, 64) filters and kernel sizes
         of (64, 32, 16) by calling
         BaseInception1D([16, 32, 64], [64, 32, 16])
    -Adding an inception-module to a network:
         To add an inception module to your network you can simply use it
         like any normal layer. I.e.:
         x = Layer(*args, **kwargs)
         x = BaseInception1D(*args, **kwargs)(x)
         x = Layer(*args, **kwargs)(x)
    -Model summary:
         The inception module will not show up as a single layer in the
         model summary but as a collection of layers.
    """
    def __init__(self, *args, **kwargs):
        self.conv_defaults = {'filters': [64, 32, 16],
                              'kernel_size': [4, 8, 16],
                              'strides': [1, 1, 1],
                              'padding': ['same'] * 3,
                              'data_format': ['channels_last'] * 3,
                              'dilation_rate': [1, 1, 1],
                              'activation': ['relu'] * 3,
                              'use_bias': [True] * 3,
                              'kernel_initializer': ['glorot_uniform'] * 3,
                              'bias_initializer': ['zeros'] * 3,
                              'kernel_regularizer': [None] * 3,
                              'bias_regularizer': [None] * 3,
                              'activity_regularizer': [None] * 3,
                              'kernel_constraint': [None] * 3,
                              'bias_constraint': [None] * 3}
        version = tf.__version__.split('.')
        version = np.array([int(pt) for pt in version])
        mult_factor = np.array([10**i for i in range(len(version)).__reversed__()])
        version = sum(version * mult_factor)
        if version > 220:
            self.conv_defaults['groups'] = [1, 1, 1]
        
        self.dimred_defaults = {'dimred_filters': [16] * 3,
                                'dimred_kernel_size': [1] * 3,
                                'dimred_padding': ['same'] * 3,
                                'dimred_activation': ['relu'] * 3}
        self.pooling_defaults = {'pool_size': 4,
                                 'pooling_strides': 1,
                                 'pooling_padding': 'same',
                                 'pooling_data_format': 'channels_last'}
        if len(args) > 0:
            if 'filters' in kwargs:
                if kwargs['filters'] != args[0]:
                    msg = f'The first argument to {type(self).__name__} '
                    msg += 'is interpreted as the number of filters.'
                    msg += ' However, the number of filters was '
                    msg += 'specified in the keyword-arguments as well '
                    msg += 'and the values do not match. (argument '
                    msg += f'value: {args[0]}, keyword-argument value: '
                    msg += f'{kwargs["filters"]})'
                    raise ValueError(msg)
            else:
                kwargs['filters'] = args[0]
        if len(args) > 1:
            if 'kernel_size' in kwargs:
                if kwargs['kernel_size'] != args[0]:
                    msg = f'The first argument to {type(self).__name__} '
                    msg += 'is interpreted as the kernel_size.'
                    msg += ' However, the kernel_size was specified in '
                    msg += 'the keyword-arguments as well and the '
                    msg += 'values do not match. (argument value: '
                    msg += f'{args[1]}, keyword-argument value: '
                    msg += f'{kwargs["kernel_size"]})'
                    raise ValueError(msg)
            else:
                kwargs['kernel_size'] = args[1]
        parsed_kwargs = self.parse_kwargs(**kwargs)
        self.conv_kwargs = parsed_kwargs[0]
        self.dimred_kwargs = parsed_kwargs[1]
        self.pool_kwargs = parsed_kwargs[2]
    
    def parse_kwargs(self, **kwargs):
        def to_list(inp):
            if isinstance(inp, list) or isinstance(inp, tuple):
                assert len(inp) == 3
                return inp
            else:
                return [inp, inp, inp]
        
        conv_kwargs = [{}, {}, {}]
        for key, val in self.conv_defaults.items():
            tmp = to_list(kwargs.get(key, val))
            for i in range(len(conv_kwargs)):
                conv_kwargs[i][key] = tmp[i]
        
        dimred_kwargs = [{}, {}, {}]
        for key, val in self.dimred_defaults.items():
            tmp = to_list(kwargs.get(key, val))
            for i in range(len(dimred_kwargs)):
                dimred_kwargs[i][key.replace('dimred_', '')] = tmp[i]
        
        pool_kwargs = {}
        if 'pooling_pool_size' in kwargs and 'pool_size' not in kwargs:
            kwargs['pool_size'] = kwargs['pooling_pool_size']
        if 'pooling_size' in kwargs and 'pool_size' not in kwargs:
            kwargs['pool_size'] = kwargs['pooling_size']
        for key, val in self.pooling_defaults.items():
            tmp = kwargs.get(key, val)
            pool_kwargs[key.replace('pooling_', '')] = tmp
        
        for key, val in kwargs.items():
            if 'conv_' in key:
                tmpkey = key.replace('conv_', '')
                if tmpkey not in conv_kwargs[0]:
                    tmpval = to_list(val)
                    for i in range(len(conv_kwargs)):
                        conv_kwargs[i][tmpkey] = tmpval[i]
            elif 'dimred_' in key:
                tmpkey = key.replace('dimred_', '')
                if tmpkey not in dimred_kwargs[0]:
                    tmpval = to_list(val)
                    for i in range(len(dimred_kwargs)):
                        dimred_kwargs[i][tmpkey] = tmpval[i]
            elif 'pooling_' in key:
                tmpkey = key.replace('pooling_', '')
                if tmpkey not in pool_kwargs:
                    pool_kwargs[tmpkey] = val
        
        return conv_kwargs, dimred_kwargs, pool_kwargs
    
    def __call__(self, input_layer):
        leftmost_conv = Conv1D(**self.conv_kwargs[0])(input_layer)
        
        left_dimred = Conv1D(**self.dimred_kwargs[0])(input_layer)
        left_conv = Conv1D(**self.conv_kwargs[1])(left_dimred)
        
        right_dimred = Conv1D(**self.dimred_kwargs[1])(input_layer)
        right_conv = Conv1D(**self.conv_kwargs[2])(right_dimred)
        
        rightmost_pool = MaxPooling1D(**self.pool_kwargs)(input_layer)
        rightmost_dimred = Conv1D(**self.dimred_kwargs[2])(rightmost_pool)
        
        return keras.layers.concatenate([leftmost_conv,
                                         left_conv,
                                         right_conv,
                                         rightmost_dimred])


class InputNormalization(Layer):
    def __init__(self, *args, **kwargs):
        self.normalize_mean = kwargs.pop('normalize_mean', True)
        self.normalize_std = kwargs.pop('normalize_std', True)
        super().__init__(*args, **kwargs)
    
    def call(self, inputs):
        if self.normalize_mean or self.normalize_std:
            mean, std = nn.moments(inputs, [1], keep_dims=True)
        if self.normalize_mean and self.normalize_std:
            return (inputs - mean) / std
        elif self.normalize_mean:
            return inputs - mean
        elif self.normalize_std:
            return inputs / std
        else:
            return inputs


class MinMaxClip(keras.constraints.Constraint):
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
    
    def __call__(self, w):
        # From keras.constraints.py function NonNeg
        # Sets the values of w to be non negative
        w *= K.cast(K.greater_equal(w, 0.), K.floatx())
        
        # From keras.constraints.py function MinMaxNorm
        # Clips the values of w to specific range
        desired = K.clip(w, self.min_val, self.max_val)
        w *= (desired / (K.epsilon() + w))
        return(w)


class FConv1D(keras.layers.Layer):
    def __init__(self, filters, frequency_low, frequency_high, activation=None,
                 number_of_cycles=1, fill=True, window='hann', **kwargs):
        self.filters = filters
        self.frequency_low = float(frequency_low)
        self.frequency_high = float(frequency_high)
        self.activation = keras.activations.get(activation)
        self.number_of_cycles = number_of_cycles
        self.dt = 0.5 / self.frequency_high
        self.rank = 1
        self.T = 1 / self.frequency_low
        self.T_LEN = int(np.floor(self.T / self.dt)) + 1
        self.fill = fill
        if window not in ['hann', None]:
            raise ValueError(('Right now only Hanning and no windowing are '
                              'supported.'))
        else:
            self.window = window
        
        super(FConv1D, self).__init__(**kwargs)
    
    def build(self, input_shape):
        length = (1, input_shape[-1], self.filters)
        
        self.frequencies = self.add_weight(name='frequencies',
                                           shape=length,
                                           trainable=True,
                                           initializer=keras.initializers.RandomUniform(minval=self.frequency_low, maxval=self.frequency_high),  # noqa: E501
                                           constraint=MinMaxClip(self.frequency_low, self.frequency_high),  # noqa: E501
                                           dtype=np.float32
                                           )
        
        self.amplitudes = self.add_weight(name='amplitudes',
                                          shape=length,
                                          trainable=True,
                                          initializer=keras.initializers.RandomUniform(minval=1.0, maxval=2.0),  # noqa: E501
                                          constraint=keras.constraints.NonNeg(),  # noqa: E501
                                          dtype=np.float32
                                          )
        
        self.phases = self.add_weight(name='phases',
                                      shape=length,
                                      trainable=True,
                                      initializer=keras.initializers.RandomUniform(minval=0.0, maxval=2 * np.pi),  # noqa: E501
                                      constraint=MinMaxClip(0, 2*np.pi),
                                      dtype=np.float32
                                      )
        
        self.kernel_shape = (self.number_of_cycles * self.T_LEN,) + length[1:]
        if self.kernel_shape[0] > input_shape[1]:
            msg  = 'A low and high frequency cutoff of (f_low, f_high) of '  # noqa: E221
            msg += '({}, {}) '.format(self.frequency_low, self.frequency_high)
            msg += 'combined with a repition rate of {}'.format(self.number_of_cycles)  # noqa: E501
            msg += ' results in a kernel of length '
            msg += '{}. The maximum length '.format(self.kernel_shape[0])
            msg += 'however must be smaller or equal '
            msg += '{}. (input_shape[1])'.format(input_shape[1])
            raise ValueError(msg)
        
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={-1: input_shape[-1]})
        
        super(FConv1D, self).build(input_shape)
    
    def call(self, x):
        np_kernel = np.zeros(self.kernel_shape, dtype=np.float32)
        if not self.window == None:
            np_window = np.zeros(self.kernel_shape, dtype=np.float32)
        
        np_freq, np_amp, np_phase = self.get_weights()
        
        for i, freq in enumerate(np_freq[0]):
            for j, f in enumerate(freq):
                if not self.fill:
                    fT_len = int(np.floor(1 / (f * self.dt))) + 1
                    args = dt * np.arange(self.number_of_cycles * fT_len)
                    pad = np.zeros(self.T_LEN - len(args))
                    vals = np.concatenate([args, pad])
                    np_kernel[:,i,j] = vals
                    if not self.window == None:
                        np_window[:,i,j] = np.concatenate([np.hanning(len(args)), pad])
                else:
                    np_kernel[:,i,j] = self.dt * np.arange(self.number_of_cycles * self.T_LEN)
                    if not self.window == None:
                        np_window[:,i,j] = np.hanning(self.number_of_cycles * self.T_LEN)
                    
        kernel = tf.convert_to_tensor(np_kernel)
        
        ones = K.ones(self.kernel_shape)
        
        a = self.frequencies * ones
        
        kernel *= a
        
        b = self.phases * ones
        
        kernel += b
        
        kernel = K.sin(kernel)
        
        b = self.amplitudes * ones
        
        kernel *= b
        
        if not self.window == None:
            kernel += tf.convert_to_tensor(np_window)
        
        output = K.conv1d(x,
                          kernel,
                          padding='same',
                          data_format='channels_last')
        
        if self.activation is not None:
            return(self.activation(output))
        
        return(output)
    
    def compute_output_shape(self, input_shape):
        return(input_shape[:-1] + (self.filters, ))
    
    def get_config(self):
        config = {
            'filters': self.filters,
            'frequency_low': self.frequency_low,
            'frequency_high': self.frequency_high,
            'activation': keras.activations.serialize(self.activation),
            'number_of_cycles': self.number_of_cycles,
            'fill': self.fill,
            'window': self.window
        }
        base_config = super(FConv1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# Wave convolution: Adds sins off different frequencies and amplitudes to
# build a filter.
class WConv1D(keras.layers.Layer):
    def __init__(self, filters, frequency_low, frequency_high, activation=None,
                 number_of_cycles=1, fill=True, window=None,
                 waves_per_filter=128, **kwargs):
        self.filters = filters
        self.waves_per_filter = waves_per_filter
        self.frequency_low = float(frequency_low)
        self.frequency_high = float(frequency_high)
        self.activation = keras.activations.get(activation)
        self.number_of_cycles = number_of_cycles
        self.dt = 0.5 / self.frequency_high
        self.rank = 1
        self.T = 1 / self.frequency_low
        self.T_LEN = int(np.floor(self.T / self.dt)) + 1
        self.fill = fill
        if not window in ['hann', None]:
            raise ValueError('Right now only Hanning and no windowing are supported.')
        else:
            self.window = window
        
        if window == 'hann':
            warnings.warn('The implementation of the Hanning window is wrong.', RuntimeWarning)
        
        super(WConv1D, self).__init__(**kwargs)
    
    def build(self, input_shape):
        length = (1, input_shape[-1], self.filters, self.waves_per_filter)
        
        self.frequencies = self.add_weight(name='frequencies',
                                           shape=length,
                                           trainable=True,
                                           initializer=keras.initializers.RandomUniform(minval=self.frequency_low, maxval=self.frequency_high),
                                           constraint=MinMaxClip(self.frequency_low, self.frequency_high),
                                           dtype=np.float32
                                           )
        
        self.amplitudes = self.add_weight(name='amplitudes',
                                          shape=length,
                                          trainable=True,
                                          initializer=keras.initializers.RandomUniform(minval=1.0, maxval=2.0),
                                          constraint=keras.constraints.NonNeg(),
                                          dtype=np.float32
                                          )
        
        self.phases = self.add_weight(name='phases',
                                      shape=length,
                                      trainable=True,
                                      initializer=keras.initializers.RandomUniform(minval=0.0, maxval=2 * np.pi),
                                      constraint=MinMaxClip(0, 2*np.pi),
                                      dtype=np.float32
                                      )
        
        self.kernel_shape = (self.number_of_cycles * self.T_LEN,) + length[1:]
        if self.kernel_shape[0] > input_shape[1]:
            msg  = 'A low and high frequency cutoff of (f_low, f_high) of '
            msg += '({}, {}) '.format(self.frequency_low, self.frequency_high)
            msg += 'combined with a repition rate of {}'.format(self.number_of_cycles)
            msg += ' results in a kernel of length '
            msg += '{}. The maximum length '.format(self.kernel_shape[0])
            msg += 'however must be smaller or equal '
            msg += '{}. (input_shape[1])'.format(input_shape[1])
            raise ValueError(msg)
        
        #print("Kernel shape: {}".format(self.kernel_shape))
        
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={-1: input_shape[-1]})
        
        super(WConv1D, self).build(input_shape)
    
    def call(self, x):
        np_kernel = np.zeros(self.kernel_shape, dtype=np.float32)
        if not self.window == None:
            np_window = np.zeros(self.kernel_shape, dtype=np.float32)
        
        np_freq, np_amp, np_phase = self.get_weights()
        
        np_freq = np_freq.transpose(0, 1, 3, 2)
        
        np_kernel = np_kernel.transpose(0, 1, 3, 2)
        
        if not self.window == None:
            np_window = np_window.transpose(0, 1, 3, 2)
        
        for i, freqs in enumerate(np_freq[0]):
            for j, freq in enumerate(freqs):
                for k, f in enumerate(freq):
                    if not self.fill:
                        fT_len = int(np.floor(1 / (f * self.dt))) + 1
                        args = dt * np.arange(self.number_of_cycles * fT_len)
                        pad = np.zeros(self.T_LEN - len(args))
                        vals = np.concatenate([args, pad])
                        np_kernel[:,i,j,k] = vals
                        if not self.window == None:
                            np_window[:,i,j,k] = np.concatenate([np.hanning(len(args)), pad])
                    else:
                        np_kernel[:,i,j,k] = self.dt * np.arange(self.number_of_cycles * self.T_LEN)
                        if not self.window == None:
                            np_window[:,i,j,k] =np.hanning(self.number_of_cycles * self.T_LEN)
        
        np_kernel = np_kernel.transpose(0, 1, 3, 2)
        
        if not self.window == None:
            np_window = np_window.transpose(0, 1, 3, 2)
                    
        kernel = tf.convert_to_tensor(np_kernel)
        
        ones = K.ones(self.kernel_shape)
        
        a = self.frequencies * ones
        
        kernel *= a
        
        b = self.phases * ones
        
        kernel += b
        
        kernel = K.sin(kernel)
        
        b = self.amplitudes * ones
        
        kernel *= b
        
        if not self.window == None:
            kernel *= tf.convert_to_tensor(np_window)
        
        kernel = K.sum(kernel, axis=-1)
        
        output = K.conv1d(x,
                          kernel,
                          padding='same',
                          data_format='channels_last')
        
        if self.activation is not None:
            return(self.activation(output))
        
        return(output)
    
    def compute_output_shape(self, input_shape):
        return(input_shape[:-1] + (self.filters, ))
    
    def get_config(self):
        config = {
            'filters': self.filters,
            'waves_per_filter': self.waves_per_filter,
            'frequency_low': self.frequency_low,
            'frequency_high': self.frequency_high,
            'activation': keras.activations.serialize(self.activation),
            'number_of_cycles': self.number_of_cycles,
            'fill': self.fill,
            'window': self.window
        }
        base_config = super(WConv1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class NetConvolve(keras.layers.Layer):
    """A layer that allows to apply networks that require a fixed
    input-size to an arbitrary input.
    
    This layer expects the input to be of shape
    (batch_size, samples, channels)
    
    For usage, a single layer has to be passed in. To apply a stack of
    layers define a new layer class:
    
    >>> class MyLayer(keras.layers.Layer):
    >>>     def __init__(self, *args, **kwargs):
    >>>         self.layer1 = keras.layers.Conv1D(filters=1, kernel_size=1)
    >>>         self.layer2 = keras.layers.Conv1D(filters=1, kernel_size=1)
    >>>     
    >>>     def call(self, x):
    >>>         x = self.layer1(x)
    >>>         x = self.layer2(x)
    >>>         return x
    
    Afterwards the convolved layer can be integrated into the network simply
    by
    
    >>> x = NetConvolve(MyLayer(), stride=2, padding=`zeros`)(x)
    
    `MyLayer` in the example above may be replaced by any singular layer or
    Model.
    
    Arguments
    ---------
    model : keras.layers.Layer or keras.models.Model
        The layer/model that should be convolved over the data.
    stride : {int or None, None}
        The stride to use for shifting the window. If set to None, it
        is assumed to be 1.
    padding : {`zeros` or `reflect` or `symmetric` or None, None}
        The kind of padding that should be applied. If set to None, no
        padding will be applied and it is not guaranteed that the entire
        input will be processed.
    squeeze : {bool, True}
        Whether the output shape is (False)
          (batch_size, nsteps, model.output_shape[1:])
        or (True)
          (batch_size, nsteps * model.output_shape[1], model.output_shape[2:])
        
    
    Notes
    -----
        -The convolved sub-network will not appear in the model summary.
         Instead only a single line for the NetConvolve layer will be
         shown. The number of trainable and non-trainable parameters
         will be preserved.
    """
    def __init__(self, model, *args, stride=None, padding=None,
                 squeeze=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.payload = model
        window_size = model.input_shape[1]
        self.window_size = window_size
        self.stride = 1 if stride is None else stride
        self.padding = padding
        self.squeeze = squeeze
        self.nslices = None
        self.cslice = 0
    
    def apply_padding(self, x):
        if self.padding is None or self.window_size is None:
            return x
        
        batch_size, nsamps, nchannels = x.shape
        if nsamps is None:
            return x
        
        optimal_nslices = int(np.ceil((int(nsamps) - self.window_size)
                                      / self.stride + 1))
        optimal_length = (optimal_nslices - 1) * self.stride + self.window_size
        padding = np.zeros((3, 2))
        padding[1, 1] = optimal_length - nsamps
        padding = tf.constant(padding, dtype=tf.int32)
        if self.padding.lower() == 'zeros':
            return tf.pad(x, padding, "CONSTANT")
        elif self.padding.lower() == 'reflect':
            return tf.pad(x, padding, "REFLECT")
        elif self.padding.lower() == 'symmetric':
            return tf.pad(x, padding, "SYMMETRIC")
    
    def n_window_positions(self, samples):
        if self.window_size is None:
            return 1
        return (samples - self.window_size) // self.stride + 1
    
    def slice_op(self, x, idx):
        nsamps, c = x.shape
        start = idx * self.stride
        x = tf.slice(x,
                     [start, 0],
                     [self.window_size, tf.shape(x)[-1]])
        x = tf.expand_dims(x, axis=0)
        ret = self.payload(x)
        return ret[0]
    
    def moving_window(self, x, idx):
        batch_size, nsamps, nchannels = x.shape
        nslices = self.n_window_positions(nsamps)
        winfunc = partial(self.slice_op, x[idx])
        ret = tf.map_fn(winfunc,
                        tf.range(nslices),
                        fn_output_signature=self.dtype)
        return ret
    
    def call(self, x):
        x = self.apply_padding(x)
        batch_size, nsamps, nchannels = x.shape
        if batch_size is None:
            # Only used to return correct shape in model summary
            out_shape = self.compute_output_shape(x.shape)
            if self.squeeze:
                out_shape = [out_shape[0],
                             out_shape[1] * out_shape[2]] + out_shape[3:]
            output = x[:, 0, 0]
            padding = [[0, 0]]
            for sh in out_shape[1:]:
                output = tf.expand_dims(output, axis=-1)
                padding.append([0, sh - 1])
            
            padding = tf.constant(padding)
            output = tf.pad(output, padding)
            return output
        
        winfunc = partial(self.moving_window, x)
        ret = tf.map_fn(winfunc,
                        tf.range(batch_size),
                        fn_output_signature=self.dtype)
        if not self.squeeze:
            return ret
        tmp = ret.shape
        return tf.reshape(ret, [tmp[0], tmp[1] * tmp[2]] + tmp[3:])
    
    def get_config(self):
        config = super().get_config()
        config.update({"model": keras.utils.serialize_keras_object(self.payload),  # noqa: E501
                       "stride": self.stride,
                       "padding": self.padding,
                       "squeeze": self.squeeze})
        return config
    
    @classmethod
    def from_config(cls, config, custom_objects=None):
        config = copy.deepcopy(config)
        model = keras.layers.deserialize(config.pop('model'),
                                         custom_objects=custom_objects)
        return cls(model, **config)
    
    def compute_output_shape(self, input_shape):
        batch_size, nsamples, nchannels = input_shape
        ret = [batch_size,
               self.n_window_positions(nsamples)]
        ret += self.payload.output_shape[1:]
        return ret
