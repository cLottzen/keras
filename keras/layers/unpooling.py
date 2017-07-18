# -*- coding: utf-8 -*-
from __future__ import absolute_import

from .. import backend as K
from ..engine import Layer
from ..engine import InputSpec
from ..utils import conv_utils
from ..legacy import interfaces

class _UnpoolingIndex2D(Layer):
    """Abstract class for different unpooling 2D layers with multiple output
       to support pool indices's as additional input
    """

    def __init__(self, pool_size=(2, 2), strides=None, padding='valid',
                 data_format=None, **kwargs):
        super(_UnpoolingIndex2D, self).__init__(**kwargs)
        data_format = conv_utils.normalize_data_format(data_format)
        if strides is None:
            strides = pool_size
        self.pool_size = conv_utils.normalize_tuple(pool_size, 2, 'pool_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = [InputSpec(), InputSpec()]

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            rows = input_shape[0][2]
            cols = input_shape[0][3]
        elif self.data_format == 'channels_last':
            rows = input_shape[0][1]
            cols = input_shape[0][2]

        rows *= self.pool_size[0]
        cols *= self.pool_size[1]

        if self.data_format == 'channels_first':
            return (input_shape[0][0], input_shape[0][1], rows, cols)
        elif self.data_format == 'channels_last':
            return (input_shape[0][0], rows, cols, input_shape[0][3])

    def _pooling_function(self, inputs, pool_size, strides,
                          padding, data_format):
        raise NotImplementedError

    def call(self, inputs):
        output = self._pooling_function(inputs=inputs,
                                        pool_size=self.pool_size,
                                        strides=self.strides,
                                        padding=self.padding,
                                        data_format=self.data_format)
        return output

    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'padding': self.padding,
                  'strides': self.strides,
                  'data_format': self.data_format}
        base_config = super(_Unpooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class UnpoolingIndex2D(_UnpoolingIndex2D):

    @interfaces.legacy_pooling2d_support
    def __init__(self, pool_size=(2, 2), strides=None, padding='valid',
                 data_format=None, **kwargs):
        super(UnpoolingIndex2D, self).__init__(pool_size, strides, padding,
                                           data_format, **kwargs)

    def _pooling_function(self, inputs, pool_size, strides,
                          padding, data_format):

        ''' not all keras standard parameters are supported yet, e.g
            data_format and padding
        '''
        #output = K.unpool2d_with_argmax(inputs, pool_size, strides,
        #                  padding, data_format)  
        output = K.unpool2d_with_argmax(inputs, pool_size, strides)

        return output
