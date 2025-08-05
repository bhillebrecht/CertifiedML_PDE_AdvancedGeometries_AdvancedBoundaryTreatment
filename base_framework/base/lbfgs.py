###################################################################################################
# Copyright (c) 2021 Jonas Nicodemus
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
###################################################################################################
#
# This file incorporates work and modifications to the originally published code
# according to the previous license by the following contributors under the following licenses
#
#   Copyright (c) 2022 Birgit Hillebrecht
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
# 
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.
#
###################################################################################################

import numpy as np
import tensorflow as tf

from base.custom_lbfgs import lbfgs, Struct
from base.lbfgs_function_factory import function_factory

class LBFGS:
    """
    Class used to represent the L-BFGS optimizer.
    """

    def minimize(self, model, loss_fcn, x, y, callback_fcn, epochs=2000, learning_rate=1.,
                 x_test=None, y_test=None, val_freq=1000, log_freq=1000, store_freq=1000, verbose=1):
        """
        Performs the Neural Network training with the L-BFGS implementation.

        :param tf.keras.Model model: an instance of `tf.keras.Model` or its subclasses
        :param object loss_fcn: a function with signature loss_value = loss(y_pred, y_true)
        :param tf.tensor x: input tensor of the training dataset
        :param tf.tensor y: output tensor of the training dataset
        :param object callback_fcn: callback function, which is called after each epoch
        :param int epochs: number of epochs
        :param tf.tensor x_test: input tensor of the test dataset, used to evaluate accuracy
        :param tf.tensor y_test: output tensor of the test dataset, used to evaluate accuracy
        """
        func = function_factory(model, loss_fcn, x, y, callback_fcn, epochs, x_test=x_test, y_test=y_test,
                                val_freq=val_freq, log_freq=log_freq, store_freq=store_freq, verbose=verbose)

        # convert initial model parameters to a 1D tf.Tensor
        init_params = tf.dynamic_stitch(func.idx, model.trainable_variables)

        nt_epochs = epochs
        nt_config = Struct()
        nt_config.learningRate = learning_rate
        nt_config.maxIter = nt_epochs
        nt_config.nCorrection = 50
        nt_config.tolFun = 1.0 * np.finfo(float).eps

        lbfgs(func, init_params, nt_config, Struct(), True, lambda x, y, z: None)