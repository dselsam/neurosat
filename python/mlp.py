# Copyright 2018 Daniel Selsam. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import tensorflow as tf
from util import decode_transfer_fn

def init_ws_bs(opts, name, d_in, d_outs):
    ws = []
    bs = []
    d = d_in

    with tf.variable_scope(name) as scope:
        for i, d_out in enumerate(d_outs):
            with tf.variable_scope('%d' % i) as scope:
                ws.append(tf.get_variable(name="w", shape=[d, d_out], initializer=tf.contrib.layers.xavier_initializer()))
                bs.append(tf.get_variable(name="b", shape=[d_out], initializer=tf.zeros_initializer()))
            d = d_out

    return (ws, bs)

class MLP(object):
    def __init__(self, opts, d_in, d_outs, name):

        (self.ws, self.bs) = init_ws_bs(opts, name, d_in, d_outs)

        self.opts = opts
        self.name = name
        self.transfer_fn = decode_transfer_fn(opts.mlp_transfer_fn)
        self.output_size = d_outs[-1]

    def forward(self, z):
        with tf.variable_scope(self.name) as scope:
            with tf.variable_scope('fwd') as scope:
                x = z
                for i in range(len(self.ws)):
                    with tf.variable_scope('%d' % i) as scope:
                        x = tf.matmul(x, self.ws[i]) + self.bs[i]
                        if i + 1 < len(self.ws):
                            x = self.transfer_fn(x)
        return x
