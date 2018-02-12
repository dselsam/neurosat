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

import tensorflow as tf

def repeat_end(val, n, k):
    return [val for i in range(n)] + [k]

def reduce_with(vec, sizes, fn, final_shape):
    n_groups = tf.shape(sizes)[0]
    start_array = tf.TensorArray(dtype=tf.float32, size=n_groups, infer_shape=False).split(value=vec, lengths=sizes)
    end_array   = tf.TensorArray(dtype=tf.float32, size=n_groups, infer_shape=True)

    result = tf.while_loop((lambda i, sa, ea: i < n_groups),
                         (lambda i, sa, ea: (i+1, sa, ea.write(i, fn(sa.read(i))))),
                         [0, start_array, end_array])[2].stack()

    return tf.reshape(result, final_shape)

def decode_final_reducer(reducer):
    if reducer == "min":
        return (lambda x: tf.reduce_min(x, axis=[1, 2]))
    elif reducer == "mean":
        return (lambda x: tf.reduce_mean(x, axis=[1, 2]))
    elif reducer == "sum":
        return (lambda x: tf.reduce_sum(x, axis=[1, 2]))
    elif reducer == "max":
        return (lambda x: tf.reduce_max(x, axis=[1, 2]))
    else:
        raise Exception("Expecting min, mean, or max")

def decode_msg_reducer(reducer):
    if reducer == "min":
        return (lambda x: tf.reduce_min(tf.concat([x, tf.zeros([1, tf.shape(x)[1]])], axis=0), axis=0))
    elif reducer == "mean":
        return (lambda x: tf.reduce_mean(tf.concat([x, tf.zeros([1, tf.shape(x)[1]])], axis=0), axis=0))
    elif reducer == "sum":
        return (lambda x: tf.reduce_sum(x, axis=0))
    elif reducer == "max":
        return (lambda x: tf.reduce_max(tf.concat([x, tf.zeros([1, tf.shape(x)[1]])], axis=0), axis=0))
    else:
        raise Exception("Expecting min, mean, or max")

def decode_transfer_fn(transfer_fn):
    if transfer_fn == "relu": return tf.nn.relu
    elif transfer_fn == "tanh": return tf.nn.tanh
    elif transfer_fn == "sig": return tf.nn.sigmoid
    else:
        raise Exception("Unsupported transfer function %s" % transfer_fn)
