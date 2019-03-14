# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def input_fn_train(train_features, train_labels, batch_size):
    """Input training function.

    :return:
    """
    dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
    dataset = dataset.repeat().shuffle(100).batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    data, labels = iterator.get_next()
    return data, labels


def input_fn_eval(eval_features, eval_labels, batch_size):
    """Input evaluation function.

    :return:
    """
    dataset = tf.data.Dataset.from_tensor_slices((eval_features, eval_labels))
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    data, labels = iterator.get_next()
    return data, labels


def serving_input_fn():
    """Create serving input function to be able to serve predictions later
    using provided inputs

    :return:
    """
    feature_placeholders = {
        'encoder': tf.placeholder(tf.string, [None]),
        'ndim': tf.placeholder(tf.string, [None])
    }
    return tf.estimator.export.ServingInputReceiver(feature_placeholders,
                                                    feature_placeholders)
