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

import argparse
import urllib

import json
import os
import adanet
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import tensorflow_hub as hub

from . import model

_DATASET_URL = 'https://storage.googleapis.com/authors-training-data/data.csv'


def get_args():
    """Collect Arguments from command line.

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        type=str,
        help='GCS location to write checkpoints and export models')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Number of records to read during each training step, default=64')
    parser.add_argument(
        '--total-steps',
        type=int,
        default=40000,
        help='Number of records to read during each training step, '
             'default=40000')
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')
    args, _ = parser.parse_known_args()
    return args


def load_data():
    """Get the data

    :return:
    """
    urllib.request.urlretrieve(_DATASET_URL, 'data.csv')

    data = pd.read_csv('data.csv')
    data = data.sample(frac=1)

    # Split into train and test sets
    train_size = int(len(data) * .8)

    train_text = data['text'][:train_size]
    train_authors = data['author'][:train_size]

    test_text = data['text'][train_size:]
    test_authors = data['author'][train_size:]

    # Turn the labels into a one-hot encoding
    encoder = LabelEncoder()
    encoder.fit_transform(np.array(train_authors))
    train_encoded = encoder.transform(train_authors)
    test_encoded = encoder.transform(test_authors)

    train_labels = np.array(train_encoded).astype(np.int32)
    eval_labels = np.array(test_encoded).astype(np.int32)

    return train_text, train_labels, test_text, eval_labels, encoder


def _get_session_config_from_env_var():
    """Returns a tf.ConfigProto instance that has appropriate device_filters
    set."""

    tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))

    if (tf_config and 'task' in tf_config and 'type' in tf_config['task'] and
        'index' in tf_config['task']):
        # Master should only communicate with itself and ps
        if tf_config['task']['type'] == 'master':
            return tf.ConfigProto(log_device_placement=False,
                                  device_filters=['/job:ps', '/job:master'])
        # Worker should only communicate with itself and ps
        elif tf_config['task']['type'] == 'worker':
            return tf.ConfigProto(log_device_placement=False,
                                  device_filters=[
                                      '/job:ps',
                                      '/job:worker/task:%d' % tf_config['task'][
                                          'index']
                                  ])
    return None


def train_and_evaluate(args):
    """

    :param args:
    :return:
    """
    
    train_text, train_labels, test_text, eval_labels, encoder = load_data()

    # Create TF Hub embedding columns using 2 different modules
    ndim_embeddings = hub.text_embedding_column(
        "ndim",
        module_spec="https://tfhub.dev/google/nnlm-en-dim128/1",
        trainable=False
    )
    encoder_embeddings = hub.text_embedding_column(
        "encoder",
        module_spec="https://tfhub.dev/google/universal-sentence-encoder/2",
        trainable=False)

    # Create a head and features dict for training
    multi_class_head = tf.contrib.estimator.multi_class_head(
        len(encoder.classes_),
        loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE
    )

    # Define the Estimators we'll be feeding into our AdaNet model
    estimator_ndim = tf.contrib.estimator.DNNEstimator(
        head=multi_class_head,
        hidden_units=[64, 10],
        feature_columns=[ndim_embeddings]
    )

    estimator_encoder = tf.contrib.estimator.DNNEstimator(
        head=multi_class_head,
        hidden_units=[64, 10],
        feature_columns=[encoder_embeddings]
    )

    # Create our AutoEnsembleEstimator from the 2 estimators above
    estimator = adanet.AutoEnsembleEstimator(
        head=multi_class_head,
        candidate_pool=[
            estimator_encoder,
            estimator_ndim
        ],
        config=tf.estimator.RunConfig(
            save_summary_steps=1000,
            save_checkpoints_steps=1000,
            model_dir=args.job_dir,
            session_config=_get_session_config_from_env_var()
        ),
        max_iteration_steps=5000)

    # Training
    train_features = {
        "ndim": train_text,
        "encoder": train_text
    }

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: model.input_fn_train(train_features, train_labels,
                                              args.batch_size),
        max_steps=args.total_steps
    )

    # Evaluation
    eval_features = {
        "ndim": test_text,
        "encoder": test_text
    }

    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: model.input_fn_eval(eval_features, eval_labels,
                                             args.batch_size),
        steps=None,
        start_delay_secs=10,
        throttle_secs=10
    )

    # Configurations for running train_and_evaluate
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    latest_ckpt = tf.train.latest_checkpoint(args.job_dir)
    last_eval = estimator.evaluate(
        lambda: model.input_fn_eval(eval_features, eval_labels,
                                    args.batch_size),
        checkpoint_path=latest_ckpt
    )

    # Export the model to GCS for serving.
    exporter = tf.estimator.LatestExporter('exporter', model.serving_input_fn,
                                           exports_to_keep=None)
    exporter.export(estimator, args.job_dir, latest_ckpt, last_eval,
                    is_the_final_export=True)


if __name__ == '__main__':
    args = get_args()
    tf.logging.set_verbosity(args.verbosity)

    # Run the training job
    train_and_evaluate(args)
