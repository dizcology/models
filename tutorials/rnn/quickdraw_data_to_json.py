# Copyright 2019 Google LLC
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

import json
import tensorflow as tf
from quickdraw.train_model import get_input_fn

batch_size = 10


def get_eval_data():
    mode = tf.estimator.ModeKeys.EVAL
    tfrecord_pattern = 'gs://cloud-samples-data/ml-engine/quickdraw_tutorial_dataset_v1/eval.tfrecord-?????-of-?????'
    
    input_fn = get_input_fn(mode, tfrecord_pattern, batch_size)

    get_features, get_labels = input_fn()

    # Get only one batch and return features.
    with tf.Session() as sess:
        features = sess.run(get_features)

    return features


def features_to_instances_json(features):
    output_filename = 'instances_input.json'

    # Restructure features into instances.
    instances = {'instances': []}

    # Send only 'ink' and 'shape'.
    for i, (single_ink, single_shape) in enumerate(zip(features['ink'], features['shape'])):
        instance = {
            'ink': single_ink.tolist(),
            'shape': single_shape.tolist(),
            'key': i
        }
        instances['instances'].append(instance)

    with open(output_filename, 'w') as f:
        f.write(json.dumps(instances))


def features_to_json(features):
    output_filename = 'input.json'

    with open(output_filename, 'w') as f:
        # Send only 'ink' and 'shape'.
        for i, (single_ink, single_shape) in enumerate(zip(features['ink'], features['shape'])):

            # TODO: look into keeping the shape as in the tf.example data.
            single_ink = single_ink.reshape((-1, 3))
            instance = {
                'ink': single_ink.tolist(),
                'shape': single_shape.tolist(),
                'key': i
            }

            f.write(json.dumps(instance))
            f.write('\n')


if __name__ == '__main__':
    features = get_eval_data()
    features_to_json(features)
    features_to_instances_json(features)
