# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import tensorflow as tf

from google.protobuf import text_format
from object_detection import exporter
from object_detection.protos import pipeline_pb2
from object_detection import model_hparams
from object_detection import model_lib
from data_preparation import txt_to_csv, generate_tfrecord

def train_model(model, pipeline_path, training_steps=10000):

    config = tf.estimator.RunConfig(model_dir=model)

    train_and_eval_dict = model_lib.create_estimator_and_inputs(
        run_config = config,
        hparams = model_hparams.create_hparams(),
        pipeline_config_path = pipeline_path,
        train_steps = training_steps,
        sample_1_of_n_eval_examples = 1,
        sample_1_of_n_eval_on_train_examples=(5))
    estimator = train_and_eval_dict['estimator']
    train_input_fn = train_and_eval_dict['train_input_fn']
    eval_input_fns = train_and_eval_dict['eval_input_fns']
    eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
    predict_input_fn = train_and_eval_dict['predict_input_fn']
    train_steps = train_and_eval_dict['train_steps']
    train_spec, eval_specs = model_lib.create_train_and_eval_specs(
        train_input_fn,
        eval_input_fns,
        eval_on_train_input_fn,
        predict_input_fn,
        train_steps,
        eval_on_train_data=False)
    # Currently only a single Eval Spec is allowed.
    print("Training starts!")
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])
    print("Training ends!")
    
def export_graph(pipeline_path, model_path, export_dir):
  pipeline_config_path = pipeline_path
  trained_checkpoint_prefix = model_path
  output_directory = export_dir
  config_override = ''
  input_shape = None

  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  with tf.gfile.GFile(pipeline_config_path, 'r') as f:
    text_format.Merge(f.read(), pipeline_config)
  text_format.Merge(config_override, pipeline_config)
  if input_shape:
    input_shape = [
        int(dim) if dim != '-1' else None
        for dim in input_shape.split(',')
    ]
  else:
    input_shape = None

  exporter.export_inference_graph(
      'image_tensor', pipeline_config, trained_checkpoint_prefix,
      output_directory, input_shape=input_shape,
      write_inference_graph=False)
  print("Successfully created inference graph!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lets train a phone detector')
    parser.add_argument('input_directory', type=str, help='Location of input data')
    args = parser.parse_args()
    INPUT_DIR = args.input_directory
    CONFIG_DIR = "training"
    MODEL_DIR = "model"
    GRAPH_DIR = "graph"
    pipeline_path = os.path.join(CONFIG_DIR, 'ssd_mobilenet_v1_coco.config')
    model_path = os.path.join(MODEL_DIR, 'model.ckpt-10000')

    txt_to_csv(INPUT_DIR)
    generate_tfrecord(INPUT_DIR, os.path.join(CONFIG_DIR,'train.csv'), 'training/train.record')
    generate_tfrecord(INPUT_DIR, os.path.join(CONFIG_DIR, 'val.csv'), 'training/val.record')
    train_model(MODEL_DIR, pipeline_path)
    export_graph(pipeline_path, model_path, GRAPH_DIR)
    print("Successfully trained and exported the model!\nNow you can run find_phone.py!")
