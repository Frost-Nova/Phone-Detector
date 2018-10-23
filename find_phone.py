#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import numpy as np
import tensorflow as tf

from distutils.version import StrictVersion
from PIL import Image
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in ['num_detections', 'detection_boxes', 'detection_scores']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
  return output_dict

def find(image_path, detection_graph):
  image = Image.open(image_path)
  image_np = load_image_into_numpy_array(image)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  # Actual detection.
  output_dict = run_inference_for_single_image(image_np, detection_graph)
  # Final result of a detection
  x_mean = (output_dict['detection_boxes'][0][1]+output_dict['detection_boxes'][0][3])/2
  y_mean = (output_dict['detection_boxes'][0][0]+output_dict['detection_boxes'][0][2])/2
  print("%.4f %.4f"% (x_mean, y_mean))

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='This is a phone detector')
  parser.add_argument('image_path', type=str, help='path to test image')
  args = parser.parse_args()
  image_path = args.image_path

  # Path to frozen detection graph. This is the actual model that is used for the object detection.
  PATH_TO_FROZEN_GRAPH = 'graph/frozen_inference_graph.pb'
  # Load a frozen Tensorflow model into memory.
  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')
  find(image_path, detection_graph)
