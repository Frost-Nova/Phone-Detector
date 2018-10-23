import os
import io
import csv
import pandas as pd
import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple

def txt_to_csv(Path):
    lamda_x = lamda_y = 25  # Width and height of bounding box
    text_file = open(Path + "/labels.txt", "r")
    lines = text_file.readlines()
    N = len(lines)
    text_file.close()
    with open('training/val.csv', 'w', newline='') as val_file:
        csv_writer = csv.writer(val_file)
        csv_writer.writerow(['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
        for i in range(13):
            line = lines[i]
            file_name = line.split(' ')[0]
            x_center = int(490*float(line.split(' ')[1]))
            y_center = int(326*float(line.split(' ')[2]))
            csv_writer.writerow([file_name, 490, 326, 'phone', x_center-lamda_x, y_center-lamda_y, x_center+lamda_x, y_center+lamda_y])
    with open('training/train.csv', 'w', newline='') as train_file:
        csv_writer = csv.writer(train_file)
        csv_writer.writerow(['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
        for i in range(13, N):
            line = lines[i]
            file_name = line.split(' ')[0]
            x_center = int(490*float(line.split(' ')[1]))
            y_center = int(326*float(line.split(' ')[2]))
            csv_writer.writerow([file_name, 490, 326, 'phone', x_center-lamda_x, y_center-lamda_y, x_center+lamda_x, y_center+lamda_y])
    print('Successfully converted txt to csv.')

def class_text_to_int(row_label):
    if row_label == 'phone':
        return 1
    else:
        None

# Create a tf instance from an image and its record
def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def generate_tfrecord(image_input, csv_input, record_file):
    writer = tf.python_io.TFRecordWriter(record_file)
   # path = os.path.join(os.getcwd(), image_input)
    examples = pd.read_csv(csv_input)
    data = namedtuple('data', ['filename', 'object'])
    gb = examples.groupby('filename')
    grouped = [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]
    for group in grouped:
        tf_example = create_tf_example(group, image_input)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print('Successfully created the TFRecords: {}'.format(record_file))

#def main():
#    INPUT_PATH = "find_phone"
#    OUTPUT_PATH = "training"
 #   txt_to_csv(INPUT_PATH)
  #  generate_tfrecord(INPUT_PATH, os.path.join(OUTPUT_PATH,'train.csv'), 'training/train.record')
   # generate_tfrecord(INPUT_PATH, os.path.join(OUTPUT_PATH, 'val.csv'), 'training/val.record')
