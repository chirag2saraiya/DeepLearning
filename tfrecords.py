import os
import sys
import tarfile

import numpy as np
from six.moves import cPickle as pickle
from six.moves import urllib
import tensorflow as tf

CIFAR_FILENAME = 'cifar-10-python.tar.gz'
CIFAR_DOWNLOAD_URL = 'http://www.cs.toronto.edu/~kriz/' + CIFAR_FILENAME
CIFAR_LOCAL_FOLDER = 'cifar-10-batches-py'

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_DEPTH = 3
NUM_CLASSES = 10

def download_and_extract(data_dir):
  tf.contrib.learn.datasets.base.maybe_download(CIFAR_FILENAME, data_dir, CIFAR_DOWNLOAD_URL)
  tarfile.open(os.path.join(data_dir, CIFAR_FILENAME), 'r:gz').extractall(data_dir)

def _get_file_names():
  """Returns the file names expected to exist in the input_dir."""
  file_names = {}
  file_names['train'] = ['data_batch_%d' % i for i in range(1, 6)]
  file_names['validation'] = ['test_batch']
  #file_names['eval'] = ['test_batch']
  return file_names

def read_pickle_from_file(filename):
  with tf.gfile.Open(filename, 'rb') as f:
    if sys.version_info >= (3, 0):
      data_dict = pickle.load(f, encoding='bytes')
    else:
      data_dict = pickle.load(f)
  return data_dict
  
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
  
def convert_to_tfrecord(input_files, output_file):
  """Converts a file to TFRecords."""
  print('Generating %s' % output_file)
  with tf.python_io.TFRecordWriter(output_file) as record_writer:
    for input_file in input_files:
      data_dict = read_pickle_from_file(input_file)
      data = data_dict[b'data']
      labels = data_dict[b'labels']
      num_entries_in_batch = len(labels)
      for i in range(num_entries_in_batch):
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'image': _bytes_feature(data[i].tobytes()),
                'label': _int64_feature(labels[i])
            }))
        record_writer.write(example.SerializeToString())
        
def create_tfrecords_files(data_dir='cifar-10'):
  print('Download from {} and extract.'.format(CIFAR_DOWNLOAD_URL))
  download_and_extract(data_dir)
  file_names = _get_file_names()
  input_dir = os.path.join(data_dir, CIFAR_LOCAL_FOLDER)
  for mode, files in file_names.items():
    input_files = [os.path.join(input_dir, f) for f in files]
    output_file = os.path.join(data_dir, mode + '.tfrecords')
    try:
      os.remove(output_file)
    except OSError:
      pass
    # Convert to tf.train.Example and write the to TFRecords.
    convert_to_tfrecord(input_files, output_file)
  print('Done!')
  
def parse_record(serialized_example):
  features = tf.parse_single_example(
    serialized_example,
    features={
      'image': tf.FixedLenFeature([], tf.string),
      'label': tf.FixedLenFeature([], tf.int64),
    })
  
  image = tf.decode_raw(features['image'], tf.uint8)
  image.set_shape([IMAGE_DEPTH * IMAGE_HEIGHT * IMAGE_WIDTH])
  image = tf.reshape(image, [IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH])
  image = tf.cast(tf.transpose(image, [1, 2, 0]), tf.float32)
  
  label = tf.cast(features['label'], tf.int32)
  label = tf.one_hot(label, NUM_CLASSES)

  return image, label

def preprocess_image(image, is_training=False):
  """Preprocess a single image of layout [height, width, depth]."""
  if is_training:
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_image_with_crop_or_pad(
        image, IMAGE_HEIGHT + 8, IMAGE_WIDTH + 8)

    # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
    image = tf.random_crop(image, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

  # Subtract off the mean and divide by the variance of the pixels.
  image = tf.image.per_image_standardization(image)
  return image

def generate_input_fn(file_names, mode='validate', batch_size=1):
  
    dataset = tf.data.TFRecordDataset(filenames=file_names)

    is_training = (mode == 'train')
    if is_training:
      buffer_size = batch_size * 2 + 1
      dataset = dataset.shuffle(buffer_size=buffer_size)

    # Transformation
    dataset = dataset.map(parse_record)
  #  dataset = dataset.map(
     # lambda image, label: (preprocess_image(image, is_training), label))

    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2 * batch_size)

    images, labels = dataset.make_one_shot_iterator().get_next()

        # Bring your picture back in shape
    images = tf.reshape(image, [-1, 32, 32, 3])
    
    # Create a one hot array for your labels
    labels = tf.one_hot(label, NUM_CLASSES)
    
    return images, labels
