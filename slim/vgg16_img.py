import os,time
import sys
import numpy as np
import h5py
import argparse
import json

import tensorflow as tf
from slim.nets import vgg
from slim.preprocessing import vgg_preprocessing

# parser = argparse.ArgumentParser(description='Preprocess image for Visual Dialogue')
# parser.add_argument('--inputJson', type=str, default='visdial_params.json', help='Path to JSON file')
# parser.add_argument('--imageRoot', type=str, default='images/', help='Path to COCO image root')
# parser.add_argument('--cnnModel', type=str, default='vgg_16.ckpt', help='Path to the CNN model')
# parser.add_argument('--batchSize', type=int, default=50, help='Batch size')
# parser.add_argument('--outName', type=str, default='vgg_img.h5', help='Output name')
# parser.add_argument('--gpuid', type=str, default='3', help='Which gpu to use.')
# parser.add_argument('--imgSize', type=int, default=224)


parser = argparse.ArgumentParser(description='Preprocess image for Visual Dialogue')
parser.add_argument('--inputJson', type=str, default='visdial_params.json', help='Path to JSON file')
parser.add_argument('--imageRoot', type=str, default='images/', help='Path to COCO image root')
parser.add_argument('--cnnModel', type=str, default='vgg_16.ckpt', help='Path to the CNN model')
parser.add_argument('--batchSize', type=int, default=50, help='Batch size')
parser.add_argument('--outName', type=str, default='vgg_img.h5.tmp', help='Output name')
parser.add_argument('--gpuid', type=str, default='3', help='Which gpu to use.')
parser.add_argument('--imgSize', type=int, default=224)

args = parser.parse_args()

if not os.path.exists('vgg_16.ckpt'):
#if True:
  os.system('wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz')
  os.system('tar -xvzf vgg_16_2016_08_28.tar.gz')
  os.system('rm vgg_16_2016_08_28.tar.gz')

def extract_feature(imgList, args):
  tf.reset_default_graph()

  queue = tf.train.string_input_producer(imgList, num_epochs=None, shuffle=False)
  reader = tf.WholeFileReader()

  img_path, img_data = reader.read(queue)
  img = vgg_preprocessing.preprocess_image(tf.image.decode_jpeg(contents=img_data, channels=3), args.imgSize, args.imgSize)
  img = tf.expand_dims(img, 0)

  _, _, features = vgg.vgg_16(img, is_training=False)
  pool5 = features['pool5'] # [1,7,7,512]
  fc7 = features['fc7'] # [1,1,1, 4096]


  saver = tf.train.Saver()
  init_op = tf.global_variables_initializer()

  pool5s = []
  fc7s = []

  with tf.Session() as sess:
    sess.run(init_op)
    saver.restore(sess, args.cnnModel)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    feats = []
    for i in range(len(imgList)):
      p, f = sess.run([pool5, fc7]) # (1, 7, 7, 512) (1, 1, 1, 4096)
      pool5s.append(p[0])
      fc7s.append(f[0][0][0])
      if (i+1)%10000==0:
        print('%s/%s'%(i,len(imgList)))
    coord.request_stop()
    coord.join(threads)
  return pool5s, fc7s

jsonFile = json.load(open(args.inputJson, 'r'))
trainList = []
print('Loading training images ...')
for img in jsonFile['unique_img_train']:
  trainList.append(os.path.join(args.imageRoot, 'train2014/COCO_train2014_%012d.jpg'%img))
print('Training images are Loaded ...')

print('Load valid images ...')
valList = []
for img in jsonFile['unique_img_val']:
  valList.append(os.path.join(args.imageRoot, 'val2014/COCO_val2014_%012d.jpg'%img))
print('Valid images are Loaded ...')

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid

print('Extracting the feature of training images ...')
trainFeats7, trainFeats1 = extract_feature(trainList, args)
print('Extracting the feature of valid images ...')

valFeats7, valFeats1 = extract_feature(valList, args)

print('Saving hdf5...')
f = h5py.File(args.outName, 'w')
f.create_dataset('images_train_7', data=trainFeats7)
f.create_dataset('images_train_1', data=trainFeats1)
f.create_dataset('images_val_7', data=valFeats7)
f.create_dataset('images_val_1', data=valFeats1)
f.close() 
