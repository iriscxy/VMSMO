import os
import sys
import numpy as np
import h5py
import argparse
import json
import cv2

import tensorflow as tf
from nets import resnet_v1
from preprocessing import vgg_preprocessing

parser = argparse.ArgumentParser(description='Preprocess image for VideoSum')
# parser.add_argument('--inputJson', type=str, default='visdial_params.json', help='Path to JSON file')
parser.add_argument('--imageRoot', type=str, default='/share/lmz/video_sum_data/video_new_img/', help='Path to video image root')
parser.add_argument('--cnnModel', type=str, default='resnet_v1_152.ckpt', help='Path to the CNN model')
parser.add_argument('--batchSize', type=int, default=50, help='Batch size')
parser.add_argument('--outName', type=str, default='fake.h5', help='Output name')
parser.add_argument('--gpuid', type=str, default='6', help='Which gpu to use.')
parser.add_argument('--imgSize', type=int, default=224)
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=10)

args = parser.parse_args()
slim = tf.contrib.slim
print(args.outName)
print(args.gpuid)
print(args.start)
print(args.end)

# if True:
#   os.system('wget http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz')
#   os.system('tar -xvzf resnet_v1_152_2016_08_28.tar.gz')
#   os.system('rm resnet_v1_152_2016_08_28.tar.gz')

def extract_feature(imgList, args):
  # tf.reset_default_graph()

  # queue = tf.train.string_input_producer(imgList, num_epochs=None, shuffle=False)
  # reader = tf.WholeFileReader()

  # img_path, img_data = reader.read(queue)
  # img = vgg_preprocessing.preprocess_image(tf.image.decode_jpeg(contents=img_data, channels=3), 128, 256)
  # img = tf.expand_dims(img, 0)
  side_batch = tf.placeholder(tf.float32, [200, 32, 64, 3], name='side_batch')
  with slim.arg_scope(resnet_v1.resnet_arg_scope()):
    net, end_points = resnet_v1.resnet_v1_152(inputs=side_batch, is_training=True)
  # feat1 = end_points['resnet_v1_152/block4']
  feat2 = end_points['global_pool']

  # saver = tf.train.Saver()
  checkpoint_exclude_scopes = 'Logits'
  exclusions = None
  if checkpoint_exclude_scopes:
      exclusions = [
          scope.strip() for scope in checkpoint_exclude_scopes.split(',')]
  variables_to_restore = []
  for var in slim.get_model_variables():
      excluded = False
      for exclusion in exclusions:
          if var.op.name.startswith(exclusion):
              excluded = True
      if not excluded:
          variables_to_restore.append(var)
  saver = tf.train.Saver(var_list=variables_to_restore)  # keep 3 checkpoints at a time

  # init_op = tf.global_variables_initializer()

  with tf.Session() as sess:
    # sess.run(init_op)
    saver.restore(sess, args.cnnModel)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    feats1 = []
    feats2 = []
    for i in range(len(imgList)):
      # f1, f2 = sess.run([feat1, feat2]) # f1: (1, 4, 8, 2048)   f2: (1, 1, 1, 2048)
      f2 = sess.run(feat2, feed_dict={
        side_batch: np.array(imgList)
      })
      # feats1.append(f1[0])
      feats2.append(f2[0][0][0])
      if (i+1)%1000==0:
        print('%s/%s'%(i+1,len(imgList)))
    coord.request_stop()
    coord.join(threads)
  # return feats1, feats2
  return feats2

# jsonFile = json.load(open(args.inputJson, 'r'))
# trainList = []
# for img in jsonFile['unique_img_train']:
#   trainList.append(os.path.join(args.imageRoot, 'train2014/COCO_train2014_%012d.jpg'%img))
# valList = []
# for img in jsonFile['unique_img_val']:
#   valList.append(os.path.join(args.imageRoot, 'val2014/COCO_val2014_%012d.jpg'%img))

# jsonFile = open(args.inputJson, 'r', encoding='utf-8')
# dirs = '/share/lmz/video_sum_data/video_img/20180513 _GgqNq4Fdv'
# new_dirs = '/share/lmz/video_sum_data/video_img/20200310 _IxXqFyzfA'
# pics = os.listdir(new_dirs)
# trainList = []
# for pic in pics:
#   trainList.append(os.path.join(args.imageRoot, 'train/{}.jpg'.format(i)))
# trainList = []
# for i in range(args.start, args.end):
# # for i in range(1, 20699089):
#     trainList.append(os.path.join(args.imageRoot, 'train/{}.jpg'.format(i)))
#     if i % 1000 == 0:
#         print(i)
# valList = []
# for i in range(1, 250996):
# for i in range(args.start, args.end):
#     valList.append(os.path.join(args.imageRoot, 'dev/{}.jpg'.format(i)))
#     if i % 1000 == 0:
#         print(i)
# testList = []
# # for i in range(1, 255301):
# for i in range(args.start, args.end):
#     testList.append(os.path.join(args.imageRoot, 'test/{}.jpg'.format(i)))
#     if i % 1000 == 0:
#         print(i)

trainList = []
for i in range(147432, 147632):
  image = cv2.resize(cv2.imread("/share/lmz/video_sum_data/video_new_img/train/{}.jpg".format(i)), (64, 32))
  trainList.append(image)
  # trainList.append(os.path.join(args.imageRoot, 'train/{}.jpg'.format(i)))


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.5
placeholder_session = tf.Session(config=config)
limit = placeholder_session.run(tf.contrib.memory_stats.BytesLimit()) / 1073741824
tf.logging.info('occupy GPU memory %f GB', limit)

# print('Extracting the feature of training images ...')
# trainFeats1 = extract_feature(trainList, args)
# print('Extracting the feature of valid images ...')
# valFeats1 = extract_feature(valList, args)
print('Extracting the feature of testing images ...')
testFeats1 = extract_feature(trainList, args)

print('Saving hdf5...')
f = h5py.File(args.outName, 'w')
# f.create_dataset('images_train_7', data=trainFeats7)
# f.create_dataset('images_train_lmz', data=trainFeats1)
# f.create_dataset('images_val_7', data=valFeats7)
# f.create_dataset('images_val_lmz', data=valFeats1)
f.create_dataset('images_test_lmz', data=testFeats1)

f.close()
