from main_loop import run_main_loop

import tensorflow as tf

from BigGAN import BigGAN
from inception_score import prefetch_inception_model

import argparse
import subprocess
import os.path
import math

import logging
logger = logging.getLogger(__name__)

from utils import *
from args  import *



def main():
  args = parse_args()
  setup_logging(args)

  gan = BigGAN(args)

  params = vars(args)
  #params['batch_size'] = params['_batch_size']
  params['batch_size'] = 1

  """
  estimator = tf.estimator.Estimator(
    model_fn=lambda features, labels, mode, params: gan.gpu_model_fn(features, labels, mode, params),
    params=vars(args),
    model_dir=model_dir(args)
  )

  run_main_loop(args, estimator, estimator)
  """

  import tensorflow as tf

  sess = tf.InteractiveSession()


  #params = EasyDict({'z_dim': 128, 'batch_size': 1, 'use_tpu': False, 'ch': 64, 'sn': True, 'layers': 5, 'use_label_cond': True, 'self_attn_res': [64], 'img_ch': 3, 'img_size': 128, 'num_labels': 1000})
  #params = args

  z = tf.random.truncated_normal(shape=[params['batch_size'], params['z_dim']], name='random_z')
  label = tf.placeholder(shape=[1], dtype=tf.int32)
  labels = tf.one_hot(label, params['num_labels'])
  with tf.variable_scope('', reuse=tf.AUTO_REUSE):
    fake_images = gan.generator(params, z, labels)
  ckpt = tf.train.latest_checkpoint(params['checkpoint_dir'])
  saver = tf.train.Saver()
  saver.restore(sess, ckpt)
  with open('test.jpg', 'wb') as f:
    img = sess.run(fake_images[0] + 1.0, {label: [6]})/2*254
    f.write(sess.run(tf.io.encode_jpeg(img)))

if __name__ == '__main__':
  main()


