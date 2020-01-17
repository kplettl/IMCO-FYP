import os
import scipy.misc
import numpy as np 

from model import DCGAN
from utils import pp, visualize, to_json

import tensorflow as tf 

flags = tf.app.flags
#flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
#flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
#flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
#flags.DEFINE_integer("image_size", 64, "the size of image to use")
#flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
#flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
#flags.DEFINE_integer("grid_height", 8, "Grid Height")
#flags.DEFINE_integer("grid_width", 8, "Grid Width")
#flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
#flags.DEFINE_integer("output_width", 64, "The size of the output images to produce. If None, same value as output_height [None]")
#flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
#flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
#flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
#flags.DEFINE_integer("sample_rate", None, "If == 5, it will take a sample image every 5 iterations")
#flags.DEFINE_integer("nbr_of_layers_d", 5, "Number of layers in Discriminator")
#flags.DEFINE_integer("nbr_of_layers_g", 5, "Number of layers in Generator")
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "Size of batch images [64]")
flags.DEFINE_integer("image_size", 64, "the size of image to use")
#flags.DEFINE_integer("image_size", 64, "the size of image to use"
flags.DEFINE_string("dataset", r"C:\Users\plettlk\DCGAN_Image_Completion\61119\scaled_tofu", "Dataset directory")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
FLAGS = flags.FLAGS


if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

config = tf.ConfigProto()
#config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
#    dcgan = DCGAN(
#          sess, image_size=FLAGS.image_size,
#          output_width=FLAGS.output_width,
#          output_height=FLAGS.output_height,
#          grid_height=FLAGS.grid_height,
#          grid_width=FLAGS.grid_width,
#          sample_size=FLAGS.batch_size,
#          checkpoint_dir=FLAGS.checkpoint_dir,
#          nbr_of_layers_d=FLAGS.nbr_of_layers_d,
#          nbr_of_layers_g=FLAGS.nbr_of_layers_g)
    dcgan = DCGAN(sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size,
                  sample_size=FLAGS.batch_size, is_crop=False, checkpoint_dir=FLAGS.checkpoint_dir)

    dcgan.train(FLAGS)