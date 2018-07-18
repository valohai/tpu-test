# From https://cloud.google.com/tpu/docs/quickstart
import os
import tensorflow as tf
from tensorflow.contrib import tpu
from tensorflow.contrib.cluster_resolver import TPUClusterResolver

tpu_name = os.environ.get('TPU_NAME')
use_tpu = bool(tpu_name)

def axy_computation(a, x, y):
  return a * x + y

inputs = [
    3.0,
    tf.ones([80, 80], tf.float32),
    tf.ones([80, 80], tf.float32),
]

if use_tpu:
  computation = tpu.rewrite(axy_computation, inputs)
  tpu_grpc_url = TPUClusterResolver(tpu=[tpu_name]).get_master()
else:
  computation = tf.py_func(axy_computation, inputs, tf.float32)
  tpu_grpc_url = None

with tf.Session(tpu_grpc_url) as sess:
  if use_tpu:
    sess.run(tpu.initialize_system())
  sess.run(tf.global_variables_initializer())
  output = sess.run(computation)
  print(output)
  
  if use_tpu:
    sess.run(tpu.shutdown_system())

print('Done!')