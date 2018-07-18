# Based on https://cloud.google.com/tpu/docs/quickstart
import os
import tensorflow as tf
from tensorflow.contrib import tpu
from tensorflow.contrib.cluster_resolver import TPUClusterResolver

tpu_name = os.environ.get('TPU_NAME')
use_tpu = bool(tpu_name)


def axy_computation(a, x, y):
    return a * x + y

output_shape = [80, 80]

inputs = [
    3.0,
    tf.ones(output_shape, tf.float32),
    tf.ones(output_shape, tf.float32),
]

if use_tpu:
    print('Setting up TPU')
    tpu_grpc_url = TPUClusterResolver(tpu=[tpu_name]).get_master()
    computation = tpu.rewrite(axy_computation, inputs)
else:
    print('TPU IS NOT ENABLED (pass a TPU name or grpc://ip:port as the TPU_NAME envvar)')
    computation = tf.py_func(axy_computation, inputs, tf.float32)
    tpu_grpc_url = None


with tf.Session(tpu_grpc_url) as sess:
    if use_tpu:
        print('Running TPU initializer')
        sess.run(tpu.initialize_system())
    sess.run(tf.global_variables_initializer())
    print('Running computation {}'.format(computation))
    output = sess.run(computation)
    print(output)
    output_var = tf.get_variable('output', output_shape)
    sess.run(tf.assign(output_var, output))

    if use_tpu:
        print('Shutting down TPU')
        sess.run(tpu.shutdown_system())

    print('Saving session information')
    saver = tf.train.Saver()
    save_path = saver.save(sess, os.environ.get('VH_OUTPUTS_DIR', '.') + '/model.ckpt')
    print('OK: {}'.format(save_path))

print('Done!')
