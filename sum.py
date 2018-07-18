# Based on https://cloud.google.com/tpu/docs/quickstart
import os
import tensorflow as tf
from tensorflow.contrib import tpu
from tensorflow.contrib.cluster_resolver import TPUClusterResolver

tpu_name = os.environ.get('TPU_NAME')
use_tpu = bool(tpu_name)
output_dir = os.environ.get('VH_OUTPUTS_DIR', '.')


def axy_computation(a, x, y):
    return a * x + y

output_shape = [80, 80]

inputs = [
    3.0,
    tf.random_uniform(output_shape, dtype=tf.float32),
    tf.random_uniform(output_shape, dtype=tf.float32),
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

    if not use_tpu:
        # For whichever reason, we can't do this in the TPU environment...
        output_var = tf.get_variable('output', output_shape)
        sess.run(tf.assign(output_var, output))
        save_path = tf.train.Saver().save(sess, output_dir + '/model.ckpt')
        print('Saved model to: {}'.format(save_path))

    with open(output_dir + '/output.txt', 'w') as outf:
        outf.write(repr(output))
        print('Saved output data to: {}'.format(outf.name))


    if use_tpu:
        print('Shutting down TPU')
        sess.run(tpu.shutdown_system())


print('Done!')
