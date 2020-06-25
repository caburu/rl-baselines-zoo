# O código abaixo foi retirado do [tutorial](https://github.com/araffin/rl-tutorial-jnrr19/blob/master/3_multiprocessing.ipynb). 
# Ele é útil para deixar as saídas do notebooks sem os warnings do TensorFlow 
# (existem muitos porque a Stable Baselines trabalha com TensorFlow 1.x, e os 
#  warnings se referem à migração para 2.x).

# Filter tensorflow version warnings
import os
# https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import warnings
# https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
import logging
tf.get_logger().setLevel(logging.ERROR)
