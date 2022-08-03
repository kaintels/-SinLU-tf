import tensorflow as tf
from tensorflow.keras import layers


class SinLU(layers.Layer):
    def __init__(self):
        super(SinLU, self).__init__()
        self.a = tf.Variable(name="a", initial_value=1.0, trainable=True)
        self.b = tf.Variable(name="b", initial_value=1.0, trainable=True)

    def call(self, x):
        return (x+self.a*tf.sin(self.b*x))*tf.sigmoid(x)