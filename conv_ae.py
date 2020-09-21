'''
  Author       : Bao Jiarong
  Creation Date: 2020-09-21
  email        : bao.salirong@gmail.com
  Task         : conv_ae for super resolution
'''

import tensorflow as tf

class Conv_ae(tf.keras.Model):
    def __init__(self, latent = 100, units = 16, input_shape = None):
        super(Conv_ae, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters = units << 0,kernel_size=(3,3),strides=(2,2),padding = 'same',activation = "relu")
        # self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))
        self.conv2 = tf.keras.layers.Conv2D(filters = units << 1,kernel_size=(3,3),strides=(2,2),padding = 'same',activation = "relu")
        # self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))

        self.la_conv = tf.keras.layers.Conv2D(filters = units << 2,kernel_size=(3,3),strides=(2,2),padding = 'same',activation = "relu")

        # self.upsampling1  = tf.keras.layers.UpSampling2D((2, 2))
        self.transpose1   = tf.keras.layers.Conv2DTranspose(filters = units << 1 , kernel_size=(3,3), strides=(2,2), padding='same')
        # self.upsampling2  = tf.keras.layers.UpSampling2D((2, 2))
        self.transpose2   = tf.keras.layers.Conv2DTranspose(filters = units << 0 , kernel_size=(3,3), strides=(2,2), padding='same')
        self.transpose3   = tf.keras.layers.Conv2DTranspose(filters = 3 , kernel_size=(3,3), strides=(2,2), padding='same')

    def call(self, inputs):
        x = inputs
        x1 = self.conv1(x)
        x3 = self.conv2(x1)

        x5 = self.la_conv(x3)

        x7 = self.transpose1(x5)
        x9 = self.transpose2(x7)
        x  = self.transpose3(x9)
        return x

#------------------------------------------------------------------------------

def Conv_AE(input_shape, latent, units):
    model = Conv_ae(latent = latent, units = units, input_shape = input_shape)
    model.build(input_shape = input_shape)
    return model
