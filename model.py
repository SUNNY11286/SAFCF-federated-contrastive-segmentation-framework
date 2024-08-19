import tensorflow as tf

class UNET3D(tf.keras.Model):
    def __init__(self, classes):
        super(UNET3D, self).__init__()
        self.classes = classes

    def conv3D(self, x, filters, filter_size, activation='relu'):
        out = tf.keras.layers.Conv3D(filters, (filter_size, filter_size, filter_size), padding="same", use_bias=True)(x)
        batch_norm_ = tf.keras.layers.BatchNormalization()(out)
        conv_batch_norm_act = tf.keras.layers.Activation(activation)(batch_norm_)
        return conv_batch_norm_act

    def upsampling3D(self, x, filters, filter_size, stride=2, activation='relu'):
        up_out = tf.keras.layers.Conv3DTranspose(filters, (filter_size, filter_size, filter_size), strides=(stride, stride, stride), padding='same')(x)
        batch_norm_ = tf.keras.layers.BatchNormalization()(up_out)
        return batch_norm_

    def concatenate(self, x1, x2):
        return tf.keras.layers.concatenate([x1, x2])

    def max_pool3D(self, x, filter_size, stride, name=None):
        return tf.keras.layers.MaxPooling3D((filter_size, filter_size, filter_size), strides=stride, name=name)(x)

    def downconv(self, x, filters, name=None):
        s1 = self.conv3D(x, filters, 3)
        s2 = self.conv3D(s1, filters, 3)
        return s2

    def upconv(self, x, filters, skip_connection):
        e1 = self.upsampling3D(x, filters, 6)
        concat = self.concatenate(e1, skip_connection)
        conv1 = self.conv3D(concat, filters, 3)
        conv2 = self.conv3D(conv1, filters, 3)
        return conv2

    def call(self, inputs):
        d1 = self.downconv(inputs, 32)
        m1 = self.max_pool3D(d1, filter_size=2, stride=2)
        d2 = self.downconv(m1, 64)
        m2 = self.max_pool3D(d2, filter_size=2, stride=2)
        d3 = self.downconv(m2, 128)
        m3 = self.max_pool3D(d3, filter_size=2, stride=2)
        d4 = self.downconv(m3, 256)
        m4 = self.max_pool3D(d4, filter_size=2, stride=2, name="layer_before_output")

        bridge1 = self.conv3D(m4, 1024, 3)
        bridge2 = self.conv3D(bridge1, 1024, 3)

        u1 = self.upconv(bridge2, 256, d4)
        u2 = self.upconv(u1, 128, d3)
        u3 = self.upconv(u2, 64, d2)
        u4 = self.upconv(u3, 32, d1)

        logits = tf.keras.layers.Conv3D(self.classes, (1, 1, 1), padding="same")(u4)
        logits = tf.nn.sigmoid(logits)

        return tf.keras.Model(inputs=[inputs], outputs=[bridge2, logits])