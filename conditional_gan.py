
#%%

import tensorflow as tf
import numpy as np
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

#%%

class cGAN():

    def __init__(self, critic, generator, gen_optimizer, critic_optimizer, input_shape):
        self.critic = critic
        self.generator = generator
        self.gen_optimizer = gen_optimizer
        self.critic_optimizer = critic_optimizer

        self.cgan = cGAN.cGAN_model(critic, generator, input_shape)
        
    @staticmethod
    def cGAN_model(critic, generator, input_shape):

        X = tf.keras.Input(shape = input_shape)

        Y = tf.keras.Input(shape = input_shape)

        critic_real_score = critic((X,Y))

        generated_y  = generator(X)

        critic_fake_score = critic((X,generated_y))

        generator_loss = -1 * critic_fake_score #optimizes for higher scores on fake items -> more believable

        critic_loss = critic_fake_score - critic_real_score #optimizes for critic_real_score > critic_fake_score

        L1_loss = tf.reduce_mean(tf.math.abs(Y - generated_y)) # MAE for generator and target image

        return tf.keras.Model(inputs = [X,Y], outputs = [critic_loss, generator_loss, L1_loss])

    def train_step(self, X,Y, lambd = 100):

        with tf.GradientTape() as tape:

            (critic_loss, generator_loss, L1_loss) = self.cgan((X, Y))

            generator_loss_adj = generator_loss + lambd * L1_loss

        critic_grads = tape.gradient(critic_loss, self.critic.trainable_weights)
        generator_grads = tape.gradient(generator_loss_adj, self.generator.trainable_weights)

        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_weights))
        self.gen_optimizer.apply_gradients(zip(generator_grads, self.generator.trainable_weights))

        return critic_loss, generator_loss_adj

#%%

## UNET Generator

def downsample_layer(num_filters, filter_size = 2, apply_BN = True):

    layer_model = tf.keras.Sequential()
    layer_model.add(tf.keras.layers.Conv2D(num_filters, filter_size, strides=2, padding='same', use_bias = False))
    if apply_BN:
        layer_model.add(tf.keras.layers.BatchNormalization())
    layer_model.add(tf.keras.layers.ReLU())
    return layer_model

def upsample_layer(num_filters, filter_size = 2, apply_dropout = False):

    layer_model = tf.keras.Sequential()
    layer_model.add(tf.keras.layers.Conv2DTranspose(num_filters, filter_size, strides = 2, padding = 'same', use_bias = False))
    layer_model.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        layer_model.add(tf.keras.layers.Dropout(0.5))
    layer_model.add(tf.keras.layers.ReLU())
    return layer_model

def unet_connection(X, num_filters, filter_size, layer_num, max_filters, output_channels = 3, first_layer = False):

    downsample_filters = min(num_filters, max_filters)
    upsample_filters = output_channels if first_layer else min(num_filters//2, max_filters)

    Y = downsample_layer(downsample_filters, filter_size, apply_BN = not first_layer)(X)

    if layer_num == 1:

        X2 = upsample_layer(upsample_filters, filter_size)(Y)

        return X2

    else:

        u_return = unet_connection(Y, num_filters * 2, filter_size, layer_num - 1,  max_filters)

        cat = tf.keras.layers.Concatenate()([Y, u_return])

        output = upsample_layer(upsample_filters, filter_size, apply_dropout = layer_num <= 3)(cat)

    return output
    

def UNET(input_shape, num_layers, filter_size = 4, max_filters = 512, initial_filters = 64, output_channels = 3):

    assert(num_layers > 2), 'Common this has to be a U-net'

    X = tf.keras.Input(shape = input_shape)

    Y = unet_connection(X, initial_filters, filter_size, num_layers - 1, max_filters, output_channels = output_channels, first_layer = True)

    return tf.keras.Model(X, Y)

#%%

# PatchGAN

def Critic(input_shape):
    
    X = tf.keras.layers.Input(shape=input_shape, name = 'X')
    Y = tf.keras.layers.Input(shape=input_shape, name = 'Y')

    cat = tf.keras.layers.concatenate([X, Y]) # (bs, 256, 256, channels*2)

    down1 = downsample_layer(64, 4, False)(cat) # (bs, 128, 128, 64)
    down2 = downsample_layer(128, 4)(down1) # (bs, 64, 64, 128)
    down3 = downsample_layer(256, 4)(down2) # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

    labels = tf.keras.layers.Conv2D(1, 4, strides=1)(zero_pad2) # (bs, 30, 30, 1)

    critic_score = tf.reduce_mean(labels, axis = (1,2,3))

    return tf.keras.Model(inputs=[X, Y], outputs=[critic_score])

