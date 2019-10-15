# class Encoder:
#     pass

# class Decoder:
#     pass

# class VariationAutoEncoder:
#     pass

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import pickle
import logging
from glob import glob
import numpy as np
from time import time
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras

if not os.path.exists("logs"):
    os.makedirs("logs")

today = datetime.now().strftime('%Y%m%d')

logger = logging.getLogger('worldmodels')
logger.setLevel(logging.DEBUG)

# Create logger
logger = logging.getLogger("worldmodels")
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s') 
logger.setLevel(logging.DEBUG)

# Uncomment to enable console logger
streamhandler = logging.StreamHandler()
streamhandler.setFormatter(formatter)
streamhandler.setLevel(logging.DEBUG)
logger.addHandler(streamhandler)

filehandler = logging.FileHandler(filename='logs/dataset.{}.log'.format(today))
filehandler.setFormatter(formatter)
filehandler.setLevel(logging.DEBUG)
logger.addHandler(filehandler)

AUTOTUNE = tf.data.experimental.AUTOTUNE


def load_preprocess_image(fname, resize_to=[64,64]):
    image = tf.io.read_file(fname)
    image = tf.image.decode_jpeg(image, channels=3)
    # image = tf.image.resize(image, [64, 64])
    image = tf.image.resize(image, resize_to)
    image /= 255.0
    return image

INPUT_SHAPE = (64,64,3)
# INPUT_SHAPE = (128,128,3)
LATENT_DIM = 32

encoder_input = keras.Input(shape=(INPUT_SHAPE), name='encoder_input_image')
x = keras.layers.Conv2D(32, 4, strides=(2,2), activation='relu', name='conv-1')(encoder_input)
x = keras.layers.Conv2D(64, 4, strides=(2,2), activation='relu', name='conv-2')(x)
x = keras.layers.Conv2D(128, 4, strides=(2,2), activation='relu', name='conv-3')(x)
x = keras.layers.Conv2D(256, 4, strides=(2,2), activation='relu', name='conv-4')(x)
# x = keras.layers.Conv2D(512, 4, strides=(2,2), activation='relu', name='conv-5')(x)
encoder_last_conv_shape = K.int_shape(x)[1:]
logger.info("encoder_last_conv_shape: {}".format(encoder_last_conv_shape))
x = keras.layers.Flatten()(x)
mu = keras.layers.Dense(LATENT_DIM, activation='linear', name="mean")(x)
logvar = keras.layers.Dense(LATENT_DIM, activation='linear', name="variance")(x)

encoder = keras.Model(encoder_input, [mu, logvar], name='encoder')
encoder.summary()

def sample(args):
    mean, logvar = args
    # reparameterizaton trick: allows gradients to pass through the sample
    # 1. sample from unit gaussian, then
    # 2. multiply it with standard deviation and add mean
    e = tf.random.normal(shape=(K.shape(mean)[0], LATENT_DIM))
    return e * tf.math.exp(logvar) + mean


sampled_latent_vector = keras.layers.Lambda(sample)([mu, logvar])

decoder_input = keras.layers.Input(shape=K.int_shape(sampled_latent_vector)[1:], name='decoder_input')
x = keras.layers.Dense(np.prod(encoder_last_conv_shape))(decoder_input)
x = keras.layers.Reshape((1,1,np.prod(encoder_last_conv_shape)))(x)
x = keras.layers.Conv2DTranspose(128, kernel_size=5, strides=(2,2), activation='relu')(x)
x = keras.layers.Conv2DTranspose(64, kernel_size=5, strides=(2,2), activation='relu')(x)
x = keras.layers.Conv2DTranspose(32, kernel_size=6, strides=(2,2), activation='relu')(x)
# x = keras.layers.Conv2DTranspose(32, kernel_size=4, strides=(2,2), activation='relu')(x)
decoder_output = keras.layers.Conv2DTranspose(3, kernel_size=6, strides=(2,2))(x)

decoder = keras.Model(decoder_input, decoder_output, name='decoder')
decoder.summary()

# Taken from tensorflow VAE example
def log_normal_pdf(sample, mean, logvar):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=1)

@tf.function
def calculate_loss(mean, logvar, labels, decoded_logits):
    xent_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=decoded_logits)
    z = sample([mean, logvar])
    logpx_z = -tf.reduce_sum(xent_loss, axis=[1,2,3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    loss = -tf.reduce_mean(logpx_z + logpz - logqz_x)
    return loss

class VAE(keras.Model):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def train_vars(self):
        return self.encoder.trainable_variables + self.decoder.trainable_variables
        
    def encode(self, x):
        encoded = self.encoder(x)
        return encoded
    
    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            return tf.sigmoid(logits)
        return logits         

@tf.function
def train_step(train_x, model, optimizer):
    with tf.GradientTape() as tape:
        # use training inputs to approximate the posterior 
        mean, logvar = model.encode(train_x)
        # sample latent vector from the learned mean and variance
        latent_z = sample([mean, logvar])
        # decode z
        decoded_logits = model.decode(latent_z)
        # calculate loss            
        loss = calculate_loss(mean, logvar, labels=train_x, decoded_logits=decoded_logits)        
        
    # calculate gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    # apply gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss            

def train(fnames, output_dirname="output", epochs=600, save_every_pct=0.3, print_every_pct=0.05):
    logger.info('Total files: {}'.format(len(fnames)))
    path_ds = tf.data.Dataset.from_tensor_slices(fnames)
    image_ds = path_ds.map(load_preprocess_image, num_parallel_calls=AUTOTUNE)
    # Dataset
    BATCH_SIZE = 64
    SHUFFLE_BUFFER_SIZE = len(fnames)
    train_dataset = image_ds \
        .shuffle(SHUFFLE_BUFFER_SIZE) \
        .repeat() \
        .batch(BATCH_SIZE) \
        .prefetch(buffer_size=AUTOTUNE)

    if not os.path.exists(output_dirname):
        os.makedirs('{}/ckpt'.format(output_dirname))
        os.makedirs('{}/imgs'.format(output_dirname))

    # Number of training epochs
    # EPOCHS = 600
    logger.info('Training epochs: {}'.format(epochs))
    # Initialize the Variational Autoencoder model 
    model = VAE(encoder, decoder)
    # Define optimizer
    optimizer = keras.optimizers.Adam(1e-4)

    # keep track of losses
    losses = []

    # How often to print the loss
    print_every = max(int(print_every_pct * epochs), 1)

    # Model Checkpoint 
    # Save model and optimizer
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    # Set save path and how many checkpoints to save
    checkpoint_path = '{}/ckpt/'.format(output_dirname)
    logger.info('Checkpoints will be stored at {}'.format(checkpoint_path))
    manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=2)
    # Load the latest checkpoint and restore
    latest_ckpt = manager.latest_checkpoint
    ckpt.restore(latest_ckpt)

    if latest_ckpt:
        logger.info('Restored from {}'.format(latest_ckpt))
    else:
        logger.info('Training from scratch')
    # How often to save the checkpoint
    save_every = max(int(save_every_pct * epochs), 1)

    # We are now ready to start the training loop
    elapsed_loop_time = time()
    for epoch in range(0, epochs):
        for train_x in train_dataset:
            loss = train_step(train_x, model, optimizer)
            losses.append(loss)
        if epoch % print_every == 0:
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            logger.info('{}:Epoch {}/{}: train loss {} in {} seconds'.format(epoch, epochs, losses[-1], time()-elapsed_loop_time))
            elapsed_loop_time = time()
        if epoch % save_every == 0:
            save_path = manager.save()
            logger.info('Saved checkpoint for step {}:{}'.format(epoch, save_path))
            
    # Final Save
    save_path = manager.save()
    logger.info('Saved checkpoint for step {}'.format(save_path))


if __name__ == "__main__":
    # Toons
    # fnames = glob('{}/*.png'.format("/mnt/bigdrive/datasets/cartoonset/cartoonset10k/"))
    # train(fnames, output_dirname="toons128")

    # Car racing
    fnames = glob('{}/*.png'.format("/mnt/bigdrive/projects/public_repos/world-models/src/imgs/"))
    train(fnames, output_dirname="car_racing")
