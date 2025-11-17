from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np
from keras import layers
from keras.models import Sequential
from bicoder import BiCoder


class Decoder(BiCoder, ABC):
    """
    Abstract base class for all decoders (Generative model p(x|z)).
    Contains shared VAE logic (Location-Scale Sampling) and fixed sigma.
    """
    def __init__(self, latent_dim, **kwargs):
        super(Decoder, self).__init__(latent_dim=latent_dim, **kwargs)
        self._fixed_std = tf.constant(0.75, dtype=tf.float32) 
        self.network = None 
        self._build_network()
        
    @property 
    def fixed_std(self):
        """ Getter for fixed_std. Makes the variable read-only for external calls. """
        return self._fixed_std 
        
    @abstractmethod
    def _build_network(self):
        """ 
        Abstract method for building the Sequential model (MLP or CNN Deconv).
        Must be overridden by subclasses.
        """
        pass 
        
    def call(self, z):
        """
        Performs the forward pass for Decoder p(x|z).
        Returns mu, fixed std, and the reconstructed sample x_sample.
        """
        mu_x = self.network(z)
        std_x = self.fixed_std 
        eps = tf.random.normal(tf.shape(mu_x)) 
        x_sample = mu_x + eps * std_x
        
        return x_sample, mu_x, std_x


class MLP_Decoder(Decoder):
    """
    Decoder for B&W images (mnist_bw). Latent dimension d=20.
    Implements the MLP architecture.
    """
    def __init__(self, latent_dim=20, units=400, activation='relu', **kwargs):
        
        self.output_dim = 28 * 28 
        self.units = units
        self.activation = activation
        
        super(MLP_Decoder, self).__init__(latent_dim=latent_dim, **kwargs)

        
    def _build_network(self):
        """
        Implements the specific MLP architecture for B&W. [1, 5]
        """
        self.network = Sequential(
            [
                layers.InputLayer(input_shape=(self.latent_dim,)),
                layers.Dense(self.units, activation=self.activation), 
                layers.Dense(self.output_dim),       
            ]
        )
        


class CNN_Decoder(Decoder):
    """
    Decoder for color images (mnist_color). Latent dimension d=50.
    Implements the CNN/Deconvolutional architecture.
    """
    def __init__(self, latent_dim=50, filters=32, kernel_size=3, activation='relu', **kwargs):
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        target_shape = (4, 4, 4 * filters)
        self.units = np.prod(target_shape) 
        self.target_shape = target_shape 
        super(CNN_Decoder, self).__init__(latent_dim=latent_dim, **kwargs)
        
    def _build_network(self):
        """
        Implements the specific CNN Deconvolution architecture. [6, 7]
        """
        self.network = Sequential(
            [
                layers.InputLayer(input_shape=(self.latent_dim,)),
                layers.Dense(units=self.units, activation=self.activation),
                layers.Reshape(target_shape=self.target_shape),

                layers.Conv2DTranspose(
                    filters=self.filters * 2, kernel_size=self.kernel_size, strides=2, padding='same', output_padding=0,
                    activation=self.activation),
                
                layers.Conv2DTranspose(
                    filters=self.filters, kernel_size=self.kernel_size, strides=2, padding='same', output_padding=1,
                    activation=self.activation),
                
                layers.Conv2DTranspose(
                    filters=3, kernel_size=self.kernel_size, strides=2, padding='same', output_padding=1),
                
                layers.Activation('linear', dtype='float32'),
            ]
        )
