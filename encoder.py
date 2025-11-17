import tensorflow as tf
from tensorflow import keras
from keras import layers
from abc import ABC, abstractmethod
from bicoder import BiCoder
from keras.models import Sequential

class Encoder(BiCoder, ABC):
    """
    Abstract base class for all encoders.
    Contains shared VAE logic (Reparameterization trick).
    """
    def __init__(self, latent_dim, **kwargs):
        
        super(Encoder, self).__init__(latent_dim=latent_dim, **kwargs)
        
        self.network = None 
        
        self._build_network()
        
    @abstractmethod
    def _build_network(self):
        """ 
        Abstract method for building the Sequential model (MLP or CNN).
        Must be overridden by subclasses.
        """
        pass 
        
    def call(self, x):
        """
        Performs the forward pass for Encoder q(z|x).
        Uses shared logic to split the output and sample z.
        """
        out = self.network(x)
        
        mu = out[:, :self.latent_dim]
        log_var = out[:, self.latent_dim:]
        
        std = tf.math.exp(0.5 * log_var)
        eps = tf.random.normal(tf.shape(mu))
        z = mu + eps * std
        
        return z, mu, log_var


class MLP_Encoder(Encoder):
    """
    Encoder for B&W images (mnist_bw). Latent dimension d=20.
    Implements the MLP architecture.
    """
    def __init__(self, latent_dim=20, units=400, activation='relu', **kwargs):
        
        self.input_dim = (28 * 28,)
        self.units = units
        self.activation = activation
        
        super(MLP_Encoder, self).__init__(latent_dim=latent_dim, **kwargs)
        
    def _build_network(self):
        """
        Implements the specific MLP architecture.
        """
        self.network = Sequential(
            [
                layers.InputLayer(input_shape=self.input_dim),
                layers.Dense(self.units, activation=self.activation), 
                layers.Dense(2 * self.latent_dim),              
            ]
        )

class CNN_Encoder(Encoder):
    """
    Encoder for color images (mnist_color). Latent dimension d=50.
    Implements the CNN architecture.
    """
    def __init__(self, latent_dim=50, filters=32, kernel_size=3, strides=2, activation='relu', **kwargs):
        
        self.input_dim = (28, 28, 3) 
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation
        
        super(CNN_Encoder, self).__init__(latent_dim=latent_dim, **kwargs)
        
    def _build_network(self):
        """
        Implements the specific CNN architecture.
        """
        self.network = Sequential(
            [
                layers.InputLayer(input_shape=self.input_dim),
                
                layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, 
                              activation=self.activation, padding='same'),
                
                layers.Conv2D(filters=2*self.filters, kernel_size=self.kernel_size, strides=self.strides, 
                              activation=self.activation, padding='same'),
                
                layers.Conv2D(filters=4*self.filters, kernel_size=self.kernel_size, strides=self.strides, 
                              activation=self.activation, padding='same'),
                
                layers.Flatten(), 
                layers.Dense(2 * self.latent_dim) 
            ]
        )
