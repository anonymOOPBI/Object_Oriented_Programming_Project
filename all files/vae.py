import tensorflow as tf

from keras.models import Model 
from encoder import Encoder
from decoder import Decoder
from losses import kl_divergence, log_diag_mvn
import tensorflow as tf
from encoder import MLP_Encoder, CNN_Encoder 
from decoder import MLP_Decoder, CNN_Decoder 


class VAE(Model): 
    """
    Orchestrates the Encoder and Decoder, and computes the total VAE loss function (-ELBO).
    Dynamically chooses between MLP and CNN architecture.
    """
    def __init__(self, latent_dim=20, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.vae_loss = None 
        
        if latent_dim == 20:
            self.encoder = MLP_Encoder(latent_dim=latent_dim, **kwargs)
            self.decoder = MLP_Decoder(latent_dim=latent_dim, **kwargs)
        elif latent_dim == 50:
            self.encoder = CNN_Encoder(latent_dim=latent_dim, **kwargs)
            self.decoder = CNN_Decoder(latent_dim=latent_dim, **kwargs)
        else:
            raise ValueError(f"Unknown latent dimension: {latent_dim}. Must be 20 or 50.")

    def call(self, x):
        """
        Performs the forward pass and computes the approximate lower bound (Negative ELBO).
        """
        
        z, mu_z, log_var_z = self.encoder(x)
        x_sample, mu_x, std_x = self.decoder(z) 
        log_sigma_x = tf.math.log(0.75)
        log_p_x_given_z = log_diag_mvn(x, mu_x, log_sigma_x) 
        
        kl_loss = kl_divergence(mu_z, log_var_z) 
        self.vae_loss = tf.reduce_mean(kl_loss - log_p_x_given_z)

        return self.vae_loss
    
    def reconstruct(self, x):
        """
        Performs the forward pass to retrieve the reconstructed mean mu_x for plotting.
        """
        
        z, _, _ = self.encoder(x)
        
        x_sample, mu_x, std_x = self.decoder(z)
        
        return mu_x
