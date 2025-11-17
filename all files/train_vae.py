

import argparse
import tensorflow as tf
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE 
import numpy as np
from vae import VAE
from data_loader import DataLoader
from utils import train, plot_grid

parser = argparse.ArgumentParser(description='Train a VAE on MNIST B&W or Color.')
parser.add_argument('--dset', type=str, default='mnist_bw', choices=['mnist_bw', 'mnist_color'],
                    help='Dataset to use (mnist_bw or mnist_color).')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for training.')
parser.add_argument('--visualize_latent', action='store_true', help='Visualizes the latent space.')
parser.add_argument('--generate_from_prior', action='store_true', help='Generates images from the prior p(z).')
parser.add_argument('--generate_from_posterior', action='store_true', help='Generates images from the posterior q(z|x).')

args = parser.parse_args()

if args.dset == 'mnist_color':
    latent_dim = 50 
    print(f"Starting training of CNN VAE for Color images (d={latent_dim})...")
else: 
    latent_dim = 20 
    print(f"Starting training of MLP VAE for B&W images (d={latent_dim})...")

BATCH_SIZE = 128
LEARNING_RATE = 1e-4

my_data_loder = DataLoader(dset=args.dset, batch_size=BATCH_SIZE)
model = VAE(latent_dim=latent_dim) 
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE) 

tr_data = my_data_loder.get_training_data()

print(f"Starting training for {args.epochs} epochs...")
for e in range(args.epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()

    for i, tr_batch in enumerate(tr_data):
     
        loss = train(model, tr_batch, optimizer) 
        
        epoch_loss_avg.update_state(loss)
        
        if i % 100 == 0:
            print(f"Epoch {e+1}, Batch {i}: Loss = {loss.numpy():.4f}")

    print(f"Epoch {e+1} Finished. Average Loss: {epoch_loss_avg.result().numpy():.4f}")

dl_test = DataLoader(dset=args.dset, batch_size=128)

if args.visualize_latent:
    print("Visualizing Latent Space (TSNE)...")
    latent_dim = model.latent_dim 
    test_data_ds = dl_test.get_test_data()
    x_test_list = [x for x in test_data_ds]
    x_test = tf.concat(x_test_list, axis=0)
    test_labels = dl_test.load_labels() 
    _, mu, log_var = model.encoder(x_test) 
    
    tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
    z_2d = tsne.fit_transform(mu.numpy())
    
    plt.figure(figsize=(10, 8))
    plt.scatter(z_2d[:, 0], z_2d[:, 1], c=test_labels, cmap='Spectral', alpha=0.6)
    plt.colorbar()
    plt.title(f'TSNE of Latent Space (d={latent_dim}) after {args.epochs} epochs')
    plt.savefig(f'latent_space_{args.dset}.pdf')
    plt.close()
    print(f"Latent space plot saved as latent_space_{args.dset}.pdf")

if args.dset == 'mnist_color':
    CHANNELS = 3
else:
    CHANNELS = 1 

if args.generate_from_prior:
    print("Generating images from Prior p(z)...")
    
    N_SAMPLES = 100 
    latent_dim = model.latent_dim
    
    z_prior = tf.random.normal(shape=(N_SAMPLES, latent_dim), dtype=tf.float32)
    
    _, mu_x_hat, _ = model.decoder(z_prior)
    mu_x_hat = tf.clip_by_value(mu_x_hat, 0.0, 1.0)
    reshaped_images = tf.reshape(mu_x_hat, shape=(N_SAMPLES, 28, 28, CHANNELS))

    plot_grid(reshaped_images.numpy(), name='prior', dset_name=args.dset)
    print(f"Generated images from prior p(z) saved as xhat_{args.dset}_prior.pdf.") 

if args.generate_from_posterior:
    print("Generating reconstructions from Posterior q(z|x)...")
    
    x_input = next(iter(dl_test.get_test_data()))
    mu_x_hat = model.reconstruct(x_input) 
    reshaped_images = tf.reshape(mu_x_hat, shape=(-1, 28, 28, CHANNELS))  
    x_hat_to_plot = reshaped_images.numpy()[:100]
    
    plot_grid(x_hat_to_plot, name='posterior', dset_name=args.dset)
    print(f"Reconstructed images from posterior q(z|x) saved as xhat_{args.dset}_posterior.pdf.")
