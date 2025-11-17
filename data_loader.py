import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import pickle
import subprocess 
from keras import layers 
MNIST_LINKS = {
    'mnist_bw': {
        'train_data': 'https://www.dropbox.com/scl/fi/fjye8km5530t9981ulrll/mnist_bw.npy?rlkey=ou7nt8t88wx1z38nodjjx6lch&st=5swdpnbr&dl=0',
        'test_data': 'https://www.dropbox.com/scl/fi/dj8vbkfpf5ey523z6ro43/mnist_bw_te.npy?rlkey=5msedqw3dhv0s8za976qlaoir&st=nmu00cvk&dl=0',
        'test_labels': 'https://www.dropbox.com/scl/fi/8kmcsy9otcxg8dbi5cqd4/mnist_bw_y_te.npy?rlkey=atou1x07fnna5sgu6vrrgt9j1&st=m05mfkwb&dl=0',
        'label_data': 'https://www.dropbox.com/scl/fi/8kmcsy9otcxg8dbi5cqd4/mnist_bw_y_te.npy?rlkey=atou1x07fnna5sgu6vrrgt9j1&st=m05mfkwb&dl=0',
        'label_file': 'mnist_bw_y_te.npy',
        'train_file': 'mnist_bw.npy',
        'test_file': 'mnist_bw_te.npy',
    },
    'mnist_color': {
        'train_data': 'https://www.dropbox.com/scl/fi/w7hjg8ucehnjfv1re5wzm/mnist_color.pkl?rlkey=ya9cpgr2chxt017c4lg52yqs9&st=ev984mfc&dl=0',
        'test_data': 'https://www.dropbox.com/scl/fi/w08xctj7iou6lqvdkdtzh/mnist_color_te.pkl?rlkey=xntuty30shu76kazwhb440abj&st=u0hd2nym&dl=0',
        'test_labels': 'https://www.dropbox.com/scl/fi/fkf20sjci5ojhuftc0ro0/mnist_color_y_te.npy?rlkey=fshs83hd5pvo81ag3z209tf6v&st=99z1o18q&dl=0',
        'label_data': 'https://www.dropbox.com/scl/fi/fkf20sjci5ojhuftc0ro0/mnist_color_y_te.npy?rlkey=fshs83hd5pvo81ag3z209tf6v&st=99z1o18q&dl=0',
        'label_file': 'mnist_color_y_te.npy',
        'train_file': 'mnist_color.pkl',
        'test_file': 'mnist_color_te.pkl',
    }
}


class DataLoader:
    def __init__(self, dset, batch_size=128):
        self.dset = dset 
        self.batch_size = batch_size
        self.links = MNIST_LINKS[dset] 
        self._cached_train_data = None
        self._cached_test_data = None 
        self._cached_labels = None 
        if dset == 'mnist_color':
             self.color_version_key = 'm1' # options: 'm0' ,'m1', 'm2', 'm3', 'm4'
        else:
             self.color_version_key = None

    def _download_data(self, data_key):
        """
        Checks if the file exists locally and downloads it if necessary,
        using the recommended 'wget' command [5].
        """
        url = self.links[data_key]
        filename = self.links[data_key.replace('_data', '_file')]

        if not os.path.exists(filename):
            print(f"Downloading {filename} from Dropbox...")
            subprocess.run(['wget', '-O', filename, url], check=True)
            print("Download completed.")
        else:
            print(f"{filename} already exists.")

        return filename

    def _load_data(self, key='train_data'):
        """ Downloads and reads the data file (.npy or .pkl). """
        
        filename = self._download_data(key)
        
        if filename.endswith('.npy'):
            return np.load(filename)
            
        elif filename.endswith('.pkl'):
            with open(filename, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            return data
        
        else:
            raise ValueError("Unknown file format.")

    def _preprocess_data(self, data):
        """ Performs specific preprocessing based on dataset. """
        
        if self.dset == 'mnist_bw':
            data = data.astype(np.float32) / 255.0 
            data = data.reshape((-1, 784)) 
            return data
            
        elif self.dset == 'mnist_color':
            return data.astype(np.float32) 
         
    def get_training_data(self):
        """ 
        Returns the fully processed training data as tf.data.Dataset.
        Uses caching to avoid repeated loading from disk.
        """
        if self._cached_train_data is None:
            raw_data = self._load_data(key='train_data') 
        
            if self.dset == 'mnist_color':
                raw_data = raw_data[self.color_version_key]
            
            processed_data = self._preprocess_data(raw_data)
            self._cached_train_data = processed_data

        processed_data = self._cached_train_data 
        dataset = tf.data.Dataset.from_tensor_slices(processed_data)
        dataset = dataset.shuffle(buffer_size=1024).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    
    def get_test_data(self):
        """ 
        Returns the fully processed test data as tf.data.Dataset.
        Uses caching to avoid repeated loading from disk.
        """
        
        if self._cached_test_data is None:
            print("Loading test data from disk...")
            raw_data = self._load_data(key='test_data') 
            
            if self.dset == 'mnist_color':
                raw_data = raw_data[self.color_version_key]

            processed_data = self._preprocess_data(raw_data)
            self._cached_test_data = processed_data
        
        processed_data = self._cached_test_data
        dataset = tf.data.Dataset.from_tensor_slices(processed_data)
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset

    
    def load_labels(self):
        """
        Loads labels for the test dataset and caches the result.
        """
        if self._cached_labels is None:
            print("Loading labels from disk...")
            
            label_key = 'label_data' 
            filename = self._download_data(label_key)
            
            try:
                labels = np.load(filename)
            except Exception as e:
                raise IOError(f"Could not load labels from {filename}: {e}")

            self._cached_labels = labels.flatten()
        return self._cached_labels
