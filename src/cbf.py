# IMPORTS
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.layers as layers
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
import pickle
from sklearn.decomposition import TruncatedSVD
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
import datetime
import argparse
import os


# ENCODER ARCHITECTURE
def encoder(input_shape, encoding_size):
    model = Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Dense(units=512, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(units=256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(units=encoding_size, activation='relu')
    ])
    
    return model


# DECODER ARCHITECTURE
def decoder(output_shape, encoding_size):
    model = Sequential([
        layers.InputLayer(input_shape=(encoding_size,)),
        layers.Dense(units=256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(units=512, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(units=output_shape[0], activation='sigmoid')
    ])
    
    return model


# FULL AUTOENCODER CLASS
class Autoencoder(Model):
    def __init__(self, input_shape, encoding_size):
        super().__init__()
        self.encoder = encoder(input_shape=input_shape, encoding_size=encoding_size)
        self.decoder = decoder(output_shape=input_shape, encoding_size=encoding_size)
        
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

def train(**kwargs):
    """
    Trains an autoencoder to reduce dimension of content vectors
    
    """

    # load data
    mat = pickle.load(open('../data/processed/tfidf_matrix.pkl', 'rb'))

    # train test split
    x_train, x_test = train_test_split(mat, test_size=0.1, random_state=42)

    # training params
    tsvd_size = kwargs['tsvd_size']
    encoding_size = kwargs['encoding_size']
    lr = kwargs['lr']
    epochs = kwargs['epochs']

    # dimensionality reduction
    tsvd_path = '../models/cbf_tsvd_{}.pkl'.format(tsvd_size)
    if os.path.exists(tsvd_path):
        tsvd = pickle.load(open(tsvd_path, 'rb'))
        x_train = tsvd.transform(x_train)
        x_test = tsvd.transform(x_test)
    else:
        tsvd = TruncatedSVD(n_components=tsvd_size)
        x_train = tsvd.fit_transform(x_train)
        x_test = tsvd.transform(x_test)
        pickle.dump(tsvd, open('../models/cbf_tsvd_{}.pkl'.format(tsvd_size), 'wb'))

    # compile and fit model
    model = Autoencoder(input_shape=(x_train.shape[1],), encoding_size=encoding_size)
    model.compile(optimizer=Adam(learning_rate=lr), loss=MeanSquaredError())
    model.fit(x_train, x_train,
              epochs=epochs,
              validation_data=(x_test, x_test),
              callbacks=[TensorBoard(log_dir='logs')])
    
    # save model
    time = str(datetime.datetime.now())
    time = time[:time.index('.')]
    time = time.replace('-', '_').replace(':', '_').replace(' ', '_')

    model.save('../models/cbf_model_{}'.format(time), save_format='tf')



if __name__ == "__main__":
    # get command line arguments
    parser = argparse.ArgumentParser(description='Train Content Based Filtering Model')
    parser.add_argument('--encoding_size', dest='encoding_size', type=int, action='store', help='encoded vector size', default=128)
    parser.add_argument('--lr', dest='lr', action='store', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epochs', dest='epochs', action='store', type=int, default=50, help='number of training epochs')
    parser.add_argument('--tsvd_size', dest='tsvd_size', action='store', default=7500, type=int, help='n_components for truncated SVD')
    args = parser.parse_args()

    # train model
    train(**vars(args))


