# IMPORTS
import pandas as pd
import numpy as np
import datetime
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import AUC
import mlflow
from tensorflow.keras.callbacks import TensorBoard
import argparse


def buildNeuMFModel(num_books, num_users, mlp_units=[256, 128, 64, 32, 8], mf_embedding_dim=32, mlp_embedding_dim=32):
    """
    Builds NeuMF network architecture

    Parameters
    -----------
    num_books : int
        number of unique books
    num_users : int
        number of unique users
    mlp_units : List[int]
        number of units in each dense layer of MLP network
    mf_embedding_dim : int
        embedding dimension of matrix factorization embeddings
    mlp_embedding_dim : int
        embedding dimension of mlp embeddings

    Returns
    --------
    model : tf.keras.Model
        NeuMF model
    """

    # inputs
    user_input = layers.Input(shape=(1,), name='user')
    book_input = layers.Input(shape=(1,), name='book')

    # embedding layers
    mf_book_embedding = layers.Embedding(input_dim=num_books, output_dim=mf_embedding_dim, input_length=1,
                                        name='mf_book_embedding_layer')
    mf_user_embedding = layers.Embedding(input_dim=num_users, output_dim=mf_embedding_dim, input_length=1)

    mlp_book_embedding = layers.Embedding(input_dim=num_books, output_dim=mlp_embedding_dim, input_length=1,
                                         name='mlp_book_embedding_layer')
    mlp_user_embedding = layers.Embedding(input_dim=num_users, output_dim=mlp_embedding_dim, input_length=1)

    # latent vectors
    mf_book_vec = layers.Flatten()(mf_book_embedding(book_input))
    mf_user_vec = layers.Flatten()(mf_user_embedding(user_input))
    gmf_vec = layers.Multiply()([mf_user_vec, mf_book_vec])

    mlp_book_vec = layers.Flatten()(mlp_book_embedding(book_input))
    mlp_user_vec = layers.Flatten()(mlp_user_embedding(user_input))
    mlp_vec = layers.Concatenate()([mlp_user_vec, mlp_book_vec])

    # MLP
    for units in mlp_units:
        dense_layer = layers.Dense(units=units, activation='relu')
        dropout = layers.Dropout(0.2)
        mlp_vec = dropout(dense_layer(mlp_vec))

    # output
    neumf_vec = layers.Concatenate()([gmf_vec, mlp_vec])
    output_layer = layers.Dense(units=5, activation='linear')
    output = output_layer(neumf_vec)
        
    # define model
    model = Model(inputs=[user_input, book_input], outputs=output)
    
    return model


def train(df, **kwargs):
    """
    Trains an NCF model using NeuMF architecture and tracks metrics/params using MLFlow

    Parameters
    -----------
    df : pd.DataFrame
        ratings dataframe
    """

    # SPLIT DATA
    x_user, x_book = df['user_id'].to_numpy().astype(int), df['book_id'].to_numpy().astype(int)
    y = pd.get_dummies(df['rating']).astype(int).to_numpy()

    x_train_user, x_test_user, x_train_book, x_test_book, y_train, y_test = train_test_split(x_user, x_book, y, 
                                                                                            test_size=0.1, random_state=42,
                                                                                            stratify=df['rating'].tolist())
    
    # TRAINING PARAMS
    exp_name = kwargs['name']
    batch_size = kwargs['batch_size']
    lr = kwargs['lr']
    mf_embedding_dim = kwargs['mf_embed_dim']
    mlp_embedding_dim = kwargs['mlp_embed_dim']
    mlp_units = kwargs['mlp_units']

    # SET EXPERIMENT
    mlflow.set_experiment(exp_name)


    # RUN TRIAL
    with mlflow.start_run():

        # construct datasets
        ds_train = tf.data.Dataset.from_tensor_slices(({'user': x_train_user, 'book': x_train_book}, y_train))
        ds_train = ds_train.batch(2048, drop_remainder=True).shuffle(y_train.shape[0]).prefetch(tf.data.experimental.AUTOTUNE).cache()

        ds_test = tf.data.Dataset.from_tensor_slices(({'user': x_test_user, 'book': x_test_book}, y_test))
        ds_test = ds_test.batch(2048).prefetch(tf.data.experimental.AUTOTUNE)

        # build model
        model = buildNeuMFModel(df['book_id'].nunique(), df['user_id'].nunique(), mlp_units=mlp_units,
                            mf_embedding_dim=mf_embedding_dim, mlp_embedding_dim=mlp_embedding_dim)
        
        # log parameters
        mlflow.log_params({
            'learning_rate' : lr,
            'batch_size' : batch_size,
            'mlp_units' : mlp_units,
            'mf_embedding_dim' : mf_embedding_dim,
            'mlp_embedding_dim' : mlp_embedding_dim,
        })
        
        # train model
        metrics = ['accuracy', AUC(curve='roc', name='aucroc', from_logits=True)]
        model.compile(optimizer=Adam(learning_rate=lr), loss=CategoricalCrossentropy(from_logits=True), metrics=metrics)
        history = model.fit(ds_train,
                            epochs=75,
                            validation_data=ds_test,
                            callbacks=[TensorBoard(log_dir='logs')])
        
        # log metrics
        train_loss = history.history['loss'][-1]
        train_acc = history.history['accuracy'][-1]
        train_auc = history.history['aucroc'][-1]
        
        test_loss = history.history['val_loss'][-1]
        test_acc = history.history['val_accuracy'][-1]
        test_auc = history.history['val_aucroc'][-1]
        
        mlflow.log_metrics({
            'train_loss' : train_loss,
            'train_acc' : train_acc,
            'train_aucroc' : train_auc,
            'test_loss' : test_loss,
            'test_acc' : test_acc,
            'test_aucroc' : test_auc
        })
        
        # save model and log artifacts
        time = str(datetime.datetime.now())
        time = time[:time.index('.')]
        time = time.replace('-', '_').replace(':', '_').replace(' ', '_')
        
        model.save('../models/ncf_model_{}.h5'.format(time))
        mlflow.log_artifact('../models/ncf_model_{}.h5'.format(time))
    mlflow.end_run()



if __name__ == "__main__":
    # read ratings data
    df = pd.read_csv('../data/raw/ratings.csv')

    # get command line arguments
    parser = argparse.ArgumentParser(description='Train Neural Collaborative Filtering Model')
    parser.add_argument('--exp_name', dest='name', default='NCF', action='store', help='mlflow experiment name')
    parser.add_argument('--batch_size', dest='batch_size', default=2048, type=int, action='store', help='training batch size')
    parser.add_argument('--lr', dest='lr', default=1e-5, type=float, action='store', help='learning rate')
    parser.add_argument('--mf_embeddding_dim', dest='mf_embed_dim', default=32, type=int, action='store', 
                        help='embedding dimension for matrix factorization vectors')
    parser.add_argument('--mlp_embeddding_dim', dest='mlp_embed_dim', default=32, type=int, action='store', 
                        help='embedding dimension for mlp vectors')
    parser.add_argument('--mlp_units', dest='mlp_units', default=[256, 128, 64, 32, 8], type=int, nargs='+',
                        help='number of neurons in each layer of the mlp')
    args = parser.parse_args()
    

    # train ncf model
    train(df, **vars(args))
