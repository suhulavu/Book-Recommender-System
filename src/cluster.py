# IMPORTS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import pickle
import pandas as pd
import tensorflow as tf
import argparse


def cluster(X):
    """
    Determines optimal amount of clusters for KMeans using silhouette score and trains KMeans

    Parameters
    -----------
    X : np.array
        2d array of data to be clustered
    
    Returns
    --------
    labels : np.array
        cluster label for each data point
    """
    scores = {}
    for n_clusters in range(2, 11):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, init='k-means++', n_init='auto').fit(X)
        sil_score = silhouette_score(X, kmeans.labels_)
        scores[n_clusters] = sil_score

    best_cluster = max(scores, key=scores.get)
    labels = KMeans(n_clusters=best_cluster, random_state=42, init='k-means++', n_init='auto').fit(X).labels_

    return labels


def clusterItems(df, autoencoder_filepath, ncf_filepath, tsvd_filepath):
    """
    Clusters book embeddings for each of the three different embeddings

    Parameters
    -----------
    df : pd.DataFrame
        dataframe containing book ids
    autoencoder_filepath : str
        path to autoencoder model
    ncf_filepath : str
        path to ncf model
    tsvd_filepath : str
        path to truncated svd model
    """

    # load models
    autoencoder = tf.keras.models.load_model(autoencoder_filepath)
    ncf = tf.keras.models.load_model(ncf_filepath)
    tsvd = pickle.load(open(tsvd_filepath, 'rb'))

    # get relevant layers
    encoder = autoencoder.encoder
    mf_embedding_layer = ncf.get_layer('mf_book_embedding_layer')
    mlp_embedding_layer = ncf.get_layer('mlp_book_embedding_layer')

    # mf embeddings
    print('Calculating Matrix Factorization Embeddings')
    mf_embeddings = mf_embedding_layer(df['work_id'].to_numpy(), training=False).numpy()
    labels = cluster(mf_embeddings)
    df['mf_cluster'] = labels

    # mlp embeddings
    print('Calculating MLP Embeddings')
    mlp_embeddings = mlp_embedding_layer(df['work_id'].to_numpy(), training=False).numpy()
    labels = cluster(mlp_embeddings)
    df['mlp_cluster'] = labels

    # encodings
    print('Calculating Content Based Filtering Encodings')
    mat = pickle.load(open('../data/processed/tfidf_matrix.pkl', 'rb'))
    mat = tsvd.transform(mat)
    encodings = encoder(mat, training=False).numpy()
    labels = cluster(encodings)
    df['cbf_cluster'] = labels

    # save embeddings and labels
    pickle.dump((mf_embeddings, mlp_embeddings, encodings), open('../results/embeddings.pkl', 'wb'))
    df.to_csv('../results/df_cluster.csv', index=False)



if __name__ == "__main__":

    # read data
    df = pd.read_csv('../data/processed/df_processed.csv')

    # parse args
    parser = argparse.ArgumentParser(description='Generate clusters from embeddings for candidate generation')
    parser.add_argument('--autoencoder_path', action='store', dest='autoencoder_path', help='path to autoencoder model')
    parser.add_argument('--ncf_path', action='store', dest='ncf_path', help='path to NCF model')
    parser.add_argument('--tsvd_path', action='store', dest='tsvd_path', help='path to TruncatedSVD model')
    args = parser.parse_args()

    # perform clustering
    clusterItems(df=df, autoencoder_filepath=args.autoencoder_path, ncf_filepath=args.ncf_path, tsvd_filepath=args.tsvd_path)


