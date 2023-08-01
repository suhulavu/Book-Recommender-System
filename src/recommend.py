# IMPORTS
import pandas as pd
import numpy as np
import pickle
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine



def cosineSimilarity(vec_1, vec_2):
    """
    Calculates cosine similarity between two vectors

    Parameters
    -----------
    vec_1 : np.array
        first vector
    vec_2 : np.array
        second vector
    
    Returns
    --------
    cos_sim : float
        cosine similarity between both vectors
    """

    cos_sim =  (vec_1 @ vec_2.T) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))
    return cos_sim


def rank(df, book_idx, cluster, embedding_type='mf'):
    """
    Ranks items similar to chosen item based on cosine similarity for a given embedding type

    Parameters
    -----------
    df : pd.DataFrame
        dataframe of candidates
    book_idx : int
        index of chosen book in dataframe
    cluster : int
        cluster label for chosen book for chosen embedding type
    embedding_type : str
        the type of embedding to use to calculate similarity scores

    Returns
    --------
    df_res : pd.DataFrame
        dataframe of cosine similarities for all candidates
    """

    # candidate generation
    df_cluster = df.loc[df['{}_cluster'.format(embedding_type)] == cluster]

    # load embeddings
    mf_embeddings, mlp_embeddings, encodings = pickle.load(open('../results/embeddings.pkl', 'rb'))
    match embedding_type:
        case 'mf':
            embeddings = mf_embeddings
        case 'mlp':
            embeddings = mlp_embeddings
        case 'cbf':
            embeddings = encodings
    
    book_embedding = embeddings[book_idx]
    embeddings = embeddings[df_cluster.index.values]

    # cosine similarities
    cos_sims = [cosineSimilarity(book_embedding, embedding) for embedding in embeddings]

    # create dataframe of results
    res = {'title' : df_cluster['title'].values, 'cos_sim' : cos_sims}
    df_res = pd.DataFrame(res)
    return df_res



def retrieveAndRank(df, book, top_k):
    """
    Aggregates recommendations across embedding types and prints top "K" recommendations

    Parameters
    -----------
    df : pd.DataFrame
        dataframe with all books and clusters
    book : str
        the book to give recommendations for
    top_k : int
        the number of recommendations to give
    """

    # find book data
    book_data = df.loc[df['title'] == book]
    book_index = book_data.index.values[0]
    mf_cluster = book_data['mf_cluster'].values[0]
    mlp_cluster = book_data['mlp_cluster'].values[0]
    cbf_cluster = book_data['cbf_cluster'].values[0]

    # rank similar items
    df_wo_book = df.loc[~(df['title'] == book)]
    df_mf = rank(df_wo_book, book_index, mf_cluster, 'mf')
    df_mlp = rank(df_wo_book, book_index, mlp_cluster, 'mlp')
    df_cbf = rank(df_wo_book, book_index, cbf_cluster, 'cbf')

    # aggregate results
    df_ranked = pd.concat([df_mf, df_mlp, df_cbf], axis=0, ignore_index=True)
    df_ranked = df_ranked.groupby('title', as_index=False).mean()
    df_ranked = df_ranked.sort_values(by='cos_sim', ascending=False)

    print(df_ranked.head(top_k))


if __name__ == "__main__":

    # load data
    df = pd.read_csv('../results/df_cluster.csv')

    # select book and number of recommendations
    parser = argparse.ArgumentParser(description='Get Recommendations for a Book')
    parser.add_argument('--book', dest='book', action='store', help='book to get recommendations for')
    parser.add_argument('--num_recs', dest='k', default=10, type=int, action='store', help='number of recommendations')
    args = parser.parse_args()
    

    # get recommendations
    retrieveAndRank(df, args.book, args.k)