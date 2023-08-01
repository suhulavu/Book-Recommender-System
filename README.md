# Book Recommender System

A hybrid book-to-book recommender system based on collaborative filtering and content based filtering.

# Data Description

This project utilizes the [goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k) dataset which contains six million ratings for ten thousand of the most popular books on Goodreads. The dataset also includes book metadata as well as tags, shelves, and genres.

# Methodology
To build a hybrid recommendation system, I combined results from a collaborative filtering model and a content-based filtering model.

## Web Scraping
Because the dataset did not include book descriptions, descriptions were scraped from Goodreads using the BeautifulSoup library. The code can be found at src/scrape.py. 

## Collaborative Filtering
The collaborative filtering approach uses similarities between users as well as similarities between items to provide recommendations. One advantage of this approach is that item and user embeddings can be learned automatically without having to manually engineer features. For this project, I implemented collaborative filtering using the NeuMF model in Tensorflow. NeuMF combines two commonly used training methods in collaborative filtering: matrix factorization and neural collaborative filtering (NCF). The architecture of the model is shown below and the full code for training the model can be found at src/ncf.py. Training run results, including losses, metrics, and hyperparameters, were tracked using MLflow.
![NeuMF Model](/imgs/neumf.png)


## Content Based Filtering
Content based filtering uses item features to recommend items similar to other items that the user likes. In order to encode every book in the dataset, I first combined all relevant information (title, author, summary, and tags) to form the "content" of each book. The content of each book was then vectorized using Term Frequency - Inverse Document Frequency (TF-IDF). An autoencoder model was then trained to reduce the dimension of the large TF-IDF vector. Due to the limits of my local machine and sparsity of the output of TF-IDF, an extra dimensionality reduction was performed prior to the autoencoder using truncated singular value decomposition. After obtaining an encoding for each book, books similar to a given book could be identified using cosine similarity. The code for preprocessing the data can be found at src/preprocess.py and the code for training the autoencoder can be found at src/cbf.py. 

## Hybrid Recommender
By training the collaborative filtering and content based filtering models, we are able to obtain three learned embeddings for each book: the matrix factorization (MF) item vector, the MLP item vector, and the encoded content vector. In order to perform efficient candidate generation, each set of embeddings for each book were clustered using K-Means, such that each book was assigned 3 clusters, one for each embedding space. The optimal number of clusters was determined using silhouette score. The code to run the clustering can be found at src/cluster.py. \
When a user prompts the program for recommendations based on a given book, first, candidates are chosen from each space based on the chosen book's cluster. Then, cosine similarity scores are calculated for all candidates and aggregated across the three spaces if a book appears in more than one space. Finally, the top "K" (as selected by the user) books are returned based on the ranked cosine similarity scores. The code for this can be found at src/recommend.py. 

# Results
### Example 1: Top 10 Recommendations for Twilight
As seen in the image, the list of recommendations include the other novels from the Twilight series as well as popular novels from the same genre (fantasy/romance/horror). \
\
![Twilight Recs](/imgs/twilight_recs.png)

### Example 2: Top 10 Recommendations for Harry Potter and the Sorcerer's Stone
As seen in the image, the list of recommendations include other novels from the Harry Potter series as well as spin off novels related to the series.\
\
![HP Recs](/imgs/harry_potter_recs.png)
