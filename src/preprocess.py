# IMPORTS
import pandas as pd
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import pkg_resources
from symspellpy.symspellpy import SymSpell
import re
import pickle
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

def mergeTags():
    """
    Merges tag data with book data

    Returns
    --------
    df_books : pd.DataFrame
        dataframe with books and tags
    """

    # READING DATA
    df_tags = pd.read_csv('../data/raw/tags.csv')
    df_book_tags = pd.read_csv('../data/raw/book_tags.csv')
    df_books = pd.read_csv('../data/raw/books.csv')


    # MERGING TAG DATA
    df_tag = df_book_tags.merge(df_tags, how='inner', on='tag_id')


    # FILTERING TAGS
    df_tag = df_tag.loc[~((df_tag['tag_name'] == 'to-read') | (df_tag['tag_name'] == 'favorites') | (df_tag['tag_name'] == 'currently-reading'))]
    df_tag = df_tag.loc[df_tag['count'] >= 1000]


    # COMBINE TAGS FOR EACH BOOK
    df_tag = df_tag.drop(columns=['tag_id', 'count']).groupby('goodreads_book_id', as_index=False).agg({'tag_name': ' '.join})


    # MERGE TAGS WITH BOOK DATA
    df_books = df_books.merge(df_tag, on='goodreads_book_id', how='left')


    # RETURN DATAFRAME
    return df_books


def checkEnglish(desc):
    """
    Checks whether a description is english

    Parameters
    -----------
    desc : str
        description
    
    Returns
    --------
    bool
        True if language detected is English
    """

    try:
        lang = detect(desc)
        return lang == 'en'
    except LangDetectException:
        return False


def processContent(df):
    """
    Preprocesses book data to train a content-based filtering model

    Parameters
    -----------
    df : pd.DataFrame
        dataframe of compiled book data
    """

    # CHANGING DTYPES
    df = df.astype({
        'authors' : str,
        'description' : str,
        'title' : str,
        'tag_name' : str
    })

    # PROCESS AUTHORS/TITLES
    df['author'] = df['authors'].map(lambda x : x[:x.index(',')] if ',' in x else x)
    df['title'] = df['title'].map(lambda x : x[:x.index('(')] if '(' in x else x)
    
    
    # FILTERING COLUMNS
    cols = ['isbn', 'isbn13', *['ratings_{}'.format(i) for i in range(1, 6)], 'image_url','small_image_url', 
            'work_ratings_count', 'work_text_reviews_count', 'ratings_count','average_rating', 'language_code', 
            'original_publication_year', 'books_count','original_title', 'authors']
    df = df.drop(columns=cols)
    
    
    # LANGUAGE DETECTION
    DetectorFactory.seed = 42
    df = df.loc[[checkEnglish(x) for x in tqdm(df['description'], desc='Language Check')]]
    
    
    # SPELL CHECK
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dictionary_path = pkg_resources.resource_filename('symspellpy', "frequency_dictionary_en_82_765.txt")
    bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)
    df['description'] = [sym_spell.lookup_compound(desc, max_edit_distance=2)[0]._term for desc in tqdm(df['description'], desc='Spell Check')]
    
    
    # TEXT PROCESSING
    # combine columns
    df['content'] = df['author'] + ' ' + df['title'] + ' ' + df['description'] + ' ' + df['tag_name']
    
    # remove punctuation
    df['content'] = df['content'].map(lambda x: re.sub('[,\.!?:)(]', '', x))

    # tokenization
    corpus = [word_tokenize(desc) for desc in df['content']]

    # removing stopwords and lemmatization
    stopwords = nltk.corpus.stopwords.words('english')
    stemmer = SnowballStemmer('english')
    corpus = [' '.join([stemmer.stem(token) for token in desc if token not in stopwords]) for desc in corpus]
    
    
    # VECTORIZATION
    tfidf = TfidfVectorizer(
        analyzer='word',
        stop_words='english',
        ngram_range=(1, 2),
        norm='l2',
        max_features=50000,
    )
    corpus = tfidf.fit_transform(corpus)
    
    
    # SAVING DATA
    pickle.dump(corpus, open('../data/processed/tfidf_matrix.pkl', 'wb'))
    pickle.dump(df, open('../data/processed/df_processed.pkl', 'wb'))



if __name__ == "__main__":

    # read scraped data
    df_scraped = pd.read_csv('../data/scraped data/scraped_data_final.csv')

    # get book data with tags
    df_books = mergeTags()

    # merge dataframes
    df_scraped = df_scraped.rename(columns={'goodreads_id' : 'goodreads_book_id'})
    df_final = df_books.merge(df_scraped, on='goodreads_book_id', how='inner')

    # process data
    processContent(df_final)


