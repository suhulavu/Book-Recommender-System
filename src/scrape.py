# IMPORTS
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import os
from tqdm import tqdm

def scrapeDesc(df):
    """
    Scrapes book descriptions

    Parameters
    -----------
    df : pd.DataFrame
        dataframe containing books with their goodread ids

    Returns
    --------
    failed_count : int
        the number of books that failed to scrape
    """

    # get ids
    new = False
    try:
        rem = df.loc[pd.isnull(df['description'])]
        ids = rem['goodreads_id']
    except KeyError:
        ids = df['goodreads_id']
        new = True

    # scrape book descriptions and genres
    descriptions = []
    failed_count = 0
    for id in tqdm(ids):

        # get content from webpage
        url = 'https://www.goodreads.com/book/show/{}'.format(id)
        try:
            page = requests.get(url)
        except requests.exceptions.ConnectionError:
            descriptions.append(None)
            continue

        # ensure successful HTTPS connection
        if page.status_code != 200:
            tqdm.write('Failed to establish connection to page: {}'.format(url))
            descriptions.append(None)
            continue

        soup = BeautifulSoup(page.content, 'lxml')
        
        # book description
        desc = soup.select('.TruncatedContent__text--large .DetailsLayoutRightParagraph__widthConstrained .Formatted')

        if len(desc) == 0:
            failed_count += 1
            tqdm.write('Failed book count: {}'.format(failed_count))
            descriptions.append(None)
            continue
        
        descriptions.append(desc[0].text)

        # pause between books
        time.sleep(1)
    
    # save data
    if new:
        df['description'] = descriptions
    else:
        df.loc[ids.index, 'description'] = descriptions

    if not os.path.exists('../data/scraped data'):
        os.mkdir('../data/scraped data')
    df.to_csv('../data/scraped data/scraped_data_{}.csv'.format(failed_count), index=False)

    return failed_count



if __name__ == '__main__':

    # read raw data
    df = pd.read_csv('../data/raw/books.csv', usecols=[1], names=['goodreads_id'], header=0)

    # scrape descriptions
    fails = scrapeDesc(df)

    # retry failed scrapes
    MAX_ITER = 30
    iters = 0
    while fails != 0:
        df = pd.read_csv('../data/scraped data/scraped_data_{}.csv'.format(fails))
        fails = scrapeDesc(df)

        iters += 1
        if iters >= MAX_ITER:
            break




