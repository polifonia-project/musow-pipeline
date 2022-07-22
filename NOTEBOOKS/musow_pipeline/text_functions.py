path = '../'
# classifier
import pandas as pd
# web scraping
import requests
from bs4 import BeautifulSoup
from langdetect import detect
import trafilatura
from transformers import pipeline
from urllib.parse import urlparse
#logreg
from musow_pipeline.logreg_prediction import *

#load positives from description training set to check for matches 
checklist = pd.read_pickle(path+'LOGREG_RELEVANCE/TRAINING_SETS/archive_desc_training_v4.pkl')
checklist = checklist.loc[checklist['Target'] == 1]
checklist = checklist['URL'].to_list()
parsed = []
for url in checklist:
    parsedurl = urlparse(url)
    parsed.append(parsedurl[1])
parsed = set(parsed)

## these functions are intended to scrape URLs for text descriptions and predict their relevance using the logistic regression pipeline in logreg_prediction.py

def detect_en(text):
    try:
        return detect(text) == 'en'
    except:
        return False

def scrape_links(link_list, pred_df, filename):
    """ Scrape links from a list, save scrapes, combine them w/ a DF of predictions and return a final DF for description classification.

    Parameters
    ----------
    link_list:
        list of comma separated urls to scrape (list of str)
    pred_df:
        dataframe of resource predictions from which link_list is extracted (e.g. tweets, github repos etc...), is rejoined on URL column. Optional, can be set to None if uneeded (var)
    filename: 
        name for saved scrapes, as pkl file to avoid having to rerun scrapes if needed (str)
    """
    links = pd.DataFrame(columns=['Title', 'Description', 'URL'])
    summarizer = pipeline("summarization", model='sshleifer/distilbart-cnn-12-6')
    counter = 0

    for link in link_list:
        URL = link
        page = None 
        status = None
        ARTICLE = ''
        try:
            x = requests.head(URL, timeout=15)
            content_type = x.headers["Content-Type"] if "Content-Type" in x.headers else "None"
            if ("text/html" in content_type.lower()):
                page = requests.get(URL, timeout=15)
                status = page.status_code
        except Exception:
            pass

        if status == 200 and page:
            soup = BeautifulSoup(page.content, "html.parser")
            title = ' '.join([t.text for t in soup.find('head').find_all('title')]).strip() \
                if soup and soup.find('head') and soup.find('body') is not None \
                else URL

            try:
                downloaded = trafilatura.fetch_url(URL)
                ARTICLE = trafilatura.extract(downloaded, include_comments=False, include_tables=True, target_language='en', deduplicate=True)
            except Exception:
                results = soup.find_all(['h1', 'p'])
                text = [result.text for result in results]
                ARTICLE = ' '.join(text)

            if ARTICLE is not None and len(ARTICLE) > 200:
                # text summarisation
                max_chunk = 500
                #removing special characters and replacing with end of sentence
                ARTICLE = ARTICLE.replace('.', '.<eos>')
                ARTICLE = ARTICLE.replace('?', '?<eos>')
                ARTICLE = ARTICLE.replace('!', '!<eos>')
                sentences = ARTICLE.split('<eos>')
                current_chunk = 0
                chunks = []

                # split text to process
                for sentence in sentences:
                    if len(chunks) == current_chunk + 1:
                        if len(chunks[current_chunk]) + len(sentence.split(' ')) <= max_chunk:
                            chunks[current_chunk].extend(sentence.split(' '))
                        else:
                            current_chunk += 1
                            chunks.append(sentence.split(' '))
                    else:
                        chunks.append(sentence.split(' '))

                for chunk_id in range(len(chunks)):
                    chunks[chunk_id] = ' '.join(chunks[chunk_id])
                try:
                    res = summarizer(chunks, min_length = 30, max_length = 120, do_sample=False)
                    # summary
                    text = ' '.join([summ['summary_text'] for summ in res])
                except Exception:
                    text = ARTICLE
                    continue
            else:
                text = ARTICLE
            #display scraped URL and increase counter
            counter += 1
            print(counter, URL)
            #create new row w/ scraped URL data then append to DF
            new_row = {'Title': title, 'Description': text, 'URL': URL.strip()}
            new_df = pd.DataFrame(data=new_row, index=[0])
            links = pd.concat([links, new_df], ignore_index=True)
    #clean DF w/ discard variable and lang detection 
    links = links.fillna('None')
    links = links[~links.Description.str.contains('|'.join(PredictPipeline.desc_discard))]
    links = links[links['Description'].apply(detect_en)]
    #check if an additional DF for rejoining was passed as parameter 
    if pred_df is not None:
        scrapes_preds = pd.merge(pred_df, links, on='URL')
        scrapes_preds.to_pickle(f'{path}LOGREG_RELEVANCE/SCRAPES/{filename}.pkl')
        print('Total links scraped:', len(scrapes_preds))
        return scrapes_preds
    else:
        scrapes_preds = links
        scrapes_preds.to_pickle(f'{path}LOGREG_RELEVANCE/SCRAPES/{filename}.pkl')
        print('Total links scraped:', len(scrapes_preds))
        return scrapes_preds
    

def resource_predictions(path, filename, p_input, p_feature, score, savefile):
    """ Predict relevant URL descriptions using a pickled model based on Logistic regression and TF-IDF.

    Parameters
    ----------
    path:
        path to saved model (str/var)
    filename: 
        model file name (str)
    p_input:
        input to predict, dataframe generated by scrape_links (var)
    p_feature:
        df column, should be Description, values should be string formatted (str)
    score: 
        which prediction score to filter the results by 1/0 (int)
    savefile: 
        name for the final csv to be saved under (str)
    """
    #catch for empty
    if len(filename) == 0:
        return 'Sorry no URLs to classify!'
    #classify reslults
    preds = lr_predict(path, filename, p_input, p_feature)
    #clean DF
    preds = preds.drop_duplicates(['Description'], keep='last')
    preds = preds.loc[preds['Description'] != '']
    #filter by score parameter
    preds = preds.loc[preds['Prediction'] == score]
    #discard based on variables
    preds = preds[~preds.URL.str.contains('|'.join(PredictPipeline.url_discard))]
    preds = preds[~preds.URL.str.contains('|'.join(PredictPipeline.whitelist))]
    preds = preds[~preds.Title.str.contains('|'.join(PredictPipeline.title_discard))]
    #check if results exist in training set and add a note
    preds.loc[preds['URL'].str.contains('|'.join(parsed), case=False, regex=True), ['Match']] = 'training set match'
    #sort by score, descending
    preds = preds.sort_values(by='Score', ascending=False).reset_index(drop=True)
    preds.to_csv(f'{path}LOGREG_RELEVANCE/PREDICTIONS/{savefile}.csv')
    return preds