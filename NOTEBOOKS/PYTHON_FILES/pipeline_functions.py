path = '../'
import csv , dateutil.parser , time
import os
# classifier
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report 
from sklearn.model_selection import cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
# web scraping
import requests
from bs4 import BeautifulSoup
#!pip3 install trafilatura
import trafilatura
from transformers import pipeline
#cleaning 
import emoji


def lr_training(t_input, t_feature, target, cv_int, score_type, filename, path):
    """ Create a text classifier based on Logistic regression and TF-IDF. Use cross validation 
    
    Parameters
    ----------
    t_input: 
        dataframe of the training set
    t_feature: 
        df column, text of tweet or description of the resource
    target: 
        df column, [0,1] values
    cv_int: int
        the number of cross validation folding
    score_type: str
        precision or recall
    filename: str
        model file name
    path: str
        parent folder
    """
    # TODO eda to define max_features=1000
      
    tfidf_transformer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, max_features=1000) 
    x_train = tfidf_transformer.fit_transform(t_input[t_feature])
    y_train = t_input[target].values
    model = LogisticRegressionCV(solver='liblinear', random_state=44, cv=cv_int, scoring=score_type)
    
    # export
    model.fit(x_train, y_train)
    export_model = f'LOGREG_RELEVANCE/MODELS/{filename}_model.pkl'
    export_vectorizer = f'LOGREG_RELEVANCE/MODELS/{filename}_vectorizer.pkl'
    pickle.dump(model, open(path+export_model, 'wb'))
    pickle.dump(tfidf_transformer, open(path+export_vectorizer, 'wb'))
    
    # report
    y_pred = cross_val_predict(model, x_train, y_train, cv=cv_int)
    report = classification_report(y_train, y_pred)
    print('report:', report, sep='\n')
    return model
    
def lr_predict(path, filename, p_input, p_feature):
    """ Classify text using a pickled model based on Logistic regression and TF-IDF.
    
    Parameters
    ----------
    p_input: 
        dataframe of the prediction set
    p_feature: 
        df column, text of tweet or description of the resource
    filename: str
        model file name
    path: str
        parent folder
    """
    export_model = f'{path}LOGREG_RELEVANCE/MODELS/{filename}_model.pkl'
    export_vectorizer = f'{path}LOGREG_RELEVANCE/MODELS/{filename}_vectorizer.pkl'
    model = pickle.load(open(export_model, 'rb'))
    tfidf_transformer = pickle.load(open(export_vectorizer, 'rb'))
  
    #result = loaded_model.score(X_test, Y_test)
    #x_new_count = count_vect.transform(p_input[p_feature])
    x_predict = tfidf_transformer.transform(p_input[p_feature])
    y_predict = model.predict(x_predict)
    scores = model.decision_function(x_predict)
    probability = model.predict_proba(x_predict)
    
    #results = [r for r in y_predict]
    result = p_input.copy()
    result['Prediction'] = y_predict
    result['Score'] = scores
    result['Probability'] = probability[:,1]
    result['Input Length'] = result[p_feature].str.len()
    return result

def create_url(keyword, start_date, end_date, max_results):
        search_url = "https://api.twitter.com/2/tweets/search/all" #Change to the endpoint you want to collect data from
        #change params based on the endpoint you are using
        query_params = {'query': keyword,
                        'start_time': start_date,
                        'end_time': end_date,
                        'max_results': max_results,
                        'expansions': 'author_id,in_reply_to_user_id,geo.place_id',
                        'tweet.fields': 'id,text,author_id,in_reply_to_user_id,geo,conversation_id,created_at,lang,public_metrics,referenced_tweets,reply_settings,source,entities',
                        'user.fields': 'id,name,username,created_at,description,public_metrics,verified',
                        'place.fields': 'full_name,id,country,country_code,geo,name,place_type',
                        'next_token': {}}
        return (search_url, query_params)
    
def connect_to_endpoint(url, headers, params, next_token = None):
    params['next_token'] = next_token   #params object received from create_url function
    response = requests.request("GET", url, headers = headers, params = params)
    print("Endpoint Response Code: " + str(response.status_code))
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()

def give_emoji_free_text(text):
    return emoji.replace_emoji(text, replace='')

def append_to_csv(json_response, fileName):
    #A counter variable
    counter = 0

    #Open OR create the target CSV file
    csvFile = open(fileName, "a", newline="", encoding='utf-8')
    csvWriter = csv.writer(csvFile)
    
    #setup usernames via includes
    username = {user['id']: user['username'] for user in json_response['includes']['users']}
    
    #Loop through each tweet
    for tweet in json_response['data']:

        # 1. Username
        author_id = tweet['author_id']
        user = username[author_id]

        # 2. Time created
        created_at = dateutil.parser.parse(tweet['created_at'])

        # 3. Language
        lang = tweet['lang']

        # 4. Tweet metrics
        retweet_count = tweet['public_metrics']['retweet_count']
        reply_count = tweet['public_metrics']['reply_count']
        like_count = tweet['public_metrics']['like_count']
        quote_count = tweet['public_metrics']['quote_count']

        #5. URLs w/ a catch for tweets w/ two links TODO: how to catch more than two links? 
        if ('entities' in tweet) and ('urls' in tweet['entities']):
            for url in tweet['entities']['urls']:
                url = [url['expanded_url'] for url in tweet['entities']['urls'] if 'twitter.com' not in url['expanded_url']]
                url = ', '.join(url)
        else:
            url = ""
        
        #6. Tweet text
        text = give_emoji_free_text(tweet['text']) 
        
        # Assemble all data in a list
        res = [user, created_at, lang, like_count, quote_count, reply_count, retweet_count, text, url]

        # Append the result to the CSV file
        csvWriter.writerow(res)
        counter += 1    
    
    # When done, close the CSV file
    csvFile.close()

    # Print the number of tweets for this iteration
    print("# of Tweets added from this response: ", counter) 

def twitter_search(token, keyword, start, end, mresults, mcount, file_name):
    
    # TODO filter tweets in english only OR tweak TF-IDF stopwords (lang detection)
    bearer_token = token
    headers = {"Authorization": "Bearer {}".format(bearer_token)} 
    start_list = start
    end_list =  end
    max_results = mresults
    total_tweets = 0

    # Create file
    csvFile = open(f'{path}TWITTER_SEARCHES/RAW_SEARCHES/{file_name}.csv', "a", newline="", encoding='utf-8')
    csvWriter = csv.writer(csvFile)
    csvWriter.writerow(['user', 'created_at', 'lang', 'like_count', 'quote_count', 'reply_count','retweet_count','tweet', 'URL'])
    csvFile.close()

    for i in range(0,len(start_list)):
        # Inputs
        count = 0 # Counting tweets per time period
        max_count = mcount # Max tweets per time period
        flag = True
        next_token = None
        
        while flag:
            # Check if max_count reached
            if count >= max_count:
                break
            print("-------------------")
            print("Token: ", next_token)
            url = create_url(keyword, start_list[i],end_list[i], max_results)
            json_response = connect_to_endpoint(url[0], headers, url[1], next_token)
            result_count = json_response['meta']['result_count']

            if 'next_token' in json_response['meta']:
                # Save the token to use for next call
                next_token = json_response['meta']['next_token']
                print("Next Token: ", next_token)
                if result_count is not None and result_count > 0 and next_token is not None:
                    print("Start Date: ", start_list[i])
                    append_to_csv(json_response, f'{path}TWITTER_SEARCHES/RAW_SEARCHES/{file_name}.csv')
                    count += result_count
                    total_tweets += result_count
                    print(f"Total # of Tweets added for '{keyword}':", total_tweets)
                    print("-------------------")
                    time.sleep(5)                
            # If no next token exists
            else:
                if result_count is not None and result_count > 0:
                    print("-------------------")
                    print("Start Date: ", start_list[i])
                    append_to_csv(json_response, f'{path}TWITTER_SEARCHES/RAW_SEARCHES/{file_name}.csv')
                    count += result_count
                    total_tweets += result_count
                    print(f"Total # of Tweets added for '{keyword}':", total_tweets)
                    print("-------------------")
                    time.sleep(5)

                #Since this is the final request, turn flag to false to move to the next time period.
                flag = False
                next_token = None
            time.sleep(5)
    print("Total number of results:", total_tweets)
    
    df = pd.read_csv(f'{path}TWITTER_SEARCHES/RAW_SEARCHES/{file_name}.csv', keep_default_na=False, dtype={"user": "string", "lang": "string", "tweet": "string", "URL": "string"})
    
    # clean the tweet from meentions, hashtags, emojis
    df['tweet'].replace( { r"@[A-Za-z0-9_]+": '' }, inplace= True, regex = True)
    df['tweet'].replace( { r"#": '' }, inplace= True, regex = True)
    
    # remove tweets that are not in english, have empty URLs, or have duplicate URLs
    df = df[df['lang'].isin(['en'])]
    df = df[df.URL != '']
    df = df.drop_duplicates(['URL'], keep='last')

    #add a column for the search keyword
    df['Search KW'] = keyword

    #pickle df for reuse
    df.to_pickle(f'{path}TWITTER_SEARCHES/RAW_SEARCHES/{file_name}.pkl')

def scrape_links(link_list, pred_df, filename):
    """ Scrape links from classified tweets, save scrapes, combine them w/ tweets and return a DF for description classification.
    
    Parameters
    ----------
    p_input: 
        dataframe of the prediction set
    p_feature: 
        df column, text of tweet or description of the resource
    filename: str
        model file name
    path: str
        parent folder
    score: int
        which prediction score to filter the results by 1/0
    discard: variable
        a list of terms to check against to remove tweets
    filter: str 
        a string against which to further filter predictions 
    """
    links = pd.DataFrame(columns=['Title', 'Description', 'URL'])
    summarizer = pipeline("summarization", model='sshleifer/distilbart-cnn-12-6')
    
    for link in link_list:
        URL = link
        page = None
        ARTICLE = ''
        try:
            x = requests.head(URL, timeout=15)
            content_type = x.headers["Content-Type"] if "Content-Type" in x.headers else "None"
            if ("text/html" in content_type.lower()):
                page = requests.get(URL, timeout=15)
        except Exception:
            pass
        
        if page:
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
                    res = summarizer(chunks, min_length = int(0.2 * len(text)), max_length = int(0.5 * len(text)), do_sample=False)
                    # summary
                    text = ' '.join([summ['summary_text'] for summ in res])
                except Exception:
                    text = ARTICLE
                    continue
            else:
                text = ARTICLE
            print(URL)
            new_row = {'Title': title, 'Description': text, 'URL': URL.strip()}
            new_df = pd.DataFrame(data=new_row, index=[0])
            links = pd.concat([links, new_df], ignore_index=True)
    discard = ['None', '! D O C T Y P E h t m l >', '! d o c t y p e h t m l >', '! D O C T Y P E H T M L >']
    links = links.fillna('None')
    links = links[~links.Description.str.contains('|'.join(discard))]
    twitter_scrapes_preds = pd.merge(pred_df, links, on='URL')
    twitter_scrapes_preds.to_pickle(f'{path}LOGREG_RELEVANCE/SCRAPES/{filename}.pkl')
    print(len(twitter_scrapes_preds))
    return twitter_scrapes_preds

def twitter_predictions(path, filename, p_input, p_feature, score, discard, filter):
    """ Predict relevant tweets using a pickled model based on Logistic regression and TF-IDF.
    
    Parameters
    ----------
    p_input: 
        dataframe of the prediction set
    p_feature: 
        df column, text of tweet or description of the resource
    filename: str
        model file name
    path: str
        parent folder
    score: int
        which prediction score to filter the results by 1/0
    discard: variable
        a list of terms to check against to remove tweets
    filter: str 
        a string against which to further filter predictions 
    """
    preds = lr_predict(path, filename, p_input, p_feature)
    preds = preds.drop_duplicates(['tweet'], keep='last')
    preds = preds.loc[preds['Prediction'] == score]
    preds = preds[~preds.URL.str.contains('|'.join(discard))]
    preds = preds.sort_values(by='Score', ascending=False).reset_index(drop=True)
    preds = preds[['tweet', 'Prediction', 'Score', 'Probability', 'Input Length', 'URL', 'Search KW']]
    if filter != '':
        preds = preds[preds['tweet'].str.contains(filter)]
        preds = preds.reset_index(drop=True)
    twitter_link_list = [link for link in preds['URL']]
    print('Total tweets classified:', len(preds))
    return preds, twitter_link_list

def resource_predictions(path, filename, p_input, p_feature, score, discard, savefile):
    """ Predict relevant URL descriptions using a pickled model based on Logistic regression and TF-IDF.
    
    Parameters
    ----------
    p_input: 
        dataframe of the prediction set
    p_feature: 
        df column, text of tweet or description of the resource
    filename: str
        model file name
    path: str
        parent folder
    score: int
        which prediction score to filter the results by 1/0
    discard: variable
        a list of terms to check against to remove tweets
    savefile: str
        name for the final csv to be saved under 
    """
    if len(filename) == 0:
        return 'Sorry no URLs to classify!'
    preds = lr_predict(path, filename, p_input, p_feature)
    preds = preds.drop_duplicates(['Description'], keep='last')
    preds = preds.loc[preds['Description'] != '']
    preds = preds.loc[preds['Prediction'] == score]
    preds = preds[~preds.URL.str.contains('|'.join(discard))]
    preds = preds[~preds.Title.str.contains('|'.join(discard))]
    preds = preds.sort_values(by='Score', ascending=False).reset_index(drop=True)
    preds.to_csv(f'{path}LOGREG_RELEVANCE/PREDICTIONS/{savefile}.csv')
    print(preds)
    return preds

def tweets_to_classify(path, filetype):   
    """ Merge all tweet searches together.
    
    Parameters
    ----------
    path: 
        for raw searches folder
    filetype: 
        the ending of the files to load, you can call just .pkl or also the date tag from file names
    """  
    raw_searches = path+'TWITTER_SEARCHES/RAW_SEARCHES/'
    result = pd.DataFrame()
    tweets_to_classify = pd.DataFrame()
    for file in os.listdir(raw_searches):
        if file.endswith(filetype):
            result = pd.read_pickle(raw_searches+file)
        tweets_to_classify = pd.concat([tweets_to_classify, result])
        tweets_to_classify = tweets_to_classify.reset_index(drop=True)
    print('Total tweets to classify:', len(tweets_to_classify))
    return tweets_to_classify