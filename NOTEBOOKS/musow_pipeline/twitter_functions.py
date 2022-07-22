#imports 
path = '../'
import csv , dateutil.parser , time
from datetime import date , timedelta
import os
import re
import pandas as pd
#querying
import requests
#cleaning
import emoji
#logreg 
from musow_pipeline.logreg_prediction import *

## These functions cover the use of twitter to search for new resources and evaluate them using the logistic regression pipeline in logreg_prediction.py

## Twitter search - five functions to cover searching twitter via API and return a csv and a dataframe for each search

def create_url(keyword, start_date, end_date, max_results):
        #Change to the endpoint you want to collect data from
        search_url = "https://api.twitter.com/2/tweets/search/all" 
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

        tweet_id = tweet['id']

        # 2. Time created
        created_at = dateutil.parser.parse(tweet['created_at'])

        # 3. Language
        lang = tweet['lang']

        # 4. Tweet metrics
        retweet_count = tweet['public_metrics']['retweet_count']
        reply_count = tweet['public_metrics']['reply_count']
        like_count = tweet['public_metrics']['like_count']
        quote_count = tweet['public_metrics']['quote_count']

        #5. URLs w/ catches for unwound and expanded URLs and multiple URLs  
        if ('entities' in tweet) and ('urls' in tweet['entities']) and ('unwound' in tweet['entities']['urls']):
            for url in tweet['entities']['urls']['unwound']:
                url = [url['url'] for url in tweet['entities']['urls']['unwound'] if 'twitter.com' not in url['url']]
                url = ', '.join(url) 
        elif ('entities' in tweet) and ('urls' in tweet['entities']):
            for url in tweet['entities']['urls']:
                url = [url['expanded_url'] for url in tweet['entities']['urls'] if 'twitter.com' not in url['expanded_url']]
                url = ', '.join(url)
        else:
            url = ""

        #6. Tweet text
        text = give_emoji_free_text(tweet['text'])

        # Assemble all data in a list
        res = [user, tweet_id, created_at, lang, like_count, quote_count, reply_count, retweet_count, text, url]

        # Append the result to the CSV file
        csvWriter.writerow(res)
        counter += 1

    # When done, close the CSV file
    csvFile.close()

    # Print the number of tweets for this iteration
    print("# of Tweets added from this response: ", counter)

def twitter_search(token, keyword, start, end, mresults, mcount, file_name):
    bearer_token = token
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    start_list = start
    end_list =  end
    max_results = mresults
    total_tweets = 0

    # Create file
    csvFile = open(f'{path}TWITTER_SEARCHES/RAW_SEARCHES/{file_name}.csv', "a", newline="", encoding='utf-8')
    csvWriter = csv.writer(csvFile)
    csvWriter.writerow(['user', 'tweet id', 'created_at', 'lang', 'like_count', 'quote_count', 'reply_count','retweet_count','tweet', 'URL'])
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

## Twitter search options - two functions (weekly or user specified time input) that call the above primary search function and return a list of filenames 

def twitter_search_weekly (token, keyword_list, max_results, max_counts):
    """ Search tweets for the last week only.

    Parameters
    ----------
    token:
        twitter search token (str)
    keyword_list:
        keywords to search for, automatically sets them to be searched for w/o RTs (list of str)
    max_results / max_counts:
        max tweets per json response, max tweets per search period. 100 is max for normal API token, 500 is max for Academic token. (int)
    """
    today = date.today()
    week_ago = today - timedelta(days=7)
    start = [week_ago.strftime("%Y-%m-%dT%H:%M:%S.000Z")]
    end = [today.strftime("%Y-%m-%dT%H:%M:%S.000Z")]
    #input max results / counts
    mresults = max_results
    mcount = max_counts
    #format keywords for search
    input_keywords = [f'\"{k}\" -is:retweet' for k in keyword_list]
    #send to search
    filenames = [] 
    for k in input_keywords:
        filename = re.sub(r"([^A-Za-z0-9]+)", '', k) + f'_{start[0][0:10]}' + f'_{end[-1][5:10]}'
        filename = re.sub(r"isretweet", '', filename)
        twitter_search(token, k, start, end, mresults, mcount, filename)
        filenames.append(filename)
    return filenames

def twitter_search_custom (token, keyword_list, start_list, end_list, max_results, max_counts):
    """ Search tweets for a custom time frame.

    Parameters
    ----------
    token:
        twitter search token (str)
    keyword_list:
        keywords to search for, automatically sets them to be searched for w/o RTs (list of str)
    start / end_list:
        dates to cycle through, should be symmetrical and formatted according to Twitter API standards: YYYY-MM-DDTHH:MM:SS.000Z (list of str)
    max_results / max_counts:
        max tweets per json response, max tweets per search period. 100 is max for normal API token, 500 is max for Academic token. (int)
    """
    start = start_list
    end = end_list
    #input max results / counts
    mresults = max_results
    mcount = max_counts
    #format keywords for search
    input_keywords = [f'\"{k}\" -is:retweet' for k in keyword_list]
    #send to search
    #send to search
    filenames = []
    for k in input_keywords:
        filename = re.sub(r"([^A-Za-z0-9]+)", '', k) + f'_{start[0][0:10]}' + f'_{end[-1][5:10]}'
        filename = re.sub(r"isretweet", '', filename)
        twitter_search(token, k, start, end, mresults, mcount, filename)
        filenames.append(filename)
    return filenames

## Twitter preparation - function to load all tweets from search options and create a single dataframe for prediction, lets user know how many tweets to process 

def tweets_to_classify(folder, filetype):
    """ Merge all tweet searches together.

    Parameters
    ----------
    folder:
        path to folder containing raw searches (str)
    filetype:
        the ending of the files to load, can be manually inputed w/ .pkl at the end or automatically generated using the filenames returned by search options (str)
    """
    raw_searches = folder
    result = pd.DataFrame()
    tweets_to_classify = pd.DataFrame()
    for file in os.listdir(raw_searches):
        if file.endswith(filetype):
            result = pd.read_pickle(raw_searches+file)
            tweets_to_classify = pd.concat([tweets_to_classify, result])
            tweets_to_classify = tweets_to_classify.reset_index(drop=True)
    print('Total tweets to classify:', len(tweets_to_classify))
    return tweets_to_classify

## Twitter prediction - function to classify tweets based on model generated with logreg pipeline returns a dataframe of predictions and a list of URLs to scrape 

def twitter_predictions(path, filename, p_input, p_feature, score):
    """ Predict relevant tweets using a pickled model based on Logistic regression and TF-IDF.

    Parameters
    ----------
    p_input:
        input to predict, dataframe generated by tweets_to_classify (var)
    p_feature:
        df column, should be tweet, values should be string formatted (str)
    filename: 
        model file name (str)
    path: 
        parent folder (str/var)
    score: 
        which prediction score to filter the results by 1/0 (int)
    """
    #catch for empty 
    if len(filename) == 0:
        return 'Sorry no tweets to classify!'
    #classify input
    preds = PredictPipeline.predict(path, filename, p_input, p_feature)
    #drop duplicate tweets 
    preds = preds.drop_duplicates(['tweet'], keep='last')
    #set filter based on parameter
    preds = preds.loc[preds['Prediction'] == score]
    #discard urls based on values in discard variables 
    preds = preds[~preds.URL.str.contains('|'.join(PredictPipeline.url_discard))]
    preds = preds[~preds.URL.str.contains('|'.join(PredictPipeline.whitelist))]
    #check for multiple URls, create dupes if needed 
    preds = preds.assign(URL = preds.URL.str.split(', ')).explode('URL', ignore_index=True)
    #sort results by score, descending
    preds = preds.sort_values(by='Score', ascending=False).reset_index(drop=True)
    #drop unneeded columns 
    preds = preds[['tweet', 'Prediction', 'Score', 'Probability', 'Input Length', 'URL', 'Search KW', 'created_at', 'user', 'tweet id']]
    #standardize tweet date 
    preds['created_at'] = preds['created_at'].astype(str)
    preds['created_at'] = preds['created_at'].str[0:10]
    preds = preds.rename({'created_at': 'tweet date'}, axis=1)
    #create additional variable for scraping 
    preds = preds.drop_duplicates(['URL'], keep='last')
    twitter_link_list = [link for link in preds['URL']]
    #give user result overview 
    print('Total tweets predicted:', len(preds))
    return preds, twitter_link_list