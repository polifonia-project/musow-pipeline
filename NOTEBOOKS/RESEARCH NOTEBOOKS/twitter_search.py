#imports
# For sending GET requests from the API
import requests
# For saving access tokens and for file management when creating and adding to the dataset
import os
# For dealing with json responses we receive from the API
# For displaying the data after
import pandas as pd
# For saving the response data in CSV format
import csv
# For parsing the dates received from twitter in readable formats
import dateutil.parser
#To add wait time between requests
import time

#LF token
#os.environ['TOKEN'] = 'AAAAAAAAAAAAAAAAAAAAADLsTwEAAAAAEAceYKKDFE0DUoWANpylf37zVuA%3DN1Arq9ZoC2LNMbKYjTtcZ2pk2C5rV806OWrxPzoP7MQgoWAuz6'

#Marilena token
os.environ['TOKEN'] = 'AAAAAAAAAAAAAAAAAAAAAJgsNAEAAAAAQcsgbUnOJJmqmU483%2F8x6n9V1i8%3Df0qaEo9cV1sWP4eyNQ6E9s8BiRjvFTSN9mSqithe8uIXSNP68x'

#Prep and functions for headers, URL, endpoint connection

def auth():
    return os.getenv('TOKEN')

def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers

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

#csv function

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

        #5. URLs 
        if ('entities' in tweet) and ('urls' in tweet['entities']):
            for url in tweet['entities']['urls']:
                url = url['expanded_url']
        else:
            url = " "

        #6. Tweet text
        text = tweet['text'] 
        
        # Assemble all data in a list
        res = [user, created_at, lang, like_count, quote_count, reply_count, retweet_count, text, url]

        # Append the result to the CSV file
        csvWriter.writerow(res)
        counter += 1    
    
    # When done, close the CSV file
    csvFile.close()

    # Print the number of tweets for this iteration
    print("# of Tweets added from this response: ", counter) 

def twitter_search(input_keyword, start, end, file_name, mresults, mcount):
    bearer_token = auth()
    headers = create_headers(bearer_token)
    keyword = input_keyword
    start_list = start

    end_list =  end

    max_results = mresults

    #Total number of tweets we collected from the loop
    total_tweets = 0

    # Create file
    csvFile = open(f'/Users/laurentfintoni/Desktop/University/COURSE DOCS/THESIS/Internship/DATA/TWITTER_SEARCHES/{file_name}.csv', "a", newline="", encoding='utf-8')
    csvWriter = csv.writer(csvFile)

    #Create headers for the data you want to save, in this example, we only want save these columns in our dataset
    csvWriter.writerow(['user', 'created_at', 'lang', 'like_count', 'quote_count', 'reply_count','retweet_count','tweet', 'url'])
    csvFile.close()

    for i in range(0,len(start_list)):

        # Inputs
        count = 0 # Counting tweets per time period
        max_count = mcount # Max tweets per time period
        flag = True
        next_token = None
        
        # Check if flag is true
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
                    append_to_csv(json_response, f'/Users/laurentfintoni/Desktop/University/COURSE DOCS/THESIS/Internship/DATA/TWITTER_SEARCHES/{file_name}.csv')
                    count += result_count
                    total_tweets += result_count
                    print("Total # of Tweets added: ", total_tweets)
                    print("-------------------")
                    time.sleep(5)                
            # If no next token exists
            else:
                if result_count is not None and result_count > 0:
                    print("-------------------")
                    print("Start Date: ", start_list[i])
                    append_to_csv(json_response, f'/Users/laurentfintoni/Desktop/University/COURSE DOCS/THESIS/Internship/DATA/TWITTER_SEARCHES/{file_name}.csv')
                    count += result_count
                    total_tweets += result_count
                    print("Total # of Tweets added: ", total_tweets)
                    print("-------------------")
                    time.sleep(5)
                
                #Since this is the final request, turn flag to false to move to the next time period.
                flag = False
                next_token = None
            time.sleep(5)
    print("Total number of results: ", total_tweets)
    pickle_df = pd.read_csv(f'/Users/laurentfintoni/Desktop/University/COURSE DOCS/THESIS/Internship/DATA/TWITTER_SEARCHES/{file_name}.csv')
    pickle_df.to_pickle(f'/Users/laurentfintoni/Desktop/University/COURSE DOCS/THESIS/Internship/DATA/TWITTER_SEARCHES/{file_name}.pkl')

