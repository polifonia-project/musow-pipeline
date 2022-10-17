from musow_pipeline.twitter_functions import *

class TwitterPipeline(object):
    def search_weekly(token, keyword_list, max_results, max_counts):
        return twitter_search_weekly(token, keyword_list, max_results, max_counts)
    
    def search_custom(token, keyword_list, start_list, end_list, max_results, max_counts):
        return twitter_search_custom(token, keyword_list, start_list, end_list, max_results, max_counts)
    
    def classify_tweets(folder, filetype):
        return tweets_to_classify(folder, filetype)

    def predict_twitter(path, filename, p_input, p_feature, score):
        return twitter_predictions(path, filename, p_input, p_feature, score)