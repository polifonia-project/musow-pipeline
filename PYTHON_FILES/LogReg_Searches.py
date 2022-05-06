from pipeline_functions import *

class LogRegSearches(object):
    def train(t_input, t_feature, target, cv_int, score_type, filename, path):
        return lr_training(t_input, t_feature, target, cv_int, score_type, filename, path)

    def predict(path, filename, p_input, p_feature):
        return lr_predict(path, filename, p_input, p_feature)

    def search_twitter(token, keyword, start, end, mresults, mcount, file_name):
        return twitter_search(token, keyword, start, end, mresults, mcount, file_name)

    def scrape_links(link_list):
        return scrape_links(link_list)
    
    def predict_twitter(path, filename, p_input, p_feature, score, discard, filter):
        return twitter_predictions(path, filename, p_input, p_feature, score, discard, filter)

    def predict_resource(path, filename, p_input, p_feature, score, discard, savefile):
        return resource_predictions(path, filename, p_input, p_feature, score, discard, savefile)

    def tweets_to_classify(path, filetype):
        return tweets_to_classify(path, filetype)