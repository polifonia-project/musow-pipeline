from musow_pipeline.logreg_functions import *

class PredictPipeline(object):
    #keywords to remove from URL and Title strings at prediction stages
    title_discard = ['404', 'Not Found', 'It needs a human touch', 'Page not found', 'We\'re sorry...', 'Not Acceptable!', 'Access denied', '412 Error', 'Unsupported browser', 'Last.fm', 'PortalTaxiMusic', 'YouTube', 'Robot or human?']
    url_discard = ['youtu', 'bandcamp', 'ebay', 'open.spotify.com', 'goo.gl', 'instagr.am', 'soundcloud', 'apple.co', 'amzn', 'masterstillmusic', 'facebook', 'last.fm', 'amazon', 'tidal.com', 'tmblr.co', 'dailymusicroll','apple.news', 'yahoo.com', 'etsy', 'nts.live', 'twitch.tv', 'radiosparx.com', 'freemusicarchive.org', 'blastradio', 'opensea', 'mixcloud', 'catalog.works', 'nft', 'allmusic.com', 'foundation.app', 'heardle', 'insession.agency', 'jobvite', 'career', 'docs.google.com/forms/', 'discogs.com', 'zora.co', 'play.google.com', 't.me', 'mintable.app', 'instagram', 'linkedin', 'forms.gle', 'vimeo', 'radioiita', 'spotify', 'event', 'mediafire', 'noodsradio', 'pinterest', 'rakuten', 'stackoverflow', 'fiverr', 'patreon', 'radio.nrk', 'bibliocommons', 'assetstore.unity', 'ebook-globalink.blogspot', 'gofund', 'rumble', 'ilovefreegle', 'pandora.app', 'solsea', 'dealsily', 'wikiart', 'dlsite', 'craigslist', 'clickasnap', 'ballsackradio', 'freedealsandoffers', 'stringsbymail', 'fileforum', 'bandlab', 'lnk.to']
    desc_discard = ['None', '! D O C T Y P E h t m l >', '! d o c t y p e h t m l >', '! D O C T Y P E H T M L >', '! D O C T Y P E h t m l P U B L I C ']
    #remove these websites from final results as they've already been assessed 
    whitelist = ['sheetmusiclibrary.website', 'sheetmusicplus', 'musicnotes', 'sheetmusic.kongashare', 'scoreexchange', 'sheetmusicdirect', 'practicingmusician', 'matonizz', 'lindseystirlingsheetmusic', 'churchofjesuschrist', 'praisecharts', 'mdmarchive', 'fabermusic', 'sheetmusicdownload', 'giamusic', 'audiosparx', 'atomicamusic', 'sounds.bl.uk', 'dancemusicarchive', 'sheetmusic.me']

    def train(t_input, t_feature, target, cv_int, max_feats, filename, path):
       return lr_training(t_input, t_feature, target, cv_int, max_feats, filename, path)
    
    def predict(path, filename, p_input, p_feature):
        return lr_predict(path, filename, p_input, p_feature)