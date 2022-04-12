from ghapi.core import *
import pandas as pd

def github_search(keyword):
    #setup api object
    api = GhApi(owner='laurentfintoni', token='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')
    
    #setup api call for searches 
    search_result = api.search.repos(keyword, 'commiter-date') 

    #create lists of the needed items from returned json object 
    name = [item['name'] for item in search_result['items']]
    desc = [item['description'] for item in search_result['items']]
    url = [item['html_url'] for item in search_result['items']]
    topics = [item['topics'] for item in search_result['items']]
    license = []
    for item in search_result['items']:
        if item['license'] is None:
            license.append('')
        elif item['license']['name'] is None:
            license.append('')
        elif ('license' in item) and ('name' in item['license']):
            license.append(item['license']['name'])

    #create df w/ lists 
    gh_df = pd.DataFrame()
    gh_df['Name'] = pd.Series(name).astype('string') 
    gh_df['Description'] = pd.Series(desc).astype('string')
    gh_df['Topics'] = pd.Series(topics) 
    gh_df['URL'] = pd.Series(url).astype('string')
    gh_df['License'] = pd.Series(license).astype('string')
    
    return gh_df


#keyword list option 

input_kws = ['music archive', 'oral history']
counter = 0
for kw in input_kws:
    result = github_search(kw)
    result.to_pickle(f'/Users/laurentfintoni/Desktop/University/COURSE DOCS/THESIS/Internship/DATA/GH_PICKLES/{input_kws[counter]}.pkl')
    counter =+ 1

#single keyword option

input_kw = ''
result = github_search(input_kw)
result.to_pickle(f'/Users/laurentfintoni/Desktop/University/COURSE DOCS/THESIS/Internship/DATA/GH_PICKLES/{input_kw}')