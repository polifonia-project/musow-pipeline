from ghapi.core import GhApi

api = GhApi(owner='laurentfintoni', token='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')

test = api.search.repos('music archive') 

for item in test['items']:
    print(item['name'])