import pandas as pd
import requests
import os


concepts_dict = {
                 'building':'http://data.europeana.eu/concept/base/29',
                 'ceramics':'http://data.europeana.eu/concept/base/31',
                 'drawing':'http://data.europeana.eu/concept/base/35',
                 'map':'http://data.europeana.eu/concept/base/43',
                 'furniture':'http://data.europeana.eu/concept/base/37',
                 'textile':'http://data.europeana.eu/concept/base/55',
                 'jewellery':'http://data.europeana.eu/concept/base/41',
                 'painting':'http://data.europeana.eu/concept/base/47',
                 'photograph':'http://data.europeana.eu/concept/base/48',
                 'postcard':'http://data.europeana.eu/concept/base/50',
                 'sculpture':'http://data.europeana.eu/concept/base/51',
                 'specimen':'http://data.europeana.eu/concept/base/167',
                 'tapestry':'http://data.europeana.eu/concept/base/54',
                 'toy':'http://data.europeana.eu/concept/base/56',
                 'woodwork':'http://data.europeana.eu/concept/base/59',
                 }

dest_path = '../data'

N = 5000

for category in concepts_dict.keys():

  skos_concept = concepts_dict[category]

  data_list = []
  response = {'nextCursor':'*'}
  while 'nextCursor' in response:
    qf_str = f'(skos_concept:"{skos_concept}" AND TYPE:IMAGE )'
    params = { 'reusability':'open','media':True,'cursor':response['nextCursor'] ,'qf':qf_str, 'query':'*', 'wskey':'api2demo'}
    response = requests.get('https://www.europeana.eu/api/v2/search.json', params = params).json()

    print(len(data_list))
    #print(response['totalResults'])
    if len(data_list)>N:
      break

    for item in response['items']:
      ID = item['id']
      URI = 'http://data.europeana.eu/item'+ID
      try:
        URL = item['edmIsShownBy'][0]
        data_dict = {'category':category,'skos_concept':skos_concept,'URI':URI,'ID':ID,'URL':URL}
        data_list.append(data_dict)
      except:
        pass

  df = pd.DataFrame(data_list)
  filename = f'{category}_data.csv'
  file_path = os.path.join(dest_path,filename)
  df.to_csv(file_path,index=False)