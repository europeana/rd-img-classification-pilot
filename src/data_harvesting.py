import pandas as pd
import requests
import os

from ds_utils import create_dir
    
def query_single_category(skos_concept, N):  
  CHO_list = []
  response = {'nextCursor':'*'}
  while 'nextCursor' in response:
    
    if len(CHO_list)>N:
      break

    params = {
        'reusability':'open',
        'media':True,
        'cursor':response['nextCursor'],
        'qf':f'(skos_concept:"{skos_concept}" AND TYPE:IMAGE )', 
        'query':'*', 
        'wskey':'api2demo'
    }
    response = requests.get('https://www.europeana.eu/api/v2/search.json', params = params).json()

    for item in response['items']:
      ID = item['id']
      URI = 'http://data.europeana.eu/item'+ID
      try:
        URL = item['edmIsShownBy'][0]
        CHO_list.append({'category':category,'skos_concept':skos_concept,'URI':URI,'ID':ID,'URL':URL})
      except:
        pass
    
  return pd.DataFrame(CHO_list[:N])
    
 

ec_vocab = {
                 'building':'http://data.europeana.eu/concept/base/29',
                 'ceramics':'http://data.europeana.eu/concept/base/31',
                 'drawing':'http://data.europeana.eu/concept/base/35',
                 'furniture':'http://data.europeana.eu/concept/base/37',
                 'jewellery':'http://data.europeana.eu/concept/base/41',
                 'map':'http://data.europeana.eu/concept/base/43',
                 'painting':'http://data.europeana.eu/concept/base/47',
                 'photograph':'http://data.europeana.eu/concept/base/48',
                 'postcard':'http://data.europeana.eu/concept/base/50',
                 'sculpture':'http://data.europeana.eu/concept/base/51',
                 'specimen':'http://data.europeana.eu/concept/base/167',
                 'tapestry':'http://data.europeana.eu/concept/base/54',
                 'textile':'http://data.europeana.eu/concept/base/55',
                 'toy':'http://data.europeana.eu/concept/base/56',
                 'woodwork':'http://data.europeana.eu/concept/base/59',
                 }


getty_vocab = {
                'print': 'http://vocab.getty.edu/aat/300041379' ,
                'building': 'http://vocab.getty.edu/aat/300004792',
                'archaeological_site': 'http://vocab.getty.edu/aat/300266151',
                'cartoon': 'http://vocab.getty.edu/aat/300123430',
                'ceramics': 'http://vocab.getty.edu/aat/300151343',
                'clothing' : 'http://vocab.getty.edu/aat/300266639' ,
                'costume_accessories': 'http://vocab.getty.edu/aat/300209273',
                'drawing': 'http://vocab.getty.edu/aat/300033973',
                'map': 'http://vocab.getty.edu/aat/300028094',
                'furniture': 'http://vocab.getty.edu/aat/300037680',
                'textile': 'http://vocab.getty.edu/aat/300231565',
                'food': 'http://vocab.getty.edu/aat/300254496',
                'glassware': 'http://vocab.getty.edu/aat/300010898',
                'inscription': 'http://vocab.getty.edu/aat/300028702' ,
                'jewellery': 'http://vocab.getty.edu/aat/300209286' ,
                'metalwork': 'http://vocab.getty.edu/aat/300015336',
                'machinery': 'http://vocab.getty.edu/aat/300024839' ,
                'medal' : 'http://vocab.getty.edu/aat/300046025' ,
                'memorabilia': 'http://vocab.getty.edu/aat/300028884' ,
                'mineral': 'http://vocab.getty.edu/aat/300011068' ,
                'musical_instrument': 'http://vocab.getty.edu/aat/300041620' ,
                'painting': 'http://vocab.getty.edu/aat/300033618' ,
                'photograph': 'http://vocab.getty.edu/aat/300046300' ,
                'postcard': 'http://vocab.getty.edu/aat/300026816' ,
                'poster': 'http://vocab.getty.edu/aat/300027221' ,
                'sculpture': 'http://vocab.getty.edu/aat/300047090' ,
                'specimen': 'http://vocab.getty.edu/aat/300235576' ,
                'tableware': 'http://vocab.getty.edu/aat/300043196' ,
                'tool': 'http://vocab.getty.edu/aat/300024841' ,
                'tapestry': 'http://vocab.getty.edu/aat/300205002' ,
                'toy': 'http://vocab.getty.edu/aat/300211037' ,
                'weaponry': 'http://vocab.getty.edu/aat/300036926' ,
                'woodwork': 'http://vocab.getty.edu/aat/300015348' ,
                'stamp': 'http://vocab.getty.edu/aat/300037321' }

if __name__ == '__main__':

    ROOT_DIR = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]


    #select some categories from getty vocab
    getty_categories = ['archaeological_site','clothing','costume_accessories','inscription','weaponry']
    vocab_dict = {k:getty_vocab[k] for k in getty_categories}
    
    #merge ec and getty
    vocab_dict.update(ec_vocab)

    N = 1000
    
    df = pd.DataFrame()
    for category in vocab_dict.keys():
      print(category)
      skos_concept = vocab_dict[category]
      df_category = query_single_category(skos_concept, N)
      df = pd.concat((df,df_category))
      #save after each category
      df.to_csv(os.path.join(ROOT_DIR,'dataset.csv'),index=False)
