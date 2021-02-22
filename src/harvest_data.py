import pandas as pd
import requests
import os
import json
import argparse

from ds_utils import create_dir

def parse_CHO(item):
  ID = item['id']
  URI = 'http://data.europeana.eu/item'+ID
  try:
    URL = item['edmIsShownBy'][0]
  except:
    URL = None

  return ID,URI,URL

    
def query_single_category(category,skos_concept, N):  
  """

  """

  params = {
      'reusability':'open',
      'media':True,
      'qf':f'(skos_concept:"{skos_concept}" AND TYPE:IMAGE )', 
      'query':'*', 
      'wskey':'api2demo',
  }

  CHO_list = []
  response = {'nextCursor':'*'}
  while 'nextCursor' in response:
    
    if len(CHO_list)>N:
      break

    params.update({'cursor':response['nextCursor']})
    response = requests.get('https://www.europeana.eu/api/v2/search.json', params = params).json()

    for CHO in response['items']:

      ID,URI,URL = parse_CHO(CHO)

      if URL:
        CHO_list.append({
          'category':category,
          'skos_concept':skos_concept,
          'URI':URI,
          'ID':ID,
          'URL':URL
          })

  return pd.DataFrame(CHO_list[:N])


def harvest_categories(vocab_dict,n,fname,saving_dir):
    df = pd.DataFrame()
    for category,skos_concept in vocab_dict.items():
      print(category)
      df_category = query_single_category(category,skos_concept, n)
      df = pd.concat((df,df_category))
      #save after each category
      df.to_csv(os.path.join(saving_dir,fname),index=False)

    return df

  
if __name__ == '__main__':

    """
    Script for assembling the image classification dataset 
    in csv format making use of Europeana's Search API 

    Usage:

      python src/harvest_data.py --vocab_json vocabulary.json --n 3000 --name dataset_3000

    Parameters:

      vocab_json: json file with categories as keys and concept URIs as values
                  Required

      saving_dir: directory for saving the csv file. 
                  If not specified this will be the root path of the repository

      name: tag for the table
             Default: dataset
      
      n: number of desired Cultural Heritage Objects per category
         Default: 1000
    """

    ROOT_DIR = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]

    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_json', required=True)
    parser.add_argument('--n', required=False, nargs = '?', const = 1000)
    parser.add_argument('--name', required=False)
    parser.add_argument('--saving_dir', required=False)

    args = parser.parse_args()
    
    with open(args.vocab_json,'r') as f:
      vocab_dict = json.load(f)
      
    if args.name:
      fname = f'{args.name}.csv'
    else:
      fname = 'dataset.csv'

    if not args.saving_dir:
      saving_dir = ROOT_DIR
    else:
      saving_dir = args.saving_dir
          
    harvest_categories(vocab_dict,int(args.n),fname,saving_dir)


    

