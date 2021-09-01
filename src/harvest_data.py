import pandas as pd
import requests
import os
import json
from tqdm import tqdm
from random import choice
from itertools import combinations

import fire
from pathlib import Path

from ds_utils import create_dir

def filename_to_id(filename):
  filename = os.path.split(filename)[1]
  return filename.replace('[ph]','/').replace('.jpg','')

def parse_CHO(item):
  ID = item['id']
  URI = 'http://data.europeana.eu/item'+ID
  try:
    URL = item['edmIsShownBy'][0]
  except:
    URL = None

  return ID,URI,URL

def combine_categories(vocab,min_n_categories,max_n_categories):
  categories_list = list(vocab.keys())
  combinations_list = []
  for n in range(min_n_categories,max_n_categories+1):
    combinations_list += list(combinations(categories_list, n))
  return combinations_list

def search_combination(concepts_list,vocab,reusability):
  skos_concepts_list = [vocab[concept] for concept in concepts_list]
  query = '"'+'"AND"'.join(skos_concepts_list)+'"'
  params = { 
      'reusability':reusability,
      'media':True,
      'qf':f'TYPE:IMAGE',
      'query':query,
      'wskey':'api2demo',
      'sort':'random,europeana_id',
      }

  response = requests.get('https://www.europeana.eu/api/v2/search.json', params = params).json()
  return response

def search_number_combinations(combinations_list,vocab,reusability):

  results = []
  for comb in tqdm(combinations_list):
    response = search_combination(comb,vocab,reusability)
    n_res = response['totalResults']
    if n_res>0:
      results.append({'labels':' '.join(comb),'results':n_res})

  return pd.DataFrame(results)


def search_CHOs(concepts_list,vocab,reusability,N):

  skos_concepts_list = [vocab[concept] for concept in concepts_list]
  query = '"'+'"AND"'.join(skos_concepts_list)+'"'
  params = { 
      'reusability':reusability,
      'media':True,
      'qf':f'TYPE:IMAGE',
      'query':query,
      'wskey':'api2demo',
      'sort':'random,europeana_id',
      }

  CHO_list = []
  response = {'nextCursor':'*'}
  while 'nextCursor' in response:
    params.update({'cursor':response['nextCursor']})

    response = requests.get('https://www.europeana.eu/api/v2/search.json', params = params).json()      
    CHO_list += response['items']
    if len(CHO_list)>N:
      break

  return CHO_list[:N]

class EuropeanaAPI:
  def __init__(self,wskey):
    self.wskey = wskey

  def record(self,id):
    params = {'wskey':self.wskey}
    response = requests.get(f'https://api.europeana.eu/record/v2/{id}.json',params=params).json()  
    try:
      return response['object']['aggregations'][0]['edmIsShownBy']
    except:
      return None

  def search(self,**kwargs):

    skos_concept = kwargs.get('skos_concept')
    query = kwargs.get('query','*')
    reusability = kwargs.get('reusability','open')
    n = kwargs.get('n',20)
    
    params = {
        'reusability':reusability,
        'media':True,
        'qf':f'(skos_concept:"{skos_concept}" AND TYPE:IMAGE )' if skos_concept else 'TYPE:IMAGE', 
        'query':"*" if skos_concept else query, 
        'wskey':self.wskey,
        'sort':'random,europeana_id',
    }

    CHO_list = []
    
    response = {'nextCursor':'*'}
    while 'nextCursor' in response:
      if len(CHO_list)>n:
        break
      params.update({'cursor':response['nextCursor']})
      response = requests.get('https://www.europeana.eu/api/v2/search.json', params = params).json()      
      CHO_list += response['items']

    return CHO_list[:n]

    
def query_single_category(**kwargs):  
  """

  """
  category = kwargs.get('category')
  skos_concept = kwargs.get('skos_concept')
  n = kwargs.get('n')
  reusability = kwargs.get('reusability','open')

  eu = EuropeanaAPI('api2demo')
  CHO_retrieved = eu.search(
     skos_concept = skos_concept,
     reusability = reusability,
     n = n,
  )

  CHO_list = []
  for CHO in CHO_retrieved:
    ID,URI,URL = parse_CHO(CHO)
    if URL:
      CHO_list.append({
        'category':category,
        'skos_concept':skos_concept,
        'URI':URI,
        'ID':ID,
        'URL':URL
        })

  return pd.DataFrame(CHO_list)

def harvest_single_label(**kwargs):
  reusability_list = kwargs.get('reusability_list')
  vocab_dict = kwargs.get('vocab_dict')
  saving_path = kwargs.get('saving_path')
  n = kwargs.get('n')

  df = pd.DataFrame()
  for category,skos_concept in vocab_dict.items():
    print(category)
    df_category = pd.DataFrame()
    for reusability in reusability_list:
      df_reusability = query_single_category(
        category = category,
        skos_concept = skos_concept,
        n = n,
        reusability = reusability,
        )
      df_category = pd.concat((df_category,df_reusability))
    df = pd.concat((df,df_category))
    #save after each category
    df.to_csv(saving_path,index=False)

   

def harvest_multilabel(**kwargs):
  reusability_list = kwargs.get('reusability_list')
  vocab_dict = kwargs.get('vocab_dict')
  saving_path = kwargs.get('saving_path')
  n = kwargs.get('n')

  min_n_categories = 2
  max_n_categories = 3
  combinations_list = combine_categories(vocab_dict,min_n_categories,max_n_categories)

  CHO_list = []
  for reusability in reusability_list:
    print(reusability)
    for combination in tqdm(combinations_list):
      concept_list = list(combination)
      retrieved_CHO_list = search_CHOs(concept_list,vocab_dict,reusability,n)
      skos_concepts_list = [vocab_dict[concept] for concept in concept_list]

      for CHO in retrieved_CHO_list:
        ID,URI,URL = parse_CHO(CHO)
        if URL:
          CHO_list.append({
            'category':' '.join(concept_list),
            'skos_concept':' '.join(skos_concepts_list),
            'URI':URI,
            'ID':ID,
            'URL':URL
            })

  print(len(CHO_list))
  df = pd.DataFrame(CHO_list)
  df.to_csv(saving_path,index=False)
   


def main(**kwargs):

  vocab_json = kwargs.get('vocab_json',None)
  n = kwargs.get('n',3000)
  name = kwargs.get('name',None)
  saving_path = kwargs.get('saving_path',None)
  reusability_list = kwargs.get('reusability',['open'])
  mode = kwargs.get('mode','single_label')

  if not isinstance(reusability_list,list):
    reusability_list = [reusability_list]

  if not vocab_json:
    raise Exception('vocab_json not provided')
  if not saving_path:
    raise Exception('saving_path not provided')

  with open(vocab_json,'r') as f:
    vocab_dict = json.load(f)

  if mode == 'single_label':
    harvest_single_label(
      vocab_dict = vocab_dict,
      reusability_list = reusability_list,
      saving_path = saving_path,
      n = n
    )
  elif mode == 'multilabel':
    harvest_multilabel(
      vocab_dict = vocab_dict,
      reusability_list = reusability_list,
      saving_path = saving_path,
      n = n
    )



  
  # # harvest single category
  # df = pd.DataFrame()
  # for category,skos_concept in vocab_dict.items():
  #   print(category)
  #   df_category = pd.DataFrame()
  #   for reusability in reusability_list:
  #     df_reusability = query_single_category(
  #       category = category,
  #       skos_concept = skos_concept,
  #       n = n,
  #       reusability = reusability,
  #       )
  #     df_category = pd.concat((df_category,df_reusability))
  #   df = pd.concat((df,df_category))
  #   #save after each category
  #   df.to_csv(saving_path,index=False)

  
if __name__ == '__main__':
    fire.Fire(main)

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

      reusability: level of copyright of the CHOs. Available: open, permission, restricted
         Default: open
    """

    

          

    

