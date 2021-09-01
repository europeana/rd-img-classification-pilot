import json
import pandas as pd
import fire
from pathlib import Path

def main(**kwargs):
    vocab_path = kwargs.get('vocab_path')
    export_path = kwargs.get('export_path')
    saving_dir = kwargs.get('saving_dir')

    saving_dir = Path(saving_dir)
    
    with open(vocab_path,'r') as f:
        vocab = json.load(f)

    with open(export_path,'r') as f:
        export = json.load(f)

    URI_list = []
    URL_list = []
    ID_list = []
    annotations_list = []
    concepts_URI_list = []

    for item in export:
        labels_list = item['annotations'][0]['result'][0]['value']['choices']

        concepts_URI = [vocab[label] for label in labels_list]
        annotations_list.append(' '.join(labels_list))
        concepts_URI_list.append(' '.join(concepts_URI))

        ID_list.append(item['data']['ID'])
        URL_list.append(item['data']['URL'])
        URI_list.append(item['data']['URI'])

    df = pd.DataFrame({
        'category':annotations_list,
        'skos_concept':concepts_URI_list,
        'ID':ID_list,
        'URL':URL_list,
        'URI':URI_list,
        })

    df.to_csv(saving_dir.joinpath('exported.csv'),index=False)

if __name__ == '__main__':
    fire.Fire(main)