import json
import fire
from pathlib import Path
import pandas as pd


def main(**kwargs):
    vocab_path = kwargs.get('vocab_path')
    data_path = kwargs.get('data_path')
    saving_dir = kwargs.get('saving_dir')
    
    saving_dir = Path(saving_dir)

    with open(vocab_path,'r') as f:
        vocab = json.load(f)

    config = '<View><Image name="$ID" value="$URL"/><Choices name="choice" toName="$ID" choice="multiple">'
    for k in vocab.keys():
        config += f'<Choice value="{k}"/>'
    config += '</Choices></View>'

    with open(saving_dir.joinpath("config.txt"), "w") as f:
        f.write(config)

    df = pd.read_csv(data_path)
    df = df[['URI', 'ID', 'URL','category']]
    df = df.dropna()

    data_json = []
    for idx,row in df.iterrows():
        data_json.append(
            {
            'ID':row['ID'],
            'URI':row['URI'],
            'URL':row['URL'],
            'category':row['category'],
            }  
        )

    with open(saving_dir.joinpath("input_data.json"),'w') as f:
        json.dump(data_json,f)


if __name__ == '__main__':
    fire.Fire(main)
