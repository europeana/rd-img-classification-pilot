import pandas as pd
import os
import argparse
import fire

def remove_test_from_train(**kwargs):

    training_df = kwargs.get('training_df')
    evaluation_df = kwargs.get('evaluation_df')
    saving_path = kwargs.get('saving_path')


    evaluation_df = evaluation_df[['URI','ID','URL','category']]
    evaluation_df = evaluation_df.dropna()

    training_df = training_df[~training_df.ID.isin(evaluation_df.ID)]

    training_df.to_csv(saving_path,index=False)

def main(**kwargs):
    data1 = kwargs.get('data1')
    data2 = kwargs.get('data2')
    saving_path = kwargs.get('saving_path')
    operation = kwargs.get('operation')

    df1 = pd.read_csv(data1)
    df2 = pd.read_csv(data2)

    n_data1 = df1.shape[0]
    n_data2 = df2.shape[0]

    if operation == 'substract':
        if n_data1 > n_data2:
            remove_test_from_train(training_df = df1, evaluation_df = df2, saving_path = saving_path)
        else:
            remove_test_from_train(training_df = df2, evaluation_df = df1, saving_path = saving_path)

    elif operation == 'add':
        df = pd.concat((df1,df2))
        df.to_csv(saving_path,index=False)





if __name__ == '__main__':

    fire.Fire(main)



