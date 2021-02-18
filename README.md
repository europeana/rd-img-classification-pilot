# Europeana Image Classification pilot

General goal

Enrichments

Vocabulary


pytorch

## Installation

Clone this repository:

`git clone https://github.com/europeana/rd-img-classification-pilot.git`

`cd rd-img-classification-pilot`

Install dependencies:

`pip install -r requirements.txt`


## Assembling the dataset

We need a json file containing the concepts and URIs for the items of the vocabulary. Our vocabulary can be found in the file [`vocabulary.json`](https://github.com/europeana/rd-img-classification-pilot/blob/main/vocabulary.json)

Now we can query the EF Search API for the different categories and build a table with information about the resulting CHOs.

We can do that from the command line by specifying the vocabulary file to consider, the maximum number of CHOs retrieved per category and an optional name for the resulting file:

`python src/harvest_data.py --vocab_json vocabulary.json --n 1000 --name dataset_1000`

The resulting table should have the columns category, skos_concept,URI,URL,ID

This allows to uniquely identify the Cultural Heritage Objects and the images, and potentially use EF Record API for retrieving further information about the objects. 

(Include sample of the table and comment on URIs, etc)

Once we have the URL for the images we will save them in disk under directories corresponding to the different categories. This step is required for training the model. 

`python src/download_images.py --csv_path dataset_1000.csv --saving_dir training_data`


## Training the model


Crossvalidation

Divide the dataset into train, validation and test splits. 


`python src/train.py --data_dir training_data --epochs 100`

Output: directories for each splits including a model checkpoint, model metadata and XAI images for the test set

['jupyter notebook training'](https://github.com/europeana/rd-img-classification-pilot/blob/main/notebooks/train.ipynb)




## Inference

`checkpoint.pth`

['jupyter notebook inference'](https://github.com/europeana/rd-img-classification-pilot/blob/main/notebooks/inference.ipynb)

#to do: link to colab




