# Europeana Image Classification pilot

General goal

Enrichments

Vocabulary


pytorch

This repo:

Assembles a dataset
Train a model

## Installation

Clone this repository:

`git clone https://github.com/europeana/rd-img-classification-pilot.git`

change to the repo directory:

`cd rd-img-classification-pilot`

Install dependencies:

`pip install -r requirements.txt`


## Assembling the dataset

We need a json file containing the concepts and URIs for the items of the vocabulary. For our experiments we used the vocabulary [`vocabulary.json`](https://github.com/europeana/rd-img-classification-pilot/blob/main/vocabulary.json)



to do: include conversion table between the different vocabularies



Now we can query the EF Search API for the different categories and build a table with information about the resulting Cultural Heritage Objects.

We can do that from the command line by specifying the vocabulary file to consider, the maximum number of CHOs retrieved per category and an optional name for the resulting file:

`python src/harvest_data.py --vocab_json vocabulary.json --n 3000 --name dataset_3000`

The resulting table should have the columns category, skos_concept,URI,URL,ID

todo: point to the dataset

This allows to uniquely identify the Cultural Heritage Objects and the images, and potentially use EF Record API for retrieving further information about the objects. 


Once we have the URL for the images we will save them in disk under directories corresponding to the different categories. This step is required for training the model. 

`python src/download_images.py --csv_path dataset_3000.csv --saving_dir training_data`


## Training the model


Crossvalidation

Divide the dataset into train, validation and test splits. 


`python src/train.py --data_dir training_data --epochs 100 --patience 10 --experiment_name model_training --img_aug 0.5`

Output: directories for each splits including a model checkpoint, model metadata and XAI images for the test set

['jupyter notebook training'](https://github.com/europeana/rd-img-classification-pilot/blob/main/notebooks/train.ipynb)




## Inference

`checkpoint.pth`

['jupyter notebook inference'](https://github.com/europeana/rd-img-classification-pilot/blob/main/notebooks/inference.ipynb)

#to do: link to colab

#to do: script for inference from a folder with images. Include confidence score




