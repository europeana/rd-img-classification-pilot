# Europeana Image Classification pilot

Our mission at Europeana Foundation (EF) is to make Cultural Heritage data findable, accessible, reusable and interoperable. The metadata included in our Cultural Heritage Objects allow us to build functionalities such as search, browsing and recommendation systems. We have a history of using automatic enrichments for generating metadata. However, so far we have focused on creating metadata from textual data or metadata. 

We would like to explore automatic enrichments based on content, and we decided to start a pilot on image classification. We were motivated by the recent advances in computer vision and the easy access to specialized hardware. 

This repository builds a training dataset using the EF Search API, and allows to train a model using the deep learning pytorch framework.

## Installation

Clone this repository:

`git clone https://github.com/europeana/rd-img-classification-pilot.git`

change to the repo directory:

`cd rd-img-classification-pilot`

Install dependencies:

`pip install -r requirements.txt`


## Assembling the dataset

Vocabularies in Cultural Heritage (CH) aim to standarize and relate concepts semantically. This makes metadata referencing standard vocabularies interoperable. We have gathered a list of common concepts in CH together with URIs pointing to different vocabularies in the file `uris_vocabularies.csv`.

For our experiments we will use a selection of terms from EF Entity Collection and the AAT Getty vocabulary, contained in the file [`vocabulary.json`](https://github.com/europeana/rd-img-classification-pilot/blob/main/vocabulary.json)

Once the vocabulary is defined we can query the EF Search API for the different categories and build a table with the information necessary to assemble an image classification dataset. We can do that from the command line by specifying the vocabulary file to consider, the maximum number of CHOs retrieved per category and an optional name for the resulting file:

`python src/harvest_data.py --vocab_json vocabulary.json --n 3000 --name dataset_3000`

The resulting table should have the columns `category`, `skos_concept`, `URI`, `URL`, `ID`. This allows to uniquely identify the Cultural Heritage Objects and the images, and potentially use EF Record API for retrieving further information about the objects. We have included the dataset `dataset.csv` as an example of querying 3000 CHOs per category.

Once we have the URL for the images we will save them in disk under directories corresponding to the different categories. This step is required for training the model. We can do that by specifying the path to the dataset in csv format and the directory for the images.

`python src/download_images.py --csv_path dataset_3000.csv --saving_dir training_data`


## Training the model

We are ready to proceed with training our model! To make sure that we evaluate the performance of the model fairly, we will consider several train, validation and test splits in a process called crossvalidation. The result will be a set of directories (one per split) containing the training history, model checkpoint and interpretable heatmaps for the test images. We can use the script `train_crossvalidation.py` by specifying the directory to the dataset and some of the training hyperparameters:

`python src/train.py --data_dir training_data --epochs 100 --patience 10 --experiment_name model_training --img_aug 0.5`

The code for training a single split is in the notebook ['jupyter notebook training'](https://github.com/europeana/rd-img-classification-pilot/blob/main/notebooks/train.ipynb)


## Inference

`checkpoint.pth`

['jupyter notebook inference'](https://github.com/europeana/rd-img-classification-pilot/blob/main/notebooks/inference.ipynb)

#to do: link to colab

#to do: script for inference from a folder with images. Include confidence score




