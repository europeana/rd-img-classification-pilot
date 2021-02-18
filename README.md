# Europeana Image Classification pilot

description

## Installation

Clone this repository:

`git clone https://github.com/europeana/rd-img-classification-pilot.git`

`cd rd-img-classification-pilot`

Install dependencies:

`pip install -r requirements.txt`


## Assembling the dataset

Vocabulary in a json file:

`vocabulary.json` (Include link)


Harvesting data: Using EF Search API for retrieving CHOs from each category of the vocabulary

`python src/data_harvesting.py --vocab_json vocabulary.json --n 1000 --name dataset_1000`

The result should be dataset_1000.csv (point to the csv file)

(Include sample of the table and comment on URIs, etc)

Motivate this format

Download images. This step is required for training the model

`python src/download_images.py --csv_path dataset_1000.csv --saving_dir training_data`

Output: images organized in subdirectories


## Training the model

Once the images are downloaded

Crossvalidation

`python src/train.py --data_dir training_data --epochs 100`

Output: directories for each splits including a model checkpoint, model metadata and XAI images for the test set


Jupyter notebook (include link)


## Inference

Checkpoint (point to checkpoint)

Jupyter notebook



