# Europeana Image Classification pilot

description

## Installation

`git clone https://github.com/europeana/rd-img-classification-pilot.git`

`cd rd-img-classification-pilot`

`pip install -r requirements.txt`


## Assembling the dataset

`python src/data_harvesting.py`

The result should be (point to the csv file)

Download images. This step is required for training the model

`python src/download_images.py`



## Training the model

Once the images are downloaded

`python src/train.py`

Jupyter notebook


## Inference

Checkpoint (point to checkpoint)



Jupyter notebook



