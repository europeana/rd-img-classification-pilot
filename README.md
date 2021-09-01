# Europeana Image Classification pilot

Our mission at Europeana is to make Cultural Heritage (CH) data findable, accessible, reusable and interoperable. The metadata provided next to Cultural Heritage Objects (CHOs) by our partners allow us to build functionalities such as search and browsing. 

We have a history of performing [automatic enrichments](https://pro.europeana.eu/page/europeana-semantic-enrichment#automatic-semantic-enrichment) to augment metadata. However, so far we have focused on creating such enrichments based on the metadata only.

We would like to explore automatic enrichments based on digital content (images), and we decided to start a pilot on image classification. We were motivated by the recent advances in computer vision and the easy access to specialized hardware. 

This repository builds a training dataset for a specific image classification goal, using the [Europeana Search API](https://pro.europeana.eu/page/search), and allows to train a model using the deep learning [pytorch](https://pytorch.org/) framework.

## Setup

Clone this repository:

`git clone https://github.com/europeana/rd-img-classification-pilot.git`

change to the repo directory:

`cd rd-img-classification-pilot`

Install dependencies:

`pip install -r requirements.txt`


## Assembling the dataset

Vocabularies in CH aim to standarize and relate concepts semantically. This makes metadata referencing standard vocabularies more interoperable. From set of concepts that Europeana has gathered from linked data sources, we have extracted a list of high-level concepts of specific interest to identify types of object in Europeana (in the file `uris_vocabularies.csv`). These concepts are identified by URIs and point to different vocabularies they have equivalence links to.

For our experiments we will use a selection of terms from that list, which are both in the vocabulary used in Europeana's metadata-based enrichment (the [Europeana Entity Collection](https://pro.europeana.eu/page/entity#entity-collection)) and the [Getty AAT vocabulary](https://www.getty.edu/research/tools/vocabularies/aat/), contained in the file [`vocabulary.json`](https://github.com/europeana/rd-img-classification-pilot/blob/main/vocabulary.json)

Once the vocabulary is defined, we can query the Europeana Search API for CHOs in the different categories and build a table with the information necessary to assemble an image classification dataset. We can do that from the command line by specifying the vocabulary file to consider, the maximum number of CHOs retrieved per category and an optional name for the resulting file:

`python src/harvest_data.py --vocab_json vocabularies/vocabulary.json --n 3000 --saving_path /home/jcejudo/projects/image_classification/data/single_label/harvested_training.csv --reusability [open,permission] --mode single_label`

The resulting table should have the columns `category`, `skos_concept`, `URI`, `URL`, `ID`. This allows to uniquely identify the CHOs and the images, and potentially use Europeana's [Record API](https://pro.europeana.eu/page/record) for retrieving further information about the objects. We have included the dataset `dataset.csv` as an example of querying 3000 CHOs per category.

Remove images present in evaluation data

`python src/data_operations.py --operation substract --data1 /home/jcejudo/projects/image_classification/data/single_label/harvested_training.csv --data2 /home/jcejudo/projects/image_classification/data/single_label/eval_dataset.csv --saving_path /home/jcejudo/projects/image_classification/data/single_label/training_dataset.csv`

Once we have the URL for the images we will save them in disk under directories corresponding to the different categories. This step is required for training the model. We can do that by specifying the path to the dataset in csv format and the directory for the images.

Download training data

`nohup python src/download_images.py --csv_path /home/jcejudo/projects/image_classification/data/single_label/training_data.csv --saving_dir /home/jcejudo/projects/image_classification/data/single_label/images_training --mode single_label`

Download evaluation data

`nohup python src/download_images.py --csv_path /home/jcejudo/projects/image_classification/data/single_label/eval_dataset.csv --saving_dir /home/jcejudo/projects/image_classification/data/single_label/images_evaluation --mode single_label &> download_single_label_evaluation.out &`

dataset statistics -> Jupyter Notebook

## Training the model

We are ready to proceed with training our model! To make sure that we evaluate the performance of the model fairly, we will consider several train, validation and test splits in a process called [cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)). The result will be a set of directories (one per split) containing the training history, model checkpoint and interpretable heatmaps for the test images. We can use the script `train_single_label.py` by specifying the directory to the dataset and some of the training hyperparameters:

`nohup python src/train_single_label.py --data_dir /home/jcejudo/projects/image_classification/data/single_label/images_training --epochs 100 --patience 10 --input_size 128 --batch_size 32 --learning_rate 1e-4 --crossvalidation True --saving_dir /home/jcejudo/projects/image_classification/results/single_label/crossvalidation &> crossvalidation_single_label.out &`

## Evaluation

`python src/evaluate_single_label.py --data_path /home/jcejudo/projects/image_classification/data/single_label/images_evaluation --results_path /home/jcejudo/projects/image_classification/results/single_label/crossvalidation/split_0 --saving_dir /home/jcejudo/projects/image_classification/results/single_label/evaluation/`

## Inference

To do: read configuration, rerun

Once the model is trained, it can be used for predicting on unseen images.

[colab notebook inference evaluation set](https://colab.research.google.com/drive/1Ma8O1hWhUNRlrJBDZO4Rhwzg4MlAVAFa?usp=sharing#offline=true&sandboxMode=true)

We can also apply the model to images from CHOs retrieved using the Search API

[colab notebook inference search API](https://colab.research.google.com/drive/1B3S_DYQ6UtCYGaScygcf_BZa0Ifml4SR?usp=sharing#offline=true&sandboxMode=true)








