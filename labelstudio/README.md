# Annotating with Labelstudio


## Installation

`conda create --name label-studio python=3.8`
`conda activate label-studio`
`pip install label-studio`

## Launch Labelstudio

`label-studio --host 127.0.0.1 --port 5050`

`http://rnd-3.eanadev.org:5050/`


## Configuration and data preparation

`python prepare.py --vocab_path ../vocabularies/vocabulary.json --saving_dir . --data_path ../data/unlabeled_data.csv`

Set up your labeling task, label the data


##  Export the results.

Once the annotations are exported you can use the following command to process the export

`python parse_export.py --vocab_path ../vocabularies/vocabulary.json --saving_dir . --export_path /home/jcejudo/rd-img-classification-pilot/labelstudio/project-11-at-2021-06-08-09-39-0c04a66d.json`

Add to previously annotated data

`python ../src/data_operations.py --operation add --data1 /home/jcejudo/rd-img-classification-pilot/labelstudio/exported.csv --data2 /home/jcejudo/rd-img-classification-pilot/data/multilabel/clean_training_data.csv --saving_path /home/jcejudo/rd-img-classification-pilot/labelstudio/training_data.csv`


Remove labeled from unlabeled

`python ../src/data_operations.py --operation substract --data1 /home/jcejudo/rd-img-classification-pilot/labelstudio/training_data.csv --data2 /home/jcejudo/projects/image_classification/data/multilabel/eval_multilabel_with_URL.csv --saving_path /home/jcejudo/rd-img-classification-pilot/labelstudio/training_data.csv`


Download data

`nohup python ../src/download_images.py --csv_path /home/jcejudo/rd-img-classification-pilot/labelstudio/training_data.csv --saving_dir /home/jcejudo/projects/image_classification/data/multilabel/labelstudio_images --mode multi_label &> download_labelstudio.out &`

## Train

`nohup python ../src/train_multilabel.py --data_dir /home/jcejudo/projects/image_classification/data/multilabel/labelstudio_images --annotations /home/jcejudo/rd-img-classification-pilot/labelstudio/training_data.csv --saving_dir /home/jcejudo/projects/image_classification/results/multilabel/labelstudio_crossvalidation --input_size 128 --batch_size 32 --learning_rate 1e-5 --resnet_size 18 --max_epochs 100 --num_workers 4 --crossvalidation True --sample 1.0 &> train_labelstudio.out &`

# Evaluation

`python ../src/multilabel/evaluate.py --data_dir /home/jcejudo/projects/image_classification/data/multilabel/images_evaluation --annotations /home/jcejudo/projects/image_classification/data/multilabel/eval_multilabel_with_URL.csv --results_path /home/jcejudo/projects/image_classification/results/multilabel/labelstudio_crossvalidation/ --saving_dir /home/jcejudo/projects/image_classification/results/multilabel/evaluation_labelstudio/ --threshold 0.8 --batch_size 32`