# Multilabel classification


Harvest data

`python src/harvest_data.py --vocab_json vocabularies/vocabulary.json --n 3000 --saving_path /home/jcejudo/projects/image_classification/data/multilabel/harvested_training.csv --reusability [open,permission] --mode multilabel &> harvest_multilabel.out &`

Remove test images from training data

`python src/data_operations.py --operation substract --data1 /home/jcejudo/projects/image_classification/data/multilabel/harvested_training.csv --data2 /home/jcejudo/projects/image_classification/data/multilabel/eval_multilabel_with_URL.csv --saving_path /home/jcejudo/projects/image_classification/data/multilabel/training_dataset.csv`

Download training images

`nohup python src/download_images.py --csv_path /home/jcejudo/projects/image_classification/data/multilabel/training_dataset.csv --saving_dir /home/jcejudo/projects/image_classification/data/multilabel/images_training --mode multi_label &> download_training_multilabel.out &`

Download evaluation images

`nohup python src/download_images.py --csv_path /home/jcejudo/projects/image_classification/data/multilabel/eval_multilabel_with_URL.csv --saving_dir /home/jcejudo/projects/image_classification/data/multilabel/images_evaluation --mode multi_label &> download_evaluation_multilabel.out &`

dataset statistics -> Jupyter Notebook

Train 

`nohup python src/train_multilabel.py --data_dir /home/jcejudo/projects/image_classification/data/multilabel/images_training --annotations /home/jcejudo/projects/image_classification/data/multilabel/training_dataset.csv --saving_dir /home/jcejudo/projects/image_classification/results/multilabel/crossvalidation --input_size 128 --batch_size 32 --learning_rate 1e-5 --resnet_size 18 --max_epochs 100 --num_workers 4 --crossvalidation True --sample 1.0 &> train_multilabel.out &`

Evaluation on test set

`python src/multilabel/evaluate.py --data_dir /home/jcejudo/projects/image_classification/data/multilabel/images_evaluation --annotations /home/jcejudo/projects/image_classification/data/multilabel/eval_multilabel_with_URL.csv --results_path /home/jcejudo/projects/image_classification/results/multilabel/crossvalidation/ --saving_dir /home/jcejudo/projects/image_classification/results/multilabel/evaluation/ --threshold 0.8 --batch_size 32`

to do: Inference

