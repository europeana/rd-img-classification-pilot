

Train self supervised model

`nohup python src/self_supervised/train.py --ss_model byol --data_path /home/jcejudo/projects/image_classification/data/single_label/images_training --saving_dir /home/jcejudo/projects/image_classification/results/self_supervised --batch_size 64 --input_size 224 --num_ftrs 512 --sample 1.0 --max_epochs 100 --model_size 18 &> train_self_supervised.out &`


Create embeddings

`python src/self_supervised/create_embeddings.py --pretrained_dir /home/jcejudo/projects/image_classification/results/self_supervised --saving_path /home/jcejudo/projects/image_classification/results/self_supervised/embeddings.csv --data_dir /home/jcejudo/projects/image_classification/data/single_label/images_training --sample 1.0`


Set up recommender API

`python src/self_supervised/flask_api.py --embeddings_path /home/jcejudo/projects/image_classification/results/self_supervised/embeddings.csv --data_path /home/jcejudo/projects/image_classification/data/single_label/images_training`


Fine tune multilabel



`nohup python src/self_supervised/finetune_multilabel.py --data_dir /home/jcejudo/projects/image_classification/data/multilabel/images_training --annotations /home/jcejudo/projects/image_classification/data/multilabel/training_dataset.csv --saving_dir /home/jcejudo/projects/image_classification/results/self_supervised_finetuning --pretrained_dir /home/jcejudo/projects/image_classification/results/self_supervised --hf_prob 0.5 --gb_prob 0.5 --cj_prob 0.5 --input_size 224 --batch_size 64 --learning_rate 1e-3 --max_epochs 100 --num_workers 8 &> finetune_self_supervised.out &`


to do: finetuning with Labelstudio

to do: add crossvalidation
add sampling
add baselines
add n_splits

`nohup python src/self_supervised/finetune_multilabel.py --data_dir /home/jcejudo/projects/image_classification/data/multilabel/labelstudio_images --annotations /home/jcejudo/rd-img-classification-pilot/labelstudio/training_data.csv --saving_dir /home/jcejudo/projects/image_classification/results/self_supervised_finetuning --pretrained_dir /home/jcejudo/projects/image_classification/results/self_supervised --hf_prob 0.5 --gb_prob 0.0 --cj_prob 0.0 --input_size 224 --batch_size 32 --learning_rate 1e-5 --max_epochs 100 --num_workers 4 &> finetune_labelstudio_self_supervised.out &`


Evaluate final model

`python src/self_supervised/evaluate.py --data_dir /home/jcejudo/projects/image_classification/data/multilabel/images_evaluation --annotations /home/jcejudo/projects/image_classification/data/multilabel/eval_multilabel_with_URL.csv --results_path /home/jcejudo/projects/image_classification/results/self_supervised_finetuning --saving_dir /home/jcejudo/projects/image_classification/results/self_supervised_evaluation/ --threshold 0.8 --batch_size 32`
