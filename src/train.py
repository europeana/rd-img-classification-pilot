from notebook_env import *

from ds_utils import *
from torch_utils import *
from imgaug import augmenters as iaa

if __name__ == '__main__':

    root_results_path = '/home/jcejudo/rd-img-classification-pilot/results'

    experiment_name = 'testing'
    learning_rate = 0.00001
    epochs = 1
    model_name = 'resnet34'
    optimizer_name = 'Adam'
    num_workers = 4
    batch_size = 64
    weighted_loss = True
    img_aug = None
    
    create_dir(root_results_path)
    results_path = os.path.join(root_results_path,experiment_name)
    create_dir(results_path)


    data_path = '/home/jcejudo/rd-img-classification-pilot/training_data/ec'
    ec_df = path2DataFrame(data_path)

    data_path = '/home/jcejudo/rd-img-classification-pilot/training_data/getty'
    getty_df = path2DataFrame(data_path)

    df = pd.concat((ec_df,getty_df))

    df = df.sample(frac=0.1)

    X = df['file_path'].values
    y = df['category'].values
    y_encoded, encoding_dict = label_encoding(y)
    weights = get_class_weights(y_encoded,encoding_dict)
    
    splits = prepare_dataset(
        X = X,
        y = y_encoded,
        img_aug = img_aug,
        num_workers = num_workers,
        batch_size = batch_size,
        splits = 10,
    )

    #crossvalidation
    for i,split in enumerate(splits):
        print(f'split {i}\n')
        split_path = os.path.join(results_path,f'split_{i}')
        
        train_data = split['trainloader']
        val_data = split['valloader']
        test_data = split['testloader']

        print('size train: {}'.format(len(train_data.dataset)))
        print('size val: {}'.format(len(val_data.dataset)))
        print('size test: {}'.format(len(test_data.dataset)))

        device, multi_gpu = check_cuda()
        net = build_model(model_name,device, multi_gpu, output_size = len(encoding_dict))
        
        #set optimizer
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        #set loss
        if weighted_loss:
            loss_function = nn.CrossEntropyLoss(reduction ='sum',weight=torch.FloatTensor(weights).to(device))           
        else:
            loss_function = nn.CrossEntropyLoss(reduction='sum')

        callback = CallBack(
                            validate = True,
                            early_stopping = True,
                            save_best = True,
                            return_best = True,
                            path = split_path)
        
        
        net, history = train(
                            epochs = epochs, 
                            trainloader = train_data, valloader = val_data,
                            loss_function = loss_function,
                            net = net,
                            model_name = model_name,
                            optimizer = optimizer,
                            device = device,
                            multi_gpu = multi_gpu,
                            callback = callback,
                            encoding_dict = encoding_dict)


        metrics_dict = validate_test(net,test_data,device,loss_function,encoding_dict)

        acc_test = metrics_dict['accuracy']
        f1_test = metrics_dict['f1']
        precision_test = metrics_dict['precision']
        recall_test = metrics_dict['recall']
        sensitivity_test = metrics_dict['sensitivity']
        specificity_test = metrics_dict['specificity']
        confusion_matrix_test = metrics_dict['confusion_matrix']
        loss_test = metrics_dict['loss']

        ROC_fig = metrics_dict['ROC_fig']

        ground_truth_list = metrics_dict['ground_truth_list']
        predictions_list = metrics_dict['predictions_list']


        save_XAI(net,test_data.dataset.X,ground_truth_list,predictions_list,split_path,device,encoding_dict)

        #plt

        print(f'acc_test: {acc_test}')
        print(f'f1_test: {f1_test}')
        print(f'precision_test: {precision_test}')
        print(f'recall_test: {recall_test}')
        print(f'sensitivity_test: {sensitivity_test}')
        print(f'specificity_test: {specificity_test}')
        print(f'loss_test: {loss_test}')
        print('confusion_matrix_test \n')
        print(confusion_matrix_test)
        
        experiment = Experiment()
        experiment.add('learning_rate',learning_rate)
        experiment.add('optimizer',optimizer)
        experiment.add('loss_function',loss_function)
        experiment.add('train_data',train_data)
        experiment.add('val_data',val_data)
        experiment.add('test_data',test_data)
        experiment.add('epochs',epochs)
        experiment.add('encoding_dict',encoding_dict)
        experiment.add('weights',weights)
        experiment.add('net',net)
        experiment.add('model_name',model_name)
        experiment.add('optimizer',optimizer)
        experiment.add('loss_function',loss_function)
        experiment.add('batch_size',batch_size)
        experiment.add('num_workers',num_workers)
        experiment.add('acc_test',acc_test)
        experiment.add('loss_test',loss_test)
        experiment.add('f1_test',f1_test)
        experiment.add('precision_test',precision_test)
        experiment.add('recall_test',recall_test)
        experiment.add('sensitivity_test',sensitivity_test)
        experiment.add('specificity_test',specificity_test)
        experiment.add('confusion_matrix_test',confusion_matrix_test)
        for k,v in history.items():
            experiment.add(k,v)
        #experiment.show()
        experiment.save(split_path)
        












