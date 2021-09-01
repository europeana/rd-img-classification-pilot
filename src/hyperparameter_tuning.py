from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from train_multilabel import *

def train_cifar(config, **kwargs):

    X_train = kwargs.get('X_train')
    X_val = kwargs.get('X_val')

    y_train = kwargs.get('y_train')
    y_val = kwargs.get('y_val')

    checkpoint_dir = kwargs.get('checkpoint_dir')
    class_index_dict = kwargs.get('class_index_dict')
    resnet_size = kwargs.get('resnet_size')
    criterion = kwargs.get('criterion')
    train_transform = kwargs.get('train_transform')
    test_transform = kwargs.get('test_transform')


    

    #tune.utils.wait_for_gpu()


    num_workers = 8


 

    trainset = MultilabelDataset(X_train,y_train,transform = train_transform)
    trainloader = DataLoader(
        trainset,
        batch_size=int(config["batch_size"]),
        shuffle=True, 
        num_workers=num_workers,
        drop_last=True
        )

    valset = MultilabelDataset(X_val,y_val,transform = test_transform)
    valloader = DataLoader(
        valset, 
        batch_size=int(config["batch_size"]),
        shuffle=True, 
        num_workers=num_workers,
        drop_last=True
        )



    # print('train:',X_train.shape[0])
    # print('val:',X_val.shape[0])



    model = MultilabelResNet(resnet_size,len(class_index_dict))

    #device = "cuda:0"

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    print(device)

    model.to(device)


    optimizer = optim.Adam(model.parameters(), lr=config["lr"])


    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # trainset, testset = load_data(data_dir)

    # test_abs = int(len(trainset) * 0.8)
    # train_subset, val_subset = random_split(
    #     trainset, [test_abs, len(trainset) - test_abs])

    # trainloader = torch.utils.data.DataLoader(
    #     train_subset,
    #     batch_size=int(config["batch_size"]),
    #     shuffle=True,
    #     num_workers=8)
    # valloader = torch.utils.data.DataLoader(
    #     val_subset,
    #     batch_size=int(config["batch_size"]),
    #     shuffle=True,
    #     num_workers=8)

    for epoch in range(30):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels,_ = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            labels = labels.type_as(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        val_metrics = evaluate(
            model = model,
            dataloader = valloader,
            loss_function = criterion,
            device = device
        )

        val_loss = val_metrics['loss']
        accuracy = val_metrics['acc']

        # # Validation loss
        # val_loss = 0.0
        # val_steps = 0
        # total = 0
        # correct = 0
        # for i, data in enumerate(valloader, 0):
        #     with torch.no_grad():
        #         inputs, labels,_ = data
        #         inputs, labels = inputs.to(device), labels.to(device)

        #         outputs = model(inputs)
        #         labels = labels.type_as(outputs)
        #         _, predicted = torch.max(outputs.data, 1)
        #         total += labels.size(0)
        #         correct += (predicted == labels).sum().item()

        #         loss = criterion(outputs, labels)
        #         val_loss += loss.cpu().numpy()
        #         val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=val_loss, accuracy=accuracy)

    print("Finished Training")


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=1):
    #data_dir = os.path.abspath("./data")
    data_dir = '/home/jcejudo/projects/image_classification/data/multilabel/labelstudio_images'
    df_path = '/home/jcejudo/rd-img-classification-pilot/labelstudio/training_data.csv'

    input_size = 224
    hf_prob = 0.5
    vf_prob = 0.0
    sample = 1.0
    n_splits = 10
    num_workers = 8
    resnet_size = 18

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    
    #make train, test splits

    data_dir = Path(data_dir)
    
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(hf_prob),
        transforms.RandomVerticalFlip(vf_prob),
        transforms.ToTensor(),
        # this normalization is required https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        # this normalization is required https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    df = pd.read_csv(df_path)
    df = df.groupby('category').apply(lambda x: x.sample(frac=sample))

    #filter images in df contained in data_path
    imgs_list = list(data_dir.iterdir())
    df['filepath'] = df['ID'].apply(lambda x:data_dir.joinpath(id_to_filename(x)+'.jpg'))
    df = df.loc[df['filepath'].apply(lambda x: Path(x) in imgs_list)]

    df['n_labels'] = df['category'].apply(lambda x: len(x.split()))
    df = df.sort_values(by='n_labels',ascending=False)
    df = df.drop_duplicates(keep='first',subset='ID')
    df = drop_categories(df,['specimen','clothing'])
    #print(df.shape)

    mlb = sklearn.preprocessing.MultiLabelBinarizer()

    imgs = np.array([str(path) for path in df['filepath'].values])

    labels = [item.split() for item in df['category'].values]
    labels = mlb.fit_transform(labels)

    class_index_dict = {i:c for i,c in enumerate(mlb.classes_)}
    # print(class_index_dict)

    skf = KFold(n_splits=n_splits)
    sk_splits = skf.split(imgs, labels)
    train_val_index, test_index = next(sk_splits)


    X_train_val, X_test = imgs[train_val_index], imgs[test_index]
    y_train_val, y_test = labels[train_val_index], labels[test_index]
    
    val_sk_splits = skf.split(X_train_val, y_train_val)
    train_index,val_index = next(val_sk_splits)
    
    X_train, X_val = X_train_val[train_index], X_train_val[val_index]
    y_train, y_val = y_train_val[train_index], y_train_val[val_index]


    weights = calculate_weights(df,class_index_dict).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight = weights)


    #load_data(data_dir)
    config = {
        "lr": tune.loguniform(1e-6, 1e-2),
        "batch_size": tune.choice([32, 64])
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])

    result = tune.run(
        partial(train_cifar, X_train=X_train, y_train=y_train,X_val=X_val,y_val=y_val,class_index_dict = class_index_dict, resnet_size = resnet_size, criterion = criterion, train_transform = train_transform, test_transform = test_transform),
        resources_per_trial={"gpu": 1},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    

    best_trained_model = MultilabelResNet(resnet_size,len(class_index_dict))

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    # evaluate



    testset = MultilabelDataset(X_test,y_test,transform = test_transform)
    testloader = DataLoader(
        testset, 
        batch_size=16,
        shuffle=True, 
        num_workers=num_workers,
        drop_last=True
        )

    val_metrics = evaluate(
        model = best_trained_model,
        dataloader = testloader,
        loss_function = criterion,
        device = device
    )

    val_loss = val_metrics['loss']
    accuracy = val_metrics['acc']


    #test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(accuracy))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=1)