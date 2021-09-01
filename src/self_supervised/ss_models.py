import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import lightly
import pytorch_lightning as pl

import lightly.utils.benchmarking as benchmarking

class MoCoModel(pl.LightningModule):
    def __init__(
        self,
        resnet,
        num_ftrs = 512,
        memory_bank_size = 4096,
        learning_rate = 0.1,
        max_epochs = 400):
        super().__init__()

        last_conv_channels = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.Conv2d(last_conv_channels, num_ftrs, 1),
        )

        self.resnet_moco = lightly.models.MoCo(self.backbone, num_ftrs=num_ftrs, m=0.99, batch_shuffle=True)
        self.criterion = lightly.loss.NTXentLoss(
            temperature=0.1,
            memory_bank_size=memory_bank_size)

        self.lr = learning_rate
        self.max_epochs = max_epochs

    def forward(self, x):
        return self.resnet_moco(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        x0, x1 = self.resnet_moco(x0, x1)
        loss = self.criterion(x0, x1)
        return loss
    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.resnet_moco.parameters(), lr=self.lr, momentum=0.9
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]



class MoCoModel_benchmarking(benchmarking.BenchmarkModule):
    def __init__(self,
                backbone,
                dataloader_kNN,
                num_classes,
                 num_ftrs = 128,
                 knn_k=20,
                 memory_bank_size = 4096,
                 learning_rate = 0.1,
                 max_epochs = 400):
      
        super().__init__(dataloader_kNN,num_classes)

        self.knn_k = knn_k

        self.max_epochs = max_epochs
        self.lr = learning_rate

        model = MoCoModel(
            backbone,
            num_ftrs = num_ftrs,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            )
        
        self.resnet_moco = model.resnet_moco
        self.backbone = model.backbone
        self.criterion = model.criterion

      
    def forward(self, x):
        return self.resnet_moco(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        x0, x1 = self.resnet_moco(x0, x1)
        loss = self.criterion(x0, x1)
        return loss
        
    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.resnet_moco.parameters(), lr=self.lr, momentum=0.9
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]

    def validation_epoch_end(self, outputs):
        device = self.dummy_param.device
        if outputs:
            total_num = torch.Tensor([0]).to(device)
            total_top1 = torch.Tensor([0.]).to(device)
            for (num, top1) in outputs:
                total_num += num[0]
                total_top1 += top1

            # if dist.is_initialized() and dist.get_world_size() > 1:
            #     dist.all_reduce(total_num)
            #     dist.all_reduce(total_top1)

            acc = float(total_top1.item() / total_num.item())
            if acc > self.max_accuracy:
                self.max_accuracy = acc
            self.log('kNN_accuracy', acc * 100.0, prog_bar=True)

class BYOLModel(pl.LightningModule):
    def __init__(
        self,
        resnet,
        num_ftrs = 512,
        memory_bank_size = 4096,
        learning_rate = 0.1,
        max_epochs = 400):
        super().__init__()

        last_conv_channels = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.Conv2d(last_conv_channels, num_ftrs, 1),
        )

        self.resnet_moco = lightly.models.BYOL(self.backbone, num_ftrs=num_ftrs, m=0.99)
        self.criterion = lightly.loss.SymNegCosineSimilarityLoss()
        # self.criterion = lightly.loss.NTXentLoss(
        #     temperature=0.1,
        #     memory_bank_size=memory_bank_size)

        self.lr = learning_rate
        self.max_epochs = max_epochs

    def forward(self, x):
        return self.resnet_moco(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        x0, x1 = self.resnet_moco(x0, x1)
        loss = self.criterion(x0, x1)
        return loss
    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.resnet_moco.parameters(), lr=self.lr, momentum=0.9
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]


class BYOLModel_benchmarking(benchmarking.BenchmarkModule):
    def __init__(self,
                backbone,
                dataloader_kNN,
                num_classes,
                num_ftrs = 128,
                knn_k=20,
                memory_bank_size = 4096,
                learning_rate = 0.1,
                max_epochs = 400):
      
        super().__init__(dataloader_kNN,num_classes)

        self.knn_k = knn_k

        self.max_epochs = max_epochs
        self.lr = learning_rate

        model = BYOLModel(
            backbone,
            num_ftrs = num_ftrs,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            )
        
        self.resnet_moco = model.resnet_moco
        self.backbone = model.backbone
        self.criterion = model.criterion

      
    def forward(self, x):
        return self.resnet_moco(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        x0, x1 = self.resnet_moco(x0, x1)
        loss = self.criterion(x0, x1)
        return loss
        
    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.resnet_moco.parameters(), lr=self.lr, momentum=0.9
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]

    def validation_epoch_end(self, outputs):
        device = self.dummy_param.device
        if outputs:
            total_num = torch.Tensor([0]).to(device)
            total_top1 = torch.Tensor([0.]).to(device)
            for (num, top1) in outputs:
                total_num += num[0]
                total_top1 += top1

            # if dist.is_initialized() and dist.get_world_size() > 1:
            #     dist.all_reduce(total_num)
            #     dist.all_reduce(total_top1)

            acc = float(total_top1.item() / total_num.item())
            if acc > self.max_accuracy:
                self.max_accuracy = acc
            self.log('kNN_accuracy', acc * 100.0, prog_bar=True)

class Classifier(nn.Module):
    def __init__(self,backbone,output_size):
        super().__init__()

        self.backbone = backbone
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, output_size)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.backbone(x)
        out = self.sm(out)
        return out


class SelfSupervisedClassifier(nn.Module):
    def __init__(self, backbone,num_ftrs,output_dim):
        super().__init__()
        
        self.backbone = backbone
        self.fc = nn.Linear(num_ftrs, output_dim)
        self.sm = nn.Softmax(dim=-1)
        
    def forward(self, x):
        y_hat = self.backbone(x).squeeze()
        y_hat = self.fc(y_hat)
        y_hat = self.sm(y_hat)
        return y_hat

class LinearClassifier(nn.Module):
    def __init__(self, backbone,num_ftrs,output_dim):
        super().__init__()
        
        self.backbone = backbone

        # freeze the layers
        for p in self.backbone.parameters():  # reset requires_grad
            p.requires_grad = False

        self.fc = nn.Linear(num_ftrs, output_dim)
        self.sm = nn.Softmax(dim=-1)
        
    def forward(self, x):
        with torch.no_grad():
            y_hat = self.backbone(x).squeeze()
            y_hat = nn.functional.normalize(y_hat, dim=1)
        y_hat = self.fc(y_hat)
        y_hat = self.sm(y_hat)
        return y_hat
