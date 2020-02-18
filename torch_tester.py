import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
from torchsummary import summary
import numpy as np
import torch.utils.data as Data
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

import torchvision.transforms as transforms

from torch_trainer import Torch_trainer_classification


from torch_trainer import Torch_trainer_regression
import copy
import time
from sklearn.model_selection import train_test_split
from network_torch import Network_regression, Network_classification


def anomaly_detection_test():
    # generate random data
    X = np.random.rand(1000, 38, 30, 3)
    y = np.random.rand(1000, 1140)

    # train, val and test: 0.8, 0.1, 0.1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)

    trainer = Torch_trainer_regression()

    dataset_train = trainer.create_torch_dataset(X_train, y_train)
    dataset_val = trainer.create_torch_dataset(X_val, y_val)
    dataset_test = trainer.create_torch_dataset(X_test, y_test)

    trainer.set_dataset(dataset_train, dataset_val, dataset_test)

    trainer.set_dataloader(batch_size_train=64)

    network = trainer.create_model(Network_regression, show_summary=True)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-2)

    trainer.set_early_stopping(monitor='val', patience=200)

    model_torch = trainer.fit(network, criterion, optimizer, epoch_size=50)
    trainer.loss_plot()

def classification_test():
    train_set = torchvision.datasets.FashionMNIST(
        root='./data/FashionMNIST',
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )

    val_set = torchvision.datasets.FashionMNIST(
        root='./data/FashionMNIST',
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )

    print(len(train_set))
    print(len(val_set))
    sample = next(iter(train_set))

    image, label = sample


    print(image.shape)
    print(label)



    X = np.random.rand(1000, 38, 30, 3)
    y = np.random.randint(2, size=1000)

    # train, val and test: 0.8, 0.1, 0.1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)

    trainer = Torch_trainer_classification()

    dataset_train = trainer.create_torch_dataset(X_train, y_train)
    dataset_val = trainer.create_torch_dataset(X_val, y_val)
    dataset_test = trainer.create_torch_dataset(X_test, y_test)

    trainer.set_dataset_(train_set, val_set)
    trainer.set_dataloader(batch_size_train=100)

    network = trainer.create_model(Network_classification, show_summary=True)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-2)

    trainer.set_early_stopping(monitor='val', patience=200)
    model_torch = trainer.fit(network, criterion, optimizer, epoch_size=50)
    trainer.acc_plot()


if __name__ == '__main__':

    # anomaly_detection_test()
    classification_test()

