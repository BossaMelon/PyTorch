import torch.nn as nn
import torch
import torch.nn.functional as F
from torchsummary import summary
import torch.optim as optim
import numpy as np
import torch.utils.data as Data
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

import copy
import time


class Network(nn.Module):
    def __init__(self, shape):
        super(Network, self).__init__()

        # Torch Tensor [B,C,H,W]
        self.data_channel = shape[-3]
        self.data_height = shape[-2]
        self.data_width = shape[-1]

        # Padding Formula: P = ((S-1)*W-S+F)//2, with F = filter size, S = stride
        # Here: padding = (F-1)//2
        conv_layers = [8, 16, 32]

        kernel_size = 3
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(in_channels=self.data_channel, out_channels=conv_layers[0], kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=conv_layers[0], out_channels=conv_layers[1], kernel_size=kernel_size,
                               padding=padding)
        self.conv3 = nn.Conv2d(in_channels=conv_layers[1], out_channels=conv_layers[2], kernel_size=kernel_size,
                               padding=padding)

        # dropout
        self.dropout = nn.Dropout(0.1)

        # max pooling
        self.max_pool = nn.MaxPool2d(2, stride=2)

        # calculate input and output shape of dense layer
        self.dense_input = (self.data_height // 2 // 2 // 2) * (self.data_width // 2 // 2 // 2) * conv_layers[2]
        self.dense_output = self.data_height * self.data_width

        # dense layer
        self.final_dense = nn.Linear(in_features=self.dense_input, out_features=self.dense_output)

    def forward(self, t):
        # (1) input layer
        t = t

        # (2) hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)
        t = self.max_pool(t)
        t = self.dropout(t)

        # (3) hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = self.max_pool(t)
        t = self.dropout(t)

        # (4) hidden conv layer
        t = self.conv3(t)
        t = F.relu(t)
        t = self.max_pool(t)
        t = self.dropout(t)

        # (5) output layer
        t = t.view(-1, self.dense_input)
        t = self.final_dense(t)

        # (6) reshape to image format
        # t = t.reshape(-1,self.data_height,self.data_width)
        return t



class Torch_trainer:
    def __init__(self):

        self._dataset_train = None
        self._dataset_val = None
        self._dataset_test = None
        self._dataloader_train = None
        self._dataloader_val = None
        self._dataloader_test = None

        self._network = None
        self._model = None

        self._X_shape = None
        self._y_shape = None

        self._train_losses = None
        self._val_losses = None

        self._n_epochs_stop_val = float("inf")
        self._n_epochs_stop_train = float("inf")

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def create_torch_tensor(self, X, y):
        # reshape and specify dtype
        X_torch = torch.tensor(X.transpose(0, 3, 1, 2), dtype=torch.float32)
        y_torch = torch.tensor(y, dtype=torch.float32)
        return X_torch, y_torch

    """""
    simple data set can directly create with torch.tensor(), it needs to load all the data to the memory.
    For large scale data, we should use torch.utils.data.Dataset, and override the generator.
    """""

    def create_torch_dataset(self, X, y):
        X_torch, y_torch = self.create_torch_tensor(X, y)
        dataset = Data.TensorDataset(X_torch, y_torch)
        self._X_shape = X_torch.shape[1:]
        self._y_shape = y_torch.shape[1:]
        return dataset

    def set_dataset(self, dataset_train, dataset_val, dataset_test=None):
        self._dataset_train = dataset_train
        self._dataset_val = dataset_val
        if dataset_test is not None:
            self._dataset_test = dataset_test

    def get_dataset(self, dataset='train'):
        if dataset == 'train':
            return self._dataset_train
        elif dataset == 'val':
            return self._dataset_val
        elif dataset == 'test':
            return self._dataset_test
        else:
            raise ValueError('only train, val or test')

    # TODO num_workers=0 fastest in windows. In linux can choose a bigger num_workers
    def set_dataloader(self, batch_size_train, batch_size_val=None, batch_size_test=1, shuffle=True, num_workers=0):
        if batch_size_val is None:
            batch_size_val = batch_size_train

        self._dataloader_train = torch.utils.data.DataLoader(self._dataset_train, batch_size_train, shuffle=shuffle,
                                                             num_workers=num_workers)
        self._dataloader_val = torch.utils.data.DataLoader(self._dataset_val, batch_size_val, shuffle=shuffle,
                                                           num_workers=num_workers)
        if self._dataloader_test is not None:
            self._dataloader_test = torch.utils.data.DataLoader(self._dataset_test, batch_size_test, shuffle=False,
                                                                num_workers=num_workers)

    def get_dataloader(self, dataloader='train'):
        if dataloader == 'train':
            return self._dataloader_train
        elif dataloader == 'val':
            return self._dataloader_val
        elif dataloader == 'test':
            return self._dataloader_test
        else:
            raise ValueError('only train, val or test')

    def create_model(self, show_summary=False):
        data_shape = self._X_shape
        self._network = Network(data_shape).to(self._device)
        if show_summary:
            summary(self._network, input_size=data_shape[-3:])
        return self._network

    def fit(self, model, criterion, optimizer, epoch_size):
        # model = self._network
        # optimizer = self._set_optimizer(optimizer, lr)
        # criterion = self._set_criterion(criterion)
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        since = time.time()

        best_loss_val = float("inf")
        best_loss_train = float("inf")

        self._train_losses = []
        self._val_losses = []

        best_model_wts = None
        epochs_no_improve_val = 0
        epochs_no_improve_train = 0
        early_stopping_flag = False

        # Print training information
        # TODO use logger?
        print('train on {} samples, validate on {} samples'.format(len(self._dataset_train), len(self._dataset_val)))
        if self._n_epochs_stop_train != float("inf") or self._n_epochs_stop_val != float("inf"):
            print('early stopping when: ', end="", flush=True)
        if self._n_epochs_stop_train != float("inf"):
            print('{} epoch without improvement in train_loss'.format(self._n_epochs_stop_train))
        elif self._n_epochs_stop_val != float("inf"):
            print('{} epoch without improvement in val_loss'.format(self._n_epochs_stop_val))
        print('{} used'.format('CPU' if self._device.type == 'cpu' else 'GPU'))
        print(64 * '-')

        # Train loop
        for epoch in range(epoch_size):
            print('Epoch {}/{}     '.format(epoch, epoch_size - 1), end="", flush=True)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                    dataloader = self._dataloader_train
                    dataset_size = len(self._dataset_train)

                else:
                    model.eval()  # Set model to evaluate mode
                    dataloader = self._dataloader_val
                    dataset_size = len(self._dataset_val)

                running_loss = 0.0

                for inputs, labels in dataloader:
                    inputs = inputs.to(self._device)
                    labels = labels.to(self._device)

                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        # backward + optimize only in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)

                # if phase == 'train':
                #     scheduler.step()

                epoch_loss = running_loss / dataset_size

                print('{} Loss: {:.6f}     '.format(
                    phase, epoch_loss), end="", flush=True)
                if phase == 'train':
                    self._train_losses.append(epoch_loss)
                else:
                    self._val_losses.append(epoch_loss)

                if phase == 'val':
                    if epoch_loss < best_loss_val:
                        epochs_no_improve_val = 0
                        best_loss_val = epoch_loss
                        # deep copy the model
                        best_model_wts = copy.deepcopy(model.state_dict())
                    else:
                        epochs_no_improve_val += 1

                    if epochs_no_improve_val == self._n_epochs_stop_val:
                        print()
                        print()
                        print('early stopping!')
                        early_stopping_flag = True
                        break

                if phase == 'train':
                    if epoch_loss < best_loss_train:
                        epochs_no_improve_train = 0
                    else:
                        epochs_no_improve_train += 1

                    if epochs_no_improve_train == self._n_epochs_stop_train:
                        print()
                        print()
                        print('early stopping!')
                        early_stopping_flag = True
                        break

            print()
            if early_stopping_flag:
                break

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Loss: {:4f}'.format(best_loss_val))

        # load best model weights
        model.load_state_dict(best_model_wts)
        self._model = model
        return self._model

    def _get_loss(self):
        return self._train_losses, self._val_losses

    def inference(self, X_test=None, model=None):
        if model is None:
            model = self._model

        if X_test is None:
            if self._X_test is None:
                raise ValueError('No test set defined')
            y_pred = model(self._X_test).detach().numpy()
            return y_pred
        elif X_test.__class__.__name__ == 'ndarray':
            X_test_torch = torch.tensor(X_test.transpose(0, 3, 1, 2), dtype=torch.float32)
            y_pred = model(X_test_torch).detach().numpy()
            return y_pred
        elif X_test.__class__.__name__ == 'Tensor':
            y_pred = model(X_test).detach().numpy()
            return y_pred
        else:
            raise ValueError('data type can only be numpy array or torch tensor')

    def loss_plot(self):
        train_losses, val_losses = self._get_loss()
        plt.plot(train_losses, label='Training loss')
        plt.plot(val_losses, label='Validation loss')
        plt.legend(frameon=False)
        plt.show()

    def set_early_stopping(self, patience, monitor='val'):
        if monitor == 'val':
            self._n_epochs_stop_val = patience
        elif monitor == 'train':
            self._n_epochs_stop_train = patience
        else:
            raise ValueError('monitor can only be train or val')


if __name__ == '__main__':
    # Create fake data set
    X_train = np.random.rand(992, 38, 30, 3)
    y_train = np.random.rand(992, 1140)
    X_val = np.random.rand(53, 38, 30, 3)
    y_val = np.random.rand(53, 1140)
    X_test = np.random.rand(589, 38, 30, 3)
    y_test = np.random.rand(589, 1140)

    trainer = Torch_trainer()

    dataset_train = trainer.create_torch_dataset(X_train,y_train)
    dataset_val = trainer.create_torch_dataset(X_val, y_val)
    dataset_test = trainer.create_torch_dataset(X_test, y_test)

    trainer.set_dataset(dataset_train, dataset_val, dataset_test)

    trainer.set_dataloader(batch_size_train=64)

    network = trainer.create_model(show_summary=True)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-2)

    trainer.set_early_stopping(monitor='val', patience=20)

    model_torch = trainer.fit(network, criterion, optimizer, epoch_size=1000)
    trainer.loss_plot()
    # y_pred = trainer.inference(X_test)
    # print(y_pred.shape)
    #
    # scores = np.max(np.abs(y_pred - y_test), axis=1)
    # print(scores.shape)
    #
    # shape_2d_sensors = X_train.shape[1:3]
    # for i in range(1):
    #     tmp = np.abs(y_pred[i] - y_test[i])
    #     diff = tmp.reshape(shape_2d_sensors)
    #     plt.imshow(diff, cmap='jet', alpha=0.3)
    #     plt.clim(0, 1)
    #     plt.show()
