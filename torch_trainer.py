import torch.nn as nn
import torch
import torch.nn.functional as F
from torchsummary import summary
import torch.optim as optim
import numpy as np
import torch.utils.data as Data
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from network_torch import Network_regression, Network_classification
import copy
import time


class Torch_trainer_regression:
    def __init__(self):

        self._dataset_train = None
        self._dataset_val = None
        self._dataset_test = None
        self._dataloader_train = None
        self._dataloader_val = None
        self._dataloader_test = None

        self._mode = 'regression'

        self._network = None
        self._model = None

        self._X_shape = None
        self._y_shape = None
        self._X_dtype = None
        self._y_dtype = None

        self._train_losses = None
        self._val_losses = None

        self._n_epochs_stop_val = float("inf")
        self._n_epochs_stop_train = float("inf")

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def create_torch_tensor(self, X, y=None):
        # reshape and specify dtype
        # numpy tensor: [B,H,W,C] to torch tensor: [B,C,H,W]
        X_torch = torch.tensor(X.transpose(0, 3, 1, 2), dtype=torch.float32)

        if y is not None:
            y_torch = torch.tensor(y, dtype=torch.float32)
            return X_torch, y_torch
        else:
            return X_torch

    """""
    simple data set can directly create with torch.tensor(), it needs to load all the data to the memory.
    For large scale data, we should use torch.utils.data.Dataset, and override the generator.
    """""

    def create_torch_dataset(self, X, y):
        X_torch, y_torch = self.create_torch_tensor(X, y)
        dataset = Data.TensorDataset(X_torch, y_torch)
        return dataset

    def set_dataset(self, dataset_train, dataset_val, dataset_test=None):
        # TODO only TensorDataset and Subset allowed, it there any other class also allowed? Need to figure out...

        # check class
        if dataset_test is not None:
            for i in [dataset_train, dataset_val, dataset_test]:
                if i.__class__.__name__ != 'TensorDataset' and i.__class__.__name__ != 'Subset':
                    raise ValueError('Dataset type invalid! Only {}, {} allowed!'.format('TensorDataset', 'Subset'))
        else:
            for i in [dataset_train, dataset_val]:
                if i.__class__.__name__ != 'TensorDataset' and i.__class__.__name__ != 'Subset':
                    raise ValueError('Dataset type invalid! Only {}, {} allowed!'.format('TensorDataset', 'Subset'))

        # check data shape and dtype
        if dataset_train.__class__.__name__ == 'TensorDataset':
            train_shape_X = dataset_train.tensors[0].shape[1:]
            train_shape_y = dataset_train.tensors[1].shape[1:]
            train_dtype_X = dataset_train.tensors[0].dtype
            train_dtype_y = dataset_train.tensors[1].dtype

            val_shape_X = dataset_val.tensors[0].shape[1:]
            val_shape_y = dataset_val.tensors[1].shape[1:]
            val_dtype_X = dataset_val.tensors[0].dtype
            val_dtype_y = dataset_val.tensors[1].dtype

            if dataset_test is not None:
                test_shape_X = dataset_test.tensors[0].shape[1:]
                test_shape_y = dataset_test.tensors[1].shape[1:]
                test_dtype_X = dataset_test.tensors[0].dtype
                test_dtype_y = dataset_test.tensors[1].dtype

                if train_shape_X == val_shape_X == test_shape_X and train_shape_y == val_shape_y == test_shape_y:
                    self._X_shape = train_shape_X
                    self._y_shape = train_shape_y
                else:
                    raise ValueError('Shape not same in train set, val set and test set!')
                if train_dtype_X == val_dtype_X == test_dtype_X and train_dtype_y == val_dtype_y == test_dtype_y:
                    self._X_dtype = train_dtype_X
                    self._y_dtype = train_dtype_y
                else:
                    raise ValueError('dtype not same in train set, val set and test set!')

            else:
                if train_shape_X == val_shape_X and train_shape_y == val_shape_y:
                    self._X_shape = train_shape_X
                    self._y_shape = train_shape_y
                else:
                    raise ValueError('Shape not same in train set and val set!')
                if train_dtype_X == val_dtype_X and train_dtype_y == val_dtype_y:
                    self._X_dtype = train_dtype_X
                    self._y_dtype = train_dtype_y
                else:
                    raise ValueError('dtype not same in train set and val set!')

        elif dataset_train.__class__.__name__ == 'Subset':
            train_shape_X = dataset_train.dataset.tensors[0].shape[1:]
            train_shape_y = dataset_train.dataset.tensors[1].shape[1:]
            train_dtype_X = dataset_train.dataset.tensors[0].dtype
            train_dtype_y = dataset_train.dataset.tensors[1].dtype

            val_shape_X = dataset_val.dataset.tensors[0].shape[1:]
            val_shape_y = dataset_val.dataset.tensors[1].shape[1:]
            val_dtype_X = dataset_val.dataset.tensors[0].dtype
            val_dtype_y = dataset_val.dataset.tensors[1].dtype

            if dataset_test is not None:
                test_shape_X = dataset_test.dataset.tensors[0].shape[1:]
                test_shape_y = dataset_test.dataset.tensors[1].shape[1:]
                test_dtype_X = dataset_test.dataset.tensors[0].dtype
                test_dtype_y = dataset_test.dataset.tensors[1].dtype

                if train_shape_X == val_shape_X == test_shape_X and train_shape_y == val_shape_y == test_shape_y:
                    self._X_shape = train_shape_X
                    self._y_shape = train_shape_y
                else:
                    raise ValueError('Shape not same in train set, val set and test set!')
                if train_dtype_X == val_dtype_X == test_dtype_X and train_dtype_y == val_dtype_y == test_dtype_y:
                    self._X_dtype = train_dtype_X
                    self._y_dtype = train_dtype_y
                else:
                    raise ValueError('dtype not same in train set, val set and test set!')

            else:
                if train_shape_X == val_shape_X and train_shape_y == val_shape_y:
                    self._X_shape = train_shape_X
                    self._y_shape = train_shape_y
                else:
                    raise ValueError('Shape not same in train set and val set!')
                if train_dtype_X == val_dtype_X and train_dtype_y == val_dtype_y:
                    self._X_dtype = train_dtype_X
                    self._y_dtype = train_dtype_y
                else:
                    raise ValueError('dtype not same in train set and val set!')

        if self._X_dtype != torch.float32 or self._y_dtype != torch.float32:
            # TODO any better way?
            """
             torch model weights have default dtype:float32, so need to cast the dataset to float32, even though
             torch model can also be casted to float64, but float64 model is much slower than float32 model.
            """
            raise ValueError('Dataset dtype should be float32')

        # set dataset
        self._dataset_train = dataset_train
        self._dataset_val = dataset_val
        if dataset_test is not None:
            self._dataset_test = dataset_test

    def get_dataset(self, dataset):
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
        """
        set dataloader parameter,
        :param num_workers:how many subprocesses to use for data loading.0 means that the data will be loaded in the main process. (default: 0)
        """
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

    def create_model(self, Network, show_summary=False):
        # summary: pip install torchsummary
        data_shape = self._X_shape
        self._network = Network(data_shape).to(self._device)
        if show_summary:
            summary(self._network, input_size=data_shape[-3:])
        return self._network

    def fit(self, model, criterion, optimizer, epoch_size, show_time=False):
        # model = self._network
        # optimizer = self._set_optimizer(optimizer, lr)
        # criterion = self._set_criterion(criterion)
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        since = time.time()

        best_loss_val = float("inf")
        best_loss_train = float("inf")

        self._train_losses = []
        self._val_losses = []

        best_model_wts = copy.deepcopy(model.state_dict())

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
            epoch_start = time.time()
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
                        early_stopping_flag = True
                        break

                if phase == 'train':
                    if epoch_loss < best_loss_train:
                        epochs_no_improve_train = 0
                    else:
                        epochs_no_improve_train += 1

                    if epochs_no_improve_train == self._n_epochs_stop_train:
                        early_stopping_flag = True
                        break

            epoch_end = time.time()
            if show_time:
                print('epoch time: {}'.format(epoch_end - epoch_start))
            else:
                print()

            if early_stopping_flag:
                print(64 * '-')
                print('early stopping!')
                break

        time_elapsed = time.time() - since
        print(64 * '-')
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Loss: {:4f}'.format(best_loss_val))

        # load best model weights
        model.load_state_dict(best_model_wts)
        self._model = model
        return self._model

    def _get_loss(self):
        return self._train_losses, self._val_losses

    def inference(self, X_test, model=None):
        X_test_torch = self.create_torch_tensor(X_test)
        if model is None:
            model = self._model
        y_pred = model(X_test_torch).detach().numpy()
        return y_pred

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


class Torch_trainer_classification:
    def __init__(self):

        self._dataset_train = None
        self._dataset_val = None
        self._dataset_test = None
        self._dataloader_train = None
        self._dataloader_val = None
        self._dataloader_test = None

        self._mode = 'classification'

        self._network = None
        self._model = None

        self._X_shape = None
        self._y_shape = None
        self._X_dtype = None
        self._y_dtype = None

        self._train_losses = None
        self._val_losses = None
        self._train_acc = None
        self._val_acc = None

        self._n_epochs_stop_val = float("inf")
        self._n_epochs_stop_train = float("inf")

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def create_torch_tensor(self, X, y=None):
        # reshape and specify dtype
        # numpy tensor: [B,H,W,C] to torch tensor: [B,C,H,W]
        X_torch = torch.tensor(X.transpose(0, 3, 1, 2), dtype=torch.float32)

        if y is not None:
            y_torch = torch.tensor(y, dtype=torch.int64)
            return X_torch, y_torch

        else:
            return X_torch

    """""
    simple data set can directly create with torch.tensor(), it needs to load all the data to the memory.
    For large scale data, we should use torch.utils.data.Dataset, and override the generator.
    """""

    def create_torch_dataset(self, X, y):
        X_torch, y_torch = self.create_torch_tensor(X, y)
        dataset = Data.TensorDataset(X_torch, y_torch)
        return dataset

    def set_dataset_(self, dataset_train, dataset_val, dataset_test=None):
        # set dataset
        self._dataset_train = dataset_train
        self._dataset_val = dataset_val
        if dataset_test is not None:
            self._dataset_test = dataset_test
        self._X_shape = torch.Size([1, 28, 28])

    def set_dataset(self, dataset_train, dataset_val, dataset_test=None):
        # TODO only TensorDataset and Subset allowed, it there any other class also allowed? Need to figure out...

        # check class
        if dataset_test is not None:
            for i in [dataset_train, dataset_val, dataset_test]:
                if i.__class__.__name__ != 'TensorDataset' and i.__class__.__name__ != 'Subset':
                    raise ValueError('Dataset type invalid! Only {}, {} allowed!'.format('TensorDataset', 'Subset'))
        else:
            for i in [dataset_train, dataset_val]:
                if i.__class__.__name__ != 'TensorDataset' and i.__class__.__name__ != 'Subset':
                    raise ValueError('Dataset type invalid! Only {}, {} allowed!'.format('TensorDataset', 'Subset'))

        # check data shape and dtype
        if dataset_train.__class__.__name__ == 'TensorDataset':
            train_shape_X = dataset_train.tensors[0].shape[1:]
            train_shape_y = dataset_train.tensors[1].shape[1:]
            train_dtype_X = dataset_train.tensors[0].dtype
            train_dtype_y = dataset_train.tensors[1].dtype

            val_shape_X = dataset_val.tensors[0].shape[1:]
            val_shape_y = dataset_val.tensors[1].shape[1:]
            val_dtype_X = dataset_val.tensors[0].dtype
            val_dtype_y = dataset_val.tensors[1].dtype

            if dataset_test is not None:
                test_shape_X = dataset_test.tensors[0].shape[1:]
                test_shape_y = dataset_test.tensors[1].shape[1:]
                test_dtype_X = dataset_test.tensors[0].dtype
                test_dtype_y = dataset_test.tensors[1].dtype

                if train_shape_X == val_shape_X == test_shape_X and train_shape_y == val_shape_y == test_shape_y:
                    self._X_shape = train_shape_X
                    self._y_shape = train_shape_y
                else:
                    raise ValueError('Shape not same in train set, val set and test set!')
                if train_dtype_X == val_dtype_X == test_dtype_X and train_dtype_y == val_dtype_y == test_dtype_y:
                    self._X_dtype = train_dtype_X
                    self._y_dtype = train_dtype_y
                else:
                    raise ValueError('dtype not same in train set, val set and test set!')

            else:
                if train_shape_X == val_shape_X and train_shape_y == val_shape_y:
                    self._X_shape = train_shape_X
                    self._y_shape = train_shape_y
                else:
                    raise ValueError('Shape not same in train set and val set!')
                if train_dtype_X == val_dtype_X and train_dtype_y == val_dtype_y:
                    self._X_dtype = train_dtype_X
                    self._y_dtype = train_dtype_y
                else:
                    raise ValueError('dtype not same in train set and val set!')

        elif dataset_train.__class__.__name__ == 'Subset':
            train_shape_X = dataset_train.dataset.tensors[0].shape[1:]
            train_shape_y = dataset_train.dataset.tensors[1].shape[1:]
            train_dtype_X = dataset_train.dataset.tensors[0].dtype
            train_dtype_y = dataset_train.dataset.tensors[1].dtype

            val_shape_X = dataset_val.dataset.tensors[0].shape[1:]
            val_shape_y = dataset_val.dataset.tensors[1].shape[1:]
            val_dtype_X = dataset_val.dataset.tensors[0].dtype
            val_dtype_y = dataset_val.dataset.tensors[1].dtype

            if dataset_test is not None:
                test_shape_X = dataset_test.dataset.tensors[0].shape[1:]
                test_shape_y = dataset_test.dataset.tensors[1].shape[1:]
                test_dtype_X = dataset_test.dataset.tensors[0].dtype
                test_dtype_y = dataset_test.dataset.tensors[1].dtype

                if train_shape_X == val_shape_X == test_shape_X and train_shape_y == val_shape_y == test_shape_y:
                    self._X_shape = train_shape_X
                    self._y_shape = train_shape_y
                else:
                    raise ValueError('Shape not same in train set, val set and test set!')
                if train_dtype_X == val_dtype_X == test_dtype_X and train_dtype_y == val_dtype_y == test_dtype_y:
                    self._X_dtype = train_dtype_X
                    self._y_dtype = train_dtype_y
                else:
                    raise ValueError('dtype not same in train set, val set and test set!')

            else:
                if train_shape_X == val_shape_X and train_shape_y == val_shape_y:
                    self._X_shape = train_shape_X
                    self._y_shape = train_shape_y
                else:
                    raise ValueError('Shape not same in train set and val set!')
                if train_dtype_X == val_dtype_X and train_dtype_y == val_dtype_y:
                    self._X_dtype = train_dtype_X
                    self._y_dtype = train_dtype_y
                else:
                    raise ValueError('dtype not same in train set and val set!')

        if self._X_dtype != torch.float32:
            # TODO any better way?
            """
             torch model weights have default dtype:float32, so need to cast the dataset to float32, even though
             torch model can also be casted to float64, but float64 model is much slower than float32 model.
            """
            raise ValueError('Dataset dtype should be float32')

        if self._y_dtype != torch.int64:
            raise ValueError('Dataset dtype should be int64')

        # set dataset
        self._dataset_train = dataset_train
        self._dataset_val = dataset_val
        if dataset_test is not None:
            self._dataset_test = dataset_test

    def get_dataset(self, dataset):
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
        """
        set dataloader parameter,
        :param num_workers:how many subprocesses to use for data loading.0 means that the data will be loaded in the main process. (default: 0)
        """
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

    def create_model(self, Network, show_summary=False):
        # summary: pip install torchsummary
        data_shape = self._X_shape
        self._network = Network(data_shape).to(self._device)
        if show_summary:
            summary(self._network, input_size=data_shape[-3:])
        return self._network

    def fit(self, model, criterion, optimizer, epoch_size, show_time=False):
        # model = self._network
        # optimizer = self._set_optimizer(optimizer, lr)
        # criterion = self._set_criterion(criterion)
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        since = time.time()

        best_loss_val = float("inf")
        best_loss_train = float("inf")

        self._train_losses = []
        self._val_losses = []
        self._train_acc = []
        self._val_acc = []

        best_acc = 0.0
        best_model_wts = copy.deepcopy(model.state_dict())

        epochs_no_improve_val = 0
        epochs_no_improve_train = 0
        early_stopping_flag = False
        epoch_acc = 0

        # Print training information
        # TODO use logger?
        print('train on {} samples, validate on {} samples'.format(len(self._dataset_train), len(self._dataset_val)))
        if self._n_epochs_stop_train != float("inf") or self._n_epochs_stop_val != float("inf"):
            print('early stopping when: ', end="", flush=True)
        if self._n_epochs_stop_train != float("inf"):
            print('{} epoch without improvement in train_acc'.format(self._n_epochs_stop_train))
        elif self._n_epochs_stop_val != float("inf"):
            print('{} epoch without improvement in val_acc'.format(self._n_epochs_stop_val))
        print('{} used'.format('CPU' if self._device.type == 'cpu' else 'GPU'))
        print(64 * '-')

        # Train loop
        for epoch in range(epoch_size):
            print('Epoch {}/{}     '.format(epoch, epoch_size - 1), end="", flush=True)
            epoch_start = time.time()
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
                running_corrects = 0.0

                for inputs, labels in dataloader:
                    inputs = inputs.to(self._device)
                    labels = labels.to(self._device)

                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        # preds is index of the max value
                        _, preds = torch.max(outputs, 1)

                        loss = criterion(outputs, labels)

                        # backward + optimize only in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                # if phase == 'train':
                #     scheduler.step()

                epoch_loss = running_loss / dataset_size
                epoch_acc = running_corrects / dataset_size

                print('{} Loss: {:.4f}     {} Acc: {:.4f}     '.format(phase, epoch_loss, phase, epoch_acc), end="", flush=True)

                if phase == 'train':
                    self._train_losses.append(epoch_loss)
                    self._train_acc.append(epoch_acc)
                else:
                    self._val_losses.append(epoch_loss)
                    self._val_acc.append(epoch_acc)

                if phase == 'val':
                    # TODO check whether epoch_loss < best_loss can work in classification
                    if epoch_acc > best_acc:
                        epochs_no_improve_val = 0
                        best_acc = epoch_acc
                        # deep copy the model
                        best_model_wts = copy.deepcopy(model.state_dict())
                    else:
                        epochs_no_improve_val += 1

                    if epochs_no_improve_val == self._n_epochs_stop_val:
                        early_stopping_flag = True
                        break

                if phase == 'train':
                    if epoch_acc > best_acc:
                        epochs_no_improve_train = 0
                    else:
                        epochs_no_improve_train += 1

                    if epochs_no_improve_train == self._n_epochs_stop_train:
                        early_stopping_flag = True
                        break

            epoch_end = time.time()
            if show_time:
                print('epoch time: {}'.format(epoch_end - epoch_start))
            else:
                print()

            if early_stopping_flag:
                print(64 * '-')
                print('early stopping!')
                break

        time_elapsed = time.time() - since
        print(64 * '-')
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        self._model = model
        return self._model

    def _get_loss(self):
        return self._train_losses, self._val_losses

    def _get_acc(self):
        return self._train_acc, self._val_acc

    def inference(self, X_test, model=None):
        X_test_torch = self.create_torch_tensor(X_test)
        if model is None:
            model = self._model
        if self._mode == 'regression':
            y_pred = model(X_test_torch).detach().numpy()
            return y_pred
        elif self._mode == 'classification':
            _, y_pred = torch.max(model(X_test_torch), 1)
            return y_pred

    def loss_plot(self):
        train_losses, val_losses = self._get_loss()
        plt.plot(train_losses, label='Training loss')
        plt.plot(val_losses, label='Validation loss')
        plt.legend(frameon=False)
        plt.show()

    def acc_plot(self):
        train_acc, val_acc = self._get_acc()
        plt.plot(train_acc, label='Training accuracy')
        plt.plot(val_acc, label='Validation accuracy')
        plt.legend(frameon=False)
        plt.show()

    def set_early_stopping(self, patience, monitor='val'):
        if monitor == 'val':
            self._n_epochs_stop_val = patience
        elif monitor == 'train':
            self._n_epochs_stop_train = patience
        else:
            raise ValueError('monitor can only be train or val')

    def set_mode(self, mode):
        if mode == 'classification' or mode == 'regression':
            self._mode = mode
        else:
            raise ValueError('mode can only be {}'.format('classification and regression'))


if __name__ == '__main__':
    # Create fake data set
    from sklearn.model_selection import train_test_split

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

    model_torch = trainer.fit(network, criterion, optimizer, epoch_size=20)
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
