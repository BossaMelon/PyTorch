import time

import torch
import numpy as np
from model_pytorch import Torch_trainer

from sklearn.model_selection import train_test_split
from models import create_conv_ad_model
from models import plot_keras_history


X = np.random.rand(1000, 38, 30, 3)
y = np.random.rand(1000, 1140)

# train, val and test: 0.8, 0.1, 0.1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)


def torch_tester(n):
    trainer = Torch_trainer()

    dataset_train = trainer.create_torch_dataset(X_train, y_train)
    dataset_val = trainer.create_torch_dataset(X_val, y_val)

    trainer.set_dataset(dataset_train, dataset_val)

    trainer.set_dataloader(batch_size_train=64)

    network = trainer.create_model(show_summary=True)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-2)

    t1 = time.time()
    trainer.fit(network, criterion, optimizer, epoch_size=n)
    t2 = time.time()
    return t2-t1

def keras_tester(n):
    model_keras, encoder = create_conv_ad_model(input_shape=X_train.shape[1:])
    t1 = time.time()
    model_keras.fit(X_train, y_train, epochs=n, batch_size=64, validation_data=(X_val, y_val), verbose=2)
    t2 = time.time()
    return t2 - t1

if __name__ == '__main__':
    n = 200
    t_torch = torch_tester(n)
    t_keras = keras_tester(n)
    print('torch: {}, keras: {}'.format(t_torch, t_keras))
