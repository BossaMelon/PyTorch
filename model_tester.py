import torch
import numpy as np
from model_pytorch import Torch_trainer

from sklearn.model_selection import train_test_split
from models import create_conv_ad_model
from models import plot_keras_history

from tensorflow.python.keras import Input, callbacks
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import optimizers

X = np.random.rand(1000, 38, 30, 3)
y = np.random.rand(1000, 1140)

# train, val and test: 0.8, 0.1, 0.1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)


def torch_tester():
    trainer = Torch_trainer()

    dataset_train = trainer.create_torch_dataset(X_train, y_train)
    dataset_val = trainer.create_torch_dataset(X_val, y_val)
    dataset_test = trainer.create_torch_dataset(X_test, y_test)

    trainer.set_dataset(dataset_train, dataset_val, dataset_test)

    trainer.set_dataloader(batch_size_train=64)

    network = trainer.create_model(show_summary=True)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-2)

    trainer.set_early_stopping(monitor='val', patience=2000)

    model_torch = trainer.fit(network, criterion, optimizer, epoch_size=1000)

def keras_tester():
    model_keras, encoder = create_conv_ad_model(input_shape=X_train.shape[1:])
    callback_list = [
        #         callbacks.ModelCheckpoint(
        #             filepath=(dir_path + '/models/anomaly_detection.h5'),
        #             monitor='val_mse',
        #             save_best_only=True
        #         ),
        callbacks.EarlyStopping(
            monitor='val_mse',
            patience=2000
        )]
    history = model_keras.fit(X_train, y_train, epochs=1000, batch_size=64, validation_data=(X_val, y_val), verbose=2,
                              callbacks=callback_list)


if __name__ == '__main__':
    keras_tester()
    # torch_tester()