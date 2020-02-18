import torch.nn as nn
import torch
import torch.nn.functional as F


class Network_regression(nn.Module):
    def __init__(self, shape):
        super(Network_regression, self).__init__()

        # Torch Tensor [B,C,H,W]
        self.data_channel = shape[-3]
        self.data_height = shape[-2]
        self.data_width = shape[-1]

        # Padding Formula: P = ((S-1)*W-S+F)//2, with F = filter size, S = stride
        # Here: padding = (F-1)//2
        conv_layers = [8, 16, 32]

        kernel_size = 3
        # same size after conv
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
        # data go through maxpooling 3 times, so height and width "// 2" for 3 times
        self.dense_input = (self.data_height // 2 // 2 // 2) * (self.data_width // 2 // 2 // 2) * conv_layers[-1]
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
        t = torch.sigmoid(t)

        # (6) reshape to image format
        # t = t.reshape(-1,self.data_height,self.data_width)
        return t


class Network_classification(nn.Module):
    def __init__(self, shape):
        super(Network_classification, self).__init__()

        # Torch Tensor [B,C,H,W]
        self.data_channel = shape[-3]
        self.data_height = shape[-2]
        self.data_width = shape[-1]

        # Padding Formula: P = ((S-1)*W-S+F)//2, with F = filter size, S = stride
        # Here: padding = (F-1)//2
        conv_layers = [8, 16, 32]

        kernel_size = 3
        # same size after conv
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
        # data go through maxpooling 3 times, so height and width "// 2" for 3 times
        self.dense_input = (self.data_height // 2 // 2 // 2) * (self.data_width // 2 // 2 // 2) * conv_layers[-1]
        self.dense_output = 10

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
        # t = torch.sigmoid(t)

        return t
