#!/usr/bin/env python
# coding: utf-8

# In[103]:

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import math

#%%

torch.set_grad_enabled(True)
batch_size=1000
epoch_size = 2
show = 5
if_use_gpu = 0

#%%

if (if_use_gpu):
    torch.cuda.init()
    print('GPU used')
else:
    print('CPU used')
    
print()
# In[105]:

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()
# In[106]:
    
class Network(nn.Module):
    def __init__(self):
        
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2= nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
 
    def forward(self, t): 
        #(1) input layer
        t=t
        
        #(2) hidden conv layer
        t=self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        #(3) hidden conv layer
        t=self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        #(4) hidden linear layer 
        t = t.reshape(-1, 12*4*4)
        t = self.fc1(t)
        t = F.relu(t)
        
        #(5) hidden linear layer 
        t = self.fc2(t)
        t = F.relu(t)
        
        #(6) output layer
        t = self.out(t)
        #t = F.softmax(t, dim=1)

        
        return t
# In[107]:
        
network = Network()

if (if_use_gpu):
    network.cuda()
    print('Cuda Network Created')
else:
    print('Network Created')

print()
# In[108]:

train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()])
    )

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,shuffle=True,)
optimizer = optim.Adam(network.parameters(), lr=0.01)
criterion=F.cross_entropy
# In[113]:
start = time.time()
lenth = len(train_set)
loss_matrix=np.zeros((epoch_size,math.ceil(lenth/batch_size)))


print('Train begin')

for epoch in range (epoch_size):
    for step, data in enumerate(train_loader):
        optimizer.zero_grad()
        images, labels = data
        optimizer.zero_grad()
    
        if (if_use_gpu):
            images=images.cuda()
            labels=labels.cuda()
            
        preds = network(images)
        loss = criterion(preds, labels)
        
        loss.backward()
        [epoch,step]=loss
        optimizer.step()
        
        end = time.time()
        interval = end-start
        
        if (step) % (round(lenth/batch_size/show)) ==0:
            accu = get_num_correct(preds, labels)/batch_size*100
            print('Epoch:', epoch+1, '  '
                  'step:', step+1 ,  '  '
                  'Loss: %.2f' % loss.item(),  '  '
                  'Accuracy: %.2f' % accu,'%', '  '
                  'Time: %.2f s' %interval)
print()       
print('Finish')      


# In[ ]:




