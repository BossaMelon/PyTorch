import torch


print('device count {}'.format(torch.cuda.device_count()))
print('cuda available {}'.format(torch.cuda.is_available()))
print('cuda device name {}'.format(torch.cuda.get_device_name()))