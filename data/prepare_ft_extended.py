import os
import torch 

NAME = 'syn-hi-small'
PERCENT_DATA = 10

train_shape = None
val_shape = None 

train_path = f'./train-data-{NAME}/'
val_path = f'./val-data-{NAME}/'
ft_train_path = f'./train-data-ft-extended-{NAME}/'
ft_val_path = f'./val-data-ft-extended-{NAME}/'

if not os.path.exists(ft_train_path):
    os.mkdir(ft_train_path)
if not os.path.exists(ft_val_path):
    os.mkdir(ft_val_path)

files = os.listdir(train_path)
for file in files:
    temp = torch.load(train_path+file)
    #print(int(temp.shape[0]*PERCENT_DATA/100),temp.shape[1]-1)
    temp = temp[:int(temp.shape[0]*PERCENT_DATA/100),:]
    print(temp.shape)
    torch.save(temp,ft_train_path+file)
    mask = torch.ones((temp.shape[0],temp.shape[1]-1))
    print(mask.shape)
    torch.save(mask, ft_train_path+file+'.mask')
files = os.listdir(val_path)
for file in files:
    os.system(f'cp {val_path+file} {ft_val_path+file}')
    temp = torch.load(ft_val_path+file)
    print(temp.shape[0],temp.shape[1]-1)
    mask = torch.ones((temp.shape[0],temp.shape[1]-1))
    torch.save(mask, ft_val_path+file+'.mask')



