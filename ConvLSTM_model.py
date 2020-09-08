import torch
import torch.nn as nn
import torch.nn.functional as F
from convlstm import ConvLSTM
class Net02(nn.Module):
    def __init__(self):
        super(Net02, self).__init__()
        #global NUM_FRAMES=5
        #self.conv1 = ConvLSTM(input_channels=1, hidden_channels=[64, 32, 32], kernel_size=3, step=5,
        #           effective_step=[4]).cuda()
        
        self.maxpool= nn.MaxPool3d((1,5,5),stride=(1,3,3))
        self.BD = nn.BatchNorm2d(16)
        self.drop = nn.Dropout(p=0.3)
        self.conv1 = ConvLSTM(input_size = (74,74), input_dim = 3, hidden_dim = [32, 16, 16], 
                              kernel_size = (3,3), num_layers = 3, batch_first=True)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3,stride=2)
        self.fc1 = nn.Linear(5*16*9*9, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        
        x = self.maxpool(x)
        #print (x.shape)
        x = self.conv1(x)     
        x = self.BD(x[0][0])    
        x = F.relu(F.max_pool2d(x,2))  
        #x = self.drop(x)
        x = self.BD(self.conv2(x))  
        x = F.relu(F.max_pool2d(x,2))  
        
        x = x.view(1,-1)        
        x = F.relu(self.fc1(x))       
        x = F.relu(self.fc2(x))
        #x = self.drop(x)
        x = self.fc3(x).reshape(-1,1,2)
        return x
class Net04(nn.Module):
    def __init__(self):
        super(Net04, self).__init__()
        global NUM_FRAMES
        #self.conv1 = ConvLSTM(input_channels=1, hidden_channels=[64, 32, 32], kernel_size=3, step=5,
        #           effective_step=[4]).cuda()
        
        self.maxpool= nn.MaxPool3d((1,5,5),stride=(1,3,3))
        self.BD = nn.BatchNorm2d(16)
        self.drop = nn.Dropout(p=0.3)
        self.conv1 = ConvLSTM(input_size = (74,74), input_dim = 3, hidden_dim = [64, 32, 16], 
                              kernel_size = (3,3), num_layers = 3, batch_first=True)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3,stride=2)
        self.fc1 = nn.Linear(5*16*9*9, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        
        x = self.maxpool(x)
        #print (x.shape)
        x = self.conv1(x)     
        x = self.BD(x[0][0])    
        x = F.relu(F.max_pool2d(x,2))  
        #x = self.drop(x)
        x = self.BD(self.conv2(x))  
        x = F.relu(F.max_pool2d(x,2))  
        
        x = x.view(1,-1)
        #x = self.drop(x)
        x = F.relu(self.fc1(x))    
        
        
        x = F.relu(self.fc2(x))
        x = self.fc3(x).reshape(-1,1,2)
        return x


class Net05(nn.Module):
    def __init__(self):
        super(Net05, self).__init__()
        
        #self.conv1 = ConvLSTM(input_channels=1, hidden_channels=[64, 32, 32], kernel_size=3, step=5,
        #           effective_step=[4]).cuda()
        
        self.maxpool= nn.MaxPool3d((1,5,5),stride=(1,3,3))
        self.BD = nn.BatchNorm2d(16)
        self.drop = nn.Dropout(p=0.3)
        self.conv1 = ConvLSTM(input_size = (74,74), input_dim = 3, hidden_dim = [64, 32, 16], 
                              kernel_size = (3,3), num_layers = 3, batch_first=True)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3,stride=2)
        self.fc1 = nn.Linear(5*16*9*9, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        
        x = self.maxpool(x)
        #print (x.shape)
        x = self.conv1(x)     
        x = self.BD(x[0][0])    
        x = F.relu(F.max_pool2d(x,2))  
        #x = self.drop(x)
        x = self.BD(self.conv2(x))  
        x = F.relu(F.max_pool2d(x,2))  
        
        x = x.view(1,-1)
        #x = self.drop(x)
        x = F.relu(self.fc1(x))    
        
        
        x = F.relu(self.fc2(x))
        x = self.fc3(x).reshape(-1,1,2)
        return x

class Net06(nn.Module):
    def __init__(self):
        super(Net06, self).__init__()
        #global NUM_FRAMES
        #self.conv1 = ConvLSTM(input_channels=1, hidden_channels=[64, 32, 32], kernel_size=3, step=5,
        #           effective_step=[4]).cuda()
        
        self.maxpool= nn.MaxPool3d((1,5,5),stride=(1,3,3))
        self.BD = nn.BatchNorm2d(16)
        self.drop = nn.Dropout(p=0.3)
        self.conv1 = ConvLSTM(input_size = (74,74), input_dim = 3, hidden_dim = [64, 32, 16], 
                              kernel_size = (3,3), num_layers = 3, batch_first=True)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3,stride=2)
        self.fc1 = nn.Linear(5*16*9*9, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)
        self.fc4 = nn.Linear(20,1)

    def forward(self, x):
        
        x = self.maxpool(x)
        #print (x.shape)
        x = self.conv1(x)     
        x = self.BD(x[0][0])    
        x = F.relu(F.max_pool2d(x,2))  
        #x = self.drop(x)
        x = self.BD(self.conv2(x))  
        x = F.relu(F.max_pool2d(x,2))  
        
        x = x.view(1,-1)
        #x = self.drop(x)
        x = F.relu(self.fc1(x))    
        
        
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x1 = self.fc4(x).reshape(1,1)
        x2 = self.fc4(x).reshape(1,1)
        return x1,x2

class Net0402(nn.Module):
    def __init__(self):
        super(Net0402, self).__init__()
        global NUM_FRAMES
        #self.conv1 = ConvLSTM(input_channels=1, hidden_channels=[64, 32, 32], kernel_size=3, step=5,
        #           effective_step=[4]).cuda()
        
        self.maxpool= nn.MaxPool3d((1,5,5),stride=(1,3,3))
        self.BD = nn.BatchNorm2d(16)
        self.drop = nn.Dropout(p=0.3)
        self.conv1 = ConvLSTM(input_size = (74,74), input_dim = 3, hidden_dim = [64, 32, 16], 
                              kernel_size = (3,3), num_layers = 3, batch_first=True)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3,stride=2)
        self.fc1 = nn.Linear(5*16*9*9, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        
        x = self.maxpool(x)
        #print (x.shape)
        x = self.conv1(x)     
        x = self.BD(x[0][0])    
        x = F.relu(F.max_pool2d(x,2))  
        #x = self.drop(x)
        x = self.BD(self.conv2(x))  
        x = F.relu(F.max_pool2d(x,2))  
        
        x = x.view(1,-1)
        #x = self.drop(x)
        x = F.relu(self.fc1(x))    
        
        
        x = F.relu(self.fc2(x))
        x = self.fc3(x).reshape(-1,1,2)
        return x




class Net14(nn.Module):
    def __init__(self):
        super(Net14, self).__init__()
        global NUM_FRAMES
        #self.conv1 = ConvLSTM(input_channels=1, hidden_channels=[64, 32, 32], kernel_size=3, step=5,
        #           effective_step=[4]).cuda()
        
        self.maxpool= nn.MaxPool3d((1,3,3),stride=(1,2,2))
        self.BD = nn.BatchNorm2d(16)
        self.drop = nn.Dropout(p=0.3)
        self.conv1 = ConvLSTM(input_size = (55,55), input_dim = 3, hidden_dim = [64, 32, 16], 
                              kernel_size = (3,3), num_layers = 3, batch_first=True)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3,stride=2)
        self.fc1 = nn.Linear(2880, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        
        x = self.maxpool(x)
        x = self.maxpool(x)
        #print (x.shape)
        x = self.conv1(x)     
        x = self.BD(x[0][0])    
        x = F.relu(F.max_pool2d(x,2))  
        #x = self.drop(x)
        x = self.BD(self.conv2(x))  
        x = F.relu(F.max_pool2d(x,2))  
        
        x = x.view(1,-1)
        #x = self.drop(x)
        x = F.relu(self.fc1(x))    
        
        
        x = F.relu(self.fc2(x))
        x = self.fc3(x).reshape(-1,1,2)
        return x

class Net015(nn.Module):
    def __init__(self):
        super(Net015, self).__init__()
        global NUM_FRAMES
        #self.conv1 = ConvLSTM(input_channels=1, hidden_channels=[64, 32, 32], kernel_size=3, step=5,
        #           effective_step=[4]).cuda()
        
        self.maxpool= nn.MaxPool3d((1,3,3),stride=(1,2,2))
        self.BD = nn.BatchNorm2d(16)
        self.drop = nn.Dropout(p=0.3)
        self.conv1 = ConvLSTM(input_size = (55,55), input_dim = 3, hidden_dim = [64, 32, 16], 
                              kernel_size = (3,3), num_layers = 3, batch_first=True)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3,stride=2)
        self.fc1 = nn.Linear(2880, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        
        x = self.maxpool(x)
        x = self.maxpool(x)
        #print (x.shape)
        x = self.conv1(x)     
        x = self.BD(x[0][0])    
        x = F.relu(F.max_pool2d(x,2))  
        #x = self.drop(x)
        x = self.BD(self.conv2(x))  
        x = F.relu(F.max_pool2d(x,2))  
        
        x = x.view(1,-1)
        #x = self.drop(x)
        x = F.relu(self.fc1(x))    
        
        
        x = F.relu(self.fc2(x))
        x = self.fc3(x).reshape(-1,1,2)
        return x


