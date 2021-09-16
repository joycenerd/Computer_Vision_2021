import torch
import torch.nn as nn

# Create CNN Model
class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        ## Convolution 0 , input_shape=(3,256,256)
        #self.cnn0 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=2, padding=3) #output_shape=(64,128,128)
        #self.bsn0 = nn.BatchNorm2d(16)
        #self.relu0 = nn.ReLU() # activation
        ## Max pool 1
        #self.maxpool0 = nn.MaxPool2d(kernel_size=2) #output_shape=(16,64,64)
        ## Convolution 1 
        #self.cnn1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1) #output_shape=(32,32,32)
        #self.bsn1 = nn.BatchNorm2d(32)
        #self.relu1 = nn.ReLU() # activation
        ## Max pool 1
        #self.maxpool1 = nn.MaxPool2d(kernel_size=2) #output_shape=(32,16,16)
        ## Convolution 2
        #self.cnn2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1) #output_shape=(64,8,8)
        #self.bsn2 = nn.BatchNorm2d(64)
        #self.relu2 = nn.ReLU() # activation
        ## Fully connected 1 ,#input_shape=(32*4*4)
        #self.fc1 = nn.Linear(64 * 8 * 8, 1024) 
        #self.fc2 = nn.Linear(1024, 128)
        #self.fc3 = nn.Linear(128, 15)
        #self.drop = nn.Dropout(0.01)
        #self.drop1 = nn.Dropout(0.02)
        # Convolution 0 , input_shape=(3,256,256)
        self.cnn0 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=2, padding=3) #output_shape=(64,128,128)
        self.bsn0 = nn.BatchNorm2d(16)
        self.relu0 = nn.ReLU() # activation
        # Max pool 1
        self.maxpool0 = nn.MaxPool2d(kernel_size=2) #output_shape=(16,64,64)
        # Convolution 1 
        self.cnn1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1) #output_shape=(32,32,32)
        self.bsn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU() # activation
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2) #output_shape=(32,16,16)
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1) #output_shape=(64,8,8)
        self.bsn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU() # activation
        # Fully connected 1 ,#input_shape=(32*4*4)
        self.fc1 = nn.Linear(64 * 8 * 8, 1024) 
        self.relu3 = nn.ReLU() # activation
        self.fc2 = nn.Linear(1024, 128)
        self.relu4 = nn.ReLU() # activation
        self.fc3 = nn.Linear(128, 15)
        self.drop = nn.Dropout(0.01)
        self.drop1 = nn.Dropout(0.02)
    
    def forward(self, x):
        # Convolution 0
        out = self.cnn0(x)
        out = self.bsn0(out)
        out = self.relu0(out)
        # Max pool 0
        out = self.maxpool0(out)
        out = self.drop(out)
        # Convolution 1
        out = self.cnn1(out)
        out = self.bsn1(out)
        out = self.relu1(out)
        # Max pool 1
        out = self.maxpool1(out)
        #out = self.drop1(out)
        # Convolution 2 
        out = self.cnn2(out)
        out = self.bsn2(out)
        out = self.relu2(out)
        # Max pool 2 
        #out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        # Linear function (readout)
        #out = self.drop(out)
        out = self.fc1(out)
        out = self.relu3(out)
        out = self.fc2(out)
        out = self.relu4(out)
        out = self.fc3(out)
        return out