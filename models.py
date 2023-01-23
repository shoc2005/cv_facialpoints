# # TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as INIT
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):
    
    def initWeights(self, layer):
        INIT.uniform_(layer.bias.data, 0.0,0.005) # inits weights from the uniform distribution
        INIT.uniform_(layer.weight.data, 0.0,0.005) # inits weights from the uniform distribution
        pass

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # an input image is 1x224x224 tensor
        conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.initWeights(conv1)
        
        conv2 = nn.Conv2d(32, 32, 5, padding=2)
        self.initWeights(conv2)
        
        conv3 = nn.Conv2d(32, 64, 3, padding=2)
        self.initWeights(conv3)
        
        conv4 = nn.Conv2d(64, 64, 3, padding=2)
        self.initWeights(conv4)
        
        conv5 = nn.Conv2d(64, 128, 2, padding=2)
        self.initWeights(conv5)
        
        conv6 = nn.Conv2d(128, 256, 2, padding=2)
        self.initWeights(conv6)
        
        conv7 = nn.Conv2d(256, 256, 1, padding=2)
        self.initWeights(conv7)
        
        maxpool2 = nn.MaxPool2d(3, stride=2, padding=0)
        maxpool = nn.MaxPool2d(2, stride=2, padding=1)
        
        dense1 = nn.Linear(6400, 2500)
        #print(type(self.dense1))
        INIT.xavier_uniform_(dense1.weight.data, gain=1.0) # init as Glorot initialization
        
        dense2 = nn.Linear(2500, 2500)
        INIT.xavier_uniform_(dense2.weight.data, gain=1.0) # init as Glorot initialization
        
        dense3 = nn.Linear(2500, 2 * 68)
        INIT.xavier_uniform_(dense3.weight.data, gain=1.0) # init as Glorot initialization
        
        self.features = nn.Sequential(
            conv1,
            nn.ELU(),
            maxpool2,
            nn.Dropout2d(p=0.1),
            conv2,
            nn.ELU(),
            maxpool,
            nn.Dropout2d(p=0.2),
            conv3,
            nn.ELU(),
            maxpool2,
            nn.Dropout2d(p=0.3),
            conv4,
            nn.ELU(),
            maxpool,
            nn.Dropout2d(p=0.4),
            conv5,
            nn.ELU(),
            maxpool2,
            nn.Dropout2d(p=0.5),
            conv6,
            nn.ELU(),
            maxpool,
            nn.Dropout2d(p=0.5),
            conv7,
            nn.ELU(),
            maxpool2,
            nn.Dropout2d(p=0.5)
        )
        self.classifier = nn.Sequential(
            dense1,
            nn.ELU(),
            nn.Dropout2d(p=0.6),
            dense2,
            nn.ELU(),
            nn.Dropout2d(p=0.6),
            dense3
        ) 
        
        
        
    def forward(self, x):
        
        out = self.features(x)
        #print(out.size())
        out = torch.flatten(out, start_dim=1)
        #print(out.size())
        out = self.classifier(out)

        #x = self.maxpool2(F.elu(self.conv1(x)))
        #x = self.dropout1(x)
        #print("Size:",x.shape)
        
        #x = self.maxpool(F.elu(self.conv2(x)))
        #x = self.dropout2(x)
        #print("Size:",x.shape)
        
        #x = self.maxpool2(F.elu(self.conv3(x)))
        #x = self.dropout3(x)
        
        #print("Size:",x.shape)
        #x = self.maxpool(F.elu(self.conv4(x)))
        #x = self.dropout4(x)
        
        #print("Size:",x.shape)
        # flattening x
        #x = x.view(x.size(0), -1)
        #print("Size:",x.shape)
        #x = F.elu(self.dense1(x))
        #x = self.dropout5(x)
        #print("Size:",x.shape)
        
        #x = F.elu(self.dense2(x))
        #x = self.dropout6(x)  
        #print("Size:",x.shape, self.dense3(x).weight.data.shape)
        
        #x = self.dense3(x)
        
        return out
