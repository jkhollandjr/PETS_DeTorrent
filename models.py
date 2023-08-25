import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch.nn as nn
import torch.nn.functional as F
import math

#model for testing a basic WF attack 
class CONV_ATTACK(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, 4, padding=0),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(.2),
            nn.Conv1d(32, 32, 4, padding=0),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(.2),
            nn.MaxPool1d(4, stride=2, padding=2),
            nn.Dropout(.1)
        )

        self.layer2 = nn.Sequential(
                nn.Conv1d(32, 64, 4, padding=0),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(.2),
                nn.Conv1d(64, 64, 4, padding=0),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(.2),
                nn.MaxPool1d(4, stride=2, padding=2),
                nn.Dropout(.1)
        )

        self.layer3 = nn.Sequential(
                nn.Conv1d(64, 64, 4, padding=0),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(.2),
                nn.Conv1d(64, 64, 4, padding=0),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(.2),
                nn.MaxPool1d(4, stride=2, padding=0),
                nn.Dropout(.1)
        )

        self.layer4 = nn.Sequential(
                nn.Conv1d(128, 256, 8, padding=0),
                nn.BatchNorm1d(256), nn.ReLU(),
                nn.Conv1d(256, 256, 8, padding=0),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.MaxPool1d(3, stride=4, padding=2),
                nn.Dropout(.1)
        )

        self.fc1 = nn.Linear(1664, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(.5)
        
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.dropout2 = nn.Dropout(.7)
        
        self.fc_final = nn.Linear(512, 95)

        self.test_conv = nn.Conv1d(1, 32, 8, padding=3)
        self.lrelu = nn.LeakyReLU(.2)

        self.conv2d = nn.Conv2d(1, 32, (2,5), padding=(0,2))

        self.first_bn = nn.BatchNorm1d(32)
        self.first_relu = nn.LeakyReLU(.2)
        self.maxpool = nn.MaxPool1d(3, stride=4, padding=2)
        self.drop = nn.Dropout(.1)
 
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        out = torch.flatten(out, start_dim=1)
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.lrelu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.dropout2(out)

        out = self.fc_final(out)
        out = F.softmax(out)

        return out

#Model for embedding trace and guessing at embedding (discriminator)
class CUSTOM_CONV_EMBEDDER(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 64, 4, padding=0),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(.2),
            nn.Conv1d(64, 64, 4, padding=0),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(.2),
            nn.MaxPool1d(4, stride=2, padding=2),
            nn.Dropout(.1)
        )

        self.layer2 = nn.Sequential(
                nn.Conv1d(64, 64, 4, padding=0),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(.2),
                nn.Conv1d(64, 64, 4, padding=0),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(.2),
                nn.MaxPool1d(4, stride=2, padding=2),
                nn.Dropout(.1)
        )

        self.layer3 = nn.Sequential(
                nn.Conv1d(64, 128, 4, padding=0),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(.2),
                nn.Conv1d(128, 128, 4, padding=0),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(.2),
                nn.MaxPool1d(4, stride=2, padding=0),
                nn.Dropout(.1)
        )

        self.layer4 = nn.Sequential(
                nn.Conv1d(128, 256, 4, padding=0),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 256, 4, padding=0),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.MaxPool1d(3, stride=4, padding=2),
                nn.Dropout(.1)
        )

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(1, 32, (2,6)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, 6),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(.1)
        )
             

        self.fc1 = nn.Linear(3328, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(.5)
        
        self.fc2 = nn.Linear(2048, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.dropout2 = nn.Dropout(.5)
        
        self.fc_final = nn.Linear(512, 256)

        self.test_conv = nn.Conv1d(1, 32, 8, padding=3)
        self.lrelu = nn.LeakyReLU(.2)
        self.conv2d = nn.Conv2d(1, 32, (2,5), padding=(0,2))

        self.first_bn = nn.BatchNorm1d(32)
        self.first_relu = nn.LeakyReLU(.2)
        self.maxpool = nn.MaxPool1d(3, stride=4, padding=2)
        self.drop = nn.Dropout(.1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        out = torch.flatten(out, start_dim=1)
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.lrelu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.dropout2(out)

        out = self.fc_final(out)

        return out

#Model for LSTM generator that creates WF defense
class LSTM_ATTACK(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(LSTM_ATTACK, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=1)
        self.fc1 = nn.Linear(hidden_size, 32) 
        self.fc2 = nn.Linear(32, 1)

        self.lrelu = torch.nn.LeakyReLU(.1)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.relu = torch.nn.ReLU()


    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)

        out = self.relu(self.bn2(self.fc2(self.lrelu(self.bn1(self.fc1(lstm_out))))))

        return out

#Old approach that would pre-generate defense
class GENERATOR(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = torch.nn.Linear(32, 128)
        self.fc2 = torch.nn.Linear(128, 256)
        self.fc3 = torch.nn.Linear(256, 256)

        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(256)

        self.lrelu = torch.nn.LeakyReLU(.5)

        self.dropout = nn.Dropout(.2)

    def forward(self, noise):
        out = self.bn1(self.lrelu(self.fc1(noise)))
        out = self.bn2(self.lrelu(self.fc2(out)))
        out = F.relu(self.fc3(out))
        return out

#Discriminator in FC setting
class FC_CONV_EMBEDDER(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, 4, padding=0),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(.2),
            nn.Conv1d(32, 32, 4, padding=0),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(.2),
            nn.MaxPool1d(4, stride=2, padding=2),
            nn.Dropout(.1)
        )

        self.layer2 = nn.Sequential(
                nn.Conv1d(32, 64, 4, padding=0),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(.2),
                nn.Conv1d(64, 64, 4, padding=0),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(.2),
                nn.MaxPool1d(4, stride=2, padding=2),
                nn.Dropout(.1)
        )

        self.layer3 = nn.Sequential(
                nn.Conv1d(64, 64, 4, padding=0),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(.2),
                nn.Conv1d(64, 64, 4, padding=0),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(.2),
                nn.MaxPool1d(4, stride=2, padding=0),
                nn.Dropout(.1)
        )

        self.layer4 = nn.Sequential(
                nn.Conv1d(128, 256, 8, padding=0),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 256, 8, padding=0),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.MaxPool1d(3, stride=4, padding=2),
                nn.Dropout(.1)
        )

        self.fc1 = nn.Linear(3328, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(.1)
        
        self.fc2 = nn.Linear(1024, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        self.dropout2 = nn.Dropout(.1)
        
        self.fc_final = nn.Linear(128, 1)

        self.test_conv = nn.Conv1d(1, 32, 8, padding=3)
        self.lrelu = nn.LeakyReLU(.2)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        x = x.reshape(40, 1, 256)
        y = y.reshape(40, 1, 256)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        first_out = torch.flatten(out, start_dim=1)

        out = self.layer1(y)
        out = self.layer2(out)
        out = self.layer3(out)
        second_out = torch.flatten(out, start_dim=1)

        all_out = torch.cat((first_out, second_out), dim=1)

        out = self.fc1(all_out)
        out = self.bn1(out)
        out = self.lrelu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.dropout2(out)

        out = self.fc_final(out)
        out = self.Sigmoid(out)

        return out
