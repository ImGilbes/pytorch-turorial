# manage a custom dataset on wine
# and handle multilabel (>= 3) prediction
import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn.modules.module import Module
from torch.utils.data import Dataset, DataLoader

class WineDataset(Dataset):

    def __init__(self) -> None:

        #perfect for loading from csv
        txt = np.loadtxt('./wine-dataset.csv', delimiter=",", dtype=np.float32, skiprows=1)

        #thse things depends on how the dataset nis written
        self.x = torch.from_numpy(txt[:, 1:]) #all rows, from the first column on
        self.y = torch.from_numpy(txt[:, [0]]) #all rows, only first column
        
        self.nsamples = self.x.shape[0]
        self.nfeatures = self.x.shape[1]

        super().__init__()

    #enables indexing dataset[0]
    def __getitem__(self, index: int):
        return self.x[index], self.y[index]

    #enables len(dataset)
    def __len__(self) -> int:
        return self.nsamples

    def numfeatures(self):
        return self.nfeatures
    
    def numclasses(self):
        return 3 #wine csv has 3 labes

dataset = WineDataset()
# print(dataset[0])

batch_size = 16
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
#loads the data in batches for better efficiency

# dataiter = iter(dataloader)
# data = dataiter.next()
# print(data)

nsamples = len(dataset)

# 2 layer feed forward nn for the wine csv dataset
class Multiclassifier(nn.Module):

    def __init__(self, ninput, nclasses):
        super(Multiclassifier, self).__init__()

        #winecsv has 13 features
        self.layer1 = nn.Linear(ninput, 8)
        self.layer2 = nn.Linear(8,4)
        self.layerout = nn.Linear(4,nclasses)
        # self.layerout = nn.Linear(ninput, nclasses)
        
        self.relu = nn.ReLU()
        # self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.2) #enahnaces learning process
        self.batchnorm1 = nn.BatchNorm1d(8) #normalization, enhances learning
        self.batchnorm2 = nn.BatchNorm1d(4)

    def forward(self, x):
        x = self.layer1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)

        x = self.layerout(x)

        return x

model = Multiclassifier(dataset.numfeatures(), dataset.numclasses())

# criterion = nn.MSELoss()
# criterion = nn.BCEWithLogitsLoss()
# criterion = nn.BCELoss()
criterion = nn.CrossEntropyLoss()
# criterion = nn.NLLLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#FIRST DIMENSION IS THE BATCH SIZE (VERTICAL)
#SECOND DIMENSION IS NUMBER OF FEATURES (HORIZONTAL)
softmax = nn.LogSoftmax(dim = 1) #DIM = 1 APPLIES HORIZONTALLY


def modelaccuracy(prediction, y):
    with torch.no_grad():

        predsoftmax = softmax(prediction)
        _, predtag = torch.max(predsoftmax, dim=1)
    
        predtag = predtag.numpy()
        predtag = predtag + 1 #i want labels from 1 to 3
                
        
        y = y.view(1, y.shape[0])
        y = y.numpy()
        y = y[0]
        p = (predtag == y)

        sum = 0
        for i in p:
            if i == True:
                sum = sum + 1

        accuracy = sum / len(p)
        # print(accuracy * 100)
    return accuracy


nepochs = 100
niters = math.ceil(nsamples/batch_size)
for epoch in range(nepochs):

    # for i, (xtrain, ytrain) in enumerate(dataloader):
    for xtrain, ytrain in dataloader:
        
        optimizer.zero_grad()

        prediction = model(xtrain)
        # print(epoch, prediction, ytrain)

        acc = modelaccuracy(prediction, ytrain)

        # _, predtag = torch.max(softmax(prediction), dim=1)
        # predtag = predtag.view(predtag.shape[0], 1)

        
        # prediction = softmax(prediction)
        # pp, pi = torch.max(prediction, dim=1)
        # prediction = torch.argmax(prediction, dim=1)
        # prediction = prediction.view(prediction.shape[0], 1)
        # print(prediction, ytrain)
        ytrain=ytrain.view(1,ytrain.shape[0])[0].long()
        ytrain = ytrain - 1 # because labels start from 0
        # print(ytrain)

        # loss = criterion(prediction.long(), ytrain.long())
        # loss = criterion(prediction, ytrain)
        loss = criterion(prediction, ytrain)
        # print(f'epoch {epoch}, loss {loss:.4f}, accuracy: {acc:0.4f}')
        
        loss.backward()

        optimizer.step()

        if epoch == 0 or epoch == 99:
            print(f'epoch {epoch}: loss= {loss.item():.6f}, accuracy={acc:.4f}')

        # if epoch %(nepochs/10) == 0:
        #     print(f'epoch: {epoch}, loss= {loss.item():.8f}')







