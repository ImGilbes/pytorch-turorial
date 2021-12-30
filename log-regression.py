import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

bc = datasets.load_breast_cancer() #binary classification problem
x, y = bc.data, bc.target

nsamples, nfeat = x.shape

#automatically split between training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=111)

#i don't know what this is for
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)

# transform into pytorch tensors
xtrain = torch.from_numpy(xtrain.astype(np.float32))
xtest = torch.from_numpy(xtest.astype(np.float32))
ytrain = torch.from_numpy(ytrain.astype(np.float32))
ytest = torch.from_numpy(ytest.astype(np.float32))

ytrain = ytrain.view(ytrain.shape[0], 1) #make a column vector
ytest = ytest.view(ytest.shape[0], 1)

# model: f(x) = wx + b and a Sigmoid at the end
class LogisticRegression(nn.Module):
    def __init__(self, ninput):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(ninput, 1) #one layer, linear; one output

    def forward(self, x):
        return torch.sigmoid(self.linear(x)) #pass through the layers (only one) and return sigmoid

model = LogisticRegression(nfeat)

#Binary Cross entropy loss
# f = (-1/n)* sumi( yi*log(p(yi)) + (1-yi)*log(1-p(yi)) ), p predicted probability
loss = nn.BCELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

nepochs = 100
for epoch in range(nepochs):

    prediction = model(xtrain)

    error = loss(prediction, ytrain)

    error.backward()

    optimizer.step()

    optimizer.zero_grad()

    if epoch %(nepochs/10) == 0:
        print(f'epoch: {epoch}, loss= {error.item():.8f}')

#when you re done you do things with no grad, otherwise stuff gets in the computational graph
with torch.no_grad():
    ypred = model(xtest)

    ypredcls = ypred.round() # this is done to classify in 0 or 1!

    acc = ypredcls.eq(ytest).sum() / float(ytest.shape[0]) #count the right guesses
    print(f'accuracy={acc:.8f}')

    #this works, but it's a mess to see :C
    yplot = ypred.detach().numpy()
    xnp = xtest.cpu().detach().numpy()
    ynp = ytest.cpu().detach().numpy()

    plt.plot(xnp, ynp, '.')
    plt.plot(xnp, yplot, 'x')
    plt.show()










