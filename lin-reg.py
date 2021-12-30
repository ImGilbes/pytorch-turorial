import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# create random dataset
xnp, ynp = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
x = torch.from_numpy(xnp.astype(np.float32))
y = torch.from_numpy(ynp.astype(np.float32))

#now x and y contain only one row, whereas they should be a column vector
y = y.view(y.shape[0], 1) #now y is a column vector [[][][]]

samples, features = x.shape

insize = features
outsize = 1
model = nn.Linear(insize, outsize)

learning_rate = 0.1
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

nepochs = 100
for epoch in range(nepochs):

    prediction = model(x)
    error = loss(prediction, y)

    #backprop
    error.backward()

    #update
    optimizer.step()

    optimizer.zero_grad()

    if epoch %(nepochs/10) == 0:
        print(f'epoch: {epoch}, loss= {error.item():.8f}')


pred = model(x).detach().numpy() #generates a new tensor without grandient required thing
plt.plot(xnp, ynp, 'ro') #plot dataset
plt.plot(xnp, pred, 'b') #plot current trained model
plt.show()





