import torch
import torch.nn as nn

x = torch.tensor([1,2,3,4], dtype=torch.float32)
# y = torch.tensor([1,2,3,4], dtype=torch.float32) #expected w = 1
y = torch.tensor([2,4,6,8], dtype=torch.float32) #expected w = 2
# y = torch.tensor([23, 11 , 20, 4], dtype=torch.float32) #mess with the expected w
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True) #requires gradient if you use only torch

learning_rate = 0.01
iter = 100

loss = nn.MSELoss() #the loss is not defined manually (Mean Squared Error)

#optimizer = torch.optim.SGD([w], lr=learning_rate, momentum=0.9)
optimizer = torch.optim.SGD([w], lr=learning_rate)

def forward(x):
    return w * x


print(f'Only Pytorch: prediction before training: f(5)={forward(5)}')

for epoch in range(iter):

    y_pred = forward(x)
    
    tmp_loss = loss(y, y_pred)
    
    tmp_loss.backward() #backprop and gradient are done automatically like this

    #update the weights
    optimizer.step()
    
    #zero gradients- clear them, they must not be accumulated
    optimizer.zero_grad()

    if epoch % (iter/10) == 0:
        print(f'{epoch}: w ={w}, loss ={tmp_loss:.8f}')
        
print(f'Only Pytorch: prediction after training: f(5)={forward(5)}')