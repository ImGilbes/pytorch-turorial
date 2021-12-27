import torch
import numpy as np
#tensors are n dimensional

#tensor init functions
t = torch.empty(1) #one element tensor
t = torch.empty(0) #empty
t= torch.zeros(3, 2, 2)
t= torch.ones(3, 2, 2)
t= torch.rand(3, 2, 2)
t= torch.empty(3, 2, dtype=torch.double) #can always specify the datatype

#these operations are element wise
x = torch.rand(2, 2)
y = torch.ones(2, 2)
z = torch.mul(x, y) #this is element wise!
y.mul_(x) #just  a synonym

#slicing operation works
x = torch.rand(4, 4)
print(x[:, 0]) #print col 0
print(x[0, :]) #print row 0

#conversion of a tensor to numpy array
x = torch.ones(3)
y = x.numpy()
#but they will share the same memory location (if they are on the cpu)
x.add_(1) #will also modify y


x = torch.randn(3, requires_grad=True)
y = x + 2 #y is the output of the computational node
z = y * y * 2 #grad funct is multplication forward
z = z.mean() #the gradient function is now th mean function

#carry out the gradients
z.backward() #dz/dx
print(x.grad) #can only access the grad attribute of leaf tensors

w = torch.ones(3, requires_grad=True)

for epoch in range(3):
    output = (w * 3).sum() #scalar output
    
    output.backward() #doutput/dw
    w.grad.zero_() #clear the gradients at each iteration, otherwise they will be accumulated at each iteration
    
x = torch.tensor(1.0)
y = torch.tensor(2.0)
w= torch.tensor(1.0, requires_grad=True)

#forward pass
y_hat = w * x
loss = (y_hat - y)**2

print(loss)

#backward pass
loss.backward()
print(w.grad)

#update weights


x = np.array([1, 2, 3, 4], dtype=np.float32)
y = np.array([1, 2, 3, 4], dtype=np.float32)

# w = torch.tensor(0.0, requires_grad=True)
w = 0.0

def forward(x): #y = 2x, approximate w st loss is minimized
    return w * x #one layer

def loss(y, y_predict): #mean squared error loss
    return ((y_predict - y)**2).mean()

#gradient
def gradient(x, y, y_predict):
    return np.dot(2*x, y_predict - y).mean()

print(f'prediction before trainig: f(5)= {forward(5):.3f}')

#training of the net

learning_rate = 0.01
iter = 50

for epoch in range(iter):
    
    y_pred = forward(x) #forward pass throught the network
    
    l = loss(y, y_pred) #epoch loss after the current pass
    
    dw = gradient(x,y,y_pred) #backward gradient computation, assign the gradient to a variable
    
    w -= learning_rate * dw #weights update
    
    if epoch % 5 == 0:
        print(f'{epoch}: w ={w}, loss ={l:.8f}')

print(f'Predictin after training: f(5)= {forward(5):.3f}')


#now do everything with pytorch

x = torch.tensor([1,2,3,4], dtype=torch.float32)
y = torch.tensor([1,2,3,4], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True) #requires gradient if you use only torch

learning_rate = 0.01
iter = 50

def forward2(x):
    return w * x

def loss2(y, y_predicted):
    return ((y_predicted - y)**2).mean()

print(f'Only Pytorch: prediction before training: f(5)={forward2(5)}')

for epoch in range(iter):

    y_pred = forward2(x)
    
    tmp_loss = loss2(y, y_pred)
    
    tmp_loss.backward() #backprop and gradient are done automatically like this

    #update the weights
    with torch.no_grad(): #it sshould not be part of the computational graph 
        w -= learning_rate * w.grad
    
    #zero gradients- clear them, they must not be accumulated
    w.grad.zero_()

    if epoch % (iter/10) == 0:
        print(f'{epoch}: w ={w}, loss ={tmp_loss:.8f}')
        
print(f'Only Pytorch: prediction after training: f(5)={forward2(5)}')

#with pytorch it is not as exact as with numpy








