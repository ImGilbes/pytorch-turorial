import torch
import torch.nn as nn

#now tensors have to be 2d
x = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
y = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32) #expected w = 2

#the input to a model always has to be a tensor
x_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = x.shape
print(n_samples,n_features)

#THE FORWARD FUNCTION NOW BECOMES A MODEL!
input_size = n_features
output_size = n_features
# model = nn.Linear(input_size, output_size)

#create my custom model class by inheriting another model 
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

model = LinearRegression(input_size ,output_size)

learning_rate = 0.01
iter = 100

loss = nn.MSELoss() #the loss is not defined manually (Mean Squared Error)

#optimizer = torch.optim.SGD([w], lr=learning_rate, momentum=0.9)
#the parameters that sgd optimizes are passed as a list!
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)



print(f'Only Pytorch: prediction before training: f(5)={model(x_test)}')

for epoch in range(iter):

    y_pred = model(x)
    
    tmp_loss = loss(y, y_pred)
    
    tmp_loss.backward() #backprop and gradient are done automatically like this

    #update the weights
    optimizer.step()
    
    #zero gradients- clear them, they must not be accumulated
    optimizer.zero_grad()

    if epoch % (iter/10) == 0:
        [w, b] = model.parameters()
        print(f'{epoch}: w ={w[0][0].item()}, loss ={tmp_loss:.8f}')
        
print(f'Only Pytorch: prediction after training: f(5)={model(x_test)}')