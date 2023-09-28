import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def loadDataset():
    dataset = np.loadtxt('TESLA.csv', delimiter=',', dtype=str)
    X = dataset[:,1:6].astype(float)
    Y = dataset[:,6].astype(float)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(Y, dtype=torch.float32)
    return (X,y)

class TeslaPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(5,8)
        self.nonlin1 = nn.ReLU()
        self.hidden2 = nn.Linear(8,5)
        self.nonlin2 = nn.ReLU()
        self.hidden3 = nn.Linear(5,1)
        self.nonlin3 = nn.Sigmoid()
    
    def forward(self,x):
        x = self.nonlin1(self.hidden1(x))
        x = self.nonlin2(self.hidden2(x))
        x = self.nonlin3(self.hidden3(x))
        return x

if __name__ == '__main__':
    inputs, outputs = loadDataset()
    # print(inputs, outputs)

    # model = TeslaPredictor()
    # print(model.parameters())

    model = nn.Sequential(
    nn.Linear(5, 12),
    nn.ReLU(),
    nn.Linear(12, 5),
    nn.ReLU(),
    nn.Linear(5, 1),
    nn.Sigmoid())

    #Define a loss function
    loss_func = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    n_epochs = 100
    batch_size = 10

    for epoch in range(n_epochs):
        for i in range(0, len(inputs), batch_size):
            inputBatch = inputs[i:i + batch_size]
            output_pred = model(inputBatch)
            # print(output_pred)
            outputBatch = outputs[i:i+batch_size]
            # print(outputBatch)
            
            loss = loss_func(output_pred, outputBatch.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch %10 == 0:
            print(f'Finished epoch {epoch}, loss: {loss}')
        