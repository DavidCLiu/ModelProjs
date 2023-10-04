import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def loadDataset():
    train = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

    # Testing data
    test = datasets.MNIST("", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

    trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
    testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)
    return (trainset,testset)

class handwriteClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(28*28,64)
        self.nonlin1 = nn.ReLU()
        self.hidden2 = nn.Linear(64,128)
        self.nonlin2 = nn.ReLU()
        self.hidden3 = nn.Linear(128,128)
        self.nonlin3 = nn.ReLU()
        # self.hidden4 = nn.Linear(128,128)
        # self.nonlin4 = nn.ReLU()
        self.hidden5 = nn.Linear(128,64)
        self.nonlin5 = nn.ReLU()
        self.hidden6= nn.Linear(64,10)
        self.nonlin6 = nn.Softmax(dim = 1)
    
    def forward(self,x):
        x = self.nonlin1(self.hidden1(x))
        x = self.nonlin2(self.hidden2(x))
        x = self.nonlin3(self.hidden3(x))
        # x = self.nonlin4(self.hidden4(x))
        x = self.nonlin5(self.hidden5(x))
        x = self.nonlin6(self.hidden6(x))
        return x

def train(trainset):
    model = handwriteClassifier()
    num_epochs = 10

    optimizer = optim.SGD(model.parameters(),lr = 0.008)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for data in trainset:
            X,y = data
            # print(X)
            pred = model.forward(X.view(-1,28*28))
            # print(pred)
            loss = loss_func(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch %2 == 0:
            print(f'Finished epoch {epoch}, loss: {loss}')
    return model

def evaluate(model, testset):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testset:
                X, y = data
                output = model(X.view(-1, 784))
                for idx, i in enumerate(output):
                    if torch.argmax(i) == y[idx]:
                        correct += 1
                    total +=1
    print(correct/total)

if __name__ == '__main__':
    trainset, testset = loadDataset()
    print(type(trainset))

    model = train(trainset)
    evaluate(model, testset)

