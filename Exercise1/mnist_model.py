import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

# Hyper Parameters
input_size = 784
num_classes = 10
batch_size = 100
num_epochs = 100


# MNIST Dataset
train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Neural Network Model
class Net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        return out


def train(network, loss_fn, opt, epochs, name):
    # Train the Model
    train_loss = []
    for epoch in range(epochs):
        epoch_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            # Convert torch tensor to Variable
            images = Variable(images.view(-1, 28*28))
            labels = Variable(labels)

            # zero the parameter gradients
            opt.zero_grad()

            # Forward + Backward + Optimize
            # TODO: implement training code
            y_pred = network(images)
            loss = loss_fn(y_pred, labels)
            # print('epoch: ', epoch, ' loss: ', loss.item())
            epoch_loss += loss.item()
            loss.backward()
            opt.step()
        print('epoch: ', epoch, ' loss: ', epoch_loss)
        train_loss += [epoch_loss]

    # Save the Model
    torch.save(network.state_dict(), 'model_{}.pkl'.format(name))
    return train_loss


def test(network):
    # Test the Model
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = Variable(images.view(-1, 28 * 28))
            # TODO: implement evaluation code - report accuracy
            y_pred = network(images)
            _, predicted = torch.max(y_pred.data, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = (100 * correct / total)
    print('Accuracy of the network on the 10000 test images: %d %%' % accuracy)

    return accuracy


# Basic Model:
learning_rate = 1e-3

# Loss and Optimizer
net = Net(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
model1_loss = train(net, criterion, optimizer, num_epochs, 'unoptimized')
model1_acc = test(net)


# Optimized Model:
learning_rate = 0.05

# Loss and Optimizer
net = Net(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adagrad(net.parameters(), lr=learning_rate)
model2_loss = train(net, criterion, optimizer, num_epochs, 'optimized')
model2_acc = test(net)


# Deep Model:
learning_rate = 0.05
hidden_size = 500

# Loss and Optimizer
net = torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_size),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_size, num_classes),
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
model3_loss = train(net, criterion, optimizer, num_epochs, 'deep')
model3_acc = test(net)


plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot(list(range(num_epochs)), model1_loss, label='unoptimized, accuracy {}%'.format(model1_acc))
plt.plot(list(range(num_epochs)), model2_loss, label='optimized, accuracy {}%'.format(model2_acc))
plt.plot(list(range(num_epochs)), model3_loss, label='deep, accuracy {}%'.format(model3_acc))
plt.legend()
plt.savefig('mnist.png')
