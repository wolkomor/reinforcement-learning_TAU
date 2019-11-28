import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt


# Hyper Parameters
input_size = 784
num_classes = 10
num_epochs = 20
batch_size = 100
learning_rate = 1e-3
hidden_layer_size = 500

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

#Deep Neural Network Model
class DeepNet(nn.Module):
    def __init__(self, input_size, hidden_layer_size ,num_classes):
        super(DeepNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, num_classes)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out

def NN_model(net, criterion, optimizer, num_epochs, name):

    # Train the Model
    epoch_loss_list = []
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        running_corrects = 0
        for i, (images, labels) in enumerate(train_loader):
            logs = {}
            # Convert torch tensor to Variable
            images = Variable(images.view(-1, 28*28))
            labels = Variable(labels)

            # Forward + Backward + Optimize

            output = net(images) #feed the input and acquire the output from network
            loss = criterion(output, labels) #calculating the predicted and the expected loss
            loss.backward() #accumulates the gradient (by addition) for each parameter.
            optimizer.step() #performs a parameter update based on the current gradient
            # zero the parameter gradients
            optimizer.zero_grad()

            _, preds = torch.max(output, 1)
            running_loss += loss.detach() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_loss_list.append(epoch_loss)

    # Save the Model
    torch.save(net.state_dict(), name+'.pkl')

    return net, epoch_loss_list


def test_NN(net):
    # Test the Model
    net.eval()
    correct = 0
    total = 0

    for images, labels in test_loader:
        images = Variable(images.view(-1, 28*28))
        total += labels.size(0)
        #with torch.no_grad():
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    return (100 * correct / total)


#create the net
net = Net(input_size, num_classes)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

#train and test the model
trained_net, epoch_loss_list = NN_model(net, criterion, optimizer, num_epochs, "unoptimized")
test_NN(trained_net)

plt.plot(epoch_loss_list, label="Unoptimized Net")
plt.legend(frameon=False)

#create the net
deep_net = DeepNet(input_size, 500, num_classes)

#train and test the Deep model
trained_net, epoch_loss_list = NN_model(deep_net, criterion, optimizer, num_epochs, "DeepNet")
test_NN(trained_net)

plt.plot(epoch_loss_list, label="DeepNet")
plt.legend(frameon=False)


#Find a better optimization configuration
list_learning_rate = [0.1, 0.01, 1e-3]
optim_list = [torch.optim.SGD, torch.optim.Adam]
optimzer_list = []
for opt in optim_list:
    for i in list_learning_rate:
        optimzer_list.append([opt(net.parameters(), lr=i), " opt_"+str(opt).replace("<", "").replace(">", "")+" lr_"+str(i)])

max_accuracy = 0
for optimizer_conf in optimzer_list:
    # train and test the model
    i = 1
    name = optimizer_conf[1]
    trained_net, epoch_loss_list = NN_model(net, criterion, optimizer_conf[0], 100, name)
    accuracy = test_NN(trained_net)
    if accuracy >= max_accuracy:
        best_acc_model = trained_net
        max_accuracy = accuracy
        best_optimizer = optimizer_conf
    plt.plot(epoch_loss_list, label=str(i))
    plt.legend(frameon=False)
    i += 1

plt.show()
plt.savefig('loss_NN.png', bbox_inches='tight')

# Print best model's state_dict
print("Model's state_dict:")
for param_tensor in best_acc_model.state_dict():
   print(param_tensor, "\t", best_acc_model.state_dict()[param_tensor].size())

# Print best optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in best_optimizer.state_dict():
   print(var_name, "\t", best_optimizer.state_dict()[var_name])