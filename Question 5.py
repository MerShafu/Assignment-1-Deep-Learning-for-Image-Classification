
import torch
from torch import nn

import torchvision
from torchvision import datasets, models
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

transform = transforms.Compose(
    [#transforms.Resize((40,40)),
    #  transforms.Augmix(),
    #  transforms.CenterCrop(),
     transforms.Resize((224,224)),
     transforms.ToTensor(),
     transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]) ]
    #  transforms.ToTensor(), #convert to 4D tensor - represent matrix tensor [B,c,h,w] - add new dimension: Batch 'B'
    #  transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]  #standard deviation
)

# train_data = datasets.CIFAR10(
#     root = 'data',
#     train = True,
#     download = True,
#     transform = transform,

# )


# test_data = datasets.CIFAR10(
#     root = 'data',
#     train = False,
#     download = True,
#     transform = transform,

# )

train_dir = r"C:\Users\User\Downloads\mv proj\DS5\train"
test_dir = r"C:\Users\User\Downloads\mv proj\DS5\test"

train_data = datasets.ImageFolder(root=train_dir,
                                  transform=transform)

test_data = datasets.ImageFolder(root=test_dir,
                                 transform=transform)

len(train_data)

img, label = train_data[0]
train_data.classes

class_names = train_data.classes
class_names

from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True) # change batch_size to reduce memory error

test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False) #turn our images into batches

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_dataloader)
images, labels = next(dataiter)

# show images
#imshow(torchvision.utils.make_grid(images))
# print labels
#print(' '.join(f'{class_names[labels[j]]:5s}' for j in range(4)))

train_imgBatch, train_labelBatch = next(iter(train_dataloader))

train_imgBatch.shape

flatten_layer = nn.Flatten()

input = flatten_layer(train_imgBatch)

input.shape

linear1 = nn.Linear(in_features=150528,out_features=8500) #(input for this layer, define yourself(neurons))
#for now the max tried out_features=8500, out_features=9000 will crashed

linear1_activation = linear1(input)
linear1_activation.shape

linear2 = nn.Linear(in_features=8500,out_features=500)

linear2_activation = linear2(linear1_activation)
linear2_activation.shape



import time
from tqdm.auto import tqdm

def train_and_validate(model, loss_criterion, optimizer, train_dataloader, test_dataloader, epochs=25, device='cuda'):
    '''
    Function to train and validate
    Parameters
        :param model: Model to train and validate
        :param loss_criterion: Loss Criterion to minimize
        :param optimizer: Optimizer for computing gradients
        :param train_dataloader: DataLoader for training data
        :param test_dataloader: DataLoader for test/validation data
        :param epochs: Number of epochs (default=25)
        :param device: Device to perform computations ('cuda' or 'cpu')

    Returns
        model: Trained Model with best validation accuracy
        history: (dict object): Having training loss, accuracy and validation loss, accuracy
    '''

    start = time.time()
    history = []
    best_acc = 0.0

    for epoch in tqdm(range(epochs)):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))

        model.train()

        train_loss = 0.0
        train_acc = 0.0

        valid_loss = 0.0
        valid_acc = 0.0

        for i, (inputs, labels) in enumerate(train_dataloader):

            inputs = inputs.to(device)
            labels = labels.to(device)

            # Clean existing gradients
            optimizer.zero_grad()

            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)

            # Compute loss
            loss = loss_criterion(outputs, labels)

            # Backpropagate the gradients
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Compute the total loss for the batch and add it to train_loss
            train_loss += loss.item() * inputs.size(0)

            # Compute the accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            # Compute total accuracy in the whole batch and add to train_acc
            train_acc += acc.item() * inputs.size(0)

        # Validation - No gradient tracking needed
        with torch.no_grad():

            model.eval()

            # Validation loop
            for j, (inputs, labels) in enumerate(test_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass - compute outputs on input data using the model
                outputs = model(inputs)

                # Compute loss
                loss = loss_criterion(outputs, labels)

                # Compute the total loss for the batch and add it to valid_loss
                valid_loss += loss.item() * inputs.size(0)

                # Calculate validation accuracy
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to valid_acc
                valid_acc += acc.item() * inputs.size(0)


        # Find average training loss and training accuracy
        avg_train_loss = train_loss / len(train_dataloader.dataset)
        avg_train_acc = train_acc / len(train_dataloader.dataset)

        # Find average validation loss and training accuracy
        avg_test_loss = valid_loss / len(test_dataloader.dataset)
        avg_test_acc = valid_acc / len(test_dataloader.dataset)

        history.append([avg_train_loss, avg_test_loss, avg_train_acc, avg_test_acc])

        epoch_end = time.time()

        print("Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation : Loss : {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(epoch, avg_train_loss, avg_train_acc * 100, avg_test_loss, avg_test_acc * 100, epoch_end - epoch_start))

        # Save if the model has best accuracy till now
        if avg_test_acc > best_acc:
            best_acc = avg_test_acc
            best_model = model
            torch.save(best_model, 'best_model.pt')

    return best_model, history



"""# **CNN Model**"""

conv1 = nn.Conv2d(in_channels=3,out_channels=10,kernel_size=5,stride=1,padding=0)

out_conv1 = conv1(train_imgBatch)

out_conv1.shape

train_imgBatch.shape

maxpool1 = nn.MaxPool2d(kernel_size=3)

out_maxpool = maxpool1(out_conv1)

out_maxpool.shape

# DEFINE OUR MODEL
class CNNmodel(nn.Module): #define/initiate the layers
  def __init__(self):
    super(CNNmodel,self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5)
    self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
    self.conv2 = nn.Conv2d(32,64,5)
    self.conv3 = nn.Conv2d(64,128,5)
    self.conv4 = nn.Conv2d(128,256,3)
    self.fc1 = nn.Linear(73728 ,120)
    self.fc2 = nn.Linear(120,10)
    self.relu = nn.ReLU()
    self.flatten = nn.Flatten()
    self.batchnorm1 = nn.BatchNorm2d(32) #num_features = output from previous layer
    self.batchnorm2 = nn.BatchNorm2d(64)
    self.dropout = nn.Dropout(0.4) #probability value = drop 50% the neurons

  def forward(self,x): #feedforward
    x = self.conv1(x)
    x = self.relu(x)
    #x = self.batchnorm1(x)
    x = self.maxpool(x)
    
    x = self.conv2(x)
    x = self.relu(x)
    #x = self.batchnorm2(x)
    x = self.maxpool(x)
    
    x = self.conv3(x)
    x = self.relu(x)
    x = self.maxpool(x)
    # x = self.conv4(x)
    # x = self.relu(x)
    x = self.flatten(x)
    x = self.fc1(x) #fully connected
    #x = self.dropout(x)
    x = self.relu(x)
    out = self.fc2(x)

    return out

class DCNNmodel(nn.Module):
    def __init__(self):
        super(DCNNmodel, self).__init__()
        self.depthwise_conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, groups=3)
        self.pointwise_conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.depthwise_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, groups=32)
        self.pointwise_conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1)
        self.depthwise_conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, groups=64)
        self.pointwise_conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 26 * 26, 120)
        self.fc2 = nn.Linear(120, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise_conv1(x)
        x = self.relu(x)
        x = self.pointwise_conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.depthwise_conv2(x)
        x = self.relu(x)
        x = self.pointwise_conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.depthwise_conv3(x)
        x = self.relu(x)
        x = self.pointwise_conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        out = self.fc2(x)

        return out

class GCNNmodel(nn.Module):
    def __init__(self):
        super(GCNNmodel, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, groups=3)  # Adjust out_channels
        self.conv1_2 = nn.Conv2d(in_channels=3, out_channels=33, kernel_size=3, groups=3)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(33, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 26 * 26, 120)
        self.fc2 = nn.Linear(120, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        out = self.fc2(x)

        return out

model = DCNNmodel()

# loss and optimizer

# cross-entropy loss

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.9)

model.to('cuda')

num_epochs = 20
trained_CNNmodel, history = train_and_validate(model,loss_fn,optimizer,
                                            train_dataloader,test_dataloader,
                                            num_epochs)

#Analyze the loss curve

def plot_loss(history):
  history = np.array(history)
  plt.plot(history[:,0:2])
  plt.legend(['Tr Loss', 'Val Loss'])
  plt.xlabel('Epoch Number')
  plt.ylabel('Loss')
  plt.ylim(0,3)
  # plt.savefig('cifar10_loss_curve.png')
  plt.show()

plot_loss(history)

def plot_accuracy(history):
  history = np.array(history)
  plt.plot(history[:,2:4])
  plt.legend(['Tr Accuracy', 'Val Accuracy'])
  plt.xlabel('Epoch Number')
  plt.ylabel('Accuracy')
  plt.ylim(0,1)
  # plt.savefig('cifar10_accuracy_curve.png')
  plt.show()

plot_accuracy(history)